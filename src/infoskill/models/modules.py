from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: Tensor) -> Tensor:
        scale = value.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (value * scale.to(value.dtype)) * self.weight


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.SiLU(),
            nn.Linear(width * 4, width),
        )

    def forward(self, state: Tensor, memory: Tensor, memory_valid: Tensor) -> Tensor:
        attended, _ = self.attention(
            self.query_norm(state),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_valid,
            need_weights=False,
        )
        state = state + attended
        return state + self.ff(self.ff_norm(state))


@dataclass(frozen=True)
class CompressorOutputs:
    state_summary: Tensor
    posterior_mu: Tensor
    posterior_logvar: Tensor
    latent: Tensor
    epsilon: Tensor


class InfoSkillCompressor(nn.Module):
    """State-token queries cross-attend to separately delimited candidate skills."""

    def __init__(
        self,
        semantic_width: int,
        *,
        model_width: int = 256,
        latent_dim: int = 32,
        attention_layers: int = 2,
        attention_heads: int = 8,
        skill_kinds: int = 3,
        max_candidate_skills: int = 17,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if model_width % attention_heads:
            raise ValueError("model_width must be divisible by attention_heads")
        self.state_projection = nn.Linear(semantic_width, model_width)
        self.skill_projection = nn.Linear(semantic_width, model_width)
        self.kind_embedding = nn.Embedding(skill_kinds, model_width)
        self.boundary_embedding = nn.Embedding(max_candidate_skills, model_width)
        self.blocks = nn.ModuleList(
            CrossAttentionBlock(model_width, attention_heads, dropout)
            for _ in range(attention_layers)
        )
        self.pool_score = nn.Linear(model_width, 1)
        self.posterior_mu = nn.Linear(model_width, latent_dim)
        self.posterior_logvar = nn.Linear(model_width, latent_dim)

    def forward(
        self,
        *,
        state_tokens: Tensor,
        state_valid: Tensor,
        skill_tokens: Tensor,
        skill_valid: Tensor,
        skill_kind_ids: Tensor,
        latent_mode: Literal["sample", "mean", "replay"] = "sample",
        replay_epsilon: Tensor | None = None,
    ) -> CompressorOutputs:
        if skill_tokens.ndim != 4 or skill_valid.ndim != 3:
            raise ValueError("skill tensors must have shapes [batch, candidates, tokens, width/mask]")
        batch, candidates, tokens, _ = skill_tokens.shape
        if candidates > self.boundary_embedding.num_embeddings:
            raise ValueError("candidate skill count exceeds configured maximum")
        state = self.state_projection(state_tokens)
        skill = self.skill_projection(skill_tokens)
        boundaries = torch.arange(candidates, device=skill.device).view(1, candidates, 1)
        skill = skill + self.kind_embedding(skill_kind_ids).unsqueeze(2)
        skill = skill + self.boundary_embedding(boundaries).expand(batch, -1, tokens, -1)
        memory = skill.reshape(batch, candidates * tokens, -1)
        memory_valid = skill_valid.reshape(batch, candidates * tokens).bool()
        if (~memory_valid.any(dim=1)).any():
            raise ValueError("every sample requires at least one valid skill token")
        for block in self.blocks:
            state = block(state, memory, memory_valid)

        scores = self.pool_score(state).squeeze(-1).masked_fill(~state_valid.bool(), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        summary = torch.sum(state * weights.unsqueeze(-1), dim=1)
        mu = self.posterior_mu(summary)
        logvar = self.posterior_logvar(summary).clamp(-10.0, 4.0)
        if latent_mode == "mean":
            epsilon = torch.zeros_like(mu)
            latent = mu
        elif latent_mode == "replay":
            if replay_epsilon is None or replay_epsilon.shape != mu.shape:
                raise ValueError("replay mode requires epsilon with posterior shape")
            epsilon = replay_epsilon
            latent = mu + torch.exp(0.5 * logvar) * epsilon
        elif latent_mode == "sample":
            epsilon = torch.randn_like(mu)
            latent = mu + torch.exp(0.5 * logvar) * epsilon
        else:
            raise ValueError(f"unsupported latent mode: {latent_mode}")
        return CompressorOutputs(summary, mu, logvar, latent, epsilon)


class StateConditionedPrior(nn.Module):
    def __init__(self, state_width: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_width, state_width),
            nn.SiLU(),
            nn.Linear(state_width, latent_dim * 2),
        )

    def forward(self, state_summary: Tensor) -> tuple[Tensor, Tensor]:
        mu, logvar = self.network(state_summary).chunk(2, dim=-1)
        return mu, logvar.clamp(-10.0, 4.0)


class LatentProjector(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        policy_hidden_size: int,
        prefix_length: int = 5,
        hidden_width: int = 256,
        initial_gate: float = 0.01,
    ) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.policy_hidden_size = policy_hidden_size
        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, prefix_length * policy_hidden_size),
        )
        self.norm = RMSNorm(policy_hidden_size)
        self.gate = nn.Parameter(torch.tensor(float(initial_gate)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, latent: Tensor) -> Tensor:
        prefix = self.network(latent).reshape(-1, self.prefix_length, self.policy_hidden_size)
        return self.norm(prefix) * self.gate


class FidelityPredictor(nn.Module):
    def __init__(self, state_width: int = 256, latent_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_width + latent_dim, state_width),
            nn.SiLU(),
            nn.Linear(state_width, 1),
        )

    def forward(self, state_summary: Tensor, latent: Tensor) -> Tensor:
        return self.network(torch.cat((state_summary, latent), dim=-1)).squeeze(-1)


class ExecutableGroundingHead(nn.Module):
    def __init__(
        self,
        *,
        semantic_width: int,
        state_width: int = 256,
        latent_dim: int = 32,
        key_width: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("grounding temperature must be positive")
        self.temperature = temperature
        self.query = nn.Sequential(
            nn.Linear(state_width + latent_dim, key_width),
            nn.SiLU(),
            nn.Linear(key_width, key_width),
        )
        self.command_key = nn.Linear(semantic_width, key_width)

    def forward(
        self,
        state_summary: Tensor,
        latent: Tensor,
        command_embeddings: Tensor,
        command_valid: Tensor,
    ) -> Tensor:
        query = F.normalize(self.query(torch.cat((state_summary, latent), dim=-1)), dim=-1)
        keys = F.normalize(self.command_key(command_embeddings), dim=-1)
        logits = torch.einsum("bd,bcd->bc", query, keys) / self.temperature
        return logits.masked_fill(~command_valid.bool(), -torch.inf)


@dataclass(frozen=True)
class AuxiliaryOutputs:
    fidelity: Tensor
    rate: Tensor
    grounding_logits: Tensor | None


def gaussian_kl(
    posterior_mu: Tensor,
    posterior_logvar: Tensor,
    prior_mu: Tensor,
    prior_logvar: Tensor,
) -> Tensor:
    """Return KL(q||p), summed over latent dimensions for every sample."""

    return 0.5 * torch.sum(
        prior_logvar
        - posterior_logvar
        + (posterior_logvar.exp() + (posterior_mu - prior_mu).square()) / prior_logvar.exp()
        - 1.0,
        dim=-1,
    )

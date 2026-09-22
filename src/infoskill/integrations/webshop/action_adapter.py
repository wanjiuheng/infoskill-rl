"""WebShop policy-action compatibility with the official baseline wrapper."""

from __future__ import annotations

from collections.abc import Mapping


class DemonstrationActionAdapter:
    """Expose official human-demo item actions over the raw WebShop environment.

    The upstream text environment exposes product ASINs as clickables, while the
    released human demonstrations were collected through ``baseline_models.WebEnv``
    with ``click_item_name=1``.  That wrapper presents ``item - <title>`` to the
    policy and translates the chosen title back to an ASIN before stepping.
    """

    def __init__(self, environment: object) -> None:
        self._environment = environment
        products = getattr(getattr(environment, "server"), "product_item_dict")
        self._asin_to_name = {
            str(asin).lower(): str(product["Title"]).lower()
            for asin, product in products.items()
        }
        # Match the released wrapper's last-title-wins behavior exactly.
        self._name_to_asin = {
            name: asin for asin, name in self._asin_to_name.items()
        }

    @property
    def observation_mode(self) -> object:
        return getattr(self._environment, "observation_mode")

    @observation_mode.setter
    def observation_mode(self, value: object) -> None:
        setattr(self._environment, "observation_mode", value)

    def get_available_actions(self) -> object:
        available = self._environment.get_available_actions()  # type: ignore[attr-defined]
        if not isinstance(available, Mapping):
            return available
        translated = dict(available)
        clickables = available.get("clickables")
        if isinstance(clickables, (list, tuple)):
            translated["clickables"] = [
                self._policy_clickable(value) for value in clickables
            ]
        return translated

    def step(self, action: str) -> object:
        return self._environment.step(self._runtime_action(action))  # type: ignore[attr-defined]

    def _policy_clickable(self, value: object) -> object:
        key = str(value).lower()
        title = self._asin_to_name.get(key)
        return f"item - {title}" if title is not None else value

    def _runtime_action(self, action: str) -> str:
        prefix = "click[item - "
        if action.lower().startswith(prefix) and action.endswith("]"):
            title = action[len(prefix) : -1].lower()
            asin = self._name_to_asin.get(title)
            if asin is not None:
                return f"click[{asin}]"
        return action

    def __getattr__(self, name: str) -> object:
        return getattr(self._environment, name)

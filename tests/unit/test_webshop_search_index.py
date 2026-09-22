from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


def _load_builder():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/build_webshop_search_index.py"
    spec = importlib.util.spec_from_file_location("build_webshop_search_index", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WebShopSearchIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_builder()

    def test_document_matches_upstream_full_index_contract(self) -> None:
        product = {
            "asin": "B012345678",
            "Title": "Travel Bag",
            "Description": "A durable bag",
            "BulletPoints": ["Water resistant", "Lightweight"],
            "options": {"color": ["red", "blue"], "size": ["small"]},
        }
        self.assertEqual(self.builder._document_from_product(product), {
            "id": "B012345678",
            "contents": (
                "travel bag a durable bag water resistant "
                "color: red, blue, and size: small"
            ),
            "product": product,
        })


if __name__ == "__main__":
    unittest.main()

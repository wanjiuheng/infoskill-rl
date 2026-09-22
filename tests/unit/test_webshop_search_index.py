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

    def test_paper1000_layout_uses_the_small_assets_and_separate_index(self) -> None:
        layout = self.builder._catalog_layout(Path("/data/webshop"), "paper1000")

        self.assertEqual(layout["products"], Path("/data/webshop/data/items_shuffle_1000.json"))
        self.assertEqual(layout["attributes"], Path("/data/webshop/data/items_ins_v2_1000.json"))
        self.assertEqual(layout["indexes"], Path("/data/webshop/search_engine/indexes_1k"))
        self.assertEqual(layout["manifest"], Path("/data/webshop/search_engine/index-1k-manifest.json"))


if __name__ == "__main__":
    unittest.main()

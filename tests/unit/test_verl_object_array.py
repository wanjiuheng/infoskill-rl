from __future__ import annotations

import unittest

from infoskill.integrations.verl.object_array import object_array


class VerlObjectArrayTests(unittest.TestCase):
    def test_bfloat16_prefixes_remain_python_objects_without_numpy_conversion(self) -> None:
        class BFloat16Prefix:
            def __array__(self, dtype=None):
                raise TypeError("Got unsupported ScalarType BFloat16")

        first = BFloat16Prefix()
        second = BFloat16Prefix()

        result = object_array((first, second, None))

        self.assertEqual(result.shape, (3,))
        self.assertIs(result[0], first)
        self.assertIs(result[1], second)
        self.assertIsNone(result[2])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from infoskill.training.m0 import _configure_training_logging


class TrainingLoggingTests(unittest.TestCase):
    def test_project_logger_does_not_propagate_and_duplicate_console_lines(self) -> None:
        with patch.object(
            logging,
            "FileHandler",
            return_value=logging.NullHandler(),
        ):
            logger = _configure_training_logging(Path("unused-run-directory"))
            try:
                self.assertFalse(logger.propagate)
            finally:
                for handler in logger.handlers:
                    handler.close()
                logger.handlers.clear()


if __name__ == "__main__":
    unittest.main()

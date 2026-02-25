#!/usr/bin/env python3

"""Invoke Community Notes scoring and user contribution algorithms.

Example Usage:
  python main.py \
    --enrollment data/userEnrollment-00000.tsv \
    --notes data/notes \
    --ratings data/ratings \
    --status data/noteStatusHistory-00000.tsv \
    --outdir data \
    --cache-dir .cache
"""

import logging
import os
from datetime import datetime

from scoring.runner import main


def _setup_logging():
  """Configure logging to write to both console and a per-run log file."""
  log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
  os.makedirs(log_dir, exist_ok=True)

  run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
  log_file = os.path.join(log_dir, f"run_{run_id}.log")

  fmt = "%(asctime)s %(levelname)s:%(name)s:%(message)s"
  datefmt = "%H:%M:%S"

  # Root logger: DEBUG level so everything is captured
  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  # Console handler — same as before
  console = logging.StreamHandler()
  console.setLevel(logging.DEBUG)
  console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
  root.addHandler(console)

  # File handler — captures everything to the per-run log file
  file_handler = logging.FileHandler(log_file, encoding="utf-8")
  file_handler.setLevel(logging.DEBUG)
  file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
  root.addHandler(file_handler)

  # Silence Numba's extremely verbose internal compiler logs
  logging.getLogger("numba").setLevel(logging.WARNING)

  logging.info(f"Logging to {log_file}")


if __name__ == "__main__":
  _setup_logging()
  main()

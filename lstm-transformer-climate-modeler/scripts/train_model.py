"""Convenience wrapper for running the training workflow from the repo root."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from climate_modeling.train import main


if __name__ == "__main__":
    main()

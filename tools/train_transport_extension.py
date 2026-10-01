"""Run the unmodified original train.py with continuation-only hooks registered."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import transport_extension_hooks  # noqa: F401
import train

if __name__ == '__main__':
    train.main()

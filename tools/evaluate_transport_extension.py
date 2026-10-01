"""Original evaluation CLI using exactly preserved predictions on NVMe."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transport_extension_hooks  # noqa: F401
from evaluate_radar_experiment import main

if __name__ == '__main__':
    main()

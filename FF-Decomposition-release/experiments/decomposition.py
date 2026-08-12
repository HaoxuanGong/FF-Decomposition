"""Run the fixed main-text Forward-Forward decomposition experiment."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from FFDecompositionBenchmark import main


if __name__ == "__main__":
    raise SystemExit(main())

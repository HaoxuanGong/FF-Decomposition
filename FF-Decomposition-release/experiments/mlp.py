"""Run MF experiments on a fully connected network."""

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from VanillaMonoForward import main


if __name__ == "__main__":
    main()

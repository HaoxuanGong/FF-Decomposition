"""Run the matched four-block MF CNN main-accuracy experiment."""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv: list[str] | None = None) -> None:
    from MFCNNBenchmark import main as run

    run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()

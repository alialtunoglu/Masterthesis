"""Package the latest matching run for transfer back to the local project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracking.external_bundle import create_external_bundle


SUMMARY_PATHS = {
    "baseline": "results/baseline/baseline_results.csv",
    "teacher_cnn": "results/teachers/cnn/teacher_results.csv",
    "teacher_vision_transformer": "results/teachers/vision_transformers/teacher_results.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=sorted(SUMMARY_PATHS), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(ROOT / SUMMARY_PATHS[args.stage])
    matches = frame[
        (frame["dataset_name"].astype(str) == args.dataset)
        & (frame["model_name"].astype(str) == args.model)
    ]
    if matches.empty:
        raise SystemExit("No completed matching result row was found.")
    row = matches.iloc[-1].to_dict()
    outputs = create_external_bundle(
        ROOT, row, source_commit=args.source_commit, output_dir=Path(args.output)
    )
    print(f"Bundle: {outputs.archive}")
    print(f"Checkpoint: {outputs.checkpoint or 'not included'}")


if __name__ == "__main__":
    main()

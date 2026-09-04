"""
Exports the frozen VQA-SUNRGBD-v2 release into the exact CSV shape the
existing training code (dataset/dataloader/**, dataset/datamodule/**)
already expects, so those scripts can retrain without being rewritten.

The existing loaders address columns by *position*, not name — e.g.
CustomSUNRGBDDatasetOneVision.__getitem__ reads
`self.df.iloc[idx, 1]` for the question, `iloc[idx, 2]` for the answer,
`iloc[idx, 3]` / `iloc[idx, 4]` for the RGB/depth paths — so pointing them at
release/VQA-SUNRGBD-v2/rule_based/*.csv directly would silently train on the
wrong columns (v2's column 0 is `question_id`, column 1 is `image_id`, not
the question) rather than error. This script is the adapter: v2 stays the
one frozen source of truth, and this just re-projects it into v1's layout
at the path those loaders read from
(`<ROOT_DATA_DIR>/SUNRGBD/csv_data/{train,val,test}_dataset.csv`), matching
v1's own column order: IDs, Questions, Answers, Image_Path, Depth_Path,
Question_Type.

Image/depth paths are copied through unchanged: v2's `image_path` /
`depth_path` already carry the `SUNRGBD/<sensor>/.../image/x.jpg` form the
loaders' own `remove_substring_from_path` expects when ROOT_DATA_DIR points
at this repo's `dataset/` directory (see .env — that's what ROOT_DATA_DIR is
set to here).

Usage::

    python dataset/dataset_creation/v2/export_v1_schema.py
"""
from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RULE_BASED_DIR = os.path.join(REPO_ROOT, "release", "VQA-SUNRGBD-v2", "rule_based")
OUTPUT_DIR = os.path.join(REPO_ROOT, "dataset", "SUNRGBD", "csv_data")

SPLIT_TO_V1_FILENAME = {
    "train": "train_dataset.csv",
    "val": "val_dataset.csv",
    "test": "test_dataset.csv",
}

# Position matters, not the header text — this must match the iloc indices
# in dataset/dataloader/OneVision/CustomSUNRGBDDatasetOneVision*.py exactly:
# 0=IDs (unused positionally), 1=Questions, 2=Answers, 3=Image_Path, 4=Depth_Path.
V1_COLUMN_ORDER = ["IDs", "Questions", "Answers", "Image_Path", "Depth_Path", "Question_Type"]
V2_TO_V1_COLUMNS = {
    "question_id": "IDs",
    "question": "Questions",
    "answer": "Answers",
    "image_path": "Image_Path",
    "depth_path": "Depth_Path",
    "question_type": "Question_Type",
}


def export_split(split: str) -> str:
    source_path = os.path.join(RULE_BASED_DIR, f"{split}.csv")
    v2_frame = pd.read_csv(source_path, dtype={"answer": str})
    v1_frame = v2_frame.rename(columns=V2_TO_V1_COLUMNS)[V1_COLUMN_ORDER]

    output_path = os.path.join(OUTPUT_DIR, SPLIT_TO_V1_FILENAME[split])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    v1_frame.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    for split in ("train", "val", "test"):
        output_path = export_split(split)
        row_count = sum(1 for _ in open(output_path)) - 1
        print(f"{split}: {row_count} rows -> {output_path}")


if __name__ == "__main__":
    main()

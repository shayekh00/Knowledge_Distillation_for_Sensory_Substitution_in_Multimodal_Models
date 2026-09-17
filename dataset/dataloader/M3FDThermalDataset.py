"""Thermal-only M³FD release loader.

This intentionally small adapter is the deployment boundary: it resolves and
opens only ``thermal_path``. RGB, annotations, boxes, evidence, and caches are
not exposed in a sample, so a thermal student cannot accidentally consume them.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


class M3FDThermalDataset:
    def __init__(self, dataset_root: str | Path, release_csv: str | Path, transform=None):
        self.dataset_root = Path(dataset_root).resolve()
        self.transform = transform
        with Path(release_csv).open(encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        required = {"question_id", "question", "answer", "thermal_path"}
        if not self.rows or not required <= set(self.rows[0]):
            raise ValueError(f"{release_csv}: missing required M3FD thermal release columns")

    def __len__(self) -> int:
        return len(self.rows)

    def _thermal_path(self, row: dict) -> Path:
        path = (self.dataset_root / row["thermal_path"]).resolve()
        try:
            path.relative_to(self.dataset_root)
        except ValueError as error:
            raise ValueError(f"thermal_path escapes dataset root: {row['thermal_path']!r}") from error
        if not path.is_file():
            raise FileNotFoundError(f"missing thermal input: {path}")
        return path

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        # PIL loads before the source file is closed; return a detached image.
        with Image.open(self._thermal_path(row)) as source:
            thermal = source.convert("RGB").copy()
        if self.transform is not None:
            thermal = self.transform(thermal)
        return {"question_id": row["question_id"], "image": thermal,
                "question": row["question"], "answer": row["answer"]}

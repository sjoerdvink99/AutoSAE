from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContrastiveDataset:
    positive: list[str]
    negative: list[str]

    def __post_init__(self) -> None:
        if len(self.positive) == 0 or len(self.negative) == 0:
            raise ValueError("Both positive and negative prompt lists must be non-empty.")
        if len(self.positive) != len(self.negative):
            raise ValueError(
                f"Positive ({len(self.positive)}) and negative ({len(self.negative)}) "
                "lists must have equal length."
            )
        if any(not s.strip() for s in self.positive) or any(not s.strip() for s in self.negative):
            raise ValueError("Prompt strings must not be blank.")

    @classmethod
    def from_json(cls, path: str | Path) -> ContrastiveDataset:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> ContrastiveDataset:
        if "positive" not in data or "negative" not in data:
            raise ValueError("JSON must contain 'positive' and 'negative' keys.")
        return cls(positive=data["positive"], negative=data["negative"])

    def __len__(self) -> int:
        return len(self.positive)

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from bcv.benchmark import BenchmarkResult


@dataclass(frozen=True)
class TripwireTrainingResult:
    examples: int
    epochs: int
    final_loss: float
    accuracy: float
    model_path: str
    device: str


def synthetic_document_examples() -> list[BenchmarkResult]:
    examples: list[BenchmarkResult] = []
    for index in range(80):
        examples.append(
            BenchmarkResult(
                candidate=f"clean_{index}",
                accepted=True,
                failure=None,
                accidental_deletions=0,
                number_drift=0,
                section_drift=0,
            )
        )
        examples.append(
            BenchmarkResult(
                candidate=f"bad_delete_{index}",
                accepted=False,
                failure="citation removed",
                accidental_deletions=1 + (index % 2),
                number_drift=0,
                section_drift=0,
            )
        )
        examples.append(
            BenchmarkResult(
                candidate=f"bad_number_{index}",
                accepted=False,
                failure="number drift",
                accidental_deletions=0,
                number_drift=1 + (index % 3),
                section_drift=0,
            )
        )
        examples.append(
            BenchmarkResult(
                candidate=f"bad_section_{index}",
                accepted=False,
                failure="section drift",
                accidental_deletions=0,
                number_drift=0,
                section_drift=1 + (index % 2),
            )
        )
    return examples


def train_document_tripwire(
    output_dir: str | Path = ".bcv_runs/tripwire",
    epochs: int = 120,
    lr: float = 0.05,
) -> TripwireTrainingResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = synthetic_document_examples()
    features = torch.tensor(
        [
            [
                float(item.accidental_deletions),
                float(item.number_drift),
                float(item.section_drift),
                1.0 if item.failure else 0.0,
            ]
            for item in examples
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[1.0 if item.accepted else 0.0] for item in examples], dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    ).to(device)
    features = features.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())

    with torch.no_grad():
        predictions = (torch.sigmoid(model(features)) >= 0.5).float()
        accuracy = float((predictions == labels).float().mean().detach().cpu())

    model_path = output_dir / "document_tripwire.pt"
    metadata_path = output_dir / "document_tripwire_metadata.json"
    torch.save(model.state_dict(), model_path)
    result = TripwireTrainingResult(
        examples=len(examples),
        epochs=epochs,
        final_loss=final_loss,
        accuracy=accuracy,
        model_path=str(model_path),
        device=device,
    )
    metadata_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> None:
    print(json.dumps(asdict(train_document_tripwire()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


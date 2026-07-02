from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.taste import TasteContext, score_taste_multi_anchor


@dataclass(frozen=True)
class TasteAdversaryCase:
    prompt: str
    domain: str
    strong: str
    generic: str
    weird: str
    slop_anchors: tuple[str, ...]


@dataclass(frozen=True)
class TasteAdversaryMetric:
    prompt: str
    strong_score: float
    generic_score: float
    weird_score: float
    strong_beats_generic: bool
    strong_beats_weird: bool


@dataclass(frozen=True)
class TasteAdversaryResult:
    cases: int
    strong_beats_generic: int
    strong_beats_weird: int
    metrics: tuple[TasteAdversaryMetric, ...]


def adversary_cases() -> tuple[TasteAdversaryCase, ...]:
    return (
        TasteAdversaryCase(
            prompt="Explain why a rough garage band recording can be appealing.",
            domain="explanation",
            generic="People like rough garage recordings because they sound raw, real, and energetic.",
            weird="A garage band is a lunar spoon orbiting a rusted microphone, and the fuzz is democracy becoming soup.",
            strong="A rough garage recording works because the flaws become evidence. The room noise, clipped vocal, and uneven drums tell you a human event happened before anyone had time to sand it smooth.",
            slop_anchors=(
                "People like rough garage recordings because they sound raw and authentic.",
                "Rough recordings can be appealing because they feel real, energetic, and less polished.",
            ),
        ),
        TasteAdversaryCase(
            prompt="Explain why simple interfaces can feel premium.",
            domain="design",
            generic="Simple interfaces feel premium because they are clean, easy to use, and less cluttered.",
            weird="A button becomes a cathedral when the menus evaporate into velvet arithmetic.",
            strong="A simple interface feels premium because it refuses to make the user manage the designer's anxiety. Every missing button says the product knows what matters.",
            slop_anchors=(
                "A simple interface feels premium because it is clean and uncluttered.",
                "Simple design is easier to use and can look more elegant.",
            ),
        ),
        TasteAdversaryCase(
            prompt="Explain why plot twists fail when they only shock.",
            domain="fiction",
            generic="Plot twists fail if they are shocking but do not make sense for the story.",
            weird="A twist is a mirror swallowing a trumpet while causality wears a fake mustache.",
            strong="A shock-only twist fails because it spends trust like cheap currency. A good twist makes the earlier story more legible; a bad one just proves the writer can yank the steering wheel.",
            slop_anchors=(
                "A plot twist needs to make sense and fit the story.",
                "A twist that only shocks can feel random or manipulative.",
            ),
        ),
    )


def run_taste_adversary_benchmark(root: str | Path = ".bcv_runs/taste_adversary") -> TasteAdversaryResult:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    metrics: list[TasteAdversaryMetric] = []
    for case in adversary_cases():
        context = TasteContext(
            prompt=case.prompt,
            domain=case.domain,  # type: ignore[arg-type]
            audience="heldout",
            mode="critique",
            target_novelty=0.48,
        )
        strong = score_taste_multi_anchor(context, case.strong, case.slop_anchors)
        generic = score_taste_multi_anchor(context, case.generic, case.slop_anchors)
        weird = score_taste_multi_anchor(context, case.weird, case.slop_anchors)
        metrics.append(
            TasteAdversaryMetric(
                prompt=case.prompt,
                strong_score=strong.final_score,
                generic_score=generic.final_score,
                weird_score=weird.final_score,
                strong_beats_generic=strong.final_score > generic.final_score,
                strong_beats_weird=strong.final_score > weird.final_score,
            )
        )
    result = TasteAdversaryResult(
        cases=len(metrics),
        strong_beats_generic=sum(1 for item in metrics if item.strong_beats_generic),
        strong_beats_weird=sum(1 for item in metrics if item.strong_beats_weird),
        metrics=tuple(metrics),
    )
    (root / "adversary_result.json").write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def main() -> None:
    print(json.dumps(asdict(run_taste_adversary_benchmark()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from bcv.taste import TasteContext, export_seed_taste_datasets, prefer_pair, score_taste


@dataclass(frozen=True)
class TastePromptSpec:
    prompt: str
    domain: str
    audience: str
    mode: str
    subject: str
    generic_frame: str
    strong_frame: str
    concrete_detail: str
    memorable_image: str


PROMPT_SPECS = (
    TastePromptSpec(
        prompt="Why do people like superhero movies?",
        domain="explanation",
        audience="technical generalist",
        mode="explanation",
        subject="superhero movies",
        generic_frame="they are exciting and show good defeating evil",
        strong_frame="they are moral navigation systems with explosions attached",
        concrete_detail="the viewer knows who to root for, when to laugh, and what victory means",
        memorable_image="a chaotic world becomes a color-coded map",
    ),
    TastePromptSpec(
        prompt="Why do people rewatch comfort shows?",
        domain="explanation",
        audience="curious adult",
        mode="explanation",
        subject="comfort shows",
        generic_frame="they are familiar and relaxing",
        strong_frame="they rent the viewer a nervous system that has already been debugged",
        concrete_detail="no new rules, no risky ending, no social homework",
        memorable_image="a predictable emotional room",
    ),
    TastePromptSpec(
        prompt="Why does generic AI prose feel bad?",
        domain="explanation",
        audience="AI builder",
        mode="critique",
        subject="generic AI prose",
        generic_frame="it lacks originality and sounds formulaic",
        strong_frame="nothing in it had to be there",
        concrete_detail="no local scar tissue, no chosen risk, no sentence that would be missed",
        memorable_image="a paragraph with no fingerprints",
    ),
    TastePromptSpec(
        prompt="Explain why horror often works better by showing less.",
        domain="explanation",
        audience="film writer",
        mode="critique",
        subject="horror restraint",
        generic_frame="mystery can be scarier than seeing the monster",
        strong_frame="withholding makes the audience finish the monster using their own private inventory of fear",
        concrete_detail="a door left half-open can recruit more dread than a full creature shot",
        memorable_image="the blank space becomes the special effect",
    ),
    TastePromptSpec(
        prompt="Explain why a good product name matters.",
        domain="design",
        audience="founder",
        mode="branding",
        subject="product names",
        generic_frame="a good name is memorable and helps people understand the product",
        strong_frame="a name is a handle for other people's memory",
        concrete_detail="it has to survive a Slack mention, a half-heard recommendation, and a search box",
        memorable_image="a tiny piece of interface for the market's mouth",
    ),
    TastePromptSpec(
        prompt="Why do jokes fail when they are overexplained?",
        domain="comedy",
        audience="writer",
        mode="critique",
        subject="overexplained jokes",
        generic_frame="explaining a joke ruins the surprise",
        strong_frame="a joke is a trapdoor; explanation installs a handrail",
        concrete_detail="the laugh needs the mind to fall one step before it catches itself",
        memorable_image="a trapdoor with a warning label",
    ),
)


def build_scaled_taste_dataset(
    root: str | Path = ".bcv_runs/taste_scaled",
    variants_per_prompt: int = 40,
) -> dict[str, object]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    preferences = []
    sft_examples = []
    score_rows = []

    for spec in PROMPT_SPECS:
        context = TasteContext(
            prompt=spec.prompt,
            domain=spec.domain,  # type: ignore[arg-type]
            audience=spec.audience,
            mode=spec.mode,
            target_novelty=0.48,
        )
        for index in range(variants_per_prompt):
            weak = _weak_answer(spec, index)
            strong = _strong_answer(spec, index)
            slop = _slop_answer(spec, index)
            preference = prefer_pair(context, weak, strong, slop)
            preferences.append(preference)
            weak_scores = score_taste(context, weak, slop)
            strong_scores = score_taste(context, strong, slop)
            score_rows.append(
                {
                    "prompt": spec.prompt,
                    "index": index,
                    "weak_score": weak_scores.final_score,
                    "strong_score": strong_scores.final_score,
                    "margin": strong_scores.final_score - weak_scores.final_score,
                    "chosen_is_strong": preference.chosen == strong,
                    "reason": preference.reason,
                }
            )
            sft_examples.append(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "Improve weak outputs using the Taste-RL verifier. Increase specificity, bounded novelty, memorability, and reflective value while reducing genericness and AI smell.",
                        },
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "prompt": spec.prompt,
                                    "weak_answer": weak,
                                    "slop_reference": slop,
                                    "verifier": {
                                        "main_failure_mode": weak_scores.main_failure_mode,
                                        "revision_instruction": weak_scores.revision_instruction,
                                    },
                                },
                                sort_keys=True,
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": strong,
                        },
                    ]
                }
            )

    preference_path = root / "taste_preferences.jsonl"
    sft_path = root / "taste_sft.jsonl"
    scores_path = root / "taste_scores.jsonl"
    preference_path.write_text(
        "\n".join(json.dumps(asdict(item), sort_keys=True) for item in preferences) + "\n",
        encoding="utf-8",
    )
    sft_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in sft_examples) + "\n",
        encoding="utf-8",
    )
    scores_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in score_rows) + "\n",
        encoding="utf-8",
    )
    chosen_strong = sum(1 for row in score_rows if row["chosen_is_strong"])
    avg_margin = sum(float(row["margin"]) for row in score_rows) / max(1, len(score_rows))
    return {
        "preference_path": str(preference_path),
        "sft_path": str(sft_path),
        "scores_path": str(scores_path),
        "preference_examples": len(preferences),
        "sft_examples": len(sft_examples),
        "chosen_strong": chosen_strong,
        "avg_strong_minus_weak_margin": avg_margin,
        "variants_per_prompt": variants_per_prompt,
        "prompt_specs": len(PROMPT_SPECS),
    }


def _weak_answer(spec: TastePromptSpec, index: int) -> str:
    starters = (
        f"People like {spec.subject} because {spec.generic_frame}.",
        f"{spec.subject.capitalize()} works because it is engaging, clear, and easy to understand.",
        f"The main reason is that {spec.subject} gives people something enjoyable and accessible.",
        f"Overall, {spec.subject} is effective because it connects with people in a simple way.",
    )
    addenda = (
        " This makes it appealing to a wide audience.",
        " It can be both entertaining and meaningful.",
        " Many people appreciate that kind of experience.",
        " It is important because it gives people what they want.",
    )
    return starters[index % len(starters)] + addenda[(index // len(starters)) % len(addenda)]


def _slop_answer(spec: TastePromptSpec, index: int) -> str:
    return (
        f"{spec.subject.capitalize()} is popular because {spec.generic_frame}. "
        f"It is fun, relatable, and easy for many people to enjoy. Overall, it works because it gives audiences a satisfying experience."
    )


def _strong_answer(spec: TastePromptSpec, index: int) -> str:
    forms = (
        f"{spec.subject.capitalize()} works because {spec.strong_frame}. {spec.concrete_detail}. The memorable part is this: {spec.memorable_image}.",
        f"The reward in {spec.subject} is not just pleasure; it is orientation. {spec.concrete_detail}. In that sense, {spec.memorable_image}.",
        f"A generic answer says {spec.generic_frame}. The sharper answer is that {spec.strong_frame}: {spec.concrete_detail}.",
        f"{spec.subject.capitalize()} earns attention when it gives the audience a clean handle on a messy feeling. {spec.concrete_detail}. That is why it sticks as {spec.memorable_image}.",
    )
    return forms[index % len(forms)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".bcv_runs/taste_scaled")
    parser.add_argument("--variants-per-prompt", type=int, default=40)
    args = parser.parse_args()
    seed = export_seed_taste_datasets(Path(args.root) / "seed")
    scaled = build_scaled_taste_dataset(args.root, args.variants_per_prompt)
    print(json.dumps({"seed": seed, "scaled": scaled}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


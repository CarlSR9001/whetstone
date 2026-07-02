from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


Domain = Literal["explanation", "fiction", "persuasion", "design", "comedy", "general"]


GENERIC_PHRASES = (
    "it depends",
    "in conclusion",
    "there are many reasons",
    "people like it because",
    "fun and exciting",
    "good and bad",
    "important to note",
    "a variety of",
    "can be beneficial",
    "overall",
    "easy to use",
    "less cluttered",
    "clean and uncluttered",
    "looks more elegant",
)

AI_SMELL_PHRASES = (
    "not only",
    "but also",
    "it's important to",
    "this highlights",
    "delve",
    "tapestry",
    "nuanced",
    "multifaceted",
    "in today's world",
)

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "being", "been", "it", "this", "that",
    "they", "you", "we", "as", "by", "from", "at", "not", "because", "when",
}


@dataclass(frozen=True)
class TasteContext:
    prompt: str
    domain: Domain = "general"
    audience: str = "general"
    mode: str = "utility"
    target_novelty: float = 0.45


@dataclass(frozen=True)
class TasteScores:
    legibility: float
    coherence: float
    specificity: float
    novelty: float
    bounded_novelty: float
    memorability: float
    grounded_transmissibility: float
    reflective_survival: float
    genericness_penalty: float
    ai_smell_penalty: float
    manipulation_penalty: float
    length_penalty: float
    conjunctive_score: float
    peak_score: float
    final_score: float
    main_failure_mode: str
    revision_instruction: str


@dataclass(frozen=True)
class TastePreference:
    prompt: str
    chosen: str
    rejected: str
    chosen_score: float
    rejected_score: float
    reason: str
    metadata: dict[str, object]


def score_taste(
    context: TasteContext,
    candidate: str,
    slop_reference: str,
) -> TasteScores:
    candidate_tokens = _content_tokens(candidate)
    slop_tokens = _content_tokens(slop_reference)
    word_count = max(1, len(re.findall(r"\b\w+\b", candidate)))

    legibility = _bounded(7.0 - max(0, _avg_sentence_len(candidate) - 24) / 8)
    coherence = _coherence(candidate)
    specificity = _specificity(candidate)
    novelty = _jaccard_distance(candidate_tokens, slop_tokens)
    bounded_novelty = _bounded(7 * math.exp(-((novelty - context.target_novelty) ** 2) / (2 * 0.30**2)) * (legibility / 7))
    memorability = _memorability(candidate)
    grounded_transmissibility = _grounded_transmissibility(candidate, specificity)
    reflective_survival = _reflective_survival(candidate)
    genericness_penalty = _phrase_penalty(candidate, GENERIC_PHRASES)
    ai_smell_penalty = _phrase_penalty(candidate, AI_SMELL_PHRASES)
    manipulation_penalty = _manipulation_penalty(candidate)
    length_penalty = _length_penalty(word_count)

    positives = (
        legibility,
        coherence,
        specificity,
        max(1.0, bounded_novelty),
        memorability,
        grounded_transmissibility,
        reflective_survival,
    )
    conjunctive_score = _ces(positives, rho=-1.0)
    peak_score = max(positives)
    residual_bonus = (
        0.12 * specificity
        + 0.10 * memorability
        + 0.08 * grounded_transmissibility
        + 0.08 * min(7.0, novelty * 7)
    )
    penalty = (
        0.95 * genericness_penalty
        + 0.45 * ai_smell_penalty
        + 0.55 * manipulation_penalty
        + 0.45 * length_penalty
    )
    final_score = _bounded(0.78 * conjunctive_score + 0.12 * peak_score + residual_bonus - penalty + 2.2)
    failure, revision = _failure_and_revision(
        {
            "legibility": legibility,
            "coherence": coherence,
            "specificity": specificity,
            "bounded_novelty": bounded_novelty,
            "memorability": memorability,
            "grounded_transmissibility": grounded_transmissibility,
            "reflective_survival": reflective_survival,
        },
        {
            "genericness": genericness_penalty,
            "ai_smell": ai_smell_penalty,
            "manipulation": manipulation_penalty,
            "length": length_penalty,
        },
    )

    return TasteScores(
        legibility=legibility,
        coherence=coherence,
        specificity=specificity,
        novelty=novelty * 7,
        bounded_novelty=bounded_novelty,
        memorability=memorability,
        grounded_transmissibility=grounded_transmissibility,
        reflective_survival=reflective_survival,
        genericness_penalty=genericness_penalty,
        ai_smell_penalty=ai_smell_penalty,
        manipulation_penalty=manipulation_penalty,
        length_penalty=length_penalty,
        conjunctive_score=conjunctive_score,
        peak_score=peak_score,
        final_score=final_score,
        main_failure_mode=failure,
        revision_instruction=revision,
    )


def score_taste_multi_anchor(
    context: TasteContext,
    candidate: str,
    slop_references: tuple[str, ...],
) -> TasteScores:
    """Score against multiple slop anchors and keep the harshest result.

    A candidate should not win just because it differs from one weak baseline while
    matching another generic attractor. The minimum final score is the conservative
    moving-anchor estimate.
    """

    if not slop_references:
        raise ValueError("at least one slop reference is required")
    scores = tuple(score_taste(context, candidate, reference) for reference in slop_references)
    return min(scores, key=lambda item: item.final_score)


def prefer_pair(
    context: TasteContext,
    candidate_a: str,
    candidate_b: str,
    slop_reference: str,
) -> TastePreference:
    score_a = score_taste(context, candidate_a, slop_reference)
    score_b = score_taste(context, candidate_b, slop_reference)
    if score_a.final_score >= score_b.final_score:
        chosen, rejected, chosen_score, rejected_score, reason = (
            candidate_a,
            candidate_b,
            score_a.final_score,
            score_b.final_score,
            _preference_reason(score_a, score_b),
        )
    else:
        chosen, rejected, chosen_score, rejected_score, reason = (
            candidate_b,
            candidate_a,
            score_b.final_score,
            score_a.final_score,
            _preference_reason(score_b, score_a),
        )
    return TastePreference(
        prompt=context.prompt,
        chosen=chosen,
        rejected=rejected,
        chosen_score=chosen_score,
        rejected_score=rejected_score,
        reason=reason,
        metadata={
            "domain": context.domain,
            "audience": context.audience,
            "mode": context.mode,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        },
    )


def seed_taste_pairs() -> list[tuple[TasteContext, str, str, str]]:
    return [
        (
            TasteContext(
                prompt="Why do people like superhero movies?",
                domain="explanation",
                audience="technical generalist",
                mode="explanation",
            ),
            "People like superhero movies because they are exciting, have action, and show good defeating evil.",
            "People like superhero movies because they are emotionally low-risk machines. You know who to root for, when to feel awe, when to laugh, and what victory is supposed to mean. The spectacle matters, but the deeper reward is orientation: the movie makes a chaotic world temporarily legible.",
            "People like superhero movies because they are fun and exciting. They have lots of action and famous actors.",
        ),
        (
            TasteContext(
                prompt="Explain why people rewatch comfort shows.",
                domain="explanation",
                audience="curious adult",
                mode="explanation",
            ),
            "People rewatch comfort shows because they are familiar and relaxing. They know what will happen and enjoy the characters.",
            "A comfort show is a predictable emotional room. Rewatching it lowers the cost of feeling: no new rules, no risky ending, no social homework. The viewer is not looking for surprise; they are renting a nervous system that has already been debugged.",
            "People rewatch shows because they like them and they are comforting.",
        ),
        (
            TasteContext(
                prompt="Write a sentence explaining why generic AI prose feels bad.",
                domain="explanation",
                audience="AI builder",
                mode="critique",
            ),
            "Generic AI prose feels bad because it lacks originality and sounds formulaic.",
            "Generic AI prose feels bad because nothing in it had to be there: no local scar tissue, no chosen risk, no sentence that would be missed if it disappeared.",
            "AI prose can feel generic because it is often repetitive, vague, and not very specific.",
        ),
    ]


def export_seed_taste_datasets(root: str | Path = ".bcv_runs/taste") -> dict[str, object]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    preferences: list[TastePreference] = []
    sft_examples: list[dict[str, object]] = []
    for context, weak, strong, slop in seed_taste_pairs():
        preference = prefer_pair(context, weak, strong, slop)
        preferences.append(preference)
        weak_scores = score_taste(context, weak, slop)
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
                                "prompt": context.prompt,
                                "weak_answer": weak,
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
    preference_path.write_text(
        "\n".join(json.dumps(asdict(item), sort_keys=True) for item in preferences) + "\n",
        encoding="utf-8",
    )
    sft_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in sft_examples) + "\n",
        encoding="utf-8",
    )
    return {
        "preference_path": str(preference_path),
        "sft_path": str(sft_path),
        "preference_examples": len(preferences),
        "sft_examples": len(sft_examples),
        "avg_margin": sum(item.chosen_score - item.rejected_score for item in preferences) / len(preferences),
    }


def _content_tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z][a-zA-Z'-]{2,}\b", text)
        if token.lower() not in STOPWORDS
    }


def _avg_sentence_len(text: str) -> float:
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    if not sentences:
        return 999.0
    words = len(re.findall(r"\b\w+\b", text))
    return words / len(sentences)


def _coherence(text: str) -> float:
    if not text.strip():
        return 1.0
    unmatched = abs(text.count("(") - text.count(")")) + abs(text.count('"') % 2)
    sentence_count = max(1, len([part for part in re.split(r"[.!?]+", text) if part.strip()]))
    return _bounded(6.4 - unmatched - max(0, sentence_count - 5) * 0.2)


def _specificity(text: str) -> float:
    concrete = len(re.findall(r"\b[A-Z][a-z]+|\b\d+|[:;]", text))
    content = len(_content_tokens(text))
    return _bounded(2.2 + min(3.0, concrete * 0.35) + min(2.0, content / 18))


def _jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / max(1, len(a | b)))


def _memorability(text: str) -> float:
    colon_bonus = 0.5 if ":" in text else 0.0
    image_bonus = min(1.4, len(re.findall(r"\bmachine|room|scar|risk|map|edge|gravity|debugged\b", text.lower())) * 0.45)
    compression_bonus = 0.8 if len(re.findall(r"\b\w+\b", text)) <= 80 else 0.2
    return _bounded(2.6 + colon_bonus + image_bonus + compression_bonus + min(1.2, len(_content_tokens(text)) / 30))


def _grounded_transmissibility(text: str, specificity: float) -> float:
    hook = 1.0 if re.search(r"\bis\b|:", text) else 0.2
    return _bounded(1.8 + hook + 0.65 * specificity)


def _reflective_survival(text: str) -> float:
    tokens = _content_tokens(text)
    if not tokens:
        return 1.0
    first_half = set(list(tokens)[: max(1, len(tokens) // 2)])
    second_half = tokens - first_half
    survival = len(first_half & tokens) / max(1, len(first_half))
    spread = min(1.0, len(second_half) / max(1, len(tokens)))
    return _bounded(2.5 + 2.2 * survival + 1.2 * spread)


def _phrase_penalty(text: str, phrases: tuple[str, ...]) -> float:
    lowered = text.lower()
    hits = sum(1 for phrase in phrases if phrase in lowered)
    return _bounded(1.0 + hits * 1.2)


def _manipulation_penalty(text: str) -> float:
    lowered = text.lower()
    hits = sum(1 for marker in ("must", "shocking", "secret", "everyone", "destroy", "insane") if marker in lowered)
    return _bounded(1.0 + hits)


def _length_penalty(word_count: int) -> float:
    if 25 <= word_count <= 90:
        return 1.0
    if word_count < 12:
        return 2.5
    return _bounded(1.0 + (word_count - 90) / 55)


def _ces(values: tuple[float, ...], rho: float) -> float:
    values = tuple(max(0.1, value) for value in values)
    return (sum(value**rho for value in values) / len(values)) ** (1 / rho)


def _failure_and_revision(positive: dict[str, float], penalty: dict[str, float]) -> tuple[str, str]:
    worst_positive = min(positive.items(), key=lambda item: item[1])
    worst_penalty = max(penalty.items(), key=lambda item: item[1])
    if worst_penalty[1] >= 3.0:
        return (
            f"high_{worst_penalty[0]}",
            f"Reduce {worst_penalty[0]} while preserving the core answer; replace generic or manipulative phrasing with grounded concrete detail.",
        )
    return (
        f"low_{worst_positive[0]}",
        f"Improve {worst_positive[0]} without adding padding; add one concrete, memorable move that fits the prompt.",
    )


def _preference_reason(chosen: TasteScores, rejected: TasteScores) -> str:
    deltas = {
        "specificity": chosen.specificity - rejected.specificity,
        "bounded_novelty": chosen.bounded_novelty - rejected.bounded_novelty,
        "memorability": chosen.memorability - rejected.memorability,
        "reflective_survival": chosen.reflective_survival - rejected.reflective_survival,
        "genericness_reduction": rejected.genericness_penalty - chosen.genericness_penalty,
        "ai_smell_reduction": rejected.ai_smell_penalty - chosen.ai_smell_penalty,
    }
    return max(deltas.items(), key=lambda item: item[1])[0]


def _bounded(value: float) -> float:
    return max(1.0, min(7.0, float(value)))


def main() -> None:
    print(json.dumps(export_seed_taste_datasets(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

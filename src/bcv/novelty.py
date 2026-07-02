"""Semantic novelty judge: is an expression outside the symbolic miner's reach?

"The model proposed something the miner didn't" is meaningless as string comparison —
`(is_tree) and (max_degree >= 3)` and `is_tree and not max_degree_le_2` are different
strings with the same extension. This module decides novelty *semantically*: an
expression's match-set over the exhaustive small-n universe is compared against the
full 1-atom and 2-atom conjunctive closure of the miner's atom vocabulary. Only an
expression whose match-set no conjunction of at most two atoms can reproduce counts
as outside the enumerable hull.

That is the earn-your-keep bar for a model in this loop: anything inside the hull is
distillation of the miner; anything outside it that also verifies is discovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from bcv.graph_agent import _atomic_refinements, _observations_for, compile_feature_expression


@dataclass(frozen=True)
class NoveltyVerdict:
    expression: str
    parseable: bool
    support: int
    in_atom_vocabulary: bool
    enumerated_one_atom: bool
    enumerated_two_atom: bool
    semantically_novel: bool


class NoveltyJudge:
    """Match-set bitmask closure over the exhaustive n <= max_n graph universe."""

    def __init__(self, max_n: int = 6) -> None:
        self.max_n = max_n
        self.observations = _observations_for(max_n)
        self.atoms = _atomic_refinements(list(self.observations))
        self._atom_masks = {atom: self._compute_mask(atom) for atom in self.atoms}
        self.universe_mask = (1 << len(self.observations)) - 1

    def mask(self, expression: str) -> int:
        return self._compute_mask(expression)

    def _compute_mask(self, expression: str) -> int:
        predicate = compile_feature_expression(expression)
        mask = 0
        for index, observation in enumerate(self.observations):
            if predicate(observation):
                mask |= 1 << index
        return mask

    @lru_cache(maxsize=64)
    def _hull(self, base_expression: str | None) -> frozenset[int]:
        """All match-sets reachable as base & (<= 2 atoms). base None = free proposals."""
        base_mask = self.universe_mask if base_expression is None else self._compute_mask(base_expression)
        masks = {base_mask}
        atom_masks = list(self._atom_masks.values())
        one_atom = [base_mask & mask for mask in atom_masks]
        masks.update(one_atom)
        for i, first in enumerate(atom_masks):
            base_and_first = base_mask & first
            if not base_and_first:
                continue
            for second in atom_masks[i + 1 :]:
                masks.add(base_and_first & second)
        return frozenset(masks)

    def judge(self, expression: str, base_expression: str | None = None) -> NoveltyVerdict:
        """Judge a proposal against the miner hull.

        base_expression scopes the hull to repairs of a specific original rule;
        None judges a free conjecture against all <=2-atom conjunctions.
        """
        try:
            mask = self._compute_mask(expression)
        except (SyntaxError, ValueError, TypeError, KeyError):
            return NoveltyVerdict(
                expression=expression,
                parseable=False,
                support=0,
                in_atom_vocabulary=False,
                enumerated_one_atom=False,
                enumerated_two_atom=False,
                semantically_novel=False,
            )
        base_mask = self.universe_mask if base_expression is None else self._compute_mask(base_expression)
        one_atom_masks = {base_mask & atom_mask for atom_mask in self._atom_masks.values()}
        one_atom_masks.add(base_mask)
        enumerated_one = mask in one_atom_masks
        enumerated_two = enumerated_one or mask in self._hull(base_expression)
        return NoveltyVerdict(
            expression=expression,
            parseable=True,
            support=mask.bit_count(),
            in_atom_vocabulary=expression.strip() in self._atom_masks,
            enumerated_one_atom=enumerated_one,
            enumerated_two_atom=enumerated_two,
            semantically_novel=not enumerated_two,
        )


@lru_cache(maxsize=4)
def default_judge(max_n: int = 6) -> NoveltyJudge:
    return NoveltyJudge(max_n=max_n)

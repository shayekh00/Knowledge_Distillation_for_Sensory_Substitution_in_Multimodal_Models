"""Genuine constrained decoding and candidate scoring (plan §6.3, WP2).

`evaluate.py --constrained` approximates constrained decoding *after the fact* by
snapping a free-form answer onto the row's answer space. The plan is explicit that
this is not the real thing, and that the real thing is one of two mechanisms:

* score every permissible answer sequence with the model, or
* restrict autoregressive decoding with a **token trie**.

Both are implemented here against a small scorer interface, so the trie logic and
the length/EOS conventions are verified on CPU with a stub. Wiring a real model in
is then a matter of supplying `next_token_logits` and `sequence_logprob`.

Conventions that must be fixed before any confirmatory run, because they change
the ranking rather than merely rescaling it:

* whether a sequence score sums token log-probabilities or applies a length
  correction (`LengthPolicy`);
* whether EOS is included in the score.

Both are chosen on validation data and then used for every model. Comparing
individual first-token logits for multi-token answers is not a valid substitute
and is not offered here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

LENGTH_POLICIES = ("sum", "mean", "none")


class SequenceScorer(Protocol):
    """What a model must expose for candidate scoring."""

    def token_logprobs(self, prefix: Sequence[int], continuation: Sequence[int]) -> list[float]:
        """Log-probability of each continuation token given the prefix."""


# ---------------------------------------------------------------------------
# Token trie for constrained generation
# ---------------------------------------------------------------------------

class TokenTrie:
    """Prefix trie over the legal answer token sequences.

    At each decoding step the trie reports exactly which tokens can legally come
    next, so the model cannot emit an illegal string at all. This is what makes
    "constrained" mean constrained, rather than repaired afterwards.
    """

    def __init__(self, sequences: Sequence[Sequence[int]]):
        if not sequences:
            raise ValueError("a token trie needs at least one legal sequence")
        self.root: dict = {}
        self.n_sequences = 0
        for sequence in sequences:
            if not sequence:
                raise ValueError("empty answer sequence is not a legal candidate")
            self._insert(list(sequence))
            self.n_sequences += 1

    def _insert(self, sequence: list[int]) -> None:
        node = self.root
        for token in sequence:
            node = node.setdefault(token, {})
        node["__end__"] = True

    def allowed_next(self, prefix: Sequence[int]) -> set[int]:
        """Tokens that may follow `prefix`. Empty when the prefix is complete or
        has left the trie."""
        node = self.root
        for token in prefix:
            if token not in node:
                return set()
            node = node[token]
        return {token for token in node if token != "__end__"}

    def is_complete(self, prefix: Sequence[int]) -> bool:
        node = self.root
        for token in prefix:
            if token not in node:
                return False
            node = node[token]
        return bool(node.get("__end__"))

    def is_prefix(self, prefix: Sequence[int]) -> bool:
        node = self.root
        for token in prefix:
            if token not in node:
                return False
            node = node[token]
        return True


def constrained_decode(trie: TokenTrie, next_token_logits, max_steps: int = 16) -> list[int]:
    """Greedy decoding restricted to the trie.

    `next_token_logits(prefix)` returns a mapping token -> score. Only tokens the
    trie permits are considered, so the output is always a legal answer. Stops as
    soon as a complete answer is reached and nothing may follow it.
    """
    prefix: list[int] = []
    for _ in range(max_steps):
        allowed = trie.allowed_next(prefix)
        if not allowed:
            break
        if trie.is_complete(prefix) and not allowed:
            break
        logits = next_token_logits(prefix)
        best = max(allowed, key=lambda token: logits.get(token, -math.inf))
        prefix.append(best)
        if trie.is_complete(prefix) and not trie.allowed_next(prefix):
            break
    if not trie.is_complete(prefix):
        raise ValueError(
            f"constrained decoding ended on an incomplete answer {prefix}; "
            "the trie or the step budget is wrong")
    return prefix


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LengthPolicy:
    """How a sequence score combines its token log-probabilities.

    `sum` favours short answers, `mean` removes the length effect entirely.
    Neither is universally right, which is exactly why it must be declared on
    validation data and then held fixed for every model.
    """
    kind: str = "sum"
    include_eos: bool = False

    def __post_init__(self):
        if self.kind not in LENGTH_POLICIES:
            raise ValueError(f"unknown length policy {self.kind!r}; expected {LENGTH_POLICIES}")

    def combine(self, token_logprobs: Sequence[float]) -> float:
        if not token_logprobs:
            raise ValueError("cannot score an empty continuation")
        if self.kind == "mean":
            return float(sum(token_logprobs) / len(token_logprobs))
        return float(sum(token_logprobs))


def score_candidates(scorer: SequenceScorer, prefix: Sequence[int],
                     candidates: dict[str, Sequence[int]],
                     policy: LengthPolicy | None = None) -> dict[str, float]:
    """Log-score for every legal answer, under one declared convention.

    Args:
        candidates: answer text -> its token ids **in that model's own
            tokenization**. Never share token ids across models.
    """
    if not candidates:
        raise ValueError("no candidates to score")
    policy = policy or LengthPolicy()
    return {answer: policy.combine(scorer.token_logprobs(prefix, tokens))
            for answer, tokens in candidates.items()}


def candidate_distribution(scores: dict[str, float], temperature: float = 1.0) -> dict[str, float]:
    """Normalize candidate log-scores into a distribution.

    This is the teacher signal cached for candidate KD, and the student side of
    the same objective. Each model reaches it through its own tokenizer, so the
    two are comparable as distributions over *answers* without any assumption
    about shared vocabulary indices.
    """
    if not scores:
        raise ValueError("no candidate scores to normalize")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = {answer: score / temperature for answer, score in scores.items()}
    highest = max(scaled.values())
    weights = {answer: math.exp(score - highest) for answer, score in scaled.items()}
    total = sum(weights.values())
    return {answer: weight / total for answer, weight in weights.items()}


def argmax_candidate(scores: dict[str, float]) -> str:
    """The chosen answer. Ties break on the answer string, so the result does not
    depend on dictionary insertion order."""
    best = max(scores.values())
    return sorted(answer for answer, score in scores.items() if score == best)[0]

"""Cross-tokenizer knowledge distillation — X-Token (plan §7.2.1, WP3b).

Ordinary logit KD assumes the teacher and student distributions live over the
same vocabulary at the same positions. Across tokenizer families that fails on
every axis at once: token boundaries, sequence lengths, vocabulary sizes, and
token IDs are all unrelated. One tokenizer emits ``"201"`` as a single token where
another emits ``"2"``, ``"0"``, ``"1"``.

The invariant two language models *do* share is the decoded text. X-Token
therefore works on text:

1. **Independent tokenization** — neither model ever consumes the other's tokens.
2. **Span alignment** (:func:`align_spans`) — teacher and student output positions
   are grouped into minimal spans whose character boundaries coincide. Explicitly
   many-to-many.
3. **Sparse vocabulary projection** (:func:`build_vocabulary_mapping`) — a
   many-to-one student→teacher map, built from exact surface matches where they
   exist and decode-and-retokenize otherwise. Sparse by construction: no dense
   ``V_student x V_teacher`` matrix is ever allocated, which at 150k-200k
   vocabularies on both sides would be infeasible.
4. **Projected KL** (:func:`projected_kl_loss`) — student mass is scattered into
   teacher token space through the map, then compared against the teacher.

What this module deliberately does **not** do: force either tokenizer onto the
other model, resize or replace a vocabulary, assume equal IDs carry equal
meaning, or touch visual tokens. X-Token operates on the autoregressive textual
answer positions only, which is what keeps it architecture-independent.

Tokenizers are duck-typed on three methods — ``get_vocab()``, ``decode(ids)`` and
``encode(text)`` — so this is testable against synthetic tokenizers with
deliberately mismatched boundaries, without downloading a model.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

DEFAULT_TOP_K = 4096
SUPPORTED_TOP_K = (2048, 4096, 8192)


# ---------------------------------------------------------------------------
# Span alignment
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Span:
    """One aligned region of the answer, as index ranges into each token list.

    ``teacher`` and ``student`` are half-open ``(start, end)`` ranges. Either may
    cover several tokens: a span is the *minimal* region whose character
    boundaries agree in both tokenizations, so it is one-to-one only when the two
    tokenizers happen to agree.
    """
    teacher: tuple[int, int]
    student: tuple[int, int]
    text: str


def align_spans(teacher_pieces: list[str], student_pieces: list[str]) -> list[Span]:
    """Group two tokenizations of the *same text* into minimal aligned spans.

    Walks both sequences accumulating character counts and closes a span each
    time the running totals coincide. ``"201"`` against ``"2","0","1"`` yields one
    span covering one teacher token and three student tokens.

    Raises:
        ValueError: if the two tokenizations do not decode to identical text.
            Silently aligning different strings would produce a meaningless
            objective, so this is an error rather than a best effort.
    """
    teacher_text, student_text = "".join(teacher_pieces), "".join(student_pieces)
    if teacher_text != student_text:
        raise ValueError(
            "teacher and student tokenizations decode to different text:\n"
            f"  teacher: {teacher_text!r}\n  student: {student_text!r}")
    if not teacher_text:
        return []

    spans: list[Span] = []
    t_i = s_i = 0
    t_chars = s_chars = 0
    t_start = s_start = 0
    while t_i < len(teacher_pieces) or s_i < len(student_pieces):
        if t_chars <= s_chars and t_i < len(teacher_pieces):
            t_chars += len(teacher_pieces[t_i])
            t_i += 1
        elif s_i < len(student_pieces):
            s_chars += len(student_pieces[s_i])
            s_i += 1
        else:                                    # pragma: no cover - guarded above
            break
        if t_chars == s_chars:
            spans.append(Span(teacher=(t_start, t_i), student=(s_start, s_i),
                              text=teacher_text[len(("".join(teacher_pieces[:t_start]))):t_chars]))
            t_start, s_start = t_i, s_i
    if t_start != len(teacher_pieces) or s_start != len(student_pieces):
        raise ValueError("alignment left trailing tokens unassigned")
    return spans


def alignment_covers_everything(spans: list[Span], n_teacher: int, n_student: int) -> bool:
    """Every teacher and student answer position belongs to exactly one span."""
    teacher_covered = sum(end - start for start, end in (s.teacher for s in spans))
    student_covered = sum(end - start for start, end in (s.student for s in spans))
    return teacher_covered == n_teacher and student_covered == n_student


# ---------------------------------------------------------------------------
# Sparse vocabulary mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VocabularyMapping:
    """Sparse many-to-one student→teacher token map for one tokenizer pair.

    Stored as two parallel index tensors, never as a dense matrix. Several student
    tokens may share a teacher target — that is expected, and projection sums
    their probability mass.
    """
    student_ids: torch.Tensor
    teacher_ids: torch.Tensor
    student_vocab_size: int
    teacher_vocab_size: int
    exact_matches: int
    retokenized: int
    unmapped: int
    student_revision: str = "unknown"
    teacher_revision: str = "unknown"
    metadata: dict = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        """Fraction of the student vocabulary with any teacher target."""
        return len(self.student_ids) / max(self.student_vocab_size, 1)

    @property
    def exact_match_fraction(self) -> float:
        """Share of mapped entries resolved by an exact surface match rather than
        by decode-and-retokenize. A low value is a signal to scrutinise results
        from this pair, not automatically a failure."""
        mapped = self.exact_matches + self.retokenized
        return self.exact_matches / mapped if mapped else 0.0

    def is_identity(self) -> bool:
        """True when the map sends every token to itself — the case that must
        reduce P-KL to ordinary token KL."""
        return (self.student_vocab_size == self.teacher_vocab_size
                and len(self.student_ids) == self.student_vocab_size
                and bool(torch.equal(self.student_ids, self.teacher_ids)))

    def content_hash(self) -> str:
        """Stable hash for run manifests. Two runs with different mappings are
        different experiments."""
        digest = hashlib.sha256()
        digest.update(f"{self.student_revision}|{self.teacher_revision}".encode())
        digest.update(self.student_ids.numpy().tobytes())
        digest.update(self.teacher_ids.numpy().tobytes())
        return digest.hexdigest()[:16]


def build_vocabulary_mapping(student_tokenizer, teacher_tokenizer,
                             student_revision: str = "unknown",
                             teacher_revision: str = "unknown") -> VocabularyMapping:
    """Construct the sparse student→teacher map for one tokenizer pair.

    For each student token, take its surface text. If some teacher token has the
    same surface, map to it (exact match). Otherwise re-tokenize the surface with
    the teacher tokenizer and map to the **first** resulting token: in next-token
    prediction, a student asserting "the continuation is X" corresponds to the
    teacher placing mass on the first token of its own tokenization of X.

    Cost is one pass over each vocabulary, so this is built once per tokenizer
    pair and cached (:func:`load_or_build_mapping`), never per batch.
    """
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()

    # Surface text -> teacher id. Built once; the reverse direction is what the
    # exact-match lookup needs.
    teacher_by_surface: dict[str, int] = {}
    for teacher_id in sorted(teacher_vocab.values()):
        surface = teacher_tokenizer.decode([teacher_id])
        teacher_by_surface.setdefault(surface, teacher_id)

    student_ids: list[int] = []
    teacher_ids: list[int] = []
    exact = retokenized = unmapped = 0
    for student_id in sorted(student_vocab.values()):
        surface = student_tokenizer.decode([student_id])
        if not surface:
            unmapped += 1
            continue
        target = teacher_by_surface.get(surface)
        if target is not None:
            exact += 1
        else:
            # A teacher may be unable to represent this text at all. That is a
            # legitimate outcome — the entry is left unmapped and its probability
            # mass is dropped at projection — never a crash, and never a guess at
            # some unrelated token. Real tokenizers usually fall back to byte
            # pieces, so this branch is rare in practice but must be safe.
            try:
                pieces = teacher_tokenizer.encode(surface)
            except Exception:
                pieces = []
            if not pieces:
                unmapped += 1
                continue
            target = int(pieces[0])
            retokenized += 1
        student_ids.append(student_id)
        teacher_ids.append(target)

    return VocabularyMapping(
        student_ids=torch.tensor(student_ids, dtype=torch.long),
        teacher_ids=torch.tensor(teacher_ids, dtype=torch.long),
        student_vocab_size=len(student_vocab),
        teacher_vocab_size=len(teacher_vocab),
        exact_matches=exact, retokenized=retokenized, unmapped=unmapped,
        student_revision=student_revision, teacher_revision=teacher_revision)


def load_or_build_mapping(student_tokenizer, teacher_tokenizer, cache_dir: str,
                          student_revision: str, teacher_revision: str) -> VocabularyMapping:
    """Cached mapping, keyed by **both** tokenizer revisions.

    A revision change produces a different cache key rather than silently reusing
    a stale map — the failure mode that would otherwise be invisible and would
    corrupt every downstream comparison.
    """
    key = hashlib.sha256(f"{student_revision}|{teacher_revision}".encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"xtoken_mapping_{key}.pt")
    if os.path.isfile(path):
        payload = torch.load(path)
        return VocabularyMapping(**payload)
    mapping = build_vocabulary_mapping(student_tokenizer, teacher_tokenizer,
                                       student_revision, teacher_revision)
    os.makedirs(cache_dir, exist_ok=True)
    torch.save({
        "student_ids": mapping.student_ids, "teacher_ids": mapping.teacher_ids,
        "student_vocab_size": mapping.student_vocab_size,
        "teacher_vocab_size": mapping.teacher_vocab_size,
        "exact_matches": mapping.exact_matches, "retokenized": mapping.retokenized,
        "unmapped": mapping.unmapped,
        "student_revision": student_revision, "teacher_revision": teacher_revision,
        "metadata": mapping.metadata,
    }, path)
    return mapping


# ---------------------------------------------------------------------------
# Projection and P-KL
# ---------------------------------------------------------------------------

def project_student_probs(student_probs: torch.Tensor,
                          mapping: VocabularyMapping) -> torch.Tensor:
    """Scatter student probability mass into teacher token space.

    Args:
        student_probs: ``[N, V_student]``, rows summing to 1.
    Returns:
        ``[N, V_teacher]``. Mass on student tokens with no teacher target is
        dropped, so rows may sum to less than 1; :func:`projected_kl_loss`
        renormalizes over the compared support.
    """
    if student_probs.size(-1) != mapping.student_vocab_size:
        raise ValueError(
            f"student_probs has {student_probs.size(-1)} columns but the mapping was "
            f"built for a {mapping.student_vocab_size}-entry student vocabulary")
    rows = student_probs.size(0)
    projected = student_probs.new_zeros((rows, mapping.teacher_vocab_size))
    contribution = student_probs.index_select(1, mapping.student_ids.to(student_probs.device))
    index = mapping.teacher_ids.to(student_probs.device).expand(rows, -1)
    projected.scatter_add_(1, index, contribution)
    return projected


def projected_kl_loss(teacher_topk_ids: torch.Tensor, teacher_topk_probs: torch.Tensor,
                      student_probs: torch.Tensor, mapping: VocabularyMapping,
                      temperature: float = 1.0) -> torch.Tensor:
    """P-KL: KL between the teacher and the projected student, over retained support.

    Both distributions are renormalized over the teacher's retained top-K support
    before the divergence is taken, so the result is an *approximate* KL whenever
    K is less than the full vocabulary. Report the omitted teacher mass alongside
    it (:func:`omitted_teacher_mass`) — a renormalized top-K distribution is not
    exact full-vocabulary KD and must never be described as such.

    Args:
        teacher_topk_ids: ``[N, K]`` teacher vocabulary indices.
        teacher_topk_probs: ``[N, K]`` teacher probabilities at those indices.
        student_probs: ``[N, V_student]``.
    """
    if teacher_topk_ids.shape != teacher_topk_probs.shape:
        raise ValueError("teacher top-K ids and probs must have the same shape")
    if teacher_topk_ids.size(0) != student_probs.size(0):
        raise ValueError(
            f"span count differs: teacher has {teacher_topk_ids.size(0)} rows, "
            f"student has {student_probs.size(0)}")

    projected = project_student_probs(student_probs.float(), mapping)
    student_at_k = projected.gather(1, teacher_topk_ids)

    teacher = teacher_topk_probs.float() / temperature if temperature != 1.0 \
        else teacher_topk_probs.float()
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-12)
    student_at_k = student_at_k / student_at_k.sum(-1, keepdim=True).clamp_min(1e-12)

    per_row = (teacher * (teacher.clamp_min(1e-12).log()
                          - student_at_k.clamp_min(1e-12).log())).sum(-1)
    return per_row.mean() * (temperature ** 2)


def omitted_teacher_mass(teacher_topk_probs: torch.Tensor) -> torch.Tensor:
    """Probability mass discarded by top-K truncation, per row.

    Must be reported with any top-K result. It is temperature-dependent: a tail
    cached at one temperature cannot be converted to another without the omitted
    logits.
    """
    return (1.0 - teacher_topk_probs.float().sum(-1)).clamp_min(0.0)


def aggregate_spans(distributions: torch.Tensor, spans: list[Span],
                    which: str = "teacher") -> torch.Tensor:
    """Reduce per-token distributions to one row per aligned span.

    A span may cover several positions on either side, so the within-span rule has
    to be fixed before the confirmatory runs; the default is the mean over the
    positions in the span. Declared in the protocol, not chosen per run.
    """
    if which not in ("teacher", "student"):
        raise ValueError(f"which must be 'teacher' or 'student', got {which!r}")
    rows = []
    for span in spans:
        start, end = span.teacher if which == "teacher" else span.student
        rows.append(distributions[start:end].mean(dim=0))
    if not rows:
        raise ValueError("no spans to aggregate")
    return torch.stack(rows)


def mapping_report(mapping: VocabularyMapping) -> dict:
    """Manifest fields for an X-Token run (protocol §10)."""
    return {
        "mapping_hash": mapping.content_hash(),
        "student_revision": mapping.student_revision,
        "teacher_revision": mapping.teacher_revision,
        "student_vocab_size": mapping.student_vocab_size,
        "teacher_vocab_size": mapping.teacher_vocab_size,
        "coverage": round(mapping.coverage, 6),
        "exact_match_fraction": round(mapping.exact_match_fraction, 6),
        "exact_matches": mapping.exact_matches,
        "retokenized": mapping.retokenized,
        "unmapped": mapping.unmapped,
        "is_identity": mapping.is_identity(),
    }


def format_mapping_report(mapping: VocabularyMapping) -> str:
    return json.dumps(mapping_report(mapping), indent=2)

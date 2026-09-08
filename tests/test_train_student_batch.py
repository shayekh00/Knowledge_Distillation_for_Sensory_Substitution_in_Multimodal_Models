"""build_batch / build_batch_with_answers: the D9 injection point.

D9 (experiment_protocol.md §8.1) needs the forced answer text to come from the
teacher's own generated completion, never from `row["answer"]`. `build_batch`
was refactored into a thin gold-answer wrapper around
`build_batch_with_answers`, which takes the answer text as an explicit
`{question_id: text}` mapping — these tests pin that (a) the refactor changed
nothing about `build_batch`'s own behaviour, and (b) the general function
actually uses the supplied mapping rather than silently falling back to gold.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from distillation.train_student import build_batch, build_batch_with_answers  # noqa: E402


class FakeTokenizer:
    eos_token = "<eos>"


class FakeProcessor:
    """Records every text it is asked to tokenize; each character becomes one
    token id so the recorded full text tells the test everything it needs."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def apply_chat_template(self, messages, add_generation_prompt, tokenize,
                            enable_thinking):
        # `question` already carries "{question}\n{PROMPT_SUFFIX}" — see
        # build_batch_with_answers's message construction.
        question = messages[0]["content"][1]["text"]
        return f"PROMPT[{question.splitlines()[0]}]"

    def __call__(self, images, text, padding, return_tensors):
        import torch
        self.calls.append(list(text))
        max_len = max(len(t) for t in text)
        ids = torch.zeros((len(text), max_len), dtype=torch.long)
        mask = torch.zeros((len(text), max_len), dtype=torch.long)
        for i, t in enumerate(text):
            ids[i, :len(t)] = 1
            mask[i, :len(t)] = 1
        return {"input_ids": ids, "attention_mask": mask, "pixel_values": None}


def make_rows():
    return [
        {"question_id": "q0", "question": "is there a chair", "answer": "yes"},
        {"question_id": "q1", "question": "where is the lamp", "answer": "left"},
    ]


def test_build_batch_uses_each_rows_own_gold_answer():
    processor = FakeProcessor()
    rows = make_rows()
    build_batch(processor, rows, images=[None, None])
    full_texts = processor.calls[1]
    assert full_texts[0] == "PROMPT[is there a chair]yes<eos>"
    assert full_texts[1] == "PROMPT[where is the lamp]left<eos>"


def test_build_batch_with_answers_uses_the_supplied_mapping_not_gold():
    """The D9 case: the mapping holds the teacher's own generated text, and
    every row's gold `answer` field is irrelevant — it could even be absent."""
    processor = FakeProcessor()
    rows = [
        {"question_id": "q0", "question": "is there a chair", "answer": "yes"},
        {"question_id": "q1", "question": "where is the lamp", "answer": "left"},
    ]
    generated = {"q0": "no", "q1": "right"}
    build_batch_with_answers(processor, rows, images=[None, None], answers=generated)
    full_texts = processor.calls[1]
    assert full_texts[0] == "PROMPT[is there a chair]no<eos>"
    assert full_texts[1] == "PROMPT[where is the lamp]right<eos>"


def test_build_batch_with_answers_works_when_gold_answer_column_is_absent():
    """§8.1: D9's training/cache interface must succeed with the gold answer
    column removed entirely, not merely unused."""
    processor = FakeProcessor()
    rows = [
        {"question_id": "q0", "question": "is there a chair"},
        {"question_id": "q1", "question": "where is the lamp"},
    ]
    generated = {"q0": "no", "q1": "right"}
    batch = build_batch_with_answers(processor, rows, images=[None, None], answers=generated)
    assert "labels" in batch

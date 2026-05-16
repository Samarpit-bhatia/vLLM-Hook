"""TruthfulQA → (prompt, label) pair builder.

label = 0 for grounded (question + correct answer), 1 for hallucinated
(question + incorrect answer). Prompts use the bare ``Q: ... A: ...`` format
the paper uses to keep activation-space signal clean; chat-format prompting
would confound the mechanistic measurement.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LabeledPrompt:
    prompt: str
    label: int  # 0 = grounded, 1 = hallucinated
    question_id: int


def _format(question: str, answer: str) -> str:
    return f"Q: {question.strip()}\nA: {answer.strip()}"


def build_truthfulqa_pairs(
    n_questions: int = 300,
    seed: int = 42,
    split: str = "validation",
) -> List[LabeledPrompt]:
    """Load TruthfulQA MC1 and emit one grounded + one hallucinated prompt per question.

    Requires the ``datasets`` library. Output is balanced and shuffled.
    """
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split=split)
    rng = random.Random(seed)

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    indices = indices[:n_questions]

    pairs: List[LabeledPrompt] = []
    for qid in indices:
        row = ds[qid]
        question = row["question"]
        choices = row["mc1_targets"]["choices"]
        labels = row["mc1_targets"]["labels"]

        correct = [c for c, lab in zip(choices, labels) if lab == 1]
        incorrect = [c for c, lab in zip(choices, labels) if lab == 0]
        if not correct or not incorrect:
            continue

        pairs.append(LabeledPrompt(
            prompt=_format(question, rng.choice(correct)),
            label=0,
            question_id=qid,
        ))
        pairs.append(LabeledPrompt(
            prompt=_format(question, rng.choice(incorrect)),
            label=1,
            question_id=qid,
        ))

    rng.shuffle(pairs)
    return pairs


def split_pairs(
    pairs: List[LabeledPrompt],
    train_frac: float = 0.7,
    seed: int = 42,
) -> Tuple[List[LabeledPrompt], List[LabeledPrompt]]:
    """Question-grouped split: all prompts from one question land in the same split."""
    rng = random.Random(seed)
    qids = sorted({p.question_id for p in pairs})
    rng.shuffle(qids)
    n_train = int(len(qids) * train_frac)
    train_qids = set(qids[:n_train])

    train = [p for p in pairs if p.question_id in train_qids]
    eval_ = [p for p in pairs if p.question_id not in train_qids]
    return train, eval_

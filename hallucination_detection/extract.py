"""Run a model under ``probe_hidden_states`` and dump labeled last-token activations.

For each prompt we run a single-token generation (prefill only — the last-token
hidden state is what the H-Node probe consumes) and collect every layer listed
in the model config. The dumped tensor file is consumed by ``train_probe.py``.
"""
from __future__ import annotations

import os
from typing import List

import torch

from hallucination_detection.data import LabeledPrompt


def extract_activations(
    llm,
    pairs: List[LabeledPrompt],
    batch_size: int = 8,
) -> dict:
    """Run prompts through ``llm`` (a HookLLM configured with probe_hidden_states)
    and return ``{layer_num: tensor(N, hidden), labels: tensor(N), question_ids: list}``.

    Activations are stacked in prompt order. The analyzer attached to ``llm``
    must be ``hidden_states`` (``analyzer_spec={"reduce": "none"}``).
    """
    per_layer: dict[int, list[torch.Tensor]] = {}
    labels: list[int] = []
    qids: list[int] = []

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p.prompt for p in batch]

        llm.generate(prompts, temperature=0.0, max_tokens=1)
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        # Map module name (0-based PyTorch index) -> paper's 1-based layer_num
        # by sorting on the trailing integer in the module name. This matches
        # ProbeHiddenStatesWorker's LAYER_PATTERNS keying.
        ordered = sorted(
            stats["hidden_states"].items(),
            key=lambda kv: int(kv[0].rsplit(".", 1)[-1]),
        )
        for module_name, tensors in ordered:
            layer_num = int(module_name.rsplit(".", 1)[-1]) + 1
            for t in tensors:
                per_layer.setdefault(layer_num, []).append(t.float().cpu())

        labels.extend(p.label for p in batch)
        qids.extend(p.question_id for p in batch)

        llm.llm_engine.reset_prefix_cache()

    stacked = {layer: torch.stack(ts) for layer, ts in per_layer.items()}
    return {
        "activations": stacked,
        "labels": torch.tensor(labels, dtype=torch.long),
        "question_ids": qids,
    }


def save_activations(bundle: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(bundle, out_path)


def load_activations(path: str) -> dict:
    return torch.load(path, map_location="cpu")

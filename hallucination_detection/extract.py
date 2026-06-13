"""Run a model under ``probe_hidden_states`` and dump labeled last-token activations.

For each prompt we run a single-token generation (prefill only — the last-token
hidden state is what the H-Node probe consumes) and collect every layer listed
in the model config. The dumped tensor file is consumed by ``train_probe.py``.

Uses the post-refactor disk path: ``HookLLM.generate(..., save_to_disk=True,
run_id=...)`` flushes the whole batch into one artifact, then
``HookLLM.analyze(run_id=...)`` (HiddenStatesAnalyzer, reduce="none") returns the
captured hidden states keyed by module name. The disk path merges all requests in
a batch deterministically, which the per-request in-memory ``output.probes`` path
does not guarantee.
"""
from __future__ import annotations

import os
from typing import List

import torch
from vllm import SamplingParams

from hallucination_detection.data import LabeledPrompt


def extract_activations(
    llm,
    pairs: List[LabeledPrompt],
    batch_size: int = 8,
) -> dict:
    """Run prompts through ``llm`` (a HookLLM configured with probe_hidden_states)
    and return ``{layer_num: tensor(N, hidden), labels: tensor(N), question_ids: list}``.

    Activations are stacked in prompt order. ``llm`` must be configured with
    ``worker_name="probe_hidden_states"``, ``analyzer_name="hidden_states"`` and
    the training config (all layers, last_token mode).
    """
    per_layer: dict[int, list[torch.Tensor]] = {}
    labels: list[int] = []
    qids: list[int] = []

    sp = SamplingParams(temperature=0.0, max_tokens=1)

    for start in range(0, len(pairs), batch_size):
        batch = pairs[start:start + batch_size]
        prompts = [p.prompt for p in batch]

        run_id = f"halludetect_extract_{start}"
        llm.generate(prompts, sp, save_to_disk=True, run_id=run_id)
        stats = llm.analyze(analyzer_spec={"reduce": "none"}, run_id=run_id)
        hs = stats["hidden_states"]  # {module_name: [tensor, ...]}, prompt order

        # Map module name (0-based PyTorch index) -> paper's 1-based layer_num,
        # iterating in stable sorted order.
        ordered = sorted(
            hs.items(),
            key=lambda kv: int(kv[0].rsplit(".", 1)[-1]),
        )
        for module_name, tensors in ordered:
            layer_num = int(module_name.rsplit(".", 1)[-1]) + 1
            for t in tensors:
                row = t if t.dim() == 1 else t[-1]
                per_layer.setdefault(layer_num, []).append(row.float().cpu())

        labels.extend(p.label for p in batch)
        qids.extend(p.question_id for p in batch)

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

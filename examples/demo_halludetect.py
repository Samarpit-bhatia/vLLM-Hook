"""End-to-end H-Node hallucination detection demo.

Three stages, each runs as a separate `python examples/demo_halludetect.py <stage>`:

  extract  — load TruthfulQA, run prompts through Qwen2.5-3B-Instruct under
             ProbeHiddenStatesWorker (vLLM-Hook), dump per-layer last-token
             hidden states + labels to artifacts/activations.pt.

  train    — fit per-layer logistic-regression probes on the dump, pick the
             best layer by held-out AUC, identify top-50 H-Nodes, save
             artifacts/probe.npz + probe.json. Rewrites the infer config so
             its hidden_states.layers matches the chosen best layer.

  detect   — reload the model with the (now-narrow) infer config and the
             registered `hallucination` analyzer; score a few example prompts.

Run all three in order:
    python examples/demo_halludetect.py extract
    python examples/demo_halludetect.py train
    python examples/demo_halludetect.py detect
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

# Make the project root importable so `hallucination_detection` resolves when
# the script is invoked as `python examples/demo_halludetect.py ...`. Append
# (not insert at 0) so the pip-installed `vllm_hook_plugins` package wins over
# the outer source dir of the same name at the project root — the outer is a
# distribution root, not the package; only the pip-installed inner package
# resolves its own absolute imports correctly.
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.append(_root)

import torch

mp.set_start_method("spawn", force=True)
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ.setdefault("VLLM_HOOK_USE_SAFETENSORS", "1")
os.environ.setdefault("VLLM_HOOK_ASYNC_SAVE", "1")
# T4 (SM 7.5) doesn't support FlashAttention 2 or 3. vllm 0.7.3 won't fall
# back automatically and dies with "Unsupported FA version: None". Force
# xformers, which works on every CUDA SM. Override only if not already set.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

MODEL = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "./cache/"
HOOK_DIR = "/dev/shm/vllm_hook"
ART_DIR = "./hallucination_detection/artifacts"
ACTIVATIONS_PATH = os.path.join(ART_DIR, "activations.pt")
PROBE_PATH = os.path.join(ART_DIR, "probe.npz")
TRAIN_CFG = "model_configs/hallucination_detection/Qwen2.5-3B-Instruct.train.json"
INFER_CFG = "model_configs/hallucination_detection/Qwen2.5-3B-Instruct.infer.json"


def _make_llm(config_file: str):
    from vllm_hook_plugins import HookLLM

    base_kwargs = dict(
        model=MODEL,
        worker_name="probe_hidden_states",
        analyzer_name="hidden_states",
        config_file=config_file,
        download_dir=CACHE_DIR,
        hook_dir=HOOK_DIR,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )
    # vLLM v0.21+ enables async scheduling by default, which desyncs the
    # framework's forward-hook capture (execute_model fires before the
    # batch's attn_metadata is ready, so query_start_loc is None and the
    # hook returns without recording — no safetensors get written).
    # Force sync scheduling on versions that accept the kwarg.
    try:
        return HookLLM(**base_kwargs, async_scheduling=False)
    except TypeError:
        return HookLLM(**base_kwargs)


def stage_extract():
    from hallucination_detection.data import build_truthfulqa_pairs
    from hallucination_detection.extract import extract_activations, save_activations

    os.makedirs(ART_DIR, exist_ok=True)

    print("Building TruthfulQA labeled pairs...")
    pairs = build_truthfulqa_pairs(n_questions=300, seed=42)
    print(f"  {len(pairs)} prompts ({sum(p.label == 0 for p in pairs)} grounded / "
          f"{sum(p.label == 1 for p in pairs)} hallucinated)")

    print("Loading model under ProbeHiddenStatesWorker (training config)...")
    llm = _make_llm(TRAIN_CFG)

    print("Extracting last-token hidden states across all layers...")
    bundle = extract_activations(llm, pairs, batch_size=8)

    save_activations(bundle, ACTIVATIONS_PATH)
    n_layers = len(bundle["activations"])
    hidden = next(iter(bundle["activations"].values())).shape[1]
    print(f"Saved → {ACTIVATIONS_PATH}  (layers={n_layers}, hidden_size={hidden}, "
          f"N={len(bundle['labels'])})")


def stage_train():
    from hallucination_detection.train_probe import train_from_bundle
    from hallucination_detection.data import build_truthfulqa_pairs, split_pairs
    import numpy as np

    # Rebuild the same pairs (same seed) so we can apply a question-grouped
    # train/eval mask matching what was extracted in prompt order.
    pairs = build_truthfulqa_pairs(n_questions=300, seed=42)
    train_pairs, eval_pairs = split_pairs(pairs, train_frac=0.7, seed=42)
    train_qids = {p.question_id for p in train_pairs}

    bundle = torch.load(ACTIVATIONS_PATH, map_location="cpu")
    qids = bundle["question_ids"]
    train_mask = np.array([q in train_qids for q in qids], dtype=bool)

    print(f"Training H-Node probe on {train_mask.sum()} samples "
          f"(eval={(~train_mask).sum()})")
    activations = {l: t.numpy() for l, t in bundle["activations"].items()}
    labels = bundle["labels"].numpy()

    from hallucination_detection.train_probe import train_h_node_probe
    artifact = train_h_node_probe(
        activations=activations,
        labels=labels,
        model_name=MODEL,
        train_mask=train_mask,
        n_h_nodes=50,
        baseline_percentile=80,
        seed=42,
    )
    artifact.save(PROBE_PATH)

    print(f"\nBest layer: {artifact.best_layer}  "
          f"(AUC={artifact.auc_per_layer[artifact.best_layer]:.3f})")
    sorted_aucs = sorted(artifact.auc_per_layer.items(), key=lambda kv: -kv[1])
    print("Top 5 layers by AUC:")
    for layer, auc in sorted_aucs[:5]:
        print(f"  layer {layer:>3d}:  AUC = {auc:.3f}")

    print(f"H-Nodes ({artifact.n_h_nodes}): "
          f"first 10 indices = {artifact.h_node_indices[:10].tolist()}")
    print(f"Saved → {PROBE_PATH}")

    # Rewrite infer config so the next stage captures only this best layer.
    with open(INFER_CFG, "r") as f:
        cfg = json.load(f)
    cfg["hidden_states"]["layers"] = [int(artifact.best_layer)]
    with open(INFER_CFG, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Updated {INFER_CFG} → layers=[{artifact.best_layer}]")


def stage_detect():
    from vllm_hook_plugins import HookLLM

    examples = [
        "Q: What is the capital of France?\nA: Paris",
        "Q: What is the capital of France?\nA: London",
        "Q: Who wrote Hamlet?\nA: William Shakespeare",
        "Q: Who wrote Hamlet?\nA: Charles Dickens",
        "Q: What is 2 + 2?\nA: 4",
        "Q: What is 2 + 2?\nA: 5",
    ]

    print("Loading model with inference config (best layer only) + hallucination analyzer...")
    llm = HookLLM(
        model=MODEL,
        worker_name="probe_hidden_states",
        analyzer_name="hallucination",
        config_file=INFER_CFG,
        download_dir=CACHE_DIR,
        hook_dir=HOOK_DIR,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        trust_remote_code=True,
        dtype=torch.float16,
        enable_prefix_caching=False,
        enable_hook=True,
        tensor_parallel_size=1,
    )

    print("Running detection on example prompts...\n")
    llm.generate(examples, temperature=0.0, max_tokens=1)
    result = llm.analyze(analyzer_spec={"probe_path": PROBE_PATH, "threshold": 0.5})

    print(f"Best layer: {result['best_layer']}  |  threshold: {result['threshold']}")
    print("-" * 78)
    for prompt, p, exc, verdict in zip(
        examples, result["probabilities"], result["h_node_excess"], result["verdicts"]
    ):
        line = prompt.replace("\n", "  ")
        print(f"[{verdict:>12s}]  P(hall)={p:.3f}  H-excess={exc:.3f}  |  {line}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "detect"
    if stage == "extract":
        stage_extract()
    elif stage == "train":
        stage_train()
    elif stage == "detect":
        stage_detect()
    else:
        print(f"Unknown stage: {stage!r}. Use extract | train | detect.")
        sys.exit(1)

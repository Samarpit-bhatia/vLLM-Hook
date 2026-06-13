"""Per-layer H-Node probe training.

Implements Phase 1 of Yocam et al. 2026 (Sections 4.2–4.4):
  1. Train a logistic-regression probe on the last-token hidden state at every
     extracted layer.
  2. Pick the best layer by held-out AUC.
  3. Take the top-N hidden-state dimensions by positive probe coefficient as
     the H-Node set.
  4. Compute per-H-Node grounded baselines as the P-th percentile of grounded
     samples' activations (paper uses P=80, N=50).

Saves a single ``probe.npz`` artifact consumed by
``vllm_hook_plugins.analyzers.hallucination_analyzer.HallucinationAnalyzer``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ProbeArtifact:
    model_name: str
    best_layer: int                  # 1-based, paper convention
    auc_per_layer: Dict[int, float]
    hidden_size: int
    weights: np.ndarray              # (hidden_size,) — best-layer probe weights
    bias: float
    scaler_mean: np.ndarray          # (hidden_size,)
    scaler_scale: np.ndarray         # (hidden_size,)
    h_node_indices: np.ndarray       # (N,) int — indices into hidden_size dim
    h_node_baselines: np.ndarray     # (N,) float — P-th percentile of grounded
    baseline_percentile: int
    n_h_nodes: int
    train_size: int
    eval_size: int

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(
            path,
            best_layer=self.best_layer,
            hidden_size=self.hidden_size,
            weights=self.weights,
            bias=self.bias,
            scaler_mean=self.scaler_mean,
            scaler_scale=self.scaler_scale,
            h_node_indices=self.h_node_indices,
            h_node_baselines=self.h_node_baselines,
            baseline_percentile=self.baseline_percentile,
            n_h_nodes=self.n_h_nodes,
            train_size=self.train_size,
            eval_size=self.eval_size,
        )
        meta_path = path.replace(".npz", ".json")
        with open(meta_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "best_layer": self.best_layer,
                "auc_per_layer": {int(k): float(v) for k, v in self.auc_per_layer.items()},
                "hidden_size": self.hidden_size,
                "n_h_nodes": self.n_h_nodes,
                "baseline_percentile": self.baseline_percentile,
                "train_size": self.train_size,
                "eval_size": self.eval_size,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProbeArtifact":
        data = np.load(path)
        meta_path = path.replace(".npz", ".json")
        with open(meta_path) as f:
            meta = json.load(f)
        return cls(
            model_name=meta["model_name"],
            best_layer=int(data["best_layer"]),
            auc_per_layer={int(k): float(v) for k, v in meta["auc_per_layer"].items()},
            hidden_size=int(data["hidden_size"]),
            weights=data["weights"],
            bias=float(data["bias"]),
            scaler_mean=data["scaler_mean"],
            scaler_scale=data["scaler_scale"],
            h_node_indices=data["h_node_indices"],
            h_node_baselines=data["h_node_baselines"],
            baseline_percentile=int(data["baseline_percentile"]),
            n_h_nodes=int(data["n_h_nodes"]),
            train_size=int(data["train_size"]),
            eval_size=int(data["eval_size"]),
        )


def _fit_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
) -> tuple[float, "LogisticRegression", "StandardScaler"]:  # noqa: F821
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xev = scaler.transform(X_eval)

    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=seed,
    ).fit(Xtr, y_train)

    scores = clf.decision_function(Xev)
    auc = float(roc_auc_score(y_eval, scores))
    return auc, clf, scaler


def train_h_node_probe(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    model_name: str,
    train_mask: Optional[np.ndarray] = None,
    n_h_nodes: int = 50,
    baseline_percentile: int = 80,
    seed: int = 42,
    layer_subset: Optional[List[int]] = None,
) -> ProbeArtifact:
    """Train per-layer probes, pick the best layer, identify H-Nodes.

    ``activations``: ``{layer_num: array(N, hidden)}``. ``labels``: ``array(N,)``
    with 0=grounded, 1=hallucinated. ``train_mask``: bool array(N,) — True for
    training samples. If None, an 80/20 random split is used (paper uses a
    question-grouped split; pass ``train_mask`` for that).
    """
    rng = np.random.default_rng(seed)
    n = len(labels)
    if train_mask is None:
        train_mask = rng.random(n) < 0.8
    eval_mask = ~train_mask

    layers = sorted(layer_subset) if layer_subset else sorted(activations.keys())

    auc_per_layer: Dict[int, float] = {}
    fits: Dict[int, tuple] = {}
    for layer in layers:
        X = np.asarray(activations[layer], dtype=np.float32)
        auc, clf, scaler = _fit_layer(
            X[train_mask], labels[train_mask],
            X[eval_mask], labels[eval_mask],
            seed=seed,
        )
        auc_per_layer[layer] = auc
        fits[layer] = (clf, scaler)

    best_layer = max(auc_per_layer, key=auc_per_layer.get)
    best_clf, best_scaler = fits[best_layer]

    weights = best_clf.coef_.ravel().astype(np.float32)
    bias = float(best_clf.intercept_.ravel()[0])
    hidden_size = weights.shape[0]

    # H-Nodes: top-N dims by positive coefficient (paper Eq. 3 — positive
    # direction = pro-hallucination).
    h_node_indices = np.argsort(weights)[::-1][:n_h_nodes].astype(np.int64)

    # Baselines: P-th percentile of grounded (label=0) raw activations per node.
    X_best = np.asarray(activations[best_layer], dtype=np.float32)
    grounded = X_best[(labels == 0) & train_mask]
    h_node_baselines = np.percentile(
        grounded[:, h_node_indices], baseline_percentile, axis=0,
    ).astype(np.float32)

    return ProbeArtifact(
        model_name=model_name,
        best_layer=int(best_layer),
        auc_per_layer=auc_per_layer,
        hidden_size=int(hidden_size),
        weights=weights,
        bias=bias,
        scaler_mean=best_scaler.mean_.astype(np.float32),
        scaler_scale=best_scaler.scale_.astype(np.float32),
        h_node_indices=h_node_indices,
        h_node_baselines=h_node_baselines,
        baseline_percentile=int(baseline_percentile),
        n_h_nodes=int(n_h_nodes),
        train_size=int(train_mask.sum()),
        eval_size=int(eval_mask.sum()),
    )


def train_from_bundle(
    bundle_path: str,
    model_name: str,
    out_path: str,
    **kwargs,
) -> ProbeArtifact:
    """Convenience wrapper: load an extracted bundle, train, save."""
    import torch
    bundle = torch.load(bundle_path, map_location="cpu")
    activations = {l: t.numpy() for l, t in bundle["activations"].items()}
    labels = bundle["labels"].numpy()
    artifact = train_h_node_probe(activations, labels, model_name=model_name, **kwargs)
    artifact.save(out_path)
    return artifact

"""Numpy-only scorer for a trained H-Node probe.

Used by ``HallucinationAnalyzer`` at inference and standalone for evaluation.
Keeps a hard dependency only on numpy so it loads in the vLLM worker process
without dragging in sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np

from hallucination_detection.train_probe import ProbeArtifact


def _sigmoid(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


@dataclass
class HallucinationScore:
    probability: float        # P(hallucinated) from probe
    h_node_excess: float      # mean max(0, h_j - baseline_j) over H-Nodes
    margin: float             # raw logit (signed distance from decision boundary)


class HNodeProbe:
    """In-memory probe — lightweight wrapper around a ``ProbeArtifact``."""

    def __init__(self, artifact: ProbeArtifact):
        self.artifact = artifact
        # Cached for speed
        self._w = artifact.weights.astype(np.float32)
        self._b = float(artifact.bias)
        self._mean = artifact.scaler_mean.astype(np.float32)
        # Guard against zero-variance dims that StandardScaler leaves at scale=1.
        self._scale = np.where(artifact.scaler_scale == 0, 1.0, artifact.scaler_scale).astype(np.float32)
        self._h_idx = artifact.h_node_indices.astype(np.int64)
        self._h_base = artifact.h_node_baselines.astype(np.float32)

    @classmethod
    def load(cls, path: str) -> "HNodeProbe":
        return cls(ProbeArtifact.load(path))

    @property
    def best_layer(self) -> int:
        return self.artifact.best_layer

    def score(self, activations: np.ndarray) -> List[HallucinationScore]:
        """Score a batch. ``activations`` has shape (batch, hidden_size).

        Returns one ``HallucinationScore`` per row.
        """
        h = activations.astype(np.float32, copy=False)
        if h.ndim == 1:
            h = h[None, :]

        std = (h - self._mean) / self._scale
        logits = std @ self._w + self._b
        probs = _sigmoid(logits)

        excess = np.maximum(0.0, h[:, self._h_idx] - self._h_base)
        mean_excess = excess.mean(axis=1)

        return [
            HallucinationScore(
                probability=float(p),
                h_node_excess=float(e),
                margin=float(l),
            )
            for p, e, l in zip(probs, mean_excess, logits)
        ]


def score_activations(
    probe_path: str,
    activations: Union[np.ndarray, Sequence[np.ndarray]],
) -> List[HallucinationScore]:
    """Convenience: load probe + score in one call."""
    probe = HNodeProbe.load(probe_path)
    if isinstance(activations, np.ndarray):
        return probe.score(activations)
    return probe.score(np.stack([np.asarray(a) for a in activations]))

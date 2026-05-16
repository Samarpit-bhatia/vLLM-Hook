"""H-Node hallucination detection for vLLM-Hook.

Implements the detection half of "H-Node Attack and Defense in Large Language
Models" (Yocam, Vaidyan, Wang, 2026): train a per-layer logistic-regression
probe on last-token hidden states, select the best layer by AUC, and identify
the top-N H-Nodes (hidden-state dimensions most associated with hallucination).
"""

from hallucination_detection.data import build_truthfulqa_pairs
from hallucination_detection.score import HNodeProbe, score_activations

__all__ = ["build_truthfulqa_pairs", "HNodeProbe", "score_activations"]

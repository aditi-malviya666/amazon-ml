from typing import Dict, List, Set
import numpy as np


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Computes F_0.5 score for a single Source 1 entity."""
    # Singleton check
    if len(true_matches) == 0:
        return 1.0 if len(pred_matches) == 0 else 0.0

    if len(pred_matches) == 0:
        return 0.0

    tp = len(true_matches.intersection(pred_matches))
    if tp == 0:
        return 0.0

    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)

    # F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    beta_sq = 0.5**2  # 0.25
    numerator = (1.0 + beta_sq) * precision * recall
    denominator = (beta_sq * precision) + recall

    return numerator / denominator if denominator > 0 else 0.0


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]], predictions: Dict[str, List[str]]
) -> float:
    """Evaluates macro-averaged F_0.5 over all Source 1 entities."""
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = set(predictions.get(s1_id, []))
        scores.append(compute_entity_f05(true_set, pred_set))
    return float(np.mean(scores)) if scores else 0.0

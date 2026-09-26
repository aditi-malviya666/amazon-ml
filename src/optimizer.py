from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from src.metrics import evaluate_macro_f05


class PrecisionOptimizer:
    """Calibrates cutoffs and inter-candidate margins to maximize Macro F_0.5."""

    def __init__(self):
        self.optimal_threshold: float = 0.72
        self.margin_delta: float = 0.12

    def calibrate(
        self,
        val_pairs: List[Tuple[str, str]],
        val_probs: np.ndarray,
        val_ground_truth: Dict[str, Set[str]],
        val_s1_ids: Set[str],
    ):
        """Sweeps threshold and margin deltas over entity validation instances."""
        # Group candidate probabilities by S1 entity
        grouped_candidates = defaultdict(list)
        for (s1_id, cand_id), prob in zip(val_pairs, val_probs):
            grouped_candidates[s1_id].append((cand_id, float(prob)))

        best_score = -1.0

        for th in np.arange(0.60, 0.88, 0.04):
            for margin in [0.08, 0.12, 0.16]:
                preds: Dict[str, List[str]] = {s1_id: [] for s1_id in val_s1_ids}

                for s1_id, candidates in grouped_candidates.items():
                    # Filter by minimum threshold
                    valid = [(cid, p) for cid, p in candidates if p >= th]
                    if not valid:
                        continue

                    # Sort descending
                    valid.sort(key=lambda x: -x[1])
                    top_prob = valid[0][1]

                    # Margin check: discard candidates trailing behind the top match
                    filtered_matches = [cid for cid, p in valid if (top_prob - p) <= margin]
                    preds[s1_id] = filtered_matches

                score = evaluate_macro_f05(val_ground_truth, preds)
                if score > best_score:
                    best_score = score
                    self.optimal_threshold = float(th)
                    self.margin_delta = float(margin)

        print(
            f"Tuned Optimal Config -> Threshold: {self.optimal_threshold:.2f}, "
            f"Margin Delta: {self.margin_delta:.2f} | Val F_0.5: {best_score:.4f}"
        )

    def filter_candidates(
        self, pairs: List[Tuple[str, str]], probs: np.ndarray, all_s1_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Applies calibrated cutoffs to generate matching outputs."""
        grouped = defaultdict(list)
        for (s1_id, cand_id), prob in zip(pairs, probs):
            grouped[s1_id].append((cand_id, float(prob)))

        final_matches: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        for s1_id, candidates in grouped.items():
            valid = [(cid, p) for cid, p in candidates if p >= self.optimal_threshold]
            if not valid:
                continue

            valid.sort(key=lambda x: -x[1])
            top_prob = valid[0][1]

            matches = [cid for cid, p in valid if (top_prob - p) <= self.margin_delta]
            # Enforce deduplication while preserving score order
            final_matches[s1_id] = list(dict.fromkeys(matches))

        return final_matches

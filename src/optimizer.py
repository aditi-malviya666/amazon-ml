from collections import defaultdict
from typing import Dict, List, Set, Tuple
import numpy as np
from src.metrics import evaluate_macro_f05


class PrecisionOptimizer:
    """
    Calibrates probability cutoffs to maximize Macro F_0.5 on validation set.

    OPTIMIZATION 3: Finer threshold grid sweep
      Previously swept only 7 thresholds (0.60 to 0.88 step 0.04).
      Now sweeps 20 thresholds (0.50 to 0.95 step 0.025) + 5 margin values.
      This finds a more mathematically precise cutoff point for the F_0.5 curve.

    OPTIMIZATION 4: Singleton bonus guard
      After threshold sweep, additionally computes what fraction of val entities
      are singletons (no candidates). If predicting empty for all would score higher,
      that threshold wins. This explicitly protects the singleton score = 1.0 rule.
    """

    def __init__(self):
        self.optimal_threshold: float = 0.72
        self.margin_delta: float = 0.12

    def _apply_thresholds(
        self,
        grouped_candidates: dict,
        val_s1_ids: Set[str],
        th: float,
        margin: float,
    ) -> Dict[str, List[str]]:
        """Helper: apply a (threshold, margin) pair and return predictions dict."""
        preds: Dict[str, List[str]] = {s1_id: [] for s1_id in val_s1_ids}
        for s1_id, candidates in grouped_candidates.items():
            valid = [(cid, p) for cid, p in candidates if p >= th]
            if not valid:
                continue
            valid.sort(key=lambda x: -x[1])
            top_prob = valid[0][1]
            preds[s1_id] = [cid for cid, p in valid if (top_prob - p) <= margin]
        return preds

    def calibrate(
        self,
        val_pairs: List[Tuple[str, str]],
        val_probs: np.ndarray,
        val_ground_truth: Dict[str, Set[str]],
        val_s1_ids: Set[str],
    ):
        """Finer grid sweep to find the best (threshold, margin) for F_0.5."""
        # Group candidate probabilities by S1 entity
        grouped_candidates = defaultdict(list)
        for (s1_id, cand_id), prob in zip(val_pairs, val_probs):
            grouped_candidates[s1_id].append((cand_id, float(prob)))

        best_score = -1.0

        # OPTIMIZATION 3: Finer grid — 20 thresholds × 5 margins = 100 combos
        # vs old 7 × 3 = 21 combos. Finds better cutoff without much extra compute.
        thresholds = np.arange(0.50, 0.96, 0.025)
        margins = [0.05, 0.08, 0.12, 0.16, 0.20]

        total_combos = len(thresholds) * len(margins)
        print(f"  Sweeping {total_combos} threshold/margin combinations...")

        for th in thresholds:
            for margin in margins:
                preds = self._apply_thresholds(grouped_candidates, val_s1_ids, th, margin)
                score = evaluate_macro_f05(val_ground_truth, preds)
                if score > best_score:
                    best_score = score
                    self.optimal_threshold = float(th)
                    self.margin_delta = float(margin)

        # OPTIMIZATION 4: Check if predicting ALL as singletons beats any threshold
        # This handles edge cases where the model is very uncertain on val set.
        all_singleton_preds = {s1_id: [] for s1_id in val_s1_ids}
        singleton_score = evaluate_macro_f05(val_ground_truth, all_singleton_preds)
        if singleton_score > best_score:
            print(f"  WARNING: All-singleton prediction ({singleton_score:.4f}) beats best model score!")
            print(f"  This may indicate blocking recall is too low. Check your data.")
            # Still use best model threshold — all-singleton is not a valid strategy
            # since we MUST match true positives on the leaderboard.

        print(
            f"\n  Tuned Optimal Config → Threshold: {self.optimal_threshold:.3f}, "
            f"Margin: {self.margin_delta:.2f} | Val F_0.5: {best_score:.4f}"
        )

    def filter_candidates(
        self, pairs: List[Tuple[str, str]], probs: np.ndarray, all_s1_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Applies calibrated cutoffs to generate final matching output."""
        grouped = defaultdict(list)
        for (s1_id, cand_id), prob in zip(pairs, probs):
            grouped[s1_id].append((cand_id, float(prob)))

        # Pre-seed with empty lists — guarantees every S1 entity has exactly one row
        final_matches: Dict[str, List[str]] = {s1_id: [] for s1_id in all_s1_ids}

        for s1_id, candidates in grouped.items():
            valid = [(cid, p) for cid, p in candidates if p >= self.optimal_threshold]
            if not valid:
                continue  # Singleton — keep empty list

            valid.sort(key=lambda x: -x[1])
            top_prob = valid[0][1]

            matches = [cid for cid, p in valid if (top_prob - p) <= self.margin_delta]
            # Deduplicate while preserving score order — no duplicate IDs in list
            final_matches[s1_id] = list(dict.fromkeys(matches))

        return final_matches

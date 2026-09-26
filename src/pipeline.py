import os
from typing import Dict, List, Set, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from src.blocking import FastSparseBlocker
from src.features import build_feature_matrix
from src.optimizer import PrecisionOptimizer


class HighRankEntityResolutionPipeline:

    def __init__(self):
        self.blocker = FastSparseBlocker()
        self.optimizer = PrecisionOptimizer()
        self.classifier = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

    def train_and_calibrate(
        self,
        df_s1: pd.DataFrame,
        df_cand: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        val_split: float = 0.2,
    ):
        """Executes entity-level stratified training and margin threshold calibration."""
        s1_entities = df_s1["entity_id"].unique()
        np.random.seed(42)
        np.random.shuffle(s1_entities)

        cutoff = int(len(s1_entities) * (1 - val_split))
        train_ids = set(s1_entities[:cutoff])
        val_ids = set(s1_entities[cutoff:])

        df_train_s1 = df_s1[df_s1["entity_id"].isin(train_ids)].reset_index(drop=True)
        df_val_s1 = df_s1[df_s1["entity_id"].isin(val_ids)].reset_index(drop=True)
        val_gt = {k: v for k, v in ground_truth.items() if k in val_ids}

        print("\n" + "="*50)
        print("[STEP 3/5] Sparse Blocking (TF-IDF & Math)")
        print("="*50)
        print(f"Generating candidates for {len(df_train_s1)} training S1 entities...")
        train_candidates = self.blocker.retrieve_candidates(df_train_s1, df_cand)

        print("\n" + "="*50)
        print("[STEP 4/5] Fuzzy Feature Engineering (RapidFuzz)")
        print("="*50)
        X_train, y_train, _ = build_feature_matrix(
            train_candidates, df_train_s1, df_cand, ground_truth
        )

        # IMPORTANT: Validation blocking uses the FULL candidate pool (df_cand)
        # to avoid data leakage from the training split
        print(f"Generating candidates for {len(df_val_s1)} validation S1 entities...")
        val_candidates = self.blocker.retrieve_candidates(df_val_s1, df_cand)
        X_val, y_val, val_pairs = build_feature_matrix(
            val_candidates, df_val_s1, df_cand, ground_truth
        )

        print("\n" + "="*50)
        print("[STEP 5/5] LightGBM Training & Threshold Tuning")
        print("="*50)
        print(f"Training on {len(X_train)} pairs | Validating on {len(X_val)} pairs")

        # SAFETY: If training data has no positive labels, skip (shouldn't happen on real data)
        if y_train.sum() == 0:
            print("WARNING: No positive labels found in training data. Skipping model fit.")
            return

        self.classifier.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=25, verbose=True),
                lgb.log_evaluation(period=50),
            ],
        )

        val_probs = self.classifier.predict_proba(X_val)[:, 1]
        self.optimizer.calibrate(val_pairs, val_probs, val_gt, val_ids)

    def generate_test_submission(
        self, df_test_s1: pd.DataFrame, df_test_cand: pd.DataFrame
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """Runs blocking, feature compilation, and matching inference on test set."""
        print("\nExecuting blocking on test partition...")
        test_candidate_pool = self.blocker.retrieve_candidates(df_test_s1, df_test_cand)

        # Build candidate_pairs map — must contain ALL test S1 IDs (even singletons with [])
        all_test_s1 = df_test_s1["entity_id"].tolist()
        candidate_pairs_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1}
        for s1_id, items in test_candidate_pool.items():
            candidate_pairs_map[s1_id] = [cid for cid, _, _ in items]

        print("Building feature matrix for test candidate pairs...")
        X_test, _, test_pairs = build_feature_matrix(
            test_candidate_pool, df_test_s1, df_test_cand
        )

        if len(X_test) > 0:
            test_probs = self.classifier.predict_proba(X_test)[:, 1]
        else:
            test_probs = np.empty(0)

        matching_map = self.optimizer.filter_candidates(test_pairs, test_probs, all_test_s1)

        # CONSTRAINT CHECK: Verify every match also exists in candidate_pairs_map
        orphan_count = 0
        for s1_id, matches in matching_map.items():
            candidates = set(candidate_pairs_map.get(s1_id, []))
            for m in matches:
                if m not in candidates:
                    orphan_count += 1
        if orphan_count > 0:
            print(f"WARNING: {orphan_count} matched IDs not in candidate_pairs! Check pipeline.")
        else:
            print("Constraint check PASSED: All matched IDs are subsets of their candidates.")

        return candidate_pairs_map, matching_map

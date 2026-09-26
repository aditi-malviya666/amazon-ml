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
            n_jobs=-1,          # Uses ALL CPU cores on Colab
        )

    def train_and_calibrate(
        self,
        df_s1: pd.DataFrame,
        df_cand: pd.DataFrame,
        ground_truth: Dict[str, Set[str]],
        val_split: float = 0.2,
    ):
        """
        Full training pipeline:
        1. Splits S1 into train/val stratified by entity
        2. Pre-fits TF-IDF ONCE on full candidate pool
        3. Runs blocking for train and val using cached TF-IDF (no rebuild!)
        4. Extracts 19-feature RapidFuzz vectors per candidate pair
        5. Trains LightGBM classifier with early stopping
        6. Calibrates precision threshold via 100-combo grid sweep
        """
        # Entity-level train/val split (never split mid-entity)
        s1_entities = df_s1["entity_id"].unique()
        np.random.seed(42)
        np.random.shuffle(s1_entities)

        cutoff = int(len(s1_entities) * (1 - val_split))
        train_ids = set(s1_entities[:cutoff])
        val_ids = set(s1_entities[cutoff:])

        df_train_s1 = df_s1[df_s1["entity_id"].isin(train_ids)].reset_index(drop=True)
        df_val_s1 = df_s1[df_s1["entity_id"].isin(val_ids)].reset_index(drop=True)
        val_gt = {k: v for k, v in ground_truth.items() if k in val_ids}

        # ── STEP 3: Blocking ──────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("[STEP 3/5]  Sparse Blocking (TF-IDF)")
        print("=" * 55)

        # SPEED OPTIMIZATION: fit TF-IDF ONCE on full candidate pool,
        # then reuse for both train and val transforms (no rebuild!).
        print(f"  Fitting TF-IDF on {len(df_cand):,} candidates (done once, reused for train+val)...")
        self.blocker.fit_candidates(df_cand)

        print(f"  Retrieving candidates for {len(df_train_s1):,} train S1 entities...")
        train_candidates = self.blocker.retrieve_candidates(df_train_s1)

        print(f"  Retrieving candidates for {len(df_val_s1):,} val S1 entities...")
        val_candidates = self.blocker.retrieve_candidates(df_val_s1)

        # ── STEP 4: Feature Engineering ───────────────────────────────────────
        print("\n" + "=" * 55)
        print("[STEP 4/5]  Fuzzy Feature Engineering (RapidFuzz)")
        print("=" * 55)

        # Pre-convert DataFrames to dicts ONCE and pass to both calls
        # (avoids rebuilding the dict inside build_feature_matrix twice)
        print("  Converting DataFrames to hash maps (done once for train+val)...")
        s1_dict = df_s1.set_index("entity_id").to_dict(orient="index")
        cand_dict = df_cand.set_index("entity_id").to_dict(orient="index")

        X_train, y_train, _ = build_feature_matrix(
            train_candidates, s1_dict, cand_dict, ground_truth
        )
        X_val, y_val, val_pairs = build_feature_matrix(
            val_candidates, s1_dict, cand_dict, ground_truth
        )

        # ── STEP 5: LightGBM ──────────────────────────────────────────────────
        print("\n" + "=" * 55)
        print("[STEP 5/5]  LightGBM Training & Threshold Tuning")
        print("=" * 55)
        print(f"  Training on {len(X_train):,} pairs | Validating on {len(X_val):,} pairs")

        if len(y_train) == 0 or y_train.sum() == 0:
            print("  WARNING: No positive labels in training data. Check data quality.")
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
        """
        Runs inference on test data.
        Fits a fresh TF-IDF on test candidate pool (different from train pool),
        then runs blocking and feature extraction.
        """
        print("\n" + "=" * 55)
        print("[TEST]  Generating Submission")
        print("=" * 55)

        # Fit new TF-IDF on test candidate pool (different data than training)
        print(f"  Fitting TF-IDF on {len(df_test_cand):,} test candidates...")
        self.blocker.fit_candidates(df_test_cand)

        print(f"  Blocking {len(df_test_s1):,} test S1 entities...")
        test_candidate_pool = self.blocker.retrieve_candidates(df_test_s1)

        # Build candidate_pairs map — ALL test S1 IDs must appear (including singletons)
        all_test_s1 = df_test_s1["entity_id"].tolist()
        candidate_pairs_map: Dict[str, List[str]] = {s1_id: [] for s1_id in all_test_s1}
        for s1_id, items in test_candidate_pool.items():
            candidate_pairs_map[s1_id] = [cid for cid, _, _ in items]

        print("  Building feature matrix for test candidate pairs...")
        test_cand_dict = df_test_cand.set_index("entity_id").to_dict(orient="index")
        test_s1_dict = df_test_s1.set_index("entity_id").to_dict(orient="index")

        X_test, _, test_pairs = build_feature_matrix(
            test_candidate_pool, test_s1_dict, test_cand_dict
        )

        if len(X_test) > 0:
            test_probs = self.classifier.predict_proba(X_test)[:, 1]
        else:
            test_probs = np.empty(0)

        matching_map = self.optimizer.filter_candidates(test_pairs, test_probs, all_test_s1)

        # CONSTRAINT CHECK: every matched ID must exist in its candidate set
        orphan_count = sum(
            1 for s1_id, matches in matching_map.items()
            for m in matches if m not in set(candidate_pairs_map.get(s1_id, []))
        )
        if orphan_count > 0:
            print(f"  ⚠ WARNING: {orphan_count} matched IDs missing from candidate_pairs!")
        else:
            print("  ✅ Constraint check PASSED: all matches are subsets of candidates.")

        return candidate_pairs_map, matching_map

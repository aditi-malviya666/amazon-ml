import os
import subprocess
import sys
import pandas as pd
from src.pipeline import HighRankEntityResolutionPipeline
from src.preprocess import prepare_dataframe


def read_csv_with_progress(path: str, desc: str) -> pd.DataFrame:
    """Reads a large TSV in chunks with a tqdm progress bar. Uses sep=TAB always."""
    from tqdm import tqdm
    print(f"  Counting lines in {path}...")
    with open(path, "rb") as f:
        num_lines = sum(1 for _ in f) - 1  # subtract header row

    chunksize = 250_000
    df_list = []
    with tqdm(total=num_lines, desc=f"  {desc}", unit="rows") as pbar:
        for chunk in pd.read_csv(path, sep="\t", chunksize=chunksize, dtype=str):
            df_list.append(chunk)
            pbar.update(len(chunk))
    return pd.concat(df_list, ignore_index=True)


def load_and_preprocess(csv_path: str, desc: str) -> pd.DataFrame:
    """
    Loads a TSV and preprocesses it with caching.
    On first run: reads CSV → preprocesses → saves compressed .cache.pkl to disk.
    On subsequent runs: loads the cached .pkl instantly (< 5 seconds).
    Cache is stored next to the source file so it is always paired correctly.
    """
    cache_path = csv_path + ".cache.pkl"
    if os.path.exists(cache_path):
        print(f"  [CACHE HIT] Fast-loading {desc} from cache...")
        return pd.read_pickle(cache_path)

    print(f"  [CACHE MISS] Processing {desc} for the first time (will be cached)...")
    raw_df = read_csv_with_progress(csv_path, desc)
    clean_df = prepare_dataframe(raw_df)

    print(f"  Saving {desc} cache to disk (compressed)...")
    # protocol=4 → efficient binary; compress via pickle (pandas default)
    clean_df.to_pickle(cache_path)
    print(f"  Cache saved: {cache_path}")
    return clean_df


def read_ground_truth(path: str) -> dict:
    """Reads ground truth TSV. Handles NaN for singletons."""
    df = pd.read_csv(
        path, sep="\t", dtype={"source1_entity_id": str, "matched_entity_ids": str}
    )
    gt = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        raw_matched = row["matched_entity_ids"]
        if pd.isna(raw_matched) or str(raw_matched).strip() in ("", "nan"):
            gt[s1] = set()
        else:
            gt[s1] = {x.strip() for x in str(raw_matched).split(",") if x.strip()}
    return gt


def export_tsv(data_map: dict, target_path: str, column_name: str):
    """
    Exports a dict {s1_id: [cand_ids]} as a TSV.
    - Every S1 entity gets exactly ONE row (including singletons with empty matched_entity_ids)
    - ID lists are comma-separated within a tab-separated file
    - No duplicate rows
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    rows = []
    seen_s1_ids = set()  # Guard against duplicate source1_entity_id rows
    for s1_id, id_list in data_map.items():
        if s1_id in seen_s1_ids:
            print(f"  WARNING: Duplicate s1_id detected and skipped: {s1_id}")
            continue
        seen_s1_ids.add(s1_id)
        # Deduplicate within the id_list while preserving order
        deduped = list(dict.fromkeys(id_list)) if id_list else []
        joined_ids = ",".join(deduped)  # Empty string for singletons — correct per spec
        rows.append({"source1_entity_id": s1_id, column_name: joined_ids})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(target_path, sep="\t", index=False)
    print(f"  Exported: {target_path} ({len(df_out)} entities)")


def main():
    # ──────────────────────────────────────────────────────
    # STEP 1 & 2: Data Loading + Text Preprocessing (Cached)
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[STEP 1 & 2 / 5]  Data Loading & Text Preprocessing")
    print("=" * 55)

    train_s1_path = "dataset/train/train_source1.tsv"
    if not os.path.exists(train_s1_path):
        print("ERROR: Dataset not found. Make sure dataset/ folder is in the project root.")
        return

    train_s1 = load_and_preprocess("dataset/train/train_source1.tsv", "Train Source1 (S1)")
    train_s2 = load_and_preprocess("dataset/train/train_source2.tsv", "Train Source2 (S2)")
    train_s3 = load_and_preprocess("dataset/train/train_source3.tsv", "Train Source3 (S3)")

    print("  Loading ground truth...")
    ground_truth = read_ground_truth("dataset/train/train_ground_truth.tsv")
    print(f"  Ground truth loaded: {len(ground_truth)} S1 entities")

    train_cand = pd.concat([train_s2, train_s3], ignore_index=True)
    print(f"  Train candidate pool size: {len(train_cand):,} records")

    # ──────────────────────────────────────────────────────
    # STEPS 3, 4, 5: Blocking → Features → LightGBM Train
    # (progress bars are printed inside pipeline/blocker/features)
    # ──────────────────────────────────────────────────────
    pipeline = HighRankEntityResolutionPipeline()
    pipeline.train_and_calibrate(train_s1, train_cand, ground_truth)

    # ──────────────────────────────────────────────────────
    # Test Data Loading (also cached)
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("[TEST DATA]  Loading test set (includes France)")
    print("=" * 55)
    test_s1 = load_and_preprocess("dataset/test/test_source1.tsv", "Test Source1 (S1)")
    test_s2 = load_and_preprocess("dataset/test/test_source2.tsv", "Test Source2 (S2)")
    test_s3 = load_and_preprocess("dataset/test/test_source3.tsv", "Test Source3 (S3)")
    test_cand = pd.concat([test_s2, test_s3], ignore_index=True)
    print(f"  Test S1 entities: {len(test_s1):,} | Test candidate pool: {len(test_cand):,}")

    # ──────────────────────────────────────────────────────
    # Inference & Output
    # ──────────────────────────────────────────────────────
    candidates_map, matches_map = pipeline.generate_test_submission(test_s1, test_cand)

    print("\n" + "=" * 55)
    print("[OUTPUT]  Writing submission files")
    print("=" * 55)
    export_tsv(candidates_map, "output/candidate_pairs.tsv", "candidate_entity_ids")
    export_tsv(matches_map, "output/matching_results.tsv", "matched_entity_ids")

    # ──────────────────────────────────────────────────────
    # Auto-run the official validator if available
    # ──────────────────────────────────────────────────────
    validator_path = os.path.join("student_resource", "utils", "validate_submission.py")
    if not os.path.exists(validator_path):
        validator_path = os.path.join("utils", "validate_submission.py")

    if os.path.exists(validator_path):
        print("\n" + "=" * 55)
        print("[VALIDATOR]  Running official submission validator")
        print("=" * 55)
        cmd = [
            sys.executable,
            validator_path,
            "--matching", "output/matching_results.tsv",
            "--candidate", "output/candidate_pairs.tsv",
            "--test-dir", "dataset/test",
        ]
        status = subprocess.run(cmd)
        if status.returncode == 0:
            print("\n✅  Validation PASSED. Ready for portal upload!")
        else:
            print("\n❌  Validation FAILED. Fix the issues above before submitting.")
    else:
        print("\n  (Validator script not found — skipping auto-validation)")

    print("\n🏁  Pipeline complete!")


if __name__ == "__main__":
    main()

from typing import Dict, List, Set, Tuple, Union
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def extract_pairwise_vector(
    s1_row: dict, cand_row: dict, cand_id: str, sparse_sim: float, dense_sim: float
) -> List[float]:
    """Generates 21 matching features for a candidate pair.

    Features cover name similarity, address similarity, cross-column matching,
    numeric overlap, length ratios, and structural indicators.
    All values are floats in [0, 1] range for LightGBM stability.
    """
    s1_name = str(s1_row.get("clean_name", "") or "")
    s1_addr = str(s1_row.get("clean_address", "") or "")
    c_name  = str(cand_row.get("clean_name", "") or "")
    c_addr  = str(cand_row.get("clean_address", "") or "")

    # ── 1. Fuzzy Name Metrics ──────────────────────────────────────────────────
    name_ratio      = fuzz.ratio(s1_name, c_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_token_set  = fuzz.token_set_ratio(s1_name, c_name) / 100.0
    # Jaro-Winkler: rewards prefix matches ("McDonald" vs "McDonalds")
    name_jaro       = JaroWinkler.normalized_similarity(s1_name, c_name)

    # ── 2. Name Token Features ────────────────────────────────────────────────
    s1_words = s1_name.split()
    c_words  = c_name.split()
    s1_first_word = s1_words[0] if s1_words else ""
    c_first_word  = c_words[0]  if c_words  else ""
    # First word exact match (brand anchor: "Acme" in "Acme Robotics Ltd")
    first_word_match = 1.0 if s1_first_word and s1_first_word == c_first_word else 0.0
    # Word-level Jaccard (handles word-order transpositions)
    s1_word_set  = set(s1_words)
    c_word_set   = set(c_words)
    union_words  = s1_word_set | c_word_set
    token_overlap = len(s1_word_set & c_word_set) / len(union_words) if union_words else 0.0

    # ── 3. Fuzzy Address Metrics ──────────────────────────────────────────────
    addr_ratio      = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_token_set  = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0

    # ── 4. Cross-Column Matching ──────────────────────────────────────────────
    # Catches cases where name & address fields were swapped in source data
    cross_name_to_addr = fuzz.token_set_ratio(s1_name, c_addr) / 100.0
    cross_addr_to_name = fuzz.token_set_ratio(s1_addr, c_name) / 100.0

    # ── 5. Numeric Overlap (PINs, door numbers, French postcodes) ─────────────
    # BUG FIX: addr_numbers stored as set — must handle case where it's NaN/None
    s1_nums = s1_row.get("addr_numbers") or set()
    c_nums  = cand_row.get("addr_numbers") or set()
    # Ensure they are actual sets (pickle may deserialize differently)
    if not isinstance(s1_nums, set):
        s1_nums = set(s1_nums) if s1_nums else set()
    if not isinstance(c_nums, set):
        c_nums = set(c_nums) if c_nums else set()

    if s1_nums and c_nums:
        intersection   = len(s1_nums & c_nums)
        num_jaccard    = intersection / len(s1_nums | c_nums)
        num_has_common = 1.0 if intersection > 0 else 0.0
    else:
        num_jaccard    = 0.5  # neutral when one side has no numbers
        num_has_common = 0.5

    # ── 6. Length Ratio Features ──────────────────────────────────────────────
    len_diff_name = abs(len(s1_name) - len(c_name)) / (max(len(s1_name), len(c_name)) + 1e-5)
    len_diff_addr = abs(len(s1_addr) - len(c_addr)) / (max(len(s1_addr), len(c_addr)) + 1e-5)

    # NEW FEATURE: Name length ratio (catches heavy abbreviation vs full name)
    # "SBI" (3) vs "State Bank of India" (19) → ratio = 3/19 ≈ 0.16 (very low)
    name_len_ratio = min(len(s1_name), len(c_name)) / (max(len(s1_name), len(c_name)) + 1e-5)

    # ── 7. Structural Flags ────────────────────────────────────────────────────
    # Exact match shortcut: forces LightGBM to 1.0 probability for identical pairs
    is_exact_match = 1.0 if (name_ratio == 1.0 and addr_ratio == 1.0) else 0.0
    # Source indicator (S2 vs S3 may have different noise profiles)
    is_source2 = 1.0 if str(cand_id).startswith("S2-") else 0.0

    return [
        sparse_sim,       # 0  TF-IDF cosine from blocker
        dense_sim,        # 1  (reserved, currently 0.0)
        name_ratio,       # 2
        name_token_sort,  # 3
        name_token_set,   # 4
        name_jaro,        # 5
        first_word_match, # 6
        token_overlap,    # 7
        addr_ratio,       # 8
        addr_token_sort,  # 9
        addr_token_set,   # 10
        cross_name_to_addr, # 11
        cross_addr_to_name, # 12
        num_jaccard,      # 13
        num_has_common,   # 14
        len_diff_name,    # 15
        len_diff_addr,    # 16
        name_len_ratio,   # 17  NEW
        is_exact_match,   # 18
        is_source2,       # 19
    ]


def build_feature_matrix(
    candidate_map: Dict[str, List[Tuple[str, float, float]]],
    s1_source:   Union[dict, pd.DataFrame],
    cand_source: Union[dict, pd.DataFrame],
    ground_truth: Dict[str, Set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Compiles candidate dictionary into feature matrix and binary training labels.

    Accepts either:
      - Pre-built dicts  {entity_id: {col: val, ...}}  — fast path (recommended)
      - Raw DataFrames                                  — auto-converts (slower)
    """
    if isinstance(s1_source, pd.DataFrame):
        print("  Converting DataFrames to hash maps...")
        s1_dict   = s1_source.set_index("entity_id").to_dict(orient="index")
        cand_dict = cand_source.set_index("entity_id").to_dict(orient="index")
    else:
        s1_dict   = s1_source
        cand_dict = cand_source

    X_list, y_list, pair_list = [], [], []

    from tqdm import tqdm
    for s1_id, candidates in tqdm(candidate_map.items(), desc="  Extracting Features"):
        if s1_id not in s1_dict:
            continue
        s1_row = s1_dict[s1_id]
        true_matches = ground_truth.get(s1_id, set()) if ground_truth is not None else None

        for cand_id, sparse_sim, dense_sim in candidates:
            if cand_id not in cand_dict:
                continue
            cand_row = cand_dict[cand_id]

            feats = extract_pairwise_vector(s1_row, cand_row, cand_id, sparse_sim, dense_sim)
            X_list.append(feats)
            pair_list.append((s1_id, cand_id))

            if true_matches is not None:
                y_list.append(1 if cand_id in true_matches else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else np.empty(0)
    return X, y, pair_list

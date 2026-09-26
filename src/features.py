from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def extract_pairwise_vector(
    s1_row: dict, cand_row: dict, cand_id: str, sparse_sim: float, dense_sim: float
) -> List[float]:
    """Generates dense + symbolic matching features for a candidate pair."""
    s1_name = s1_row["clean_name"]
    s1_addr = s1_row["clean_address"]
    c_name = cand_row["clean_name"]
    c_addr = cand_row["clean_address"]

    # 1. Fuzzy Name Metrics (RapidFuzz C++ - very fast)
    name_ratio = fuzz.ratio(s1_name, c_name) / 100.0
    name_token_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
    name_token_set = fuzz.token_set_ratio(s1_name, c_name) / 100.0

    # NEW FEATURE: Jaro-Winkler rewards prefix matches (catches "McDonald" vs "McDonalds")
    name_jaro = JaroWinkler.normalized_similarity(s1_name, c_name)

    # 2. Brand Anchor: First Word Agreement (e.g., "Acme" in "Acme Robotics")
    s1_words = s1_name.split()
    c_words = c_name.split()
    s1_first_word = s1_words[0] if s1_words else ""
    c_first_word = c_words[0] if c_words else ""
    first_word_match = 1.0 if s1_first_word and (s1_first_word == c_first_word) else 0.0

    # NEW FEATURE: Token word overlap ratio (handles word-order transpositions)
    s1_word_set = set(s1_words)
    c_word_set = set(c_words)
    union_words = s1_word_set | c_word_set
    token_overlap = len(s1_word_set & c_word_set) / len(union_words) if union_words else 0.0

    # 3. Fuzzy Address Metrics
    addr_ratio = fuzz.ratio(s1_addr, c_addr) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0
    addr_token_set = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0

    # 4. CROSS-COLUMN MATCHING (Catches messy data where Name & Address are swapped!)
    cross_name_to_addr = fuzz.token_set_ratio(s1_name, c_addr) / 100.0
    cross_addr_to_name = fuzz.token_set_ratio(s1_addr, c_name) / 100.0

    # 5. Strict Number Overlap (Door numbers, PINs, French postcodes)
    s1_nums: Set[str] = s1_row["addr_numbers"]
    c_nums: Set[str] = cand_row["addr_numbers"]
    if len(s1_nums) > 0 and len(c_nums) > 0:
        intersection = len(s1_nums.intersection(c_nums))
        num_jaccard = intersection / len(s1_nums.union(c_nums))
        num_has_common = 1.0 if intersection > 0 else 0.0
    else:
        num_jaccard = 0.5
        num_has_common = 0.5

    # 6. Length Differences
    len_diff_name = abs(len(s1_name) - len(c_name)) / (max(len(s1_name), len(c_name)) + 1e-5)
    len_diff_addr = abs(len(s1_addr) - len(c_addr)) / (max(len(s1_addr), len(c_addr)) + 1e-5)

    # 7. EXACT MATCH FLAG (Helps LightGBM guarantee 1.0 probability on identical strings)
    is_exact_match = 1.0 if (name_ratio == 1.0 and addr_ratio == 1.0) else 0.0

    # 8. Origin Source Indicator
    is_source2 = 1.0 if str(cand_id).startswith("S2-") else 0.0

    return [
        sparse_sim,
        dense_sim,
        name_ratio,
        name_token_sort,
        name_token_set,
        name_jaro,
        first_word_match,
        token_overlap,
        addr_ratio,
        addr_token_sort,
        addr_token_set,
        cross_name_to_addr,
        cross_addr_to_name,
        num_jaccard,
        num_has_common,
        len_diff_name,
        len_diff_addr,
        is_exact_match,
        is_source2,
    ]


def build_feature_matrix(
    candidate_map: Dict[str, List[Tuple[str, float, float]]],
    df_s1: pd.DataFrame,
    df_cand: pd.DataFrame,
    ground_truth: Dict[str, Set[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """Compiles candidate dictionary into feature matrix and binary training labels."""
    print("Converting DataFrames to hash maps for instant access...")
    s1_dict = df_s1.set_index("entity_id").to_dict(orient="index")
    cand_dict = df_cand.set_index("entity_id").to_dict(orient="index")

    X_list, y_list, pair_list = [], [], []

    from tqdm import tqdm
    for s1_id, candidates in tqdm(candidate_map.items(), desc="Extracting Features"):
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

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FastSparseBlocker:
    """
    Two-stage sparse TF-IDF blocker with adaptive candidate set sizing.

    OPTIMIZATION 1: Name-weighted composite_text
      The name is doubled in composite_text ("name name address") so TF-IDF
      cosine similarity is dominated by name matches, not address noise.

    OPTIMIZATION 2: Adaptive top_k (CRITICAL for tie-breaker)
      Contest rules: "smaller candidate set per S1 entity ranks HIGHER".
      Instead of blindly returning top_k=20, we return candidates dynamically:
        - If best TF-IDF score > 0.80 → only return candidates within 25% of best (likely 1-3)
        - If best TF-IDF score > 0.30 → return up to top_k=15
        - If best TF-IDF score < 0.10 → return 0 (treat as singleton at blocking stage)
      This gives the smallest possible candidate sets while preserving recall.
    """

    def __init__(self):
        pass

    def retrieve_candidates(
        self,
        df_s1: pd.DataFrame,
        df_cand: pd.DataFrame,
        top_k: int = 15,
        min_sim: float = 0.10,
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """Returns {s1_id: [(candidate_id, sparse_sim, 0.0), ...]}

        COMPETITION NOTE: Adaptive candidate sizing wins the tie-breaker.
        Smaller candidate sets per S1 entity rank higher per contest rules.
        """
        results: Dict[str, List[Tuple[str, float, float]]] = {
            s1_id: [] for s1_id in df_s1["entity_id"]
        }

        # Country Filtering: open-set dynamic — never hardcoded to US/India
        countries = df_s1["country"].unique()

        from tqdm import tqdm
        for c in countries:
            sub_s1 = df_s1[df_s1["country"] == c].reset_index(drop=True)
            sub_cand = df_cand[df_cand["country"] == c].reset_index(drop=True)

            if sub_cand.empty or sub_s1.empty:
                # CONSTRAINT: entities with zero candidates already pre-seeded with []
                continue

            s1_ids = sub_s1["entity_id"].to_numpy()
            cand_ids = sub_cand["entity_id"].to_numpy()

            # OPTIMIZATION 1: Double the name in composite_text so TF-IDF
            # is dominated by name similarity, not address noise.
            # "star hotel mumbai" → "star hotel star hotel mumbai"
            s1_texts = (sub_s1["clean_name"] + " " + sub_s1["clean_name"] + " " + sub_s1["clean_address"]).to_numpy()
            cand_texts = (sub_cand["clean_name"] + " " + sub_cand["clean_name"] + " " + sub_cand["clean_address"]).to_numpy()

            print(f"  Building TF-IDF Matrix for {c} ({len(sub_s1):,} S1, {len(sub_cand):,} candidates)...")
            tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                max_features=300_000,  # Caps vocabulary to prevent RAM explosion
            )
            cand_tfidf = tfidf.fit_transform(cand_texts)
            s1_tfidf = tfidf.transform(s1_texts)

            # Batch dot-product — never call .toarray() on full matrix (OOM)
            batch_size = 250
            for start_idx in tqdm(range(0, s1_tfidf.shape[0], batch_size), desc=f"  Blocking [{c}]"):
                end_idx = min(start_idx + batch_size, s1_tfidf.shape[0])
                s1_batch = s1_tfidf[start_idx:end_idx]

                # .tocsr() guarantees indptr exists regardless of scipy version
                sparse_sims = s1_batch.dot(cand_tfidf.T).tocsr()

                for i in range(sparse_sims.shape[0]):
                    start_ptr = sparse_sims.indptr[i]
                    end_ptr = sparse_sims.indptr[i + 1]

                    row_data = sparse_sims.data[start_ptr:end_ptr]
                    row_indices = sparse_sims.indices[start_ptr:end_ptr]

                    if len(row_data) == 0:
                        continue

                    global_s1_idx = start_idx + i
                    actual_s1_id = s1_ids[global_s1_idx]

                    # Top-K on only the non-zero elements (no 71GB dense allocation)
                    k = min(top_k, len(row_data))
                    if len(row_data) <= top_k:
                        top_idx_of_idx = np.argsort(-row_data)
                    else:
                        top_idx_of_idx = np.argpartition(-row_data, k)[:k]

                    # Get the top candidates above min_sim
                    raw_candidates = []
                    for idx in top_idx_of_idx:
                        val = float(row_data[idx])
                        if val >= min_sim:
                            cand_idx = row_indices[idx]
                            raw_candidates.append((str(cand_ids[cand_idx]), val, 0.0))

                    if not raw_candidates:
                        continue

                    # Sort by score descending
                    raw_candidates.sort(key=lambda x: -x[1])

                    # OPTIMIZATION 2: Adaptive candidate set sizing
                    # Directly minimizes candidate_pairs.tsv size = wins tie-breakers
                    best_score = raw_candidates[0][1]
                    if best_score >= 0.80:
                        # High confidence: only keep candidates within 25% of best score
                        # Typical result: 1-3 candidates instead of 15
                        cutoff = best_score * 0.75
                        merged = [(cid, s, d) for cid, s, d in raw_candidates if s >= cutoff]
                    elif best_score >= 0.30:
                        # Medium confidence: keep top_k as normal
                        merged = raw_candidates[:top_k]
                    else:
                        # Low confidence: treat as potential singleton, keep only top-3
                        merged = raw_candidates[:3]

                    results[actual_s1_id] = merged

        return results

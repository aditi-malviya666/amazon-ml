from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FastSparseBlocker:
    """Uses optimized word-level TF-IDF blocking to process 26 Million records efficiently."""

    def __init__(self):
        pass

    def retrieve_candidates(
        self,
        df_s1: pd.DataFrame,
        df_cand: pd.DataFrame,
        top_k: int = 20,
        min_sim: float = 0.10,
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """Returns {s1_id: [(candidate_id, sparse_sim, 0.0), ...]}

        COMPETITION NOTE: top_k=20 and min_sim=0.10 are intentionally conservative.
        Smaller candidate sets per S1 entity rank higher in tie-breakers per contest rules.
        """
        results: Dict[str, List[Tuple[str, float, float]]] = {
            s1_id: [] for s1_id in df_s1["entity_id"]
        }

        # 1. Country Filtering: open-set dynamic — never hardcoded to US/India
        countries = df_s1["country"].unique()

        from tqdm import tqdm
        for c in countries:
            sub_s1 = df_s1[df_s1["country"] == c].reset_index(drop=True)
            sub_cand = df_cand[df_cand["country"] == c].reset_index(drop=True)

            if sub_cand.empty or sub_s1.empty:
                # CONSTRAINT: Even entities with zero candidates must appear in final output.
                # results already pre-seeded with empty lists — no action needed here.
                continue

            s1_ids = sub_s1["entity_id"].to_numpy()
            cand_ids = sub_cand["entity_id"].to_numpy()

            # 2. Word-level TF-IDF with bigrams — good recall, low memory vs char n-grams
            print(f"Building TF-IDF Matrix for {c} ({len(sub_s1)} S1 entities, {len(sub_cand)} candidates)...")
            tfidf = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            cand_tfidf = tfidf.fit_transform(sub_cand["composite_text"])
            s1_tfidf = tfidf.transform(sub_s1["composite_text"])

            # 3. Batch dot-product — never call .toarray() on the full matrix (OOM risk)
            batch_size = 250
            for start_idx in tqdm(range(0, s1_tfidf.shape[0], batch_size), desc=f"Blocking {c}"):
                end_idx = min(start_idx + batch_size, s1_tfidf.shape[0])
                s1_batch = s1_tfidf[start_idx:end_idx]

                # .tocsr() guarantees indptr attribute exists regardless of scipy version
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

                    # 4. Top-K on only the non-zero elements (no 71GB dense allocation)
                    k = min(top_k, len(row_data))
                    if len(row_data) <= top_k:
                        top_idx_of_idx = np.argsort(-row_data)
                    else:
                        top_idx_of_idx = np.argpartition(-row_data, k)[:k]

                    merged = []
                    for idx in top_idx_of_idx:
                        val = float(row_data[idx])
                        if val >= min_sim:
                            cand_idx = row_indices[idx]
                            merged.append((str(cand_ids[cand_idx]), val, 0.0))

                    # Sort by score descending for cleaner candidate_pairs.tsv
                    merged.sort(key=lambda x: -x[1])
                    results[actual_s1_id] = merged

        return results

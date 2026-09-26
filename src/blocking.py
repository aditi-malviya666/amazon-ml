from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class FastSparseBlocker:
    """
    Sparse TF-IDF blocker with pre-fitted candidate matrices.

    KEY SPEED OPTIMIZATION:
      The TF-IDF vocabulary and candidate matrix are built ONCE via fit_candidates().
      All subsequent calls to retrieve_candidates() only run tfidf.transform() on S1,
      which is ~10x faster than a full fit_transform().

      This saves ~33% of total blocking time since retrieve_candidates() was previously
      called 3 times (train, val, test) with full matrix rebuild each time.

    ACCURACY OPTIMIZATION:
      Name is doubled in composite text ("name name address") so TF-IDF cosine
      similarity is dominated by name matches rather than address noise.

    TIE-BREAKER OPTIMIZATION:
      Adaptive candidate set sizing based on best TF-IDF score minimizes
      candidate_pairs.tsv size, which ranks higher per contest tie-breaker rules.
    """

    def __init__(self):
        # Per-country fitted vectorizers and pre-built candidate matrices
        self._vectorizers: Dict[str, TfidfVectorizer] = {}
        self._cand_matrices: Dict = {}       # sparse CSR matrices
        self._cand_ids_map: Dict[str, np.ndarray] = {}

    @staticmethod
    def _make_text(df: pd.DataFrame) -> np.ndarray:
        """Doubles name to make TF-IDF name-dominant. Reused in fit and transform."""
        return (df["clean_name"] + " " + df["clean_name"] + " " + df["clean_address"]).to_numpy()

    def fit_candidates(self, df_cand: pd.DataFrame):
        """
        Pre-build TF-IDF vectorizer and candidate matrix for each country.
        Call this ONCE before any retrieve_candidates() calls.
        Subsequent retrieve_candidates() calls only run tfidf.transform() — ~10x faster.
        """
        from tqdm import tqdm
        countries = df_cand["country"].unique()
        print(f"  Pre-fitting TF-IDF matrices for {len(countries)} countries...")

        for c in tqdm(countries, desc="  Fitting TF-IDF"):
            sub_cand = df_cand[df_cand["country"] == c].reset_index(drop=True)
            if sub_cand.empty:
                continue

            cand_texts = self._make_text(sub_cand)
            tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                max_features=300_000,  # Caps vocab to prevent RAM explosion
            )
            cand_matrix = tfidf.fit_transform(cand_texts)

            self._vectorizers[c] = tfidf
            self._cand_matrices[c] = cand_matrix
            self._cand_ids_map[c] = sub_cand["entity_id"].to_numpy()

        print(f"  TF-IDF fit complete. Matrices cached for fast reuse.")

    def retrieve_candidates(
        self,
        df_s1: pd.DataFrame,
        df_cand: Optional[pd.DataFrame] = None,  # Only used if fit_candidates() not called
        top_k: int = 15,
        min_sim: float = 0.10,
    ) -> Dict[str, List[Tuple[str, float, float]]]:
        """
        Returns {s1_id: [(candidate_id, sparse_sim, 0.0), ...]}.

        If fit_candidates() was called first: uses cached TF-IDF (FAST — no rebuild).
        If not: falls back to building TF-IDF from df_cand (slower, for compatibility).
        """
        results: Dict[str, List[Tuple[str, float, float]]] = {
            s1_id: [] for s1_id in df_s1["entity_id"]
        }

        # Determine which countries to process
        countries = df_s1["country"].unique()

        from tqdm import tqdm
        for c in countries:
            sub_s1 = df_s1[df_s1["country"] == c].reset_index(drop=True)
            if sub_s1.empty:
                continue

            # Use pre-fitted matrices if available, else build on-the-fly
            if c in self._vectorizers:
                tfidf = self._vectorizers[c]
                cand_tfidf = self._cand_matrices[c]
                cand_ids = self._cand_ids_map[c]
            elif df_cand is not None:
                # Fallback: build TF-IDF on the fly (slower path)
                sub_cand = df_cand[df_cand["country"] == c].reset_index(drop=True)
                if sub_cand.empty:
                    continue
                cand_texts = self._make_text(sub_cand)
                print(f"  [FALLBACK] Building TF-IDF for {c} (call fit_candidates() first for speed)...")
                tfidf = TfidfVectorizer(
                    analyzer="word", ngram_range=(1, 2),
                    min_df=1, sublinear_tf=True, max_features=300_000,
                )
                cand_tfidf = tfidf.fit_transform(cand_texts)
                cand_ids = sub_cand["entity_id"].to_numpy()
            else:
                # Country in test that wasn't in train (e.g. France edge case)
                print(f"  WARNING: No candidate data for country '{c}'. Skipping.")
                continue

            s1_ids = sub_s1["entity_id"].to_numpy()
            s1_texts = self._make_text(sub_s1)
            s1_tfidf = tfidf.transform(s1_texts)

            # Larger batch_size = fewer loop iterations = faster (safe now — no .toarray())
            batch_size = 500
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

                    actual_s1_id = s1_ids[start_idx + i]

                    # Top-K on sparse elements only (no 71GB dense allocation)
                    k = min(top_k, len(row_data))
                    if len(row_data) <= top_k:
                        top_idx_of_idx = np.argsort(-row_data)
                    else:
                        top_idx_of_idx = np.argpartition(-row_data, k)[:k]

                    raw_candidates = []
                    for idx in top_idx_of_idx:
                        val = float(row_data[idx])
                        if val >= min_sim:
                            raw_candidates.append((str(cand_ids[row_indices[idx]]), val, 0.0))

                    if not raw_candidates:
                        continue

                    raw_candidates.sort(key=lambda x: -x[1])
                    best_score = raw_candidates[0][1]

                    # Adaptive candidate sizing: wins tie-breaker per contest rules
                    if best_score >= 0.80:
                        cutoff = best_score * 0.75
                        merged = [(cid, s, d) for cid, s, d in raw_candidates if s >= cutoff]
                    elif best_score >= 0.30:
                        merged = raw_candidates[:top_k]
                    else:
                        merged = raw_candidates[:3]

                    results[actual_s1_id] = merged

        return results

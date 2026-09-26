import pandas as pd
import jellyfish
import lightgbm as lgb

def compute_string_similarities(str1: str, str2: str) -> dict:
    if not str1 and not str2:
        return {'jaro_winkler': 1.0, 'levenshtein': 0.0, 'jaccard': 1.0}
    if not str1 or not str2:
        return {'jaro_winkler': 0.0, 'levenshtein': 999.0, 'jaccard': 0.0}
        
    set1, set2 = set(str1.split()), set(str2.split())
    jaccard = len(set1.intersection(set2)) / max(len(set1.union(set2)), 1)
    
    return {
        'jaro_winkler': jellyfish.jaro_winkler_similarity(str1, str2),
        'levenshtein': jellyfish.damerau_levenshtein_distance(str1, str2),
        'jaccard': jaccard
    }

def extract_features(pairs_df: pd.DataFrame, data_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts features for each candidate pair."""
    data_dict = data_df.set_index('entity_id').to_dict('index')
    
    features = []
    for _, row in pairs_df.iterrows():
        id1, id2 = row['source1_entity_id'], row['candidate_entity_id']
        rec1, rec2 = data_dict[id1], data_dict[id2]
        
        name_sim = compute_string_similarities(rec1['clean_name'], rec2['clean_name'])
        addr_sim = compute_string_similarities(rec1['clean_address'], rec2['clean_address'])
        
        feat = {
            'source1_entity_id': id1,
            'candidate_entity_id': id2,
            'name_jaro': name_sim['jaro_winkler'],
            'name_lev': name_sim['levenshtein'],
            'name_jaccard': name_sim['jaccard'],
            'addr_jaro': addr_sim['jaro_winkler'],
            'addr_lev': addr_sim['levenshtein'],
            'addr_jaccard': addr_sim['jaccard'],
            'blocking_score': row.get('blocking_score', 0.0)
        }
        features.append(feat)
        
    return pd.DataFrame(features)

def train_model(features_df: pd.DataFrame, labels: pd.Series):
    """Trains a LightGBM model for matching."""
    X = features_df.drop(columns=['source1_entity_id', 'candidate_entity_id'])
    y = labels
    
    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

def predict_matches(model, features_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    X = features_df.drop(columns=['source1_entity_id', 'candidate_entity_id'])
    probs = model.predict_proba(X)[:, 1]
    
    results = features_df[['source1_entity_id', 'candidate_entity_id']].copy()
    results['match_prob'] = probs
    results['is_match'] = probs >= threshold
    return results[results['is_match']]

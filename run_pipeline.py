import os
import pandas as pd
from src.preprocessing import load_data, preprocess_dataframe
from src.blocking import generate_candidates
from src.features_and_matching import extract_features, train_model, predict_matches

def load_split(split_dir: str, prefix: str) -> pd.DataFrame:
    """Loads and concatenates source 1, 2, 3 files from a directory."""
    dfs = []
    for i in range(1, 4):
        path = os.path.join(split_dir, f'{prefix}_source{i}.tsv')
        if os.path.exists(path):
            df = load_data(path)
            dfs.append(df)
        else:
            print(f"File not found: {path}")
    if dfs:
        return preprocess_dataframe(pd.concat(dfs, ignore_index=True))
    return None

def format_output_file(pairs_df: pd.DataFrame, all_s1_ids: list, col_name: str) -> pd.DataFrame:
    """Groups candidate IDs by source1_entity_id and ensures all S1 IDs are present."""
    if len(pairs_df) > 0:
        grouped = pairs_df.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ",".join(sorted(list(set(x))))
        ).reset_index()
        grouped.rename(columns={'candidate_entity_id': col_name}, inplace=True)
    else:
        grouped = pd.DataFrame(columns=['source1_entity_id', col_name])
        
    all_s1_df = pd.DataFrame({'source1_entity_id': all_s1_ids})
    merged = all_s1_df.merge(grouped, on='source1_entity_id', how='left').fillna('')
    return merged

def main():
    print("Initializing pipeline...")
    os.makedirs('output', exist_ok=True)
    
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    gt_path = 'dataset/train/train_ground_truth.tsv'
    
    # 1. Load Data
    print("Loading training data...")
    df_train = load_split(train_dir, 'train')
    
    if df_train is None:
        print("Training data not found. Please place the dataset files to run.")
        return
        
    df_test = load_split(test_dir, 'test')
    
    # 2. Blocking (Candidate Generation)
    print("Generating candidates for training...")
    train_pairs = generate_candidates(df_train)
    
    # Label the train_pairs using ground truth
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path, sep='\t').fillna('')
        gt_dict = {}
        for _, row in gt_df.iterrows():
            matches = str(row.get('matched_entity_ids', '')).split(',')
            gt_dict[row['source1_entity_id']] = set([m for m in matches if m])
            
        labels = []
        for _, row in train_pairs.iterrows():
            s1_id, c_id = row['source1_entity_id'], row['candidate_entity_id']
            if c_id in gt_dict.get(s1_id, set()):
                labels.append(1)
            else:
                labels.append(0)
        train_pairs['label'] = labels
    else:
        print("No ground truth found, using dummy labels for compilation test.")
        train_pairs['label'] = 0 
    
    # 3. Feature Extraction
    print("Extracting features for training...")
    train_features = extract_features(train_pairs, df_train)
    
    # 4. Training
    print("Training LightGBM model...")
    model = train_model(train_features, train_pairs['label'])
    
    # 5. Inference
    if df_test is not None:
        print("Generating candidates for test set...")
        test_pairs = generate_candidates(df_test)
        
        s1_test_ids = df_test[df_test['entity_id'].str.startswith('S1-')]['entity_id'].unique()
        candidate_out = format_output_file(test_pairs, s1_test_ids, 'candidate_entity_ids')
        candidate_out.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
        
        print("Extracting test features & predicting matches...")
        test_features = extract_features(test_pairs, df_test)
        predictions = predict_matches(model, test_features, threshold=0.5)
        
        matching_out = format_output_file(predictions, s1_test_ids, 'matched_entity_ids')
        matching_out.to_csv('output/matching_results.tsv', sep='\t', index=False)
        print("Inference complete. Results saved to output/")
        
if __name__ == "__main__":
    main()

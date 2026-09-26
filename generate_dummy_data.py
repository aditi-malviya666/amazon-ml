import os
import pandas as pd

def create_dirs():
    os.makedirs('dataset/train', exist_ok=True)
    os.makedirs('dataset/test', exist_ok=True)

def generate_train_data():
    s1_data = [
        {"entity_id": "S1-001", "business_name": "Acme Robotics Inc", "business_address": "500 Market St, San Jose", "country": "US"},
        {"entity_id": "S1-002", "business_name": "Delta Foods", "business_address": "8 Oak Ave, Austin", "country": "US"},
        {"entity_id": "S1-003", "business_name": "Zen Trailers", "business_address": "4 Hill Rd, Boise", "country": "US"},
        {"entity_id": "S1-004", "business_name": "Tech Corp Pvt Ltd", "business_address": "MG Road, Bangalore", "country": "India"},
        {"entity_id": "S1-005", "business_name": "Singleton Bakery", "business_address": "123 Main St", "country": "US"} # Singleton
    ]
    
    s2_data = [
        {"entity_id": "S2-001", "business_name": "Acme Robotics Incorporated", "business_address": "500 Market Street, San Jose CA", "country": "US"},
        {"entity_id": "S2-002", "business_name": "Delta Foods Co", "business_address": "8 Oak Avenue, Austin", "country": "US"},
        {"entity_id": "S2-003", "business_name": "Tech Corporation", "business_address": "MG Rd, Bengaluru", "country": "India"}
    ]
    
    s3_data = [
        {"entity_id": "S3-001", "business_name": "Acme Robotics", "business_address": "Nr City Hall, San Jose", "country": "US"},
        {"entity_id": "S3-002", "business_name": "Delta Foods Ltd", "business_address": "8 Oak Av, Austin", "country": "US"},
        {"entity_id": "S3-003", "business_name": "Zen Trailers LLC", "business_address": "4 Hill Road, Boise", "country": "US"}
    ]

    gt_data = [
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-001,S3-001"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": "S2-002,S3-002"},
        {"source1_entity_id": "S1-003", "matched_entity_ids": "S3-003"},
        {"source1_entity_id": "S1-004", "matched_entity_ids": "S2-003"},
        {"source1_entity_id": "S1-005", "matched_entity_ids": ""} # Singleton
    ]

    pd.DataFrame(s1_data).to_csv('dataset/train/train_source1.tsv', sep='\t', index=False)
    pd.DataFrame(s2_data).to_csv('dataset/train/train_source2.tsv', sep='\t', index=False)
    pd.DataFrame(s3_data).to_csv('dataset/train/train_source3.tsv', sep='\t', index=False)
    pd.DataFrame(gt_data).to_csv('dataset/train/train_ground_truth.tsv', sep='\t', index=False)

def generate_test_data():
    s1_data = [
        {"entity_id": "S1-T01", "business_name": "Le Parisien SARL", "business_address": "10 Rue de Rivoli, Paris", "country": "France"},
        {"entity_id": "S1-T02", "business_name": "Global Traders", "business_address": "Connaught Place, Delhi", "country": "India"},
        {"entity_id": "S1-T03", "business_name": "Seattle Coffee", "business_address": "1st Ave, Seattle", "country": "US"},
        {"entity_id": "S1-T04", "business_name": "L'Amour", "business_address": "15 bd Saint-Germain", "country": "France"} # Singleton
    ]
    
    s2_data = [
        {"entity_id": "S2-T01", "business_name": "Le Parisien", "business_address": "10 r. de Rivoli, Paris", "country": "France"},
        {"entity_id": "S2-T02", "business_name": "Global Traders Pvt Ltd", "business_address": "CP, New Delhi", "country": "India"}
    ]
    
    s3_data = [
        {"entity_id": "S3-T01", "business_name": "Parisien SARL", "business_address": "10 Rue Rivoli", "country": "France"},
        {"entity_id": "S3-T02", "business_name": "Seattle Coffee Co", "business_address": "First Avenue, Seattle", "country": "US"}
    ]

    pd.DataFrame(s1_data).to_csv('dataset/test/test_source1.tsv', sep='\t', index=False)
    pd.DataFrame(s2_data).to_csv('dataset/test/test_source2.tsv', sep='\t', index=False)
    pd.DataFrame(s3_data).to_csv('dataset/test/test_source3.tsv', sep='\t', index=False)

if __name__ == "__main__":
    print("Generating dummy data...")
    create_dirs()
    generate_train_data()
    generate_test_data()
    print("Dummy dataset created at dataset/train/ and dataset/test/!")
    print("NOTE: Run this command to delete the dummy data before the competition:")
    print("rmdir /s /q dataset")

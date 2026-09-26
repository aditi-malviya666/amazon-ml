import pandas as pd
import re

def load_data(filepath: str) -> pd.DataFrame:
    """Loads TSV data into a pandas DataFrame."""
    return pd.read_csv(filepath, sep='\t', dtype=str).fillna('')

def clean_text(text: str) -> str:
    """Normalizes and cleans text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Applies cleaning to name and address columns."""
    if 'business_name' in df.columns:
        df['clean_name'] = df['business_name'].apply(clean_text)
    if 'business_address' in df.columns:
        df['clean_address'] = df['business_address'].apply(clean_text)
    if 'country' in df.columns:
        df['country'] = df['country'].str.upper().str.strip()
    return df

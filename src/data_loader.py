import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    if not (file_path.endswith('.csv') or file_path.endswith('.txt')):
        raise ValueError("Unsupported format. Only .csv and .txt are supported.")
    
    try:
        df = pd.read_csv(file_path, sep=',')
        if df.shape[1] <= 1:
            # If only one column detected, maybe it's semicolon separated
            df = pd.read_csv(file_path, sep=';')
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Handle missing values (mean imputation)
    df_numeric = df_numeric.fillna(df_numeric.mean())
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    df_scaled = pd.DataFrame(scaled_data, columns=df_numeric.columns)
    return df_scaled

def load_and_preprocess(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    df_clean = preprocess_data(df)
    return df_clean

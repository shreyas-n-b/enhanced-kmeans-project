import sys
import os
import numpy as np

# Add the project root directory to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.kmeans_baseline import run_kmeans
from src.enhanced_kmeans import enhanced_kmeans
from src.data_loader import load_and_preprocess
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_experiment(file_path: str, use_enhanced: bool):
    # Load and preprocess data
    try:
        df_clean = load_and_preprocess(file_path)
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")
        
    X = df_clean.values
    if X.shape[0] < 2:
        raise ValueError("Dataset too small for clustering.")
        
    k = 3 # Default k
    
    if not use_enhanced:
        labels, centroids = run_kmeans(X, k=k)
        
        try:
            sil_score = float(silhouette_score(X, labels))
            db_score = float(davies_bouldin_score(X, labels))
        except ValueError:
            sil_score = 0.0
            db_score = 0.0
            
        return {
            "algorithm": "baseline",
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score
        }
    else:
        res = enhanced_kmeans(X, k=k)
        labels = res["labels"]
        outliers = res["outliers"]
        
        # Calculate metrics only for valid (non-outlier) points
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        try:
            if len(np.unique(labels_valid)) > 1:
                sil_score = float(silhouette_score(X_valid, labels_valid))
                db_score = float(davies_bouldin_score(X_valid, labels_valid))
            else:
                sil_score = 0.0
                db_score = 0.0
        except ValueError:
            sil_score = 0.0
            db_score = 0.0
            
        return {
            "algorithm": "enhanced",
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "num_outliers": int(np.sum(outliers))
        }

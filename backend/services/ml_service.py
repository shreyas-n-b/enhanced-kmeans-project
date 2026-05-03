import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.kmeans_baseline import run_kmeans
from src.enhanced_kmeans import enhanced_kmeans
from src.data_loader import load_and_preprocess
from sklearn.metrics import silhouette_score, davies_bouldin_score

def run_experiment(file_path: str, use_enhanced: bool, k: int = 3):
    try:
        df_clean = load_and_preprocess(file_path)
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")
        
    X = df_clean.values
    if X.shape[0] < k:
        raise ValueError("Dataset too small for clustering.")
        
    if not use_enhanced:
        res = run_kmeans(X, k=k)
        labels = res["labels"]
        variance = res["variance"]
        iterations = res["iterations"]
        num_outliers = 0
        
        try:
            if len(np.unique(labels)) > 1:
                sil_score = float(silhouette_score(X, labels))
                db_score = float(davies_bouldin_score(X, labels))
            else:
                sil_score = 0.0
                db_score = 0.0
        except ValueError:
            sil_score = 0.0
            db_score = 0.0
            
        return {
            "algorithm": "baseline",
            "silhouette_score": sil_score,
            "davies_bouldin_score": db_score,
            "num_outliers": num_outliers,
            "variance": variance,
            "iterations": iterations,
            "history": res.get("history", [])
        }
    else:
        res = enhanced_kmeans(X, k=k)
        labels = res["labels"]
        outliers = res["outliers"]
        variance = res["variance"]
        iterations = res["iterations"]
        num_outliers = int(np.sum(outliers))
        
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
            "num_outliers": num_outliers,
            "variance": variance,
            "iterations": iterations,
            "history": res.get("history", [])
        }

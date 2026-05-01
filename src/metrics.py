from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np

def compute_metrics(X, labels):
    # Ignore outliers (labels == -1)
    valid_mask = labels != -1
    
    if np.sum(valid_mask) <= 1 or len(np.unique(labels[valid_mask])) <= 1:
        return {
            "silhouette": 0.0,
            "db_index": 0.0
        }
        
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    # Compute metrics
    silhouette = silhouette_score(X_valid, labels_valid)
    db_index = davies_bouldin_score(X_valid, labels_valid)
    
    return {
        "silhouette": silhouette,
        "db_index": db_index
    }

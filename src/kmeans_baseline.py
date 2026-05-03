from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from src.anomaly_detection import compute_distances

def run_kmeans(X, k=3):
    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    
    # Compute variance
    distances = compute_distances(X, centroids, labels)
    variance = float(np.mean(distances ** 2))
    
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
        
    history = [{
        "iteration": 1,
        "variance": variance,
        "silhouette": sil_score,
        "davies_bouldin": db_score
    }]
    
    return {
        "labels": labels,
        "centroids": centroids,
        "outliers": None,
        "variance": variance,
        "iterations": 1,
        "history": history
    }

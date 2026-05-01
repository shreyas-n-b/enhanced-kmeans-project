from sklearn.cluster import KMeans

def run_kmeans(X, k=3):
    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    
    # Return labels and centroids
    return kmeans.labels_, kmeans.cluster_centers_

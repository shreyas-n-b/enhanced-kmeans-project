import numpy as np

def compute_distances(X, centroids, labels):
    # compute Euclidean distance of each point to its assigned centroid
    distances = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        diff = X[i] - centroids[labels[i]]
        distances[i] = np.linalg.norm(diff)
    return distances

def detect_outliers(distances, threshold_factor=3.0):
    # threshold = mean + factor * std
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + threshold_factor * std_dist
    
    # return boolean array of outliers
    return distances > threshold

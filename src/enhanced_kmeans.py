import numpy as np
from src.kmeans_baseline import run_kmeans
from src.anomaly_detection import compute_distances, detect_outliers

def enhanced_kmeans(X, k=3, threshold_factor=3.0, max_iter=10, max_removal_pct=0.05):
    # Mask to keep track of points that are not outliers
    valid_mask = np.ones(X.shape[0], dtype=bool)
    
    for iteration in range(max_iter):
        X_valid = X[valid_mask]
        
        # 1. Run initial K-Means
        labels_valid, centroids = run_kmeans(X_valid, k=k)
        
        # 2. Compute distances
        distances = compute_distances(X_valid, centroids, labels_valid)
        
        # 3. Detect outliers
        current_outliers_valid = detect_outliers(distances, threshold_factor=threshold_factor)
        
        # Compute threshold for logging
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + threshold_factor * std_dist
        
        outlier_indices = np.where(current_outliers_valid)[0]
        
        # If no outliers detected, stop
        if len(outlier_indices) == 0:
            print(f"Iteration {iteration}: No outliers detected above threshold {threshold:.4f}. Stopping.")
            break
            
        # Limit number of outliers removed per iteration
        # Only remove top extreme points
        max_removal_count = max(1, int(len(X_valid) * max_removal_pct))
        
        if len(outlier_indices) > max_removal_count:
            # Sort the outlier distances in descending order
            outlier_distances = distances[outlier_indices]
            sorted_outlier_idx = np.argsort(outlier_distances)[::-1]
            
            # Keep only the top 'max_removal_count' most extreme points
            top_outlier_indices = outlier_indices[sorted_outlier_idx[:max_removal_count]]
            
            # Update current_outliers_valid mask
            current_outliers_valid = np.zeros_like(current_outliers_valid)
            current_outliers_valid[top_outlier_indices] = True
            
            num_removed = max_removal_count
        else:
            num_removed = len(outlier_indices)
            
        # Safety condition: Stop removing if remaining points < k * 2
        if len(X_valid) - num_removed < k * 2:
            print(f"Iteration {iteration}: Safety condition triggered. Stopping outlier removal (remaining points would be < {k * 2}).")
            break
            
        # 4. Log the threshold and removed count
        print(f"Iteration {iteration}: Threshold = {threshold:.4f} | Outliers removed = {num_removed}")
            
        # 5. Remove outliers
        # Find indices of valid points, then mark the new outliers as invalid
        valid_indices = np.where(valid_mask)[0]
        new_outlier_indices = valid_indices[current_outliers_valid]
        valid_mask[new_outlier_indices] = False

    # Final pass to get final centroids and labels for valid points
    X_valid = X[valid_mask]
    final_labels_valid, final_centroids = run_kmeans(X_valid, k=k)
    
    # Prepare return structure
    final_labels = np.full(X.shape[0], -1)  # Initialize with -1 for outliers
    valid_indices = np.where(valid_mask)[0]
    final_labels[valid_indices] = final_labels_valid
    
    outliers_mask = ~valid_mask
    
    return {
        "labels": final_labels,
        "outliers": outliers_mask,
        "centroids": final_centroids
    }

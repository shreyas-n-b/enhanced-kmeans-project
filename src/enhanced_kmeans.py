import numpy as np
from src.kmeans_baseline import run_kmeans
from src.anomaly_detection import compute_distances, detect_outliers

def compute_variance(distances):
    return float(np.mean(distances ** 2))

def enhanced_kmeans(X, k=3, threshold_factor=3.0, max_iter=10, max_removal_pct=0.05, tol=1e-4):
    valid_mask = np.ones(X.shape[0], dtype=bool)
    prev_variance = float('inf')
    num_iterations = 0
    final_variance = 0.0
    iteration_history = []
    
    for iteration in range(max_iter):
        num_iterations += 1
        X_valid = X[valid_mask]
        
        # 1. Run initial K-Means
        res_baseline = run_kmeans(X_valid, k=k)
        labels_valid = res_baseline["labels"]
        centroids = res_baseline["centroids"]
        
        # 2. Extract metrics
        current_variance = res_baseline["variance"]
        final_variance = current_variance
        
        b_hist = res_baseline["history"][0]
        iteration_history.append({
            "iteration": num_iterations,
            "variance": current_variance,
            "silhouette": b_hist["silhouette"],
            "davies_bouldin": b_hist["davies_bouldin"]
        })
        
        # Convergence check
        if abs(prev_variance - current_variance) < tol:
            print(f"Iteration {iteration}: Converged based on variance.")
            break
        prev_variance = current_variance
        
        # 3. Detect outliers
        distances = compute_distances(X_valid, centroids, labels_valid)
        current_outliers_valid = detect_outliers(distances, threshold_factor=threshold_factor)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist + threshold_factor * std_dist
        
        outlier_indices = np.where(current_outliers_valid)[0]
        
        if len(outlier_indices) == 0:
            print(f"Iteration {iteration}: No outliers detected above threshold {threshold:.4f}. Stopping.")
            break
            
        max_removal_count = max(1, int(len(X_valid) * max_removal_pct))
        
        if len(outlier_indices) > max_removal_count:
            outlier_distances = distances[outlier_indices]
            sorted_outlier_idx = np.argsort(outlier_distances)[::-1]
            top_outlier_indices = outlier_indices[sorted_outlier_idx[:max_removal_count]]
            current_outliers_valid = np.zeros_like(current_outliers_valid)
            current_outliers_valid[top_outlier_indices] = True
            num_removed = max_removal_count
        else:
            num_removed = len(outlier_indices)
            
        if len(X_valid) - num_removed < k * 2:
            print(f"Iteration {iteration}: Safety condition triggered. Stopping outlier removal.")
            break
            
        print(f"Iteration {iteration}: Threshold = {threshold:.4f} | Outliers removed = {num_removed} | Variance = {current_variance:.4f}")
            
        valid_indices = np.where(valid_mask)[0]
        new_outlier_indices = valid_indices[current_outliers_valid]
        valid_mask[new_outlier_indices] = False

    # Final pass
    X_valid = X[valid_mask]
    final_res = run_kmeans(X_valid, k=k)
    final_labels_valid = final_res["labels"]
    final_centroids = final_res["centroids"]
    final_variance = final_res["variance"]
    
    final_labels = np.full(X.shape[0], -1)
    valid_indices = np.where(valid_mask)[0]
    final_labels[valid_indices] = final_labels_valid
    outliers_mask = ~valid_mask
    
    return {
        "labels": final_labels,
        "centroids": final_centroids,
        "outliers": outliers_mask,
        "variance": final_variance,
        "iterations": num_iterations,
        "history": iteration_history
    }

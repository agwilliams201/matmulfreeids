import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import rbf_kernel

def process_file_in_chunks(file_path, label_column, chunk_size=100000):
    """Process a single CSV file in chunks."""
    processed_chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk.columns = chunk.columns.str.strip().str.lower()
        label_column = label_column.strip().lower()
        
        if label_column not in chunk.columns:
            raise ValueError(f"Label column '{label_column}' not found in {file_path}.")

        X = chunk.drop(columns=[label_column])
        y = chunk[label_column]
        
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y[X.index]

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        X = X.select_dtypes(include=[np.number])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        processed_chunks.append((X_scaled, y, X.columns))

    return processed_chunks

def batch_kernel_lda(X, y, batch_size=1000, kernel='rbf', gamma=1.0):
    """Apply kernel LDA using batch processing to handle large datasets."""
    n_samples = X.shape[0]
    n_classes = len(np.unique(y))
    n_features = X.shape[1]
    
    class_means = np.zeros((n_classes, n_features))
    total_mean = np.zeros(n_features)
    class_counts = np.bincount(y)
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]
        batch_y = y[i:end_idx]
        
        for c in range(n_classes):
            mask = batch_y == c
            if np.any(mask):
                class_means[c] += np.sum(batch_X[mask], axis=0)
        total_mean += np.sum(batch_X, axis=0)
    
    for c in range(n_classes):
        class_means[c] /= class_counts[c]
    total_mean /= n_samples
    
    S_b = np.zeros((n_features, n_features))
    S_w = np.zeros((n_features, n_features))
    
    for i in range(0, n_samples, batch_size):
        end_idx = min(i + batch_size, n_samples)
        batch_X = X[i:end_idx]
        batch_y = y[i:end_idx]
        
        for c in range(n_classes):
            mask = batch_y == c
            if np.any(mask):
                diff = class_means[c] - total_mean
                S_b += class_counts[c] * np.outer(diff, diff)
                
                class_diff = batch_X[mask] - class_means[c]
                S_w += class_diff.T @ class_diff
    
    S_w += np.eye(n_features) * 1e-5
    
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.inv(S_w) @ S_b)
        
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return np.abs(eigenvectors[:, 0])
    except np.linalg.LinAlgError:
        print("Warning: Linear algebra error occurred. Returning uniform weights.")
        return np.ones(n_features) / n_features

def kernel_lda_on_chunks(processed_chunks, batch_size=1000, kernel='rbf', gamma=1.0):
    """Apply kernel LDA on processed chunks using batch processing."""
    accumulated_weights = None
    feature_names = None
    
    for X, y, columns in processed_chunks:
        weights = batch_kernel_lda(X, y, batch_size=batch_size, kernel=kernel, gamma=gamma)
        
        if accumulated_weights is None:
            accumulated_weights = weights
            feature_names = columns
        else:
            accumulated_weights += weights
    
    return accumulated_weights, feature_names

def rank_features_by_importance(lda_weights, feature_names):
    """Rank features by their contribution to separability."""
    feature_importance = sorted(zip(feature_names, lda_weights), key=lambda x: x[1], reverse=True)
    return feature_importance

if __name__ == "__main__":
    data_dir = "data/"
    label_column = "Label"
    chunk_size = 100000
    batch_size = 1000
    
    print("Loading and processing data in chunks...")
    processed_chunks = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            print(f"Processing {file_path}...")
            chunks = process_file_in_chunks(file_path, label_column, chunk_size)
            processed_chunks.extend(chunks)
    
    print("Applying Batch Kernel LDA...")
    lda_weights, feature_names = kernel_lda_on_chunks(
        processed_chunks, 
        batch_size=batch_size,
        kernel='rbf', 
        gamma=1.0
    )
    
    print("Ranking features...")
    feature_importance = rank_features_by_importance(lda_weights, feature_names)
    
    print("\nTop 10 features:")
    for feature, importance in feature_importance[:10]:
        print(f"{feature}: {importance:.4f}")
    
    output_file = "ranked_features.csv"
    pd.DataFrame(feature_importance, columns=["Feature", "Importance"]).to_csv(output_file, index=False)
    print(f"\nRanked features saved to {output_file}.")
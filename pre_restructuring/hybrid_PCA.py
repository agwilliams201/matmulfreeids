import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from scipy import stats

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

def calculate_security_importance(X, y, columns):
    """Calculate security importance scores for features."""
    scores = {}
    
    mi_scores = mutual_info_classif(X, y)
    
    z_scores = np.abs(stats.zscore(X, axis=0))
    rare_event_scores = np.mean(z_scores > 3, axis=0)
    
    normal_variance = np.var(X[y==0], axis=0)
    normalized_variance = normal_variance / np.max(normal_variance)
    stability_scores = 1 - normalized_variance
    
    for i, column in enumerate(columns):
        scores[column] = {
            'mutual_info': mi_scores[i],
            'rare_events': rare_event_scores[i],
            'stability': stability_scores[i],
            'combined_score': (0.4 * mi_scores[i] + 
                             0.4 * rare_event_scores[i] + 
                             0.2 * stability_scores[i])
        }
    
    return scores

def hybrid_feature_selection(processed_chunks, n_components=0.95, importance_threshold=0.7):
    """Perform hybrid feature selection combining importance scoring and PCA."""
    all_scores = {}
    feature_names = None
    
    print("Calculating security importance scores...")
    for X, y, columns in processed_chunks:
        chunk_scores = calculate_security_importance(X, y, columns)
        
        if not all_scores:
            all_scores = chunk_scores
            feature_names = columns
        else:
            for feature in all_scores:
                for metric in all_scores[feature]:
                    all_scores[feature][metric] += chunk_scores[feature][metric]
    
    n_chunks = len(processed_chunks)
    for feature in all_scores:
        for metric in all_scores[feature]:
            all_scores[feature][metric] /= n_chunks
    
    high_importance_features = [
        feature for feature in all_scores 
        if all_scores[feature]['combined_score'] > importance_threshold
    ]
    
    print(f"Identified {len(high_importance_features)} high-importance features")
    
    remaining_features = [f for f in feature_names if f not in high_importance_features]
    
    if remaining_features:
        X_remaining = np.vstack([chunk[0][:, [i for i, f in enumerate(feature_names) 
                                            if f in remaining_features]] 
                               for chunk in processed_chunks])
        
        print("Applying PCA to remaining features...")
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_remaining)
        
        pca_importance = np.abs(pca.components_).mean(axis=0)
        pca_features = [f"PC_{i+1}" for i in range(pca_result.shape[1])]
    else:
        pca_importance = []
        pca_features = []
    
    feature_ranking = []
    
    for feature in high_importance_features:
        feature_ranking.append({
            'feature': feature,
            'importance': all_scores[feature]['combined_score'],
            'source': 'security_score',
            'details': all_scores[feature]
        })
    
    for i, importance in enumerate(pca_importance):
        feature_ranking.append({
            'feature': f"PC_{i+1}",
            'importance': importance,
            'source': 'pca',
            'details': {
                'explained_variance_ratio': pca.explained_variance_ratio_[i],
                'cumulative_variance_ratio': np.sum(pca.explained_variance_ratio_[:i+1])
            }
        })
    
    return feature_ranking, pca if remaining_features else None, high_importance_features

if __name__ == "__main__":
    data_dir = "data/"
    label_column = "Label"
    chunk_size = 100000
    
    print("Loading and processing data in chunks...")
    processed_chunks = []
    for file in os.listdir(data_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(data_dir, file)
            print(f"Processing {file_path}...")
            chunks = process_file_in_chunks(file_path, label_column, chunk_size)
            processed_chunks.extend(chunks)
    
    print("Applying hybrid feature selection...")
    feature_ranking, pca_model, high_importance_features = hybrid_feature_selection(
        processed_chunks,
        n_components=0.95,
        importance_threshold=0.7
    )
    
    results_df = pd.DataFrame(feature_ranking)
    results_df.to_csv("feature_analysis.csv", index=False)
    
    print("\nTop security-critical features:")
    for feature in feature_ranking[:10]:
        if feature['source'] == 'security_score':
            print(f"{feature['feature']}: {feature['importance']:.4f}")
    
    if pca_model:
        print("\nPCA Components retained:", pca_model.n_components_)
        print("Total variance explained:", 
              sum(pca_model.explained_variance_ratio_) * 100, "%")
    
    print(f"\nResults saved to feature_analysis.csv")
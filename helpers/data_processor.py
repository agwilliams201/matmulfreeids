import gc

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, RobustScaler


class MemoryEfficientIDSDataProcessor:
    """
    Improved data processor with better preprocessing and memory management.
    """
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()  # Changed to RobustScaler for better handling of outliers
        self.feature_stats = {}
        self.attack_mapping = {
            'BENIGN': 'Benign',
            'FTP-Patator': 'Brute Force',
            'SSH-Patator': 'Brute Force',
            'DoS GoldenEye': 'DoS',
            'DoS Hulk': 'DoS',
            'DoS Slowhttptest': 'DoS',
            'DoS slowloris': 'DoS',
            'Heartbleed': 'Heartbleed',
            'Web Attack - Brute Force': 'Web Attack',
            'Web Attack - Sql Injection': 'Web Attack',
            'Web Attack - SQL Injection': 'Web Attack',
            'Web Attack - XSS': 'Web Attack',
            'Infiltration': 'Infiltration',
            'Bot': 'Bot',
            'PortScan': 'PortScan',
            'DDoS': 'DDoS'
        }

    def preprocess_chunk(self, chunk):
        """
        Preprocess a single chunk of data with improved cleaning.
        """
        # Make a copy to avoid modifying the original
        processed_chunk = chunk.copy()
        
        # Get numeric columns, excluding 'Label' if it exists
        numeric_cols = processed_chunk.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        # Handle numeric columns only
        for col in numeric_cols:
            try:
                # Replace inf values
                processed_chunk[col] = processed_chunk[col].replace([np.inf, -np.inf], np.nan)
                
                # Calculate outlier bounds
                q1 = processed_chunk[col].quantile(0.25)
                q3 = processed_chunk[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                # Clip outliers
                processed_chunk[col] = processed_chunk[col].clip(lower_bound, upper_bound)
                
                # Handle skewness only if the column has no NaN values
                if not processed_chunk[col].isna().any():
                    skewness = processed_chunk[col].skew()
                    if abs(skewness) > 1:
                        # Ensure all values are positive before log transform
                        min_val = processed_chunk[col].min()
                        if min_val < 0:
                            processed_chunk[col] = processed_chunk[col] - min_val + 1
                        processed_chunk[col] = np.log1p(processed_chunk[col])
            
            except Exception as e:
                print(f"Warning: Error processing column {col}: {str(e)}")
                # If there's an error processing the column, keep it as is
                continue

        return processed_chunk

    def process_file_in_chunks(self, file_path, chunk_size=100000):
        """
        Process file in chunks with improved error handling and monitoring.
        """
        chunks = []
        total_rows = 0
        corrupted_rows = 0

        try:
            # Read CSV in chunks
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                try:
                    # Track original row count
                    chunk_rows = len(chunk)
                    total_rows += chunk_rows

                    # Basic cleaning
                    chunk.columns = chunk.columns.str.strip()
                    
                    # Preprocess chunk
                    cleaned_chunk = self.preprocess_chunk(chunk)
                    
                    if not cleaned_chunk.empty:
                        chunks.append(cleaned_chunk)
                    else:
                        corrupted_rows += chunk_rows

                except Exception as e:
                    print(f"Warning: Error processing chunk: {str(e)}")
                    corrupted_rows += chunk_rows

                # Force garbage collection
                gc.collect()

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

        # Report statistics
        if total_rows > 0:
            print(f"Processed {total_rows} total rows")
            print(f"Removed {corrupted_rows} corrupted rows ({(corrupted_rows/total_rows)*100:.2f}%)")

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def load_and_preprocess_data(self, data_dir, chunk_size=100000):
        """
        Load and preprocess data with improved monitoring and validation.
        """
        processed_data = []
        total_samples = 0
        attack_distribution = {}

        # Process only Tuesday's data first
        tuesday_file = "Tuesday-WorkingHours.pcap_ISCX.csv"
        file_path = Path(data_dir) / tuesday_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {tuesday_file} in {data_dir}")
        
        print(f"\nProcessing {tuesday_file}...")
            
        df = self.process_file_in_chunks(file_path, chunk_size)
        if not df.empty:
            # Track attack distribution
            if 'Label' in df.columns:
                attack_counts = df['Label'].value_counts()
                for attack, count in attack_counts.items():
                    attack_distribution[attack] = attack_distribution.get(attack, 0) + count
                total_samples += len(df)
            
            processed_data.append(df)
            
        gc.collect()

        # Print data statistics
        print("\nData Statistics:")
        print(f"Total samples: {total_samples}")
        print("\nAttack distribution:")
        for attack, count in attack_distribution.items():
            percentage = (count/total_samples)*100
            print(f"{attack}: {count} samples ({percentage:.2f}%)")

        # Combine processed data (just Tuesday in this case)
        print("\nCombining processed data...")
        full_data = processed_data[0]  # Take only Tuesday's data
        del processed_data
        gc.collect()

        if full_data.empty:
            raise ValueError("No data was successfully processed")

        # Encode labels
        print("Encoding labels...")
        full_data['Attack_Category'] = full_data['Label'].replace(self.attack_mapping)
        full_data['Attack_Category'] = full_data['Attack_Category'].fillna('Unknown')
        full_data['Label_Binary'] = (full_data['Attack_Category'] != 'Benign').astype(np.float32)

        # Select features
        feature_columns = full_data.select_dtypes(include=[np.number]).columns
        feature_columns = feature_columns.drop(['Label_Binary'])

        # Extract features and handle NaN values
        print("Handling missing values in features...")
        X = full_data[feature_columns].values
        
        # Fill NaN values with column medians
        for col_idx in range(X.shape[1]):
            col_median = np.nanmedian(X[:, col_idx])
            mask = np.isnan(X[:, col_idx])
            X[mask, col_idx] = col_median
        
        y = full_data['Label_Binary'].values

        # Verify no NaN values remain
        assert not np.isnan(X).any(), "NaN values remain after median filling"
        assert not np.isnan(y).any(), "NaN values found in labels"

        # Store feature statistics
        self.feature_stats = {
            'columns': feature_columns,
            'means': np.mean(X, axis=0),
            'stds': np.std(X, axis=0),
            'mins': np.min(X, axis=0),
            'maxs': np.max(X, axis=0)
        }

        print(f"Final dataset shape: {X.shape}")
        print(f"Number of features: {len(feature_columns)}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")

        return X, y, feature_columns
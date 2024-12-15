=============================
Memory-Efficient Data Processor
=============================

This section details the **MemoryEfficientIDSDataProcessor** class, which is responsible for preprocessing and managing network intrusion dataset efficiently. The processor emphasizes memory optimization, robust handling of outliers, and thorough validation of data integrity.

Overview
--------

The MemoryEfficientIDSDataProcessor provides:

- Robust preprocessing of raw network traffic data.
- Scalable processing with chunked data handling.
- Memory-efficient techniques to avoid resource exhaustion.
- Encoding of categorical labels for anomaly detection.

Key Features
------------

- **Outlier Handling:** Uses robust methods to clip outliers based on interquartile ranges.
- **Chunked Processing:** Processes data in manageable chunks to minimize memory usage.
- **Label Encoding:** Maps attack labels to categories and encodes them for binary classification.
- **Data Integrity Validation:** Ensures no missing or invalid values remain after preprocessing.

Code Components
----------------

Initialization
~~~~~~~~~~~~~~

The processor initializes key components such as:

- A `RobustScaler` for scaling features.
- A `LabelEncoder` for label encoding.
- A predefined mapping of attack labels to categories.

**Code Snippet:**

.. code-block:: python

    class MemoryEfficientIDSDataProcessor:
        """
        Improved data processor with better preprocessing and memory management.
        """
        def __init__(self):
            self.label_encoder = LabelEncoder()
            self.scaler = RobustScaler()  # Better handling of outliers
            self.feature_stats = {}
            self.attack_mapping = {
                'BENIGN': 'Benign',
                'FTP-Patator': 'Brute Force',
                ...  # Additional mappings
            }

Preprocessing Chunks
~~~~~~~~~~~~~~~~~~~~

The `preprocess_chunk` method performs the following steps:

1. **Outlier Clipping:** Clips numeric features to exclude extreme values.
2. **Log Transformation:** Reduces skewness in feature distributions.
3. **Error Handling:** Ignores columns with errors during processing.

**Code Snippet:**

.. code-block:: python

    def preprocess_chunk(self, chunk):
        """
        Preprocess a single chunk of data with improved cleaning.
        """
        processed_chunk = chunk.copy()
        numeric_cols = processed_chunk.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        # Handle numeric columns only
        for col in numeric_cols:
            try:
                ...  # Cleaning, calculate outlier quartile bounds (25%, 75%)
                
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
                ...  # Error handling

        return processed_chunk

Chunked File Processing
~~~~~~~~~~~~~~~~~~~~~~~~

The `process_file_in_chunks` method reads large CSV files in chunks to reduce memory overhead. It preprocesses each chunk and tracks statistics about corrupted rows.

**Code Snippet:**

.. code-block:: python

    def process_file_in_chunks(self, file_path, chunk_size=100000):
        """
        Process file in chunks with improved error handling and monitoring.
        """
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            ... # Basic cleaning
            self.preprocess_chunk(chunk)
            ... # Storage & error handling
        return pd.concat(chunks, ignore_index=True)

Data Loading and Preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `load_and_preprocess_data` method orchestrates the data processing pipeline. It:

1. Processes Tuesday's network traffic data from a specified directory.
2. Encodes labels into attack categories and binary outcomes.
3. Handles missing values by replacing them with column medians.
4. Stores feature statistics such as means, standard deviations, and ranges.

**Code Snippet:**

.. code-block:: python

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

Conclusion
----------

The MemoryEfficientIDSDataProcessor ensures that raw network traffic data is cleaned, scaled, and encoded effectively for intrusion detection tasks. By leveraging chunked processing and robust statistical techniques, it provides a scalable and reliable preprocessing pipeline.

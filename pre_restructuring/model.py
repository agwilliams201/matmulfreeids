import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, average_precision_score
import torch
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import gc
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import joblib
import warnings
from tqdm import tqdm

class TernaryLinear(nn.Module):
    """
    Improved TernaryLinear layer with better initialization and normalization.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = len(in_features) if type(in_features) != int else 1
        self.m = len(in_features[0]) if type(in_features) != int else 1
        # Initialize weights and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.alpha = sum(self.weight) * 1 / (self.n * self.m)
        # Learnable scaling factor
        self.scaling_factor = nn.Parameter(torch.Tensor(1))
        
        # Initialize parameters
        self.reset_parameters()
        
        # Batch normalization for input stabilization
        self.input_norm = nn.BatchNorm1d(in_features)
        
    def reset_parameters(self):
        """Initialize parameters with improved scaling."""
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in = self.weight.size(1)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.constant_(self.scaling_factor, 1.0)
        
    def constrain_weights(self):
        """Apply constraints to weights during training."""
        with torch.no_grad():
            # L2 normalize weights
            norm = self.weight.norm(p=2, dim=1, keepdim=True)
            self.weight.div_(norm.clamp(min=1e-12))
            
    def ternarize_weights(self):
        """Convert weights to ternary values with learned scaling."""
        # Calculate threshold based on weight distribution
        if self.alpha.device != self.weight.device:
            self.alpha = self.alpha.to(self.weight.device)
            
        w_ternary = torch.where(
            self.weight - self.alpha > 0, 1,
            torch.where(self.weight - self.alpha < 1, -1, 0)
        )
        return w_ternary
        # threshold = 0.7 * torch.std(self.weight)
        
        # # Ternarize weights
        # w_ternary = torch.zeros_like(self.weight)
        # w_ternary = torch.where(self.weight > threshold, 1.0, w_ternary)
        # w_ternary = torch.where(self.weight < -threshold, -1.0, w_ternary)
        
        # # Apply learned scaling
        # return w_ternary * self.scaling_factor
        
    def forward(self, x):
        # Apply input normalization
        x = self.input_norm(x)
        
        # Get ternary weights
        w_ternary = self.ternarize_weights()
        
        # Efficient matrix multiplication alternative
        pos_contrib = torch.zeros(x.size(0), self.out_features, device=x.device)
        neg_contrib = torch.zeros(x.size(0), self.out_features, device=x.device)
        
        # Process positive weights
        pos_mask = (w_ternary == 1.0)
        if pos_mask.any():
            pos_contrib = torch.sum(x.unsqueeze(2) * pos_mask.t().unsqueeze(0), dim=1)
            
        # Process negative weights
        neg_mask = (w_ternary == -1.0)
        if neg_mask.any():
            neg_contrib = torch.sum(x.unsqueeze(2) * neg_mask.t().unsqueeze(0), dim=1)
        
        # Combine contributions
        out = pos_contrib - neg_contrib + self.bias
        
        return out

class MatMulFreeGRU(nn.Module):
    """
    Improved MatMul-free GRU with better regularization and stability.
    """
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Gates using improved TernaryLinear
        self.update_gate = TernaryLinear(input_size + hidden_size, hidden_size)
        self.reset_gate = TernaryLinear(input_size + hidden_size, hidden_size)
        self.hidden_transform = TernaryLinear(input_size + hidden_size, hidden_size)
        
        # Additional regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, h=None):
        batch_size = x.size(0)
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Combine input and hidden state
        combined = torch.cat([x, h], dim=1)
        
        # Apply dropout to combined input
        combined = self.dropout(combined)
        
        # Compute gates with regularization
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        # Compute candidate hidden state
        combined_reset = torch.cat([x, reset * h], dim=1)
        candidate = torch.tanh(self.hidden_transform(combined_reset))
        
        # Update hidden state
        h_new = (1 - update) * h + update * candidate
        
        # Apply layer normalization
        h_new = self.layer_norm(h_new)
        
        return h_new, h_new

class EfficientIDS(nn.Module):
    """
    Improved hardware-efficient Intrusion Detection System with better architecture.
    """
    def __init__(self, num_features, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        layer_sizes = [num_features] + [hidden_size] * num_layers
        
        # Feature extraction layers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                TernaryLinear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ])
        
        # Temporal modeling
        self.gru = MatMulFreeGRU(hidden_size, hidden_size, dropout_rate)
        
        # Classification head with attention
        # self.attention = nn.Sequential(
        #     TernaryLinear(hidden_size, hidden_size // 2),
        #     nn.Tanh(),
        #     TernaryLinear(hidden_size // 2, 1)
        # )
        
        self.classifier = TernaryLinear(hidden_size, 1)
        
        # Additional regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, h=None):
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
        
        # Temporal modeling
        temporal_features, h_new = self.gru(x, h)
        
        # Apply attention
        # attention_weights = F.softmax(self.attention(temporal_features), dim=1)
        # attended_features = temporal_features * attention_weights
        attended_features = temporal_features
        # Classification with dropout
        features = self.dropout(attended_features)
        logits = self.classifier(features)
        
        return logits, h_new

class IDSProcessor:
    """
    Improved IDSProcessor with better training and evaluation capabilities.
    """
    def __init__(self, model_config=None):
        self.model = None
        self.config = model_config or {
            'hidden_size': 256,
            'num_layers': 2,
            'dropout_rate': 0.3
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': []
        }
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
    def train_model(self, X_train, y_train, X_val, y_val,
                epochs=20, batch_size=64, learning_rate=1e-3,
                model_type='matmul_free', callback=None):
        """Train the model with improved training loop and monitoring."""
        # Calculate class weights for balanced sampling
        class_counts = np.bincount(y_train.astype(int))
        weights = 1.0 / class_counts
        samples_weight = torch.FloatTensor(weights[y_train.astype(int)])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Initialize model if not already created
        if self.model is None:
            ModelClass = EfficientIDS if model_type == 'matmul_free' else StandardIDS
            self.model = ModelClass(
                num_features=X_train.shape[1],
                **self.config
            ).to(self.device)
        
        # Initialize optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Loss function with class weights
        class_weights = torch.FloatTensor([1.0, sum(y_train == 0) / sum(y_train == 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_x, batch_y in train_loader:
                if callback:
                    callback()
                    
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output, _ = self.model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    if callback:
                        callback()
                        
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output, _ = self.model(batch_x)
                    loss = criterion(output.squeeze(), batch_y)
                    val_loss += loss.item()
            
            avg_train_loss = total_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")
            
            # Update best model if validation loss improved
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict()
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
    def evaluate(self, data_loader):
        """Evaluate the model on a dataset."""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        criterion = nn.BCEWithLogitsLoss()
        
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output, _ = self.model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                total_loss += loss.item()
                
                pred = torch.sigmoid(output.squeeze()) > 0.5
                predictions.extend(pred.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        return {
            'loss': total_loss / len(data_loader),
            'precision': precision_score(targets, predictions),
            'recall': recall_score(targets, predictions),
            'f1': f1_score(targets, predictions)
        }
                
    def detect_anomalies(self, X_test, return_scores=False, callback=None):
        """Perform anomaly detection with optional score output and CPU measurement."""
        self.model.eval()
        predictions = []
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(X_test), 1000):
                if callback:
                    callback()
                    
                batch_x = torch.FloatTensor(X_test[i:i+1000]).to(self.device)
                output, _ = self.model(batch_x)
                scores.extend(torch.sigmoid(output.squeeze()).cpu().numpy())
                predictions.extend((torch.sigmoid(output.squeeze()) > 0.5).cpu().numpy())
                
        if return_scores:
            return np.array(predictions), np.array(scores)
        return np.array(predictions)

class StandardIDS(EfficientIDS):
    """Standard IDS model using regular PyTorch layers."""
    def __init__(self, num_features, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super(EfficientIDS, self).__init__()
        
        self.num_layers = num_layers
        layer_sizes = [num_features] + [hidden_size] * num_layers
        
        # Replace TernaryLinear with standard Linear layers
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ])
        
        # Standard GRU layer
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Standard attention and classifier
        # self.attention = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size // 2),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size // 2, 1)
        # )
        
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

# Import necessary metrics
from sklearn.metrics import precision_score, recall_score, f1_score

# Implementation of the data processing class and integration function continues...
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

def integrate_with_hardware_efficient_ids(data_dir, binary_classification=True, 
                                       chunk_size=100000, perform_hyperparameter_tuning=False,
                                       load_model=False, model_type='matmul_free',
                                       epochs=20, callback=None):
    """
    Improved integration function with measurement callback.
    """
    print("\nInitializing data processor...")
    data_processor = MemoryEfficientIDSDataProcessor()

    print("Loading and preprocessing data...")
    try:
        X, y, feature_columns = data_processor.load_and_preprocess_data(data_dir, chunk_size)
    except Exception as e:
        raise RuntimeError(f"Error during data processing: {str(e)}")

    # Data validation
    print("\nValidating data...")
    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isinf(X).any(), "X contains infinite values"
    print("Data validation passed")

    # Split data
    print("\nSplitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    # Initialize processor and train/load model
    print("\nInitializing model...")
    ids_processor = IDSProcessor()
    
    if callback:
        callback()  # Initial measurement
    
    if load_model:
        checkpoint_file = 'trained_ids_model.pth' if model_type == 'matmul_free' else 'trained_standard_model.pth'
        if Path(checkpoint_file).exists():
            print(f"Loading saved {model_type} model...")
            checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
            ids_processor.model = (EfficientIDS if model_type == 'matmul_free' else StandardIDS)(
                num_features=X.shape[1],
                **checkpoint['hyperparameters']
            ).to(ids_processor.device)
            ids_processor.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            print(f"No checkpoint found for {model_type}. Training from scratch...")
    else:
        print("Training new model...")
        def training_callback():
            if callback:
                callback()
        
        ids_processor.train_model(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            model_type=model_type,
            callback=training_callback
        )
        
        # Save model
        checkpoint = {
            'model_state_dict': ids_processor.model.state_dict(),
            'hyperparameters': ids_processor.config,
            'feature_stats': data_processor.feature_stats
        }
        if model_type == 'matmul_free':
            print("Saving to matmul_free...")
            torch.save(checkpoint, 'trained_ids_model.pth')
        elif model_type == 'standard':
            print("Saving to standard...")
            print(ids_processor.model.state_dict())
            torch.save(checkpoint, 'trained_standard_model.pth')

    if callback:
        callback()  # Final measurement

    return ids_processor, {
        'X_test': X_test,
        'y_test': y_test,
        'feature_columns': feature_columns
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate IDS model')
    parser.add_argument('--data_dir', type=str, default='data/',
                    help='Directory containing the data files')
    parser.add_argument('--chunk_size', type=int, default=100000,
                    help='Size of data chunks for processing')
    parser.add_argument('--model_type', type=str, choices=['matmul_free', 'standard'],
                    default='matmul_free', help='Type of model to use')
    parser.add_argument('--load_model', action='store_true',
                    help='Load existing model instead of training')
    parser.add_argument('--epochs', type=int, default=20,
                    help='Number of training epochs')
    
    args = parser.parse_args()
    
    try:
        # Train/load and evaluate model
        ids_processor, test_data = integrate_with_hardware_efficient_ids(
            args.data_dir,
            chunk_size=args.chunk_size,
            load_model=args.load_model,
            model_type=args.model_type,
            epochs=args.epochs
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        predictions = ids_processor.detect_anomalies(test_data['X_test'])
        
        # Print results
        print("\nModel Performance:")
        print(classification_report(test_data['y_test'], predictions))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
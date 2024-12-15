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
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n = len(in_features) if type(in_features) != int else 1
        self.m = len(in_features[0]) if type(in_features) != int else 1
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.alpha = sum(self.weight) * 1 / (self.n * self.m)
        self.scaling_factor = nn.Parameter(torch.Tensor(1))
        
        self.reset_parameters()
        
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
            norm = self.weight.norm(p=2, dim=1, keepdim=True)
            self.weight.div_(norm.clamp(min=1e-12))
            
    def ternarize_weights(self):
        """Convert weights to ternary values with learned scaling."""
        if self.alpha.device != self.weight.device:
            self.alpha = self.alpha.to(self.weight.device)
            
        w_ternary = torch.where(
            self.weight - self.alpha > 0, 1,
            torch.where(self.weight - self.alpha < 1, -1, 0)
        )
        return w_ternary

    def forward(self, x):
        x = self.input_norm(x)
        
        w_ternary = self.ternarize_weights()
        
        pos_contrib = torch.zeros(x.size(0), self.out_features, device=x.device)
        neg_contrib = torch.zeros(x.size(0), self.out_features, device=x.device)
        
        pos_mask = (w_ternary == 1.0)
        if pos_mask.any():
            pos_contrib = torch.sum(x.unsqueeze(2) * pos_mask.t().unsqueeze(0), dim=1)
            
        neg_mask = (w_ternary == -1.0)
        if neg_mask.any():
            neg_contrib = torch.sum(x.unsqueeze(2) * neg_mask.t().unsqueeze(0), dim=1)
        
        out = pos_contrib - neg_contrib + self.bias
        
        return out

class MatMulFreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        self.update_gate = TernaryLinear(input_size + hidden_size, hidden_size)
        self.reset_gate = TernaryLinear(input_size + hidden_size, hidden_size)
        self.hidden_transform = TernaryLinear(input_size + hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, h=None):
        batch_size = x.size(0)
        if h is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        combined = torch.cat([x, h], dim=1)
        
        combined = self.dropout(combined)
        
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        
        combined_reset = torch.cat([x, reset * h], dim=1)
        candidate = torch.tanh(self.hidden_transform(combined_reset))
        
        h_new = (1 - update) * h + update * candidate
        
        h_new = self.layer_norm(h_new)
        
        return h_new, h_new

class EfficientIDS(nn.Module):
    def __init__(self, num_features, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super().__init__()
        
        self.num_layers = num_layers
        layer_sizes = [num_features] + [hidden_size] * num_layers
        
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                TernaryLinear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ])
        
        self.gru = MatMulFreeGRU(hidden_size, hidden_size, dropout_rate)
        
        self.classifier = TernaryLinear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, h=None):
        for layer in self.feature_layers:
            x = layer(x)
        
        temporal_features, h_new = self.gru(x, h)

        attended_features = temporal_features
        features = self.dropout(attended_features)
        logits = self.classifier(features)
        
        return logits, h_new

class IDSProcessor:
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
        class_counts = np.bincount(y_train.astype(int))
        weights = 1.0 / class_counts
        samples_weight = torch.FloatTensor(weights[y_train.astype(int)])
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        
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
        
        if self.model is None:
            ModelClass = EfficientIDS if model_type == 'matmul_free' else StandardIDS
            self.model = ModelClass(
                num_features=X_train.shape[1],
                **self.config
            ).to(self.device)

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
        
        class_weights = torch.FloatTensor([1.0, sum(y_train == 0) / sum(y_train == 1)]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        
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
            
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.best_model_state = self.model.state_dict()
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
    def evaluate(self, data_loader):
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
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        return {
            'loss': total_loss / len(data_loader),
            'precision': precision_score(targets, predictions),
            'recall': recall_score(targets, predictions),
            'f1': f1_score(targets, predictions)
        }
                
    def detect_anomalies(self, X_test, return_scores=False, callback=None):
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
    def __init__(self, num_features, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super(EfficientIDS, self).__init__()
        
        self.num_layers = num_layers
        layer_sizes = [num_features] + [hidden_size] * num_layers
        
        self.feature_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_sizes[i], layer_sizes[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(layer_sizes[i+1]),
                nn.Dropout(dropout_rate)
            ) for i in range(num_layers)
        ])
        
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        self.classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout_rate)

from sklearn.metrics import precision_score, recall_score, f1_score

class MemoryEfficientIDSDataProcessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
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
        processed_chunk = chunk.copy()
        
        numeric_cols = processed_chunk.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        
        for col in numeric_cols:
            try:
                processed_chunk[col] = processed_chunk[col].replace([np.inf, -np.inf], np.nan)
                
                q1 = processed_chunk[col].quantile(0.25)
                q3 = processed_chunk[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                
                processed_chunk[col] = processed_chunk[col].clip(lower_bound, upper_bound)
                
                if not processed_chunk[col].isna().any():
                    skewness = processed_chunk[col].skew()
                    if abs(skewness) > 1:
                        min_val = processed_chunk[col].min()
                        if min_val < 0:
                            processed_chunk[col] = processed_chunk[col] - min_val + 1
                        processed_chunk[col] = np.log1p(processed_chunk[col])
            
            except Exception as e:
                print(f"Warning: Error processing column {col}: {str(e)}")
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
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                try:
                    chunk_rows = len(chunk)
                    total_rows += chunk_rows

                    chunk.columns = chunk.columns.str.strip()
                    
                    cleaned_chunk = self.preprocess_chunk(chunk)
                    
                    if not cleaned_chunk.empty:
                        chunks.append(cleaned_chunk)
                    else:
                        corrupted_rows += chunk_rows

                except Exception as e:
                    print(f"Warning: Error processing chunk: {str(e)}")
                    corrupted_rows += chunk_rows

                gc.collect()

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

        if total_rows > 0:
            print(f"Processed {total_rows} total rows")
            print(f"Removed {corrupted_rows} corrupted rows ({(corrupted_rows/total_rows)*100:.2f}%)")

        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def load_and_preprocess_data(self, data_dir, chunk_size=100000):
        processed_data = []
        total_samples = 0
        attack_distribution = {}

        tuesday_file = "Tuesday-WorkingHours.pcap_ISCX.csv"
        file_path = Path(data_dir) / tuesday_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Could not find {tuesday_file} in {data_dir}")
        
        print(f"\nProcessing {tuesday_file}...")
            
        df = self.process_file_in_chunks(file_path, chunk_size)
        if not df.empty:
            if 'Label' in df.columns:
                attack_counts = df['Label'].value_counts()
                for attack, count in attack_counts.items():
                    attack_distribution[attack] = attack_distribution.get(attack, 0) + count
                total_samples += len(df)
            
            processed_data.append(df)
            
        gc.collect()

        print("\nData Statistics:")
        print(f"Total samples: {total_samples}")
        print("\nAttack distribution:")
        for attack, count in attack_distribution.items():
            percentage = (count/total_samples)*100
            print(f"{attack}: {count} samples ({percentage:.2f}%)")

        print("\nCombining processed data...")
        full_data = processed_data[0]
        del processed_data
        gc.collect()

        if full_data.empty:
            raise ValueError("No data was successfully processed")

        print("Encoding labels...")
        full_data['Attack_Category'] = full_data['Label'].replace(self.attack_mapping)
        full_data['Attack_Category'] = full_data['Attack_Category'].fillna('Unknown')
        full_data['Label_Binary'] = (full_data['Attack_Category'] != 'Benign').astype(np.float32)

        feature_columns = full_data.select_dtypes(include=[np.number]).columns
        feature_columns = feature_columns.drop(['Label_Binary'])

        print("Handling missing values in features...")
        X = full_data[feature_columns].values
        
        for col_idx in range(X.shape[1]):
            col_median = np.nanmedian(X[:, col_idx])
            mask = np.isnan(X[:, col_idx])
            X[mask, col_idx] = col_median
        
        y = full_data['Label_Binary'].values

        assert not np.isnan(X).any(), "NaN values remain after median filling"
        assert not np.isnan(y).any(), "NaN values found in labels"

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

    print("\nInitializing data processor...")
    data_processor = MemoryEfficientIDSDataProcessor()

    print("Loading and preprocessing data...")
    try:
        X, y, feature_columns = data_processor.load_and_preprocess_data(data_dir, chunk_size)
    except Exception as e:
        raise RuntimeError(f"Error during data processing: {str(e)}")

    print("\nValidating data...")
    assert not np.isnan(X).any(), "X contains NaN values"
    assert not np.isinf(X).any(), "X contains infinite values"
    print("Data validation passed")

    print("\nSplitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, stratify=y_train_val
    )

    print("\nInitializing model...")
    ids_processor = IDSProcessor()
    
    if callback:
        callback()
    
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
            torch.save(checkpoint, 'trained_standard_model.pth')

    if callback:
        callback()

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
        ids_processor, test_data = integrate_with_hardware_efficient_ids(
            args.data_dir,
            chunk_size=args.chunk_size,
            load_model=args.load_model,
            model_type=args.model_type,
            epochs=args.epochs
        )
        
        print("\nEvaluating model...")
        predictions = ids_processor.detect_anomalies(test_data['X_test'])
        
        print("\nModel Performance:")
        print(classification_report(test_data['y_test'], predictions))
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
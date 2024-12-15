import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import numpy as np

from models.baseline import StandardIDS
from models.efficient_ids import EfficientIDS


class IDSProcessor:
    """
    Improved IDSProcessor with better training and evaluation capabilities (Glorified model wrapper).
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
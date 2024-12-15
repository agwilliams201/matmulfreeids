import torch
import torch.nn as nn
import numpy as np


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
import torch.nn as nn

from models.efficient_ids import EfficientIDS


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
=============================
Baseline IDS Model
=============================

This section describes the **StandardIDS**, our baseline model for intrusion detection. It uses conventional PyTorch layers without the hardware-specific optimizations found in the EfficientIDS.

Overview
--------

The StandardIDS serves as a reference implementation, utilizing:

- Standard `Linear` layers for feature extraction.
- A traditional GRU layer for temporal modeling.
- Standard PyTorch layers for classification.

Key Features
------------

- **Feature Extraction:** Extracts features using stacked `Linear` layers with ReLU activation and batch normalization.
- **Temporal Modeling:** Implements a standard GRU layer for sequential data processing.
- **Classification:** Uses a `Linear` layer to classify the processed features.
- **Dropout:** Provides regularization to prevent overfitting.

Code Components
----------------

Initialization
~~~~~~~~~~~~~~

The StandardIDS replaces hardware-efficient layers in the EfficientIDS with their standard counterparts, offering a straightforward baseline for performance comparison.

**Code Snippet:**

.. code-block:: python

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

            # Standard classifier
            self.classifier = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(dropout_rate)

Feature Extraction
~~~~~~~~~~~~~~~~~~

The model uses stacked `Linear` layers for feature extraction, with each layer followed by:

- **ReLU Activation:** Introduces non-linearity.
- **Batch Normalization:** Normalizes feature distributions.
- **Dropout:** Prevents overfitting by randomly setting activations to zero.

Temporal Modeling
~~~~~~~~~~~~~~~~~

A standard GRU layer processes temporal dependencies in the data, leveraging:

- Input size equal to the hidden size of the previous layer.
- Hidden state propagation across time steps.

Classification
~~~~~~~~~~~~~~~

The classification head comprises:

- A `Linear` layer to map temporal features to anomaly scores.
- Dropout for regularization.

Conclusion
----------

The StandardIDS model serves as a straightforward, hardware-independent baseline for intrusion detection. By comparing its performance with EfficientIDS, researchers can quantify the benefits of hardware-efficient optimizations.

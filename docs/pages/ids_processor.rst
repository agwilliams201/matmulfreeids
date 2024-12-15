=============================
IDS Wrapper
=============================

This section introduces the **IDSProcessor**, which serves as a model wrapper for training, evaluating, and detecting anomalies using various IDS models. The processor integrates efficient training techniques, balanced sampling, and robust evaluation metrics.

Overview
--------

The IDSProcessor provides:

- Model training with support for balanced class sampling.
- Efficient anomaly detection using trained models.
- Evaluation tools for computing loss and classification metrics.
- Device compatibility (CPU and GPU).

Key Features
------------

- **Training Loop:** Implements a custom training loop with monitoring and gradient clipping.
- **Evaluation Metrics:** Calculates precision, recall, F1-score, and loss.
- **Anomaly Detection:** Supports large-scale data detection with optional score outputs.
- **Model Checkpointing:** Automatically saves and restores the best model state based on validation loss.

Code Components
----------------

Initialization
~~~~~~~~~~~~~~

The IDSProcessor initializes:

- A model configuration, including hyperparameters like hidden size, number of layers, and dropout rate.
- A device setting (CPU or GPU) based on system availability.
- Training history to track metrics like loss and F1-score.

**Code Snippet:**

.. code-block:: python

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

Training
~~~~~~~~

The `train_model` method handles:

- **Class Balancing:** Uses a weighted random sampler to address class imbalances.
- **Gradient Clipping:** Prevents exploding gradients during training.
- **Learning Rate Scheduling:** Employs a `OneCycleLR` scheduler for dynamic learning rates.

**Code Snippet:**

.. code-block:: python

    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, learning_rate=1e-3):
        """
        Train the model with improved training loop and monitoring.
        """
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
            ModelClass = EfficientIDS
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

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([sum(y_train == 0) / sum(y_train == 1)]).to(self.device))

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                output, _ = self.model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output, _ = self.model(batch_x)
                    loss = criterion(output.squeeze(), batch_y)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {avg_val_loss:.4f}")

Evaluation
~~~~~~~~~~

The `evaluate` method calculates:

- **Loss:** Average binary cross-entropy loss.
- **Precision, Recall, F1-Score:** Metrics for model performance.

**Code Snippet:**

.. code-block:: python

    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset.
        """
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

Anomaly Detection
~~~~~~~~~~~~~~~~~

The `detect_anomalies` method performs:

- **Batch-Wise Detection:** Processes data in batches to optimize memory usage.
- **Optional Score Output:** Returns anomaly scores for further analysis.

**Code Snippet:**

.. code-block:: python

    def detect_anomalies(self, X_test, return_scores=False):
        """
        Perform anomaly detection with optional score output.
        """
        self.model.eval()
        predictions = []
        scores = []

        with torch.no_grad():
            for i in range(0, len(X_test), 1000):
                batch_x = torch.FloatTensor(X_test[i:i+1000]).to(self.device)
                output, _ = self.model(batch_x)
                scores.extend(torch.sigmoid(output.squeeze()).cpu().numpy())
                predictions.extend((torch.sigmoid(output.squeeze()) > 0.5).cpu().numpy())

        if return_scores:
            return np.array(predictions), np.array(scores)
        return np.array(predictions)

Conclusion
----------

The IDSProcessor acts as a comprehensive wrapper for managing IDS models, ensuring efficient training, robust evaluation, and scalable anomaly detection. Its modular design supports a variety of use cases in intrusion detection.

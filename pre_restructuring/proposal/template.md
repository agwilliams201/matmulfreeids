# Network Anomaly Detection with BitNet

## Project Participants

| Name | Surname | CNET ID | Project Role |
|------|----------|----------|--------------|
| Alexander | Baumgartner | abaumgartner | Model development and testing |
| Alexander | Williams | agwilliams200 | Quantization and model training |
| Alejandro | Alonso | aalonso20 | Model testing and writeup |

## Project Description

One of the main problems with machine learning models on a large scale is their resource-intensive nature. This can cause massive spikes in both energy usage and runtime, which is especially problematic in fields such as networking, where latency is of the utmost importance. Our goal with this project is to transfer the principles of BitNet, a recent model which does away with resource-intensive matrix multiplications via the quantization of weights, to a networking context. Specifically, we will build a simple regression model with this method, which we will then use on a dataset of packet traces during which several intrusions were attempted. Given enough time, we will experiment with extending the methodology to more complex machine learning models in an effort to maximize our accuracy and efficiency. Our hope is that our model can quickly and accurately predict when an anomaly or intrusion is occurring.

### Proposed Models

1. **Supervised Binary-Weight Temporal Classifier**
   - Adapts BitNet's 1-bit weight architecture but adds temporal convolution layers to process network packet windows
   - Uses labeled anomaly data to train a binary classifier (normal/anomaly), with the model maintaining BitNet's memory efficiency while learning from known attack patterns
   - Strategy: Train on sliding windows of network metrics where some days contain labeled anomalies, using BitNet's group quantization for efficient batch processing

2. **Unsupervised BitNet Autoencoder**
   - Implements BitNet's binary weight approach in an autoencoder architecture to learn normal network behavior patterns
   - Detects anomalies by measuring reconstruction error between input and output traffic patterns
   - Strategy: Train only on normal traffic days to establish baseline behavior, then flag deviations during inference, using BitNet's efficient memory footprint to process longer historical windows

3. **Semi-Supervised Hybrid Detector**
   - Combines a BitNet-based supervised classifier for known attack patterns with an unsupervised component for detecting novel anomalies
   - Uses binary weights in both components but maintains separate detection pathways
   - Strategy: Leverage labeled data where available while still being able to detect unknown anomalies, using BitNet's scaling properties to handle the larger dual architecture efficiently

## Data

The project will utilize the Network Intrusion Dataset available at:
[Network Intrusion Dataset on Kaggle](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)

## Deliverables

A Jupyter notebook containing:

1. Original regression model without quantized weights
2. Performance statistics for the non-quantized model:
   - Confusion matrix evaluated using an 80/20 train-test split
   - Timing measurements
   - CPU/GPU usage statistics
3. Mathematical documentation:
   - Weight quantization process
   - Implementation of quantized weights for row operations instead of matmul operations
4. Performance statistics for the quantized model:
   - Confusion matrix evaluated using an 80/20 train-test split
   - Timing measurements
   - CPU/GPU usage statistics
5. A brief writeup (Sphinx) detailing:
   - Encountered obstacles
   - Model performance analysis
   - Success evaluation

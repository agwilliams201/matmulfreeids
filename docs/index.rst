.. Network Anomaly Detection with BitNet documentation master file, created by
   sphinx-quickstart on Tue Dec 10 17:09:48 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

========================================================================
Network Anomaly Detection with BitNet | Intrusion Detection System (IDS)
========================================================================

Final project for the University of Chicago's CMSC 25422 Machine Learning for Computer Systems course.

Collaborators: Alexander Baumgartner, Alexander Williams, Alejandro Alonso

Professor: Nick Feamster


Introduction
============

Project Description: One of the main problems with machine learning models on a large scale is their resource-intensive nature. This can cause massive spikes in both energy usage and runtime, which is especially problematic in fields such as networking, where latency is of the utmost importance. Our goal with this project is to transfer the principles of BitNet, a recent model which does away with resource-intensive matrix multiplications via the quantization of weights, to a networking context. Specifically, we build a simple regression model with this method, which we then use on a dataset of packet traces during which several intrusions were attempted.


Contents
========

- **Models:** Learn about the various IDS models implemented in this project, including both baseline and hardware-efficient approaches.
- **Data Processing:** Understand the preprocessing techniques employed to prepare network traffic data for model training and evaluation.
- **Performance Evaluation:** Explore the evaluation metrics, runtime analyses, and energy consumption insights that highlight the strengths of each model.


Project Highlights
===================

- **Hardware Efficiency:** The project emphasizes the use of hardware-efficient techniques, leveraging ternary weights and reduced-complexity operations.
- **Performance Comparison:** Provides a robust framework for comparing baseline and optimized IDS models.
- **Scalability:** Demonstrates scalable data processing and evaluation for large datasets.


Getting Started
===============

To get started with this project, refer to the following resources:

1. **Project Notebook:**
   - Provides an end-to-end demonstration of the IDS evaluation pipeline.
   - Combines training, evaluation, and performance measurement in a clear and executable format.
   - In-depth visual analyses of precision-recall curves and CPU usage over time for each model.

2. **Models Documentation:**
   - Overview of the implemented IDS models, including the StandardIDS (baseline model) and EfficientIDS (our model).
   - Insights into their architectural choices and use cases.

3. **Data Processor Documentation:**
   - Overview of data preprocessing techniques, including handling outliers and preparing datasets for anomaly detection.

4. **Model Wrapper Documentation:**
   - Overview of the `IDSProcessor`, which integrates training, evaluation, and anomaly detection functionalities.

.. toctree::
   :maxdepth: 2
   :caption: Resources:

   pages/main
   pages/efficient_ids
   pages/ids_data_processor
   pages/ids_processor
   pages/standard_ids
   pages/proposal


References
==========

For further details, please refer to the project guidelines and documentation files listed above.

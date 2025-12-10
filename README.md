Multimodal Stress & Depression Detection System

This project introduces an AI-based system that detects stress and depression using three behavioral modalities:
Facial expressions
Body posture
Gait dynamics (accelerometer + pose CSVs)
The model combines CNN image features with MLP gait features through a fusion neural architecture, enabling non-invasive and objective mental-health analysis.

Features
Multimodal input: Face + Posture + Gait
Automated gait CSV feature extraction
CNN-based visual feature learning
MLP branch for gait statistics
Unified fusion classifier (binary or multi-class)
Google Colab–ready training pipeline

Architecture Overview

Face Images  → CNN ┐

Posture Images → CNN ┤ → Fusion Layer → Dense Classifier → Output

Gait Features → MLP ┘

TechStack

TensorFlow / Keras
MobileNetV2 + Custom CNNs
MLP for gait modeling
NumPy, Pandas, Scikit-Learn
Matplotlib
Platform: Google Colab GPU

Model Performances

Accuracy: 95–98% (varies by dataset)
Outputs:
Confusion matrix
Classification report
Accuracy & loss curves

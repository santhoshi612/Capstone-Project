# Multimodal Stress & Depression Detection System

This project presents an advanced AI-based system designed to detect stress and depression using three behavioral modalities:

Facial expressions

Body posture

Gait dynamics (accelerometer + pose-based CSV features)

The model fuses CNN-based visual features with MLP-based gait embeddings, forming a unified neural architecture that enables objective, non-invasive mental-health assessment.

<br>

**Features**

Multimodal input: Face + Posture + Gait

Automated gait CSV feature extraction

Custom CNN architecture for image learning

MLP branch for gait statistics modeling

Fusion classifier supporting binary or multi-class tasks

Optimized & Colab-ready training pipeline

Evaluation metrics: Accuracy, Confusion Matrix, Classification Report

<br>

**Architecture Overview**

Face Images     →     CNN ┐

Posture Images  →     CNN ┤ → Fusion Layer → Dense Classifier → Output

Gait Features   →     MLP ┘


Each modality is independently processed using a specialized neural branch, and the extracted embeddings are fused for final prediction.

<br>

**Tech Stack**

TensorFlow / Keras

Custom CNN Architectures

MLP for Gait Modeling

NumPy, Pandas, Scikit-Learn

Matplotlib for Visualization

Platform: Google Colab (GPU-accelerated)

<br>

**Model Performance**

Accuracy: 95% – 98% (varies by dataset and modality mix)

Outputs Generated:

Confusion Matrix

Precision/Recall/F1 Classification Report

Training Accuracy & Loss Curves

<br>

**How to Run**

Clone or download the repository

Upload to Google Colab

Set the correct dataset paths

Run the multimodal pipeline script

<br>

**Dataset Requirements**

Facial images: Organized by class folders

Posture images: Keypoint/skeleton images

Gait CSVs: Accelerometer or pose-based numeric files

The system automatically extracts, pads, and normalizes gait features.

<br>

**Output**

Clean training/validation accuracy curves

Fusion-model predictions

Full evaluation summary

<br>

**License**

This project is intended for research and academic use.

<br>

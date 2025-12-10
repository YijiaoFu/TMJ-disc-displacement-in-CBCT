# CBCT-based Diagnosis System for Temporomandibular Joint Disc Displacement

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A two-stage deep learning system for assisting in the diagnosis of Temporomandibular Joint (TMJ) disc displacement from Cone-Beam Computed Tomography (CBCT) images. This repository contains the complete pipeline from data preprocessing to clinical application.

## 📋 Overview

This project implements a novel two-stage approach for TMJ disc displacement screening:

1. **Region Detection Stage**: Uses YOLOv11 to automatically locate TMJ regions in CBCT slices
2. **Classification Stage**: Employs vision transformer models (FastViT, EfficientViT) and CNNs to classify TMJ regions as normal or abnormal

The system demonstrates promising performance with **73.3% AUC** and **66.9% accuracy**, offering a practical screening tool for orthodontic clinics.

## 🏗️ Architecture

```
Input CBCT DICOM
     ↓
YOLOv11 Region Detection
     ↓
TMJ ROI Extraction
     ↓
Deep Learning Classification
     ↓
Risk Assessment & Visualization
```

## 📁 Repository Structure

```
TMJ-Disc-Displacement-CBCT/
├── data_processing/
│   ├── YOLO_press.py              # YOLO-based TMJ region detection and cropping
│   └── patient_info.xlsx          # Patient metadata and labels (template)
├── model_training/
│   ├── TMJDD_classification_train.py    # Classification model training
│   └── TMJDD_classification_test.py     # Model evaluation and patient-level aggregation
├── clinical_application/
│   └── TMJDD_app.py              # Streamlit-based clinical interface
├── models/
│   ├── YOLO_tmd/                  # YOLO model weights
│   └── classifier_weights/        # Trained classification model weights
├── dataset/
│   ├── TMD/                       # Raw CBCT data (organized by patient)
│   └── TMD_yolo_pre/              # Preprocessed ROI images
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## ⚙️ Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/TMJ-disc-displacement-CBCT.git
cd TMJ-disc-displacement-CBCT

# Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
- PyTorch >= 2.0.0
- Ultralytics YOLO >= 8.0.0
- timm >= 0.9.0
- SimpleITK >= 2.2.0
- Streamlit >= 1.28.0
- OpenCV >= 4.8.0

## 🚀 Usage

### 1. Data Preparation
Organize your CBCT data in the following structure:
```
dataset/TMD/
├── class0/           # Normal cases
│   ├── train/        # Training patients
│   │   ├── patient001/
│   │   └── patient002/
│   └── test/         # Testing patients
└── class1/           # Abnormal cases
    ├── train/
    └── test/
```

### 2. TMJ Region Detection and Cropping
```bash
# Process CBCT images and extract TMJ regions
python data_processing/YOLO_press.py
```
This script:
- Reads DICOM files from patient directories
- Applies windowing (center=150, width=1500)
- Detects TMJ regions using YOLOv11
- Crops and saves ROI images to `dataset/TMD_yolo_pre/`

### 3. Model Training
```bash
# Train classification model (example with EfficientViT-m5)
python model_training/TMJDD_classification_train.py
```
Available models (uncomment in code):
- `mobilenetv3_small_100`
- `efficientnet_b1`
- `fastvit_t8`, `fastvit_t12`, `fastvit_s12`, `fastvit_sa36`
- `efficientvit_m0`, `efficientvit_m2`, `efficientvit_m5`

### 4. Model Evaluation
```bash
# Evaluate model performance at patient level
python model_training/TMJDD_classification_test.py
```
Outputs CSV file with metrics: Accuracy, AUC, AUPR, Sensitivity, Specificity, and confusion matrix.

### 5. Clinical Application
```bash
# Launch the Streamlit web application
streamlit run clinical_application/TMJDD_app.py
```
Or with custom input directory:
```bash
streamlit run clinical_application/TMJDD_app.py -- --input_dir /path/to/dicom/folder
```

## 🖥️ Clinical Interface Features

The Streamlit application provides:
- **DICOM folder upload** and batch processing
- **Automatic TMJ detection** using YOLOv11
- **Risk probability calculation** (percentage of abnormal slices)
- **Interactive slice navigation** with "Previous/Next" buttons
- **High-risk slice highlighting** for clinical review
- **Support for various CBCT scanner resolutions**

## 📊 Performance Metrics

| Model | AUC | Accuracy | Sensitivity | Specificity | AUPR |
|-------|-----|----------|-------------|-------------|------|
| FastViT-t8 | 0.733 | 0.669 | 0.848 | 0.660 | 0.716 |
| EfficientViT-m5 | 0.718 | 0.657 | 0.830 | 0.642 | 0.698 |
| MobileNetV3-small | 0.701 | 0.641 | 0.812 | 0.630 | 0.642 |
| EfficientNet-b1 | 0.710 | 0.641 | 0.821 | 0.635 | 0.624 |

*Metrics evaluated on patient-level aggregated predictions*

## 🔍 Model Interpretability

The system includes Grad-CAM visualization to explain model decisions. Heatmaps show that:
- **Normal cases**: Attention distributed over condylar contour
- **Abnormal cases**: Focus on anterosuperior condylar margin and anterior joint space
- **Clinical alignment**: Attention patterns correspond to known indirect signs of disc displacement

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{fu2025cbct,
  title={CBCT Assisted Diagnosis System for Temporomandibular Joint Disc Displacement Based on Deep Learning},
  author={Fu, Yijiao and others},
  journal={Journal of Medical Imaging},
  year={2025}
}
```

## 👥 Authors

- **Yijiao Fu** - Primary Developer
- Additional contributors welcome

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Supported by Clinical Research Program of Shanghai Ninth People's Hospital, Shanghai Jiao Tong University School of Medicine (Grant No. JYLJ202114)
- 2022 Excellent Research-Oriented Physician Training Program from Shanghai Ninth People's Hospital

## 📧 Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

**Disclaimer**: This tool is intended for research and screening purposes only. Clinical decisions should always be made by qualified healthcare professionals in conjunction with comprehensive patient evaluation.

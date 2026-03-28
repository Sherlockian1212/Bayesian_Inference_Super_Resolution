# Bayesian Inference Super Resolution

A deep learning–based **medical image super-resolution** framework integrating **Bayesian inference** for reconstruction and predictive uncertainty estimation.

This project focuses on improving reconstruction fidelity while providing **pixel-wise uncertainty quantification**, enabling more reliable clinical interpretation of super-resolved medical images.

---

## 📌 Overview

Medical image super-resolution (SR) reconstructs high-resolution (HR) images from low-resolution (LR) acquisitions. However, reconstruction quality is often affected by acquisition noise, scanner variability, and protocol differences.

This repository implements:

* Deep learning–based Super-Resolution
* Bayesian inference for probabilistic modeling
* Predictive uncertainty estimation
* Training, evaluation, and inference pipelines

---

## 🧠 Key Features

* ✅ Bayesian deep learning framework for Super-Resolution
* ✅ Pixel-wise uncertainty estimation
* ✅ Support for medical images (`.nii` / NIfTI format)
* ✅ Modular and reproducible training pipeline
* ✅ Designed for medical imaging research

---

## 📂 Project Structure

```
Bayesian_Inference_Super_Resolution/
│
├── data/           # Dataset loading and preprocessing
├── model/          # Network architectures
├── train/          # Training scripts
├── eval/           # Evaluation & testing scripts
├── utils/          # Helper functions and utilities
├── weights/        # Saved model checkpoints
├── logs/           # Training logs
└── README.md
```

---

## 🗂 Dataset Structure

The dataset must follow the hierarchical structure:

```
{DATA_FOLDER_PATH_FULL}/
    └── {Study}/
        └── {Scan Type}/
            └── {File Name}.nii
```

### Example

```
dataset/
├── Study_001/
│   ├── T1/
│   │   ├── image_01.nii
│   │   └── image_02.nii
│   ├── T2/
│   │   └── image_01.nii
│
├── Study_002/
│   └── FLAIR/
│       └── scan_01.nii
```

### Description

| Component     | Description                             |
| ------------- | --------------------------------------- |
| **Study**     | Patient ID or acquisition session       |
| **Scan Type** | Imaging modality (T1, T2, FLAIR, etc.)  |
| **File Name** | 3D medical image stored in NIfTI format |

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/Sherlockian1212/Bayesian_Inference_Super_Resolution.git
cd Bayesian_Inference_Super_Resolution
```

### 2. Create Environment

```bash
conda create -n bayesian_sr python=3.10
conda activate bayesian_sr
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run training:

```bash
python train/train.py
```

Outputs:

* Training logs → `logs/`
* Model checkpoints → `weights/`

---

## 🔎 Evaluation

```bash
python eval/evaluate.py
```

Evaluation includes:

* Reconstruction performance metrics
* Visual comparison
* Predictive uncertainty analysis

---

## 📊 Output

The framework generates:

* Super-resolved images
* Predictive mean reconstruction
* Pixel-wise uncertainty maps

---

## 🧩 Methodology

The proposed framework integrates:

1. Deep Super-Resolution Network
2. Bayesian Variational Inference
3. Predictive Uncertainty Quantification

Uncertainty estimation provides additional information about model confidence and reconstruction reliability.

---

## 📄 Citation

```bibtex
@article{bayesian_sr,
  title={Bayesian Inference for Medical Image Super-Resolution with Predictive Uncertainty},
  author={Author},
  year={2026}
}
```

---

## 👨‍💻 Author

**Sherlockian1212**

---

## 📜 License

MIT License

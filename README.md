# 🫁 Pneumonia Detection from Chest X-ray Images

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.55.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📌 Project Overview

This project uses **Convolutional Neural Networks (CNN)** to detect pneumonia from chest X-ray images. The model achieves **96.93% validation accuracy** and can predict with 99-100% confidence on test images.

### 🎯 Key Features
- ✅ **96.93% Validation Accuracy**
- ✅ Real-time predictions on chest X-rays
- ✅ Web interface using Streamlit
- ✅ Data augmentation for better generalization
- ✅ Easy-to-use command line interface

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **96.93%** |
| Training Accuracy | 96.60% |
| Best Validation | 97.03% (Epoch 5) |
| Model Parameters | 5,977,185 |
| Inference Time | <0.2 seconds |

### Test Results
| Image Type | Prediction | Confidence |
|-----------|------------|------------|
| Normal X-ray | ✅ Normal | 99.69% |
| Pneumonia X-ray | ✅ Pneumonia | 100.00% |

---

## 🏗️ Model Architecture

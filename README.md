# 🏦 Intelligent Cheque Data Extractor

### AI-Powered End-to-End Cheque Processing & Information Extraction System

An intelligent document processing system that automates cheque digitization using deep learning–based object detection, hybrid OCR, and banking-specific validation techniques. The system accurately extracts handwritten and printed cheque information, validates critical fields, and generates structured outputs suitable for financial automation workflows.

---

## 🚀 Overview

Manual cheque processing is time-consuming, error-prone, and requires significant human effort. Traditional OCR solutions struggle with handwritten text, varying cheque layouts, and noisy document images.

**Intelligent Cheque Data Extractor** addresses these challenges by combining modern Computer Vision and OCR models into a unified AI pipeline capable of:

- Detecting important cheque fields
- Extracting handwritten and printed information
- Selecting the most reliable OCR output
- Validating banking-specific fields
- Producing structured machine-readable data

The project demonstrates an end-to-end Intelligent Document Processing (IDP) workflow designed for real-world banking applications.

---

# ✨ Features

- 🏦 Automatic cheque field localization
- 🎯 Seven custom-trained YOLOv8 detection models
- 📝 Hybrid OCR pipeline
- ✍️ Handwritten & printed text recognition
- 🔍 Confidence-based OCR selection
- 📅 Date extraction and validation
- 💰 Numeric and written amount verification
- 🔢 MICR band extraction & validation
- 🖋 Signature region detection
- 📊 Structured JSON/CSV output
- ⚡ GPU accelerated inference

---

# AI Pipeline

```

Input Cheque Image
│
▼
Image Preprocessing
(OpenCV)
│
▼
YOLOv8 Detection
│
├── Payee Name
├── Date
├── Account Number
├── Amount (Words)
├── Amount (Digits)
├── Signature
└── MICR Band
│
▼
Crop Individual Regions
│
▼
Hybrid OCR Engine
├── PaddleOCR
├── EasyOCR
└── Fine-tuned TrOCR
│
▼
Confidence-Based Selection
│
▼
Field Validation
├── MICR Validation
├── Date Validation
├── Amount Verification
└── OCR Confidence Check
│
▼
Structured Output (JSON / CSV)

```

---

# Technologies Used

| Category | Technology |
|----------|------------|
| Language | Python |
| Object Detection | YOLOv8 |
| OCR | PaddleOCR |
| OCR | EasyOCR |
| Handwritten OCR | Fine-tuned TrOCR |
| Computer Vision | OpenCV |
| Deep Learning | PyTorch |
| Image Processing | NumPy |
| Data Handling | Pandas |

---

# Model Architecture

## 1️⃣ Field Localization

The first stage uses **seven independently trained YOLOv8 models** to accurately detect and crop important cheque regions.

Detected fields include:

- Payee Name
- Date
- Amount in Words
- Amount in Numbers
- Account Number
- MICR Band
- Signature Region

---

## 2️⃣ Image Enhancement

Each cropped field undergoes preprocessing to improve OCR performance.

Techniques include:

- Grayscale conversion
- Noise removal
- Adaptive thresholding
- Morphological operations
- Contrast enhancement
- ROI normalization

---

## 3️⃣ Hybrid OCR Engine

Instead of relying on a single OCR engine, the system evaluates outputs from multiple OCR models.

### PaddleOCR

Optimized for printed text with high recognition accuracy.

### EasyOCR

Provides additional robustness for noisy document regions.

### Fine-tuned TrOCR

Used for handwritten fields such as:

- Payee Name
- Amount in Words

---

## 4️⃣ Confidence-Based OCR Selection

Each OCR engine returns both text predictions and confidence scores.

The system automatically selects the most reliable prediction using confidence-based decision logic, improving overall extraction accuracy.

---

## 5️⃣ Banking Validation Layer

The extracted fields undergo multiple validation checks before producing the final result.

### MICR Validation

- Character format verification
- MICR consistency checks

### Amount Verification

Cross-validates:

- Numeric amount
- Amount written in words

to detect inconsistencies.

### Date Validation

Checks:

- Date format
- Invalid values
- Missing fields

---

# Workflow

```

Cheque Image

│

▼

YOLOv8 Detection

│

▼

Crop Individual Fields

│

▼

Image Enhancement

│

▼

Hybrid OCR Pipeline

│

▼

Confidence Selection

│

▼

Field Validation

│

▼

Structured Output

```

---

# Performance

| Metric | Result |
|---------|--------|
| Detection Models | 7 YOLOv8 Models |
| OCR Engines | 3 |
| Evaluation Dataset | 500 Cheque Images |
| Character Error Rate (CER) | < 3% |

The hybrid OCR pipeline significantly improves recognition accuracy by leveraging the strengths of multiple OCR engines while reducing errors through confidence-based prediction selection.

---

# Folder Structure

```

Intelligent_Cheque_Data_Extractor/
│
├── datasets/
│
├── models/
│ ├── YOLO/
│ ├── TrOCR/
│ └── Weights/
│
├── preprocessing/
│
├── detection/
│
├── ocr/
│
├── validation/
│
├── inference/
│
├── outputs/
│
├── notebooks/
│
└── README.md

```

---

# Installation

```bash
git clone https://github.com/kajaredhruv433/Intelligent_Cheque_Data_Extractor.git

cd Intelligent_Cheque_Data_Extractor

pip install -r requirements.txt
```

---

# Requirements

```
torch
ultralytics
opencv-python
paddleocr
easyocr
transformers
numpy
pandas
Pillow
matplotlib
```

---

# Running

Train YOLO models

```bash
python train_yolo.py
```

Run cheque detection

```bash
python detect.py
```

Extract cheque information

```bash
python extract.py
```

Run complete pipeline

```bash
python inference.py
```

---

# Applications

🏦 Banking Automation

📄 Intelligent Document Processing (IDP)

💳 Financial Record Digitization

📑 Cheque Clearing Systems

📊 Enterprise OCR Solutions

🤖 AI-powered Banking Workflows

---

# Future Improvements

- LayoutLM-based document understanding
- Transformer-based end-to-end document extraction
- Signature verification module
- Fraud detection
- Multi-bank cheque template support
- Real-time API deployment
- Cloud-native inference pipeline
- Web dashboard for cheque processing

---

# Key Learnings

- Custom Object Detection
- YOLOv8 Training
- Hybrid OCR Systems
- Handwritten Text Recognition
- TrOCR Fine-tuning
- OpenCV Image Processing
- Banking Document Intelligence
- Intelligent Document Processing (IDP)
- Confidence-based Decision Systems
- End-to-End AI Pipeline Development

---

# Acknowledgements

This project was developed to explore modern AI techniques for automating cheque processing by integrating object detection, handwritten text recognition, hybrid OCR, and banking-specific validation into a production-oriented Intelligent Document Processing pipeline.

---

## ⭐ If you found this project useful, consider giving it a star!

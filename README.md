# ChequeTrak — Intelligent Cheque Data Extractor

> An end-to-end AI pipeline for automated cheque field detection, OCR extraction, text normalization, and validation using YOLOv8 and multiple OCR models.

---

## Table of Contents

1. [Overview](#overview)
2. [Research Publication](#research-publication)
3. [How It Works](#how-it-works)
4. [System Flowchart](#system-flowchart)
5. [Detected Fields & Sample Output](#detected-fields--sample-output)
6. [Model Performance](#model-performance-yolov8)
7. [Tech Stack](#tech-stack)
8. [Repository Structure](#repository-structure)
9. [Installation](#installation)
10. [How to Run](#how-to-run)
11. [Output Format](#output-format)
12. [Documentation](#documentation)
13. [Notes](#notes)

---

## Overview

ChequeTrak is a modular, end-to-end cheque processing system designed to automate cheque data extraction and validation. The system:

- **Detects** key cheque fields using 8 specialized YOLOv8 models
- **Extracts** text using PaddleOCR, EasyOCR, and TrOCR on detected regions
- **Normalizes** raw OCR output (corrects dates, numeric amounts, OCR character errors)
- **Validates** cheques by cross-checking amount-in-words vs. amount-in-digits
- **Exports** results as a structured CSV for downstream processing

The pipeline achieves an **average mAP@0.5 of ~0.960** across all 8 detection models.

---

## Research Publication

ChequeTrak was developed as part of research presented in:

> **ChequeTrak: An Integrated Deep Learning Pipeline for Handwritten Text Extraction and Signature Region Extraction in Indian Bank Cheques**

Published at the **2026 2nd International Conference on Computing, Communication and Green Engineering (CCGE)**.

The research focuses on applying deep learning, object detection, OCR, and image-processing techniques to automated cheque information extraction.

---

## How It Works

The system operates through a structured 4-stage pipeline:

```text
Cheque Image
     │
     ▼
[Stage 1] Cheque Field Detection  — 8 specialized YOLO models locate each field
     │
     ▼
[Stage 2] OCR Extraction          — PaddleOCR / EasyOCR / TrOCR extracts text
     │
     ▼
[Stage 3] Text Normalization      — cleans dates, fixes OCR errors, converts word amounts
     │
     ▼
[Stage 4] Validation              — cross-checks amount-in-words == amount-in-digits
     │
     ▼
  CSV Output + Streamlit UI
```

### Pipeline Stages

- **Stage 1 — Cheque Field Detection**: 8 specialized YOLOv8 models detect and crop distinct bounding regions from the cheque image (payee name, amounts, date, account holder, account number, signature, and MICR band).
- **Stage 2 — OCR Extraction**: Text extraction is executed on the cropped regions using a multi-model OCR architecture.
- **Stage 3 — Text Normalization**: Raw OCR strings are cleaned, correcting character errors, parsing date formats into standard `dd-mm-yyyy` representation, and converting written amount words to numeric values.
- **Stage 4 — Validation**: The system cross-references the numeric value derived from words against the numerical amount digits to verify consistency.

### OCR Architecture & Dual-Path Design

The OCR processing layer incorporates multiple engines:
- **PaddleOCR**: Primary OCR engine for fast and accurate text recognition across detected fields.
- **EasyOCR**: Fallback recognition path used when detection confidence or alternative extraction is needed.
- **TrOCR**: Transformer-based OCR model utilized for handwriting-oriented recognition on cheque fields.

| Path | When Used | Technologies |
|------|-----------|-------------|
| **Primary** | YOLO successfully detects fields | YOLOv8 + PaddleOCR |
| **Fallback** | YOLO detection confidence is too low | EasyOCR + TrOCR |

Both paths merge into the same normalization, validation, and export pipeline.

> For full technical details, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## System Flowchart

The diagram below shows the complete processing flow from cheque input to final validation:

![System Flowchart](Assets/flowchart.jpeg)

Key decision points:
- **Detect?** - Routes to YOLO+PaddleOCR or fallback OCR
- **Amounts Match?** - Cross-validation for word and digit amounts
- **User MICR Match?** - Cheque routing and identity verification

---

## Detected Fields & Sample Output

The pipeline detects and extracts **8 fields** from every cheque:

| Field | Model | Description |
|-------|-------|-------------|
| Pay Name | `pay.pt` | Payee / recipient name |
| Amount in Words | `amt_word.pt` | Written amount (e.g. "Five Hundred Fifty") |
| Amount in Digits | `amt_digit.pt` | Numeric amount (e.g. "550.00") |
| Date | `date.pt` | Cheque date (normalized to `dd-mm-yyyy`) |
| Account Holder | `acc_holder.pt` | Name of account holder |
| Account Number | `acc_no.pt` | Bank account number |
| Signature | `sign.pt` | Signature presence (`Yes` / `No`) |
| MICR | `micr.pt` | Machine-readable routing code |

### Sample Detection

The image below shows the system detecting all 8 fields on a specimen cheque, annotated with bounding boxes and confidence scores:

![Detection Output](Assets/metrics.jpeg)

---

## Model Performance (YOLOv8)

All models were trained using **YOLOv8 Nano** — one model per cheque field. Below is a summary of the precision-recall evaluation:

| Model | Field | mAP@0.5 |
|-------|-------|---------|
| `amt_word.pt` | Amount in Words | **0.991** |
| `acc_no.pt` | Account Number | **0.990** |
| `pay.pt` | Payee Name | **0.984** |
| `date.pt` | Date | **0.984** |
| `amt_digit.pt` | Amount in Digits | **0.970** |
| `micr.pt` | MICR Band | **0.961** |
| `acc_holder.pt` | Account Holder | **0.919** |
| `sign.pt` | Signature | **0.876** |

**Average mAP@0.5: ~0.960**

### Precision-Recall Curves

````carousel
![Amount in Words - mAP@0.5: 0.991](Assets/AMT_WORD.png)
<!-- slide -->
![Account Number - mAP@0.5: 0.990](Assets/ACC_no.png)
<!-- slide -->
![Payee Name - mAP@0.5: 0.984](Assets/Pay.png)
<!-- slide -->
![Date Field - mAP@0.5: 0.984](Assets/date.png)
<!-- slide -->
![Amount in Digits - mAP@0.5: 0.970](Assets/AMT_digit.png)
<!-- slide -->
![MICR Band - mAP@0.5: 0.961](Assets/MICR.png)
<!-- slide -->
![Account Holder - mAP@0.5: 0.919](Assets/ACC_holder.png)
<!-- slide -->
![Signature - mAP@0.5: 0.876](Assets/sign.png)
````

> For detailed analysis of each model, see [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md).

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.x | Core language |
| YOLOv8 (Ultralytics) | Cheque field detection |
| PaddleOCR | Primary text recognition engine |
| EasyOCR | Fallback text recognition |
| TrOCR | Transformer-based handwriting recognition |
| OpenCV | Image loading, cropping, annotation |
| Streamlit | Web interface for batch processing |
| NumPy | Array operations |
| Pandas | Tabular result export |
| PyTorch | Deep learning backend |
| Python `difflib` | Fuzzy word matching for normalization |

---

## Repository Structure

```text
Intelligent_Cheque_Data_Extractor/
├── app.py                   # Streamlit web app for batch processing
├── extractor.py             # Core cheque extraction pipeline (ChequeExtractor class)
├── combine_extractor.py     # Standalone single-image extraction script
├── micr.py                  # MICR-specific extraction (MICRExtractor class)
├── text_normalizer.py       # Date & amount normalization helpers
├── test_normalizer.py       # Unit tests for normalizer functions
├── labeller.py              # YOLO labeling tool (single-region annotation)
├── labeller2.py             # YOLO labeling tool (multi-box annotation)
├── conbiner.py              # Experimental extractor variant
├── combiner2.py             # Experimental extractor variant
├── sign_majorpro.ipynb      # Notebook for prototyping & experimentation
│
├── Models/                  # Trained YOLO model weights
│   ├── acc_holder.pt        # Account holder detector
│   ├── acc_no.pt            # Account number detector
│   ├── amt_digit.pt         # Amount in digits detector
│   ├── amt_word.pt          # Amount in words detector
│   ├── date.pt              # Date field detector
│   ├── micr.pt              # MICR band detector
│   ├── pay.pt               # Payee name detector
│   └── sign.pt              # Signature detector
│
├── Assets/                  # Images and diagrams
│   ├── flowchart.jpeg       # System flowchart
│   ├── metrics.jpeg         # Sample detection output on specimen cheque
│   ├── ACC_holder.png       # PR curve - Account Holder model
│   ├── ACC_no.png           # PR curve - Account Number model
│   ├── AMT_WORD.png         # PR curve - Amount in Words model
│   ├── AMT_digit.png        # PR curve - Amount in Digits model
│   ├── MICR.png             # PR curve - MICR model
│   ├── Pay.png              # PR curve - Payee Name model
│   ├── date.png             # PR curve - Date model
│   └── sign.png             # PR curve - Signature model
│
├── README.md                # This file
├── ARCHITECTURE.md          # Deep-dive: system design & pipeline details
└── MODEL_PERFORMANCE.md     # YOLO model evaluation & precision-recall analysis
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install ultralytics paddleocr paddlepaddle opencv-python streamlit numpy pandas
```

> On GPU machines, install `paddlepaddle-gpu` instead of `paddlepaddle` for faster OCR.

---

## How to Run

### Option 1 — Batch Processing (Streamlit Web App)

```bash
streamlit run app.py
```

Then in the browser:
1. Enter the folder path containing your cheque images
2. Click **Start Processing**
3. Watch results populate in real-time
4. A CSV file is automatically saved to the input folder

### Option 2 — Single Image (Command Line)

Edit the `SOURCE` path inside `combine_extractor.py`, then run:

```bash
python combine_extractor.py
```

This will print extracted data to the console and save an annotated image (`output_with_boxes.jpg`).

### Option 3 — MICR Only

```bash
python micr.py
```

Extracts only the MICR band from a cheque. Edit `sample_image` inside the `__main__` block.

---

## Output Format

For each processed image, the pipeline produces a row with these columns:

| Column | Example Value | Description |
|--------|--------------|-------------|
| `Filename` | `cheque_001.jpg` | Source file name |
| `Match` | `Yes` | Amount cross-check result |
| `Time (s)` | `2.34` | Processing time per image |
| `Pay Name` | `John Doe` | Payee name |
| `Amount in Word` | `550.0` | Normalized word amount |
| `Amount in Digit` | `550.0` | Normalized digit amount |
| `Date` | `20-09-2025` | Normalized date |
| `Account Holder Name` | `Jane Smith` | Account holder |
| `Account Number` | `X110X00X000X1XXX` | Account number |
| `Sign` | `Yes` | Signature present? |

Results are saved as `<folder_name>.csv` inside the input folder.

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | Project overview, quick-start guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Full system design, pipeline stages, component breakdown, technology rationale |
| [MODEL_PERFORMANCE.md](MODEL_PERFORMANCE.md) | YOLO model evaluation, precision-recall curves, metric interpretation |

---

## Notes

- All model weights are stored locally in `Models/` — no internet connection needed at inference time.
- The normalization layer is designed to be robust against common OCR errors on handwritten cheques.
- `conbiner.py` and `combiner2.py` are earlier experimental versions kept for reference.
- Confidence threshold defaults to `0.25`; increase it to reduce false positives on ambiguous fields.

---

*Built with YOLOv8, OCR, and deep learning for automated cheque processing and validation.*

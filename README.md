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
7. [Intelligent Text Normalization](#intelligent-text-normalization)
8. [Validation & Fraud Prevention](#validation--fraud-prevention)
9. [Tech Stack](#tech-stack)
10. [Repository Structure](#repository-structure)
11. [Installation & Setup](#installation--setup)
12. [How to Run](#how-to-run)
13. [Output Format & Data Schema](#output-format--data-schema)
14. [Performance & Benchmark Characteristics](#performance--benchmark-characteristics)
15. [Documentation Suite](#documentation-suite)
16. [Authors & Acknowledgments](#authors--acknowledgments)
17. [Notes & Future Enhancements](#notes--future-enhancements)

---

## Overview

ChequeTrak is an enterprise-grade cheque processing pipeline designed to automate field localization, optical character recognition, textual normalization, and fraud cross-validation from scanned cheque images.

Financial clearing institutions handle massive volumes of paper cheques every business day. Manual sorting, visual verification, and manual data entry introduce severe operational bottlenecks, high labour costs, human transcription errors, and prolonged settlement cycles. ChequeTrak eliminates these inefficiencies through a modular, four-stage neural pipeline:

- Detects key cheque regions using 8 specialized YOLOv8 nano models
- Extracts text using a multi-model OCR architecture combining PaddleOCR, EasyOCR, and TrOCR
- Normalizes raw OCR output by correcting visual character confusions, standardizing complex date formats, and converting written verbal amounts into numerical values
- Validates cheques by cross-referencing written amounts against numerical amount digits
- Exports structured records into CSV files for direct integration into banking clearance systems

The pipeline delivers an average mAP@0.5 of ~0.960 across all 8 detection models, completing end-to-end inference and validation in under 2.5 seconds per cheque.

---

## Research Publication

ChequeTrak was developed as part of research presented in:

> ChequeTrak: An Integrated Deep Learning Pipeline for Handwritten Text Extraction and Signature Region Extraction in Indian Bank Cheques

Published at the 2026 2nd International Conference on Computing, Communication and Green Engineering (CCGE).

The research focuses on combining modern single-shot object detection architectures with multi-model optical character recognition and algorithmic normalization rules to process complex handwritten banking instruments.

---

## How It Works

The system operates through a structured four-stage execution pipeline:

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

### Pipeline Stages

1. Stage 1 — Cheque Field Detection:
   Eight specialized YOLOv8 nano models scan the incoming cheque image simultaneously. Each detector outputs bounding box coordinates and confidence ratings for its designated field: payee name, amount in words, amount in digits, date, account holder name, account number, signature zone, and the MICR band.

2. Stage 2 — OCR Extraction:
   The system crops each localized field region from the source image. Each crop is passed to the OCR processing engine. PaddleOCR with angle classification handles print and handwriting orientation, while TrOCR provides handwriting attention on difficult text strokes.

3. Stage 3 — Text Normalization:
   Raw OCR output frequently contains character misclassifications, inconsistent delimiters, or colloquial date formats. The normalization engine cleans the strings, maps optical digit substitutions, parses 8-digit and 7-digit date strings into standard dd-mm-yyyy formats, and translates verbal English numerical strings into floating-point numbers.

4. Stage 4 — Validation and Parity Check:
   The validation module conducts mathematical cross-verification. If the numeric quantity extracted from Amount in Words matches the numerical value parsed from Amount in Digit, the cheque receives a positive verification flag. Discrepancies immediately trigger a fraud or mismatch review alert.

### OCR Architecture & Dual-Path Design

To ensure high availability and robust recognition across varied scan qualities, ChequeTrak implements a dual-path OCR strategy:

| Path | Operating Condition | Core Technologies | Primary Function |
| :--- | :--- | :--- | :--- |
| Primary Path | YOLO successfully localizes fields with high confidence | YOLOv8 + PaddleOCR | High-speed, targeted text reading on clean cropped segments |
| Fallback Path | YOLO confidence falls below threshold or boundaries degrade | EasyOCR + TrOCR | Full-context OCR and sequence-to-sequence handwriting extraction |

Both paths feed into identical normalization and validation modules, ensuring consistent data contracts across all execution scenarios.

---

## System Flowchart

The diagram below illustrates the end-to-end lifecycle of a cheque transaction, encompassing ingestion, bank routing, detection arbitration, dual-path OCR, normalization, cryptographic verification, and validation output:

![System Flowchart](Assets/flowchart.jpeg)

### Decision Workflow Summary

- Step 1: Cheque arrives from physical scanning hardware or mobile client with user-declared MICR metadata.
- Step 2: The client or bank branch transmits the image payload through a secure gateway to the processing hub.
- Step 3: The Hub runs the YOLOv8 detection suite. If field boundaries meet confidence thresholds, execution proceeds via the Primary Path.
- Step 4: If scan degradation or skew prevents localization, execution routes to the Fallback Path (EasyOCR and TrOCR).
- Step 5: Normalized text strings are generated for all fields.
- Step 6: Parity check validates whether Amount in Words equals Amount in Digits. If values diverge, the cheque is flagged as Invalid.
- Step 7: Normalized MICR text is compared against user-declared bank records to ensure cheque routing authenticity.
- Step 8: Validated, structured records are encrypted and dispatched to banking core databases.

---

## Detected Fields & Sample Output

ChequeTrak isolates and transcribes 8 essential fields from each cheque:

| Field Name | Detection Model | Field Purpose | Output Transformation |
| :--- | :--- | :--- | :--- |
| Pay Name | pay.pt | Identifies beneficiary or payee entity | Raw text string, whitespace trimmed |
| Amount in Words | amt_word.pt | Legal written amount in English words | Normalized floating-point numeric value |
| Amount in Digits | amt_digit.pt | Courtesy numeric amount inside box | Normalized floating-point numeric value |
| Date | date.pt | Cheque issuance date | Standardized dd-mm-yyyy format |
| Account Holder | acc_holder.pt | Printed or written drawer name | Cleaned alphabetical string |
| Account Number | acc_no.pt | Bank account number string | Digit sequence, symbol sanitized |
| Signature | sign.pt | Presence of drawer authorization signature | Binary indicator (Yes or No) |
| MICR Band | micr.pt | Magnetic ink routing and cheque serial | Formatted numeric routing code |

### Sample Detection

Below is an annotated specimen cheque processed by the detection suite. Each localized region is bounded by coordinates corresponding to the highest-confidence bounding box:

![Detection Output](Assets/metrics.jpeg)

---

## Model Performance (YOLOv8)

Each model was trained on real-world Indian bank cheques utilizing the YOLOv8 Nano architecture. Individual single-class detectors prevent cross-class interference and permit fine-tuned confidence thresholds tailored to the geometric characteristics of each field.

### Evaluation Metrics Summary

| Model File | Field Target | mAP@0.5 | Precision Characteristics | Performance Class |
| :--- | :--- | :--- | :--- | :--- |
| amt_word.pt | Amount in Words | 0.991 | Consistent precision across broad recall range | Exceptional |
| acc_no.pt | Account Number | 0.990 | Near-perfect localization on standard account boxes | Exceptional |
| pay.pt | Payee Name | 0.984 | Highly robust across varied font styles and cursive scripts | Exceptional |
| date.pt | Date Field | 0.984 | Sharp bounding on partitioned and unpartitioned date boxes | Exceptional |
| amt_digit.pt | Amount in Digits | 0.970 | High precision on compact rectangular boxes | Excellent |
| micr.pt | MICR Band | 0.961 | Strong bottom-band localization with geometric fallback | Excellent |
| acc_holder.pt | Account Holder | 0.919 | Handles positional variations above or below signature | Very Good |
| sign.pt | Signature Region | 0.876 | High variance detection; serves presence verification | Good |

Average mAP@0.5 Across All 8 Models: ~0.960

### Precision-Recall Curves

Below are the precision-recall evaluation curves for all eight trained models:

#### 1. Amount in Words (amt_word.pt - mAP@0.5: 0.991)
![Amount in Words PR Curve](Assets/AMT_WORD.png)

#### 2. Account Number (acc_no.pt - mAP@0.5: 0.990)
![Account Number PR Curve](Assets/ACC_no.png)

#### 3. Payee Name (pay.pt - mAP@0.5: 0.984)
![Payee Name PR Curve](Assets/Pay.png)

#### 4. Date Field (date.pt - mAP@0.5: 0.984)
![Date Field PR Curve](Assets/date.png)

#### 5. Amount in Digits (amt_digit.pt - mAP@0.5: 0.970)
![Amount in Digits PR Curve](Assets/AMT_digit.png)

#### 6. MICR Band (micr.pt - mAP@0.5: 0.961)
![MICR Band PR Curve](Assets/MICR.png)

#### 7. Account Holder (acc_holder.pt - mAP@0.5: 0.919)
![Account Holder PR Curve](Assets/ACC_holder.png)

#### 8. Signature Region (sign.pt - mAP@0.5: 0.876)
![Signature Region PR Curve](Assets/sign.png)

For an in-depth breakdown of confusion matrices, IoU thresholds, and training methodologies, consult MODEL_PERFORMANCE.md.

---

## Intelligent Text Normalization

Real-world cheques suffer from handwriting artifacts, background patterns, stamp overlap, and OCR character confusions. ChequeTrak includes an algorithmic normalization engine in text_normalizer.py to ensure reliable downstream consumption.

### 1. Optical Digit Confusion Repair

Optical recognition models frequently mistake visually similar letters and numbers when reading digit boxes. The system applies a deterministic mapping table:

- Lowercase and uppercase s/S maps to digit 5
- Lowercase and uppercase b/B maps to digit 8
- Lowercase and uppercase o/O maps to digit 0
- Letters l, I, and i map to digit 1
- Lowercase and uppercase z/Z maps to digit 2
- Lowercase g and q map to digit 9
- Lowercase and uppercase y/Y and letter A map to digit 4

### 2. Date String Standardization

The date normalizer accepts messy date representations and converts them into ISO-aligned dd-mm-yyyy strings:

- Delimiter alignment: Periods, slashes, and whitespace are standardized into hyphens.
- Month vocabulary conversion: Names like Jan, January, Sept, September are mapped to 01 through 12.
- 8-digit numeric sequences: Strings like 20092025 are parsed directly into 20-09-2025.
- 7-digit numeric sequences: Ambiguous outputs such as 2092025 are dynamically resolved into 20-09-2025 by validating valid calendar month ranges.
- Two-digit year expansion: Trailing two-digit years (such as 25) are expanded to four-digit format (2025).

### 3. Written Amount to Numerical Computation

Cheque verbal amounts (such as Two Lakh Fifty Thousand Only) are evaluated through a robust two-pass accumulator:

- Lexical cleaning: Words such as only, rupees, rs, and currency marks are stripped.
- Word compound splitting: Run-together strings like Fouvthousend are split into constituent parts.
- Fuzzy spelling correction: Python SequenceMatcher performs fuzzy matching with a 0.6 cutoff against standard numeral vocabularies (handling common mistakes like thousend, fve, lack).
- Two-pass hierarchical accumulation: Hundreds act as local multipliers, while major units (thousand, lakh, crore, million, billion) trigger global accumulator resets.

Example: Fouvthousend two hundred yields 4200.0

---

## Validation & Fraud Prevention

Cheque fraud remains a primary concern in financial transactions. ChequeTrak implements automated cross-checks to eliminate manual verification vulnerability:

### Amount Parity Cross-Check

The system compares the numeric float derived from the verbal amount against the numeric float parsed from the numerical amount box:

- Condition: Amount_Word_Value equals Amount_Digit_Value AND Value greater than 0
- Positive Outcome: Match column set to Yes. Cheque marked as legally congruent.
- Negative Outcome: Match column set to No. Discrepancies indicate possible tampering, alteration, or severe OCR misreading, warranting immediate supervisory review.

### Signature Presence Verification

The signature detector confirms that drawer authorization is physically present on the instrument. If bounding box confidence meets the operational threshold, Sign is recorded as Yes; otherwise No.

### MICR Integrity Routing Check

The standalone MICRExtractor isolates the magnetic ink character recognition line containing the cheque number, bank sorting code, and account routing key. When compared against transaction metadata, it prevents counterfeit routing attacks.

---

## Tech Stack

The architecture utilizes proven computer vision, deep learning, and engineering libraries:

| Layer | Component | Version / Library | Architectural Role |
| :--- | :--- | :--- | :--- |
| Deep Learning | YOLOv8 Nano | Ultralytics YOLOv8 | Single-class field localization models |
| Deep Learning Backend | PyTorch | Torch 2.x | Neural network inference engine |
| OCR Engine (Primary) | PaddleOCR | PaddlePaddle | Angular-classified multilingual OCR |
| OCR Engine (Fallback) | EasyOCR & TrOCR | EasyOCR, Transformers | Fallback recognition and handwriting ViT |
| Computer Vision | OpenCV | opencv-python | Image decoding, color space transformations, cropping |
| UI Framework | Streamlit | Streamlit 1.x | Web-based batch processing dashboard |
| Data Processing | Pandas & NumPy | pandas, numpy | Structured tabular ingestion and CSV export |
| Fuzzy Matching | Python difflib | Standard Library | Levenshtein-based spelling error correction |

---

## Repository Structure

- app.py : Streamlit web application for batch processing
- extractor.py : Core ChequeExtractor engine coordinating YOLO and OCR
- combine_extractor.py : Standalone single-image extraction and debugging script
- micr.py : Standalone MICRExtractor class with bottom-crop fallback
- text_normalizer.py : Algorithmic date, digit, and verbal amount normalizer
- test_normalizer.py : Unit test suite verifying normalization logic
- labeller.py : OpenCV interactive labeling tool for single-box datasets
- labeller2.py : Multi-box annotation script for dataset expansion
- conbiner.py : Earlier experimental extraction pipeline variant
- combiner2.py : Second experimental extraction pipeline variant
- sign_majorpro.ipynb : Prototyping notebook for signature experiments
- Models/ : Directory containing trained YOLOv8 Nano model weights (.pt files)
  - acc_holder.pt : Account holder name detector
  - acc_no.pt : Account number detector
  - amt_digit.pt : Amount in digits detector
  - amt_word.pt : Amount in words detector
  - date.pt : Date field detector
  - micr.pt : MICR band detector
  - pay.pt : Payee name detector
  - sign.pt : Signature region detector
- Assets/ : Visual documentation assets and PR evaluation curves
  - flowchart.jpeg : End-to-end system and banking integration flowchart
  - metrics.jpeg : Specimen detection output on sample cheque
  - ACC_holder.png : Precision-Recall curve: Account Holder model
  - ACC_no.png : Precision-Recall curve: Account Number model
  - AMT_WORD.png : Precision-Recall curve: Amount in Words model
  - AMT_digit.png : Precision-Recall curve: Amount in Digits model
  - MICR.png : Precision-Recall curve: MICR model
  - Pay.png : Precision-Recall curve: Payee Name model
  - date.png : Precision-Recall curve: Date model
  - sign.png : Precision-Recall curve: Signature model
- README.md : Primary project overview and operations guide
- ARCHITECTURE.md : Comprehensive technical architecture and design document
- MODEL_PERFORMANCE.md : Detailed evaluation metrics and PR analysis

---

## Installation & Setup

### Prerequisites

- Python 3.8, 3.9, 3.10, or 3.11
- Pip package manager
- Recommended: 4GB or more RAM; optional CUDA-compatible GPU

### Step-by-Step Installation

1. Clone the repository into your local environment:
   git clone https://github.com/kajaredhruv433/Intelligent_Cheque_Data_Extractor.git
   cd Intelligent_Cheque_Data_Extractor

2. Create and activate a virtual environment:
   python -m venv venv
   On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   On Linux / macOS:
   source venv/bin/activate

3. Install required production dependencies:
   pip install ultralytics paddleocr paddlepaddle opencv-python streamlit numpy pandas

   Note for GPU systems: To enable GPU hardware acceleration for PaddleOCR, install paddlepaddle-gpu instead of paddlepaddle:
   pip install paddlepaddle-gpu

4. Verify normalization unit tests:
   python test_normalizer.py
   All tests should pass without assertion errors.

---

## How to Run

### Option 1: Batch Processing Dashboard (Streamlit App)

The primary interface for batch operations is the Streamlit web application.

1. Launch the server from your terminal:
   streamlit run app.py

2. Open the displayed local browser URL (typically http://localhost:8501).
3. In the input box, supply the directory path containing your cheque image files (supports png, jpg, jpeg, tiff, bmp).
4. Click Start Processing.
5. The dashboard streams progress in real time, displaying:
   - Dynamic tabular view of extracted fields per image
   - Execution duration and moving average speed per cheque
   - Parity verification match status (Yes or No)
6. Upon completion, a structured CSV file named after the folder is automatically written into the target directory.

### Option 2: Single-Image Extraction (Command Line)

For headless execution or pipeline integration:

1. Open combine_extractor.py and configure the SOURCE file path:
   SOURCE = "path/to/your/sample_cheque.jpg"

2. Run the script:
   python combine_extractor.py

3. The script prints extracted fields and normalization results to the console, and writes an annotated visualization image named output_with_boxes.jpg.

### Option 3: Dedicated MICR Extraction

To test or isolate the MICR band reader independently:

1. Open micr.py and specify the test image path in the execution block.
2. Run the extractor:
   python micr.py

   If YOLO fails to isolate the band due to print fading, the automatic fallback crops the bottom 15 percent of the image and performs OCR.

---

## Output Format & Data Schema

When processing batches through app.py or extractor.py, results are formatted into standardized tabular rows. Below is the data schema definition:

| Column Header | Data Type | Sample Output Value | Field Explanation |
| :--- | :--- | :--- | :--- |
| Filename | String | cheque_1042.jpg | Originating file name |
| Match | String | Yes | Amount cross-validation parity verdict (Yes or No) |
| Time (s) | Float | 2.14 | Total processing duration in seconds |
| Pay Name | String | State Electricity Board | Payee recipient or corporate entity |
| Amount in Word | Float | 150000.0 | Normalized numerical value from written words |
| Amount in Digit | Float | 150000.0 | Normalized numerical value from numerical box |
| Date | String | 15-08-2025 | Standardized dd-mm-yyyy date |
| Account Holder Name | String | Rahul S Sharma | Drawer account holder name |
| Account Number | String | 3098101002345 | Bank account identifier |
| Sign | String | Yes | Drawer signature verification status (Yes or No) |

---

## Performance & Benchmark Characteristics

ChequeTrak is optimized to operate on commodity CPU hardware as well as GPU accelerators:

- Localization Speed: 8 single-class YOLOv8n models execute in ~120ms total on CPU per image.
- OCR Inference: PaddleOCR processing across all 7 text crops averages ~1.8 seconds on multi-core CPU.
- Normalization & Validation: Sub-millisecond algorithmic execution (<5ms per image).
- Total Throughput: Approximately 2.0 to 2.5 seconds per cheque on an Intel Core i7 CPU; under 0.6 seconds on an NVIDIA T4 GPU.
- Model Memory Footprint: Each YOLOv8n model requires ~6MB of storage, allowing all 8 models to reside comfortably in low-memory environments.

---

## Documentation Suite

The repository contains documentation covering all aspects of system design:

| Document Link | Primary Content |
| :--- | :--- |
| README.md | Executive summary, operational instructions, pipeline flow, and quickstart guide |
| ARCHITECTURE.md | Deep technical dive into component classes, algorithms, error handling, and design decisions |
| MODEL_PERFORMANCE.md | Comprehensive YOLOv8 evaluation, Precision-Recall curves, and detection analysis |

---

## Authors & Acknowledgments

### Lead Developer
Dhruv Kajare

### Team Members
Juhi Nale
Neha Tarmale

### Acknowledgments
Gratitude is extended to the open-source artificial intelligence communities behind Ultralytics YOLOv8, PaddlePaddle / PaddleOCR, and Streamlit for providing the foundations that made this research possible.

---

## Notes & Future Enhancements

- Local Model Weights: All 8 models reside locally in the Models directory, enabling fully offline, air-gapped banking deployments.
- Experimental Variants: Earlier prototyping files (conbiner.py and combiner2.py) are retained for architectural lineage.
- Future Roadmap:
  - Integration of automated signature verification using Siamese neural networks to cross-check signatures against bank specimen databases.
  - Integration of synthetic training data generation to cover obscure regional cheque formats across multiple national jurisdictions.
  - Containerization using Docker for cloud-native orchestration across high-volume clearing hubs.

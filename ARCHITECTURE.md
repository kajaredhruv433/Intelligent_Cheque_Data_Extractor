# System Architecture — Intelligent Cheque Data Extractor

This document provides a comprehensive technical specification of the ChequeTrak system, detailing its layered architecture, component responsibilities, mathematical algorithms, data contracts, and operational fault tolerance.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [System Flowchart](#2-system-flowchart)
3. [Pipeline Stages & Technical Lifecycle](#3-pipeline-stages--technical-lifecycle)
4. [Component Breakdown](#4-component-breakdown)
5. [YOLO Model Design & Detection Strategy](#5-yolo-model-design--detection-strategy)
6. [OCR Layer & Dual-Path Fallback](#6-ocr-layer--dual-path-fallback)
7. [Text Normalization Engine](#7-text-normalization-engine)
8. [Validation & Fraud Detection Logic](#8-validation--fraud-detection-logic)
9. [Data Flow Diagram](#9-data-flow-diagram)
10. [Failure Modes & Resilience Matrix](#10-failure-modes--resilience-matrix)
11. [Technology Choices & Rationale](#11-technology-choices--rationale)
12. [Authors & Acknowledgments](#12-authors--acknowledgments)

---

## 1. High-Level Architecture

ChequeTrak implements a decoupled, sequential artificial intelligence pipeline designed to ingest unstructured cheque images and transform them into structured, mathematically validated records. 

The architecture is partitioned into four major operational layers:

+---------------------------------------------------------------+
|                     CHEQUE IMAGE INPUT                        |
+---------------------------------------------------------------+
                           |
                           v
+---------------------------------------------------------------+
|           LAYER 1: YOLO FIELD DETECTION                       |
|   (8 independent YOLOv8 nano models — one per cheque field)   |
+---------------------------------------------------------------+
                           | Bounding boxes + confidence scores
                           v
+---------------------------------------------------------------+
|           LAYER 2: PaddleOCR TEXT EXTRACTION                  |
|     (Run per cropped region from YOLO detection output)       |
+---------------------------------------------------------------+
                           | Raw OCR strings
                           v
+---------------------------------------------------------------+
|           LAYER 3: TEXT NORMALIZATION                         |
|     (Date parsing, numeric conversion, OCR error correction)  |
+---------------------------------------------------------------+
                           | Normalized structured data
                           v
+---------------------------------------------------------------+
|           LAYER 4: VALIDATION (Amount Cross-Check)            |
|     (Compare amount in words vs. digits; flag mismatches)     |
+---------------------------------------------------------------+
                           |
                           v
                  +-----------------+
                  |  STRUCTURED CSV  |
                  |  OUTPUT / UI     |
                  +-----------------+

### Architectural Principles

- Separation of Concerns: Geometric field localization is decoupled from textual transcription. Object detection isolates regions of interest without concerning itself with character semantics; OCR extracts text without having to parse full-page layouts.
- Single-Class Model Isolation: Rather than forcing a single multi-class detector to balance disparate aspect ratios, 8 dedicated single-class YOLO models operate concurrently.
- Algorithmic Normalization: Raw OCR outputs undergo rule-based and fuzzy-string normalization before validation, decoupling visual noise from business logic.
- Defense-in-Depth Validation: Discrepancies between verbal and numerical amounts trigger automated fraud alerts prior to database commits.

---

## 2. System Flowchart

The flowchart below documents the complete operational topology, covering scanner ingestion, secure hub transmission, dual-path OCR execution, lexical normalization, and banking validation:

![System Flowchart](Assets/flowchart.jpeg)

### End-to-End Workflow Breakdown

| Step | Component | Process Description | Output State |
| :--- | :--- | :--- | :--- |
| 1 | Ingestion Client | Cheque scanned via flatbed or mobile app along with declared MICR | Unprocessed image array |
| 2 | Bank Gateway | Securely routes image payload to ChequeTrak inference hub | Verified ingestion queue |
| 3 | Detection Arbiter | Hub tests whether YOLOv8 successfully detects cheque fields | Route decision (Primary vs Fallback) |
| 4A | Primary Path | Runs PaddleOCR with angle classification on localized crops | Targeted OCR tokens |
| 4B | Fallback Path | Initiates EasyOCR full-scan and TrOCR handwriting transformer | Fallback OCR tokens |
| 5 | Normalizer | Cleans characters, maps OCR substitutions, parses dates, computes amounts | Normalized key-value map |
| 6 | Amount Parity Gate | Tests whether verbal amount matches numerical amount | Congruence flag (Yes or No) |
| 7 | MICR Match Gate | Cross-checks extracted MICR against bank routing database | Identity confirmation |
| 8 | Export & Dispatch | Encrypts structured records and writes CSV output for core banking | Clearance transaction complete |

---

## 3. Pipeline Stages & Technical Lifecycle

### Stage 1: Ingestion & Geometric Standardization

Cheque images originate in various resolutions, color depths, and orientations. The ingestion block in extractor.py normalizes inputs into a uniform 3-channel BGR representation:

Input standardization logic:
- Handles file paths (string) or pre-loaded NumPy arrays.
- Standardizes single-channel or grayscale inputs to 3-channel BGR using cv2.cvtColor(img, cv2.COLOR_GRAY2BGR).
- Verifies image integrity and raises descriptive exceptions on read failure.

Supported formats include PNG, JPG, JPEG, TIFF, BMP, and TIF.

### Stage 2: YOLO Field Detection

The system iterates through all 8 YOLO models stored in the Models directory. Each model performs forward inference on the full image:

Inference flow:
- Model forward pass executed with confidence threshold (default 0.25).
- Candidate bounding boxes evaluated; highest-confidence box selected.
- Bounding coordinates extracted: x1, y1, x2, y2.

Selecting the highest-confidence bounding box eliminates spurious edge detections and stabilizes the cropped region.

### Stage 3: Region Cropping & OCR Extraction

When a bounding box is localized, the target region is cropped using NumPy array slicing:

Extraction flow:
- Crop sub-image: crop = img[y1:y2, x1:x2].
- Forward crop to PaddleOCR with angle classification enabled (use_angle_cls=True).
- Concatenate recognized word tokens into a trimmed text string.

For the signature field (sign.pt), text recognition is omitted; localization alone serves as positive presence verification (Yes).

### Stage 4: Lexical & Numerical Normalization

The extracted text is dispatched to text_normalizer.py, where field-specific normalization routines transform raw characters into validated structures:
- Date string: Converted to dd-mm-yyyy.
- Amount in digits: Alphanumeric characters corrected and cast to float.
- Amount in words: Run-together strings split, spellings fuzzy-corrected, and accumulated to float.

### Stage 5: Validation & Export

The normalized fields are cross-referenced to verify amount congruency. The compiled record is written to a Pandas DataFrame and saved as a persistent CSV file.

---

## 4. Component Breakdown

### extractor.py — Core Orchestrator

The ChequeExtractor class coordinates model loading, image loading, inference orchestration, and result assembly:

| Method / Attribute | Responsibility |
| :--- | :--- |
| __init__(models_dir, confidence) | Initializes model repository path and sets detection threshold (default 0.25) |
| _load_models() | Scans Models directory, loads all .pt files, and instantiates PaddleOCR |
| process_image(image_source) | Coordinates full 5-stage lifecycle; returns structured dictionary and annotated image |
| FIELD_MAP | Maps model weight prefixes to standardized human-readable field labels |

### app.py — Streamlit Batch Processing Dashboard

A reactive web interface tailored for bank operations:
- Utilizes @st.cache_resource on load_extractor() to prevent model re-instantiation across browser interactions.
- Traverses user-specified image folders and processes valid files iteratively.
- Employs Streamlit placeholders (st.empty()) to render real-time streaming data tables and latency indicators.
- Automatically generates and saves a <folder_name>.csv file in the source directory upon completion.

### micr.py — MICRExtractor Specialist

The MICRExtractor provides dedicated extraction for the Magnetic Ink Character Recognition line:
- First attempts targeted localization via micr.pt.
- If YOLO localization fails due to ink wear or background interference, it invokes a geometric fallback by cropping the bottom 15 percent of the cheque image:
  y1 = int(height x 0.85)
  crop = img[y1:height, 0:width]
- Executes PaddleOCR on the crop to isolate routing and account tokens.

### text_normalizer.py — Algorithmic Normalization Engine

Houses pure, stateless functions for string cleaning, OCR error recovery, date parsing, and verbal amount conversion:
- clean_ocr_text(): Trims whitespace and strips unprintable characters.
- split_joined_words(): Inserts spaces around words like thousand, hundred, lakh, crore.
- fuzzy_match_word(): Uses difflib SequenceMatcher to match misspelled words against numbers dictionary.
- normalize_amount_numeric(): Applies OCR digit confusion map and returns float.
- normalize_date(): Multi-format date parsing resolving 8-digit, 7-digit, and month-name dates.
- normalize_amount_in_words(): Two-pass verbal amount accumulator supporting international and Indian numbering systems.

### labeller.py & labeller2.py — Dataset Annotation Tools

Interactive OpenCV utilities utilized during model training:
- labeller.py: Facilitates single-box region drawing and export to normalized YOLO format [class x_center y_center width height].
- labeller2.py: Multi-box annotation script allowing annotators to label multiple distinct fields across a single specimen.

---

## 5. YOLO Model Design & Detection Strategy

### Architectural Choice: YOLOv8 Nano

The system deploys the YOLOv8 Nano (YOLOv8n) architecture across all 8 detectors:
- Minimal Parameter Count: Approximately 3.2 million parameters per model (~6MB file size).
- High Inference Throughput: Executes forward passes in under 15ms per model on standard CPU hardware.
- Decoupled Head Design: Separates classification and bounding-box regression, accelerating single-class convergence.

### Single-Class Model Specialization

Rather than training a single multi-class model to locate all 8 fields simultaneously, ChequeTrak trains 8 distinct single-class models:

| Comparison Dimension | Single Multi-Class Model | Eight Single-Class Models (ChequeTrak) |
| :--- | :--- | :--- |
| Class Imbalance | Fields with subtle features get dominated by large fields | Every field receives dedicated gradient optimization |
| Aspect Ratio Variance | Must reconcile 1:1 boxes (date) and 10:1 boxes (words, MICR) | Feature extractors optimize specifically for their field geometry |
| Threshold Tuning | Single global confidence or complex per-class heuristics | Independent confidence thresholds per field |
| Maintenance Overhead | Retraining requires annotating all 8 classes across all data | Surgical retraining: update only the affected model |
| False Positive Risk | Class confusion between amount in words and payee lines | Harder to confuse; each model recognizes only its specific target |

### Model Performance Metrics

| Model File | Field Target | mAP@0.5 Score | Visual & Geometric Complexity |
| :--- | :--- | :--- | :--- |
| amt_word.pt | Amount in Words | 0.991 | Wide horizontal span, highly distinct visual structure |
| acc_no.pt | Account Number | 0.990 | Standardized boxed or dotted line region |
| pay.pt | Payee Name | 0.984 | Distinct Pay prefix with high spatial consistency |
| date.pt | Date Field | 0.984 | Compact top-right corner region |
| amt_digit.pt | Amount in Digits | 0.970 | Compact currency box with distinct border |
| micr.pt | MICR Band | 0.961 | Highly consistent bottom-border placement |
| acc_holder.pt | Account Holder | 0.919 | Variable placement (above or below signature line) |
| sign.pt | Signature Region | 0.876 | High variance in drawer strokes, ink, and orientation |

Below is a specimen detection run illustrating all 8 detectors firing simultaneously with green bounding coordinates:

![Sample YOLO Detection Output](Assets/metrics.jpeg)

---

## 6. OCR Layer & Dual-Path Fallback

### Primary Engine: PaddleOCR

PaddleOCR serves as the primary character extraction engine:
- DBNet Text Detection: Real-time differentiable binarization text detector.
- SVTR / CRNN Text Recognition: Sequence-to-sequence recognition converting pixel features into character predictions.
- Angle Classification: Activated via use_angle_cls=True to automatically correct 90, 180, and 270 degree text tilt.

### Fallback Engine: EasyOCR + TrOCR

When cheques exhibit extreme background noise, paper crumpling, or faint ink, YOLO field localization can fall below confidence thresholds. The architecture specifies a dual-path fallback:
- EasyOCR: Robust deep learning engine based on CRAFT (Character Region Awareness for Text Detection) capable of locating text without explicit YOLO boundaries.
- TrOCR (Transformer OCR): An encoder-decoder vision transformer (ViT encoder + RoBERTa/BART decoder) pretrained on handwritten text, ideal for reading cursive and idiosyncratic handwriting.

---

## 7. Text Normalization Engine

### 1. Optical Digit Confusion Map

When reading numeric amount boxes, OCR engines frequently substitute characters with similar contours. The normalizer applies substitution mapping:

| OCR Character | Mapped Digit | Visual Confusion Rationale |
| :--- | :--- | :--- |
| s, S | 5 | Rounded top and bottom stroke resemblance |
| b, B | 8 | Closed loops resemble figure eight |
| o, O | 0 | Oval shape identical to numeral zero |
| l, I, i | 1 | Vertical stroke identical to numeral one |
| z, Z | 2 | Horizontal and diagonal stroke geometry |
| g, q | 9 | Descender loop resembles numeral nine |
| y, Y, A | 4 | Angled junctions resemble open or closed numeral four |

### 2. Multi-Format Date Parser

The date parser executes sequential evaluation rules:
1. Standardizes all punctuation (. / space) to hyphens.
2. Replaces alphabetical month names (both full and abbreviated) with 2-digit equivalents.
3. Collapses consecutive hyphens.
4. If an 8-digit contiguous number remains (e.g., 20092025), formats to dd-mm-yyyy (20-09-2025).
5. If a 7-digit contiguous number remains (e.g., 2092025), evaluates month boundaries to infer whether it represents 20-09-2025 or 02-09-2025.
6. Resolves two-digit years by prepending century prefix 20.

### 3. Two-Pass Verbal Amount Accumulator

Converting verbal text into numbers requires handling both Western numbering (million, billion) and Indian numbering systems (lakh, crore):

- Step A (Sanitization): Strip currency tokens (rupees, rs, only).
- Step B (Token Splitting): Split joined words (e.g., Fouvthousend into Fouv and thousend).
- Step C (Fuzzy Correction): Match tokens against NUMBER_WORDS and MULTIPLIERS dictionaries using Python SequenceMatcher with cutoff 0.6.
- Step D (Accumulator Execution):
  - Local multiplier (hundred): multiplies current accumulator.
  - Global multipliers (thousand, lakh, crore, million, billion): multiply current accumulator, add to final sum, and reset current accumulator to zero.

Example trace for "two lakh fifty thousand three hundred":
- Word 'two' -> current = 2
- Multiplier 'lakh' -> final = 0 + (2 x 100000) = 200000, current = 0
- Word 'fifty' -> current = 50
- Multiplier 'thousand' -> final = 200000 + (50 x 1000) = 250000, current = 0
- Word 'three' -> current = 3
- Multiplier 'hundred' -> current = 3 x 100 = 300
- Termination -> final = 250000 + 300 = 250300.0

---

## 8. Validation & Fraud Detection Logic

Cheque validation acts as a deterrent against check alteration and forgery:

Logic specification:
- Amount in Digit and Amount in Word parsed into floating-point numbers.
- If val_digit equals val_word and val_digit is greater than 0: Match is Yes.
- Otherwise: Match is No.

### Fraud Scenarios Caught:
- Amount Alteration: If a fraudulent party appends a zero to the digit box (changing 500 to 5000), but cannot cleanly alter the words (Five Hundred), the parity check fails (val_digit != val_word) and marks the cheque as Match = No.
- Missing Signature: If sign.pt fails to detect a signature above the account holder line, Sign is recorded as No, halting clearance.
- Fake Cheque Routing: If the extracted MICR string does not correspond to valid routing records, the transaction is rejected.

---

## 9. Data Flow Diagram

Cheque Image
     |
     +---> acc_holder.pt --> OCR --> normalize --> "Account Holder Name"
     +---> acc_no.pt     --> OCR ----------------> "Account Number"
     +---> amt_word.pt   --> OCR --> normalize --> 550.0  ----+
     +---> amt_digit.pt  --> OCR --> normalize --> 550.0  ----+--> Match? --> CSV
     +---> date.pt       --> OCR --> normalize --> "20-09-2025"
     +---> pay.pt        --> OCR ----------------> "Pay Name"
     +---> sign.pt       -----------------------> "Yes" / "No"
     +---> micr.pt       --> OCR ----------------> "1234567890"

---

## 10. Failure Modes & Resilience Matrix

The table below outlines potential pipeline failures and their automated mitigation strategies:

| Pipeline Stage | Failure Scenario | Automated Recovery Strategy |
| :--- | :--- | :--- |
| Ingestion | Corrupted image file or unsupported header | Checks cv2.imread validity; raises explicit descriptive error without terminating batch |
| Ingestion | Grayscale or single-channel cheque scan | Automatic cv2.cvtColor expansion to 3-channel BGR |
| Localization | Low contrast prevents YOLO MICR detection | Fallback geometric slicer crops bottom 15 percent of cheque |
| Localization | Multiple bounding boxes detected for a single field | max() selector picks the candidate with highest confidence score |
| OCR | Angled or upside-down handwriting | PaddleOCR angle classification (use_angle_cls=True) corrects orientation |
| Normalization | Misspelled verbal numerals (e.g. 'thousend') | Fuzzy matching via SequenceMatcher with 0.6 similarity threshold |
| Normalization | Optical letter-for-number substitution | Alphanumeric confusion map replaces letters with target digits |
| Normalization | 7-digit compressed date string | Heuristic calendar month validator infers day/month boundaries |
| Validation | Amount disparity between words and digits | Flags cheque as Match = No for manual security inspection |

---

## 11. Technology Choices & Rationale

| Component | Selected Technology | Alternative Evaluated | Selection Rationale |
| :--- | :--- | :--- | :--- |
| Object Detection | YOLOv8 Nano | Faster R-CNN / SSD | Superior inference latency (<15ms per model), small memory footprint (~6MB), easy deployment |
| OCR Engine | PaddleOCR | Tesseract OCR | Native angle classification, higher accuracy on complex scripts, direct NumPy input |
| Fallback Handwriting OCR | TrOCR | Standard CNN-RNN OCR | Vision transformer architecture excels at unconstrained handwriting strokes |
| String Normalization | Python difflib | SymSpell / Levenshtein C | Standard library module, eliminates external binary dependencies, deterministic execution |
| Web Dashboard | Streamlit | Flask / React | Rapid reactive prototyping, built-in session state caching, no frontend build pipeline required |
| Tabular Data | Pandas | Polars / SQLite | Industry standard for financial CSV generation, native integration with Streamlit |

---

## 12. Authors & Acknowledgments

### Lead Developer
Dhruv Kajare

### Team Members
Juhi Nale
Neha Tarmale

### Acknowledgments
Gratitude is extended to the research collaborators and open-source contributors of Ultralytics YOLOv8, PaddleOCR, and Streamlit whose foundational frameworks enabled the realization of the ChequeTrak pipeline.

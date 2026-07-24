# ChequeTrak – Intelligent Cheque Processing System

## Overview
ChequeTrak is an AI-based cheque processing pipeline designed to detect and extract key cheque fields from scanned images. The current implementation uses YOLOv8 for field localization, PaddleOCR for text recognition, and a normalization layer for cleaning extracted values such as dates and amounts.

The project is organized around a modular pipeline that can process a single cheque image or a batch of cheque images from a folder.

## Tech Stack
- Python
- OpenCV
- Ultralytics YOLOv8
- PaddleOCR
- Streamlit
- NumPy
- Pandas
- PyTorch

## Repository Structure

```text
.
├── app.py                   # Streamlit web app for batch processing
├── extractor.py             # Main cheque extraction pipeline
├── combine_extractor.py     # Standalone script for single-image extraction
├── conbiner.py              # Alternative/experimental extractor script
├── combiner2.py             # Alternative/experimental extractor script
├── micr.py                  # MICR-specific extraction logic
├── text_normalizer.py       # Date and amount normalization helpers
├── test_normalizer.py       # Unit tests for normalizer functions
├── labeller.py              # YOLO labeling tool for one-region annotation
├── labeller2.py             # YOLO labeling tool for multi-box annotation
├── sign_majorpro.ipynb      # Notebook for experimentation and prototyping
├── Models/                  # YOLO model weights for cheque field detection
└── Models.zip               # Archived model files
```

## Main Components

### 1. Application entry point
- app.py
  - Provides a Streamlit interface for uploading or selecting a folder of cheque images.
  - Runs the extraction pipeline on each image.
  - Displays results in a table and saves them as a CSV file.

### 2. Core extraction engine
- extractor.py
  - Loads YOLOv8 detection models from the Models folder.
  - Detects cheque fields such as payee name, amount in words, amount in digits, date, account holder, account number, and signature presence.
  - Crops detected regions and runs OCR on them.
  - Applies normalization logic to improve extracted values.

### 3. Single-image processing script
- combine_extractor.py
  - A lightweight script for running the extraction pipeline on one image.
  - Useful for testing and debugging before batch execution.

### 4. MICR extraction
- micr.py
  - Focuses on extracting MICR text from the cheque band.
  - Uses YOLO-based localization and OCR on the detected region.

### 5. Text normalization
- text_normalizer.py
  - Cleans and normalizes OCR output for dates and amounts.
  - Handles common OCR errors and spelling variations.

### 6. Dataset labeling utilities
- labeller.py and labeller2.py
  - Helper scripts for manually annotating cheque images for YOLO training.
  - Useful when expanding or retraining the detection models.

## Processing Workflow
1. Load the trained YOLOv8 models from the Models directory.
2. Detect relevant cheque regions in the image.
3. Crop each detected region.
4. Run OCR to extract text.
5. Normalize the extracted information.
6. Return structured results for downstream validation or export.

## How to Run

### Batch processing with Streamlit
```bash
streamlit run app.py
```
Then provide a folder path containing cheque images and start processing.

### Single-image testing
Update the input path in combine_extractor.py and run:
```bash
python combine_extractor.py
```

## Notes
- The project currently relies on local model files stored in the Models directory.
- Some scripts such as conbiner.py and combiner2.py appear to be alternative or experimental versions of the main extraction logic.
- The normalization layer is designed to improve OCR robustness for dates and amount values.

## Expected Output
For each processed image, the pipeline produces structured extracted values and saves a CSV file containing the results when run through the Streamlit app.

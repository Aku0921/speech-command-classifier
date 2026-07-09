# Lightweight Speech Command Classifier

A lightweight semantic command classifier designed for offline edge deployment. The system classifies 14 predefined voice commands based on their semantic meaning using Sentence Transformers and a cosine similarity-based classifier.

---

## Features

- Semantic command classification, support for 14 semantic voice commands
- Sentence embedding generation using all-MiniLM-L6-v2
- Cosine similarity-based prediction
- Unknown command rejection using confidence threshold
- Evaluation using Accuracy, Precision, Recall, F1-score and Confusion Matrix
- ASR-inspired noise simulation for robustness testing
- ONNX model export for edge deployment
- INT8 dynamic quantization
- ONNX and INT8 model verification against the original PyTorch model
- Offline inference
- Lightweight architecture for edge devices

---

## Supported Commands

The classifier currently supports the following semantic commands:

1. Activate Do Not Disturb
2. Deactivate Do Not Disturb
3. Pick Up Call
4. Decline Call
5. Play Music
6. Pause Music
7. Play Next Song
8. Play Previous Song
9. Increase Volume
10. Decrease Volume
11. Increase Brightness
12. Decrease Brightness
13. Start Vehicle
14. Stop Vehicle

Commands outside these categories are rejected as **unknown** using cosine similarity thresholding.

---

## Inference Pipeline

```text
Text Input
      ↓
Sentence Embedding (all-MiniLM-L6-v2)
      ↓
Cosine Similarity Classifier
      ↓
Threshold-based OOS Rejection
      ↓
Predicted Command / Unknown
```
---

## Evaluation Pipeline

```text
Test Dataset
      ↓
ASR Noise Simulation
      ↓
Sentence Embedding
      ↓
Cosine Similarity Classifier
      ↓
Threshold-based OOS Rejection
      ↓
Performance Evaluation
      ├── Accuracy
      ├── Precision / Recall / F1-score
      ├── Confusion Matrix
      └── OOS Metrics
```

---

## Model Deployment Pipeline

```text
PyTorch Sentence Transformer
            ↓
Export to ONNX
            ↓
INT8 Dynamic Quantization
            ↓
ONNX Runtime Verification
            ↓
Edge Deployment (Android / iOS / Embedded)
```
---

## Training

No fine-tuning is performed.

The project uses the pretrained
all-MiniLM-L6-v2 Sentence Transformer.

Training consists of generating
embeddings for the training dataset
and storing them for cosine similarity.

---

## Project Structure

```text
speech-command-classifier/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── onnx/
│   │   ├── model.onnx
│   │   └── model_int8.onnx
│   ├── 1_Pooling/
│   ├── 2_Normalize/
│   └── ...
│
├── results/
│   ├── clean_confusion_matrix.png
│   ├── clean_evaluation_report.txt
│   ├── noisy_confusion_matrix.png
│   └── noisy_evaluation_report.txt
│
├── src/
│   ├── benchmark.py
│   ├── classifier.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── export_onnx.py
│   ├── main.py
│   ├── model_size.py
│   ├── noise_generator.py
│   ├── quantize_model.py
│   ├── verify_model.py
│   └── predict.py
│
├── README.md
├── requirements.txt
└── .gitignore
```
---

## Technologies Used

- Python 3.11
- Sentence Transformers
- all-MiniLM-L6-v2
- Scikit-learn
- NumPy
- Pandas
- Matplotlib

---

## Current Progress

### Milestone 1 - Baseline Classifier ✅

- [x] Completed

### Milestone 2 - Noise Robustness ✅

- [x] ASR noise simulation
- [x] Evaluation on noisy inputs

### Milestone 3 – Model Compression and Edge Export ✅

- [x] Benchmarked the original PyTorch embedding model.
- [x] Exported the Sentence Transformer to ONNX format.
- [x] Applied INT8 dynamic quantization.
- [x] Compared original and quantized model sizes.
- [x] Verified prediction consistency between the PyTorch, ONNX, and INT8 models.

#### Model Compression Results

| Metric | Value |
|---------|------:|
| Original ONNX Model | 86.18 MB |
| Quantized INT8 Model | 21.92 MB |
| Model Size Reduction | 74.56% |
| Average PyTorch Latency | 9.14 ms |
| PyTorch vs ONNX Prediction Match | 100% |
| PyTorch vs INT8 Prediction Match | 95% |

### Milestone 4 – Extension Commands and Extensibility ✅

- [x] Extended the classifier from 10 to 14 semantic commands.
- [x] Evaluated the extended classifier on clean and noisy datasets.
- [x] Verified that no changes to the classifier architecture were required.
- [x] Demonstrated the extensibility of the data-driven pipeline.

### Milestone 5 – End-to-End Integration ✅

- [x] Implemented end-to-end inference using `predict.py`.
- [x] Added support for PyTorch, ONNX and INT8 backends.
- [x] Implemented threshold-based Out-of-Scope (OOS) rejection.
- [x] Evaluated the complete pipeline on clean and noisy datasets.
- [x] Measured OOS rejection and false rejection rates.
- [x] Verified end-to-end inference across all supported backends.

---

## End-to-End Inference

The classifier provides a single-command interface for semantic command classification.

Run inference using:

```bash
python src/predict.py "increase the brightness" --backend int8
```

---

## How to Run

### Requirements

The project was developed and tested using Python 3.11.

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the classifier:

```bash
python src/main.py
```

Run inference on a single command:

```bash
python src/predict.py "increase the brightness" --backend pytorch
```
Available backends:

- `pytorch`
- `onnx`
- `int8`

Benchmark the PyTorch model:

```bash
python src/benchmark.py
```

Generate ONNX model:

```bash
python src/export_onnx.py
```

Generate the INT8 model:

```bash
python src/quantize_model.py
```

Compare model sizes:

```bash
python src/model_size.py
```

Verify ONNX predictions:

```bash
python src/verify_model.py
```

---

## Example Inference

### Example 1

```bash
python src/predict.py "could you please increase the brightness" --backend pytorch
```
```text
============================================================
Speech Command Classifier
============================================================
Backend    : PYTORCH
Input      : could you please increase the brightness
Prediction : increase_brightness
Similarity : 0.9298
Status     : ACCEPTED
```

### Example 2

```bash
python src/predict.py "what is the weather like today" --backend pytorch    
```      
```text
============================================================
Speech Command Classifier
============================================================
Backend    : PYTORCH
Input      : what is the weather like today
Prediction : unknown
Similarity : 0.1407
Status     : REJECTED (Out-of-Scope)
```

### Example 3

```bash
python src/predict.py "would you decrease the volume" --backend onnx  
```            
```text
============================================================
Speech Command Classifier
============================================================
Backend    : ONNX
Input      : would you decrease the volume
Prediction : decrease_volume
Similarity : 0.8330
Status     : ACCEPTED
```

### Example 4

```bash
python src/predict.py "please start the vehicle" --backend int8 
```     
```text             
============================================================
Speech Command Classifier
============================================================
Backend    : INT8
Input      : please start the vehicle
Prediction : start_vehicle
Similarity : 0.9840
Status     : ACCEPTED
```

### Example 5

```bash
python src/predict.py "set and alarm" --backend int8  
```     
```text
============================================================
Speech Command Classifier
============================================================
Backend    : INT8
Input      : set and alarm
Prediction : unknown
Similarity : 0.4314
Status     : REJECTED (Out-of-Scope)
```

---

## Edge Deployment

The exported INT8 ONNX model is intended for deployment on edge devices such as smartphones.

While this repository provides a Python reference implementation, the quantized ONNX model can be integrated into Android or iOS applications using ONNX Runtime Mobile without modifying the classifier architecture.

---

## Limitations

- The classifier supports a predefined set of commands.
- Only the Sentence Transformer embedding model is exported to ONNX. The cosine similarity classifier remains implemented in Python.
- Dynamic INT8 quantization may introduce small prediction differences for inputs near the cosine similarity threshold.
- The project is benchmarked on a CPU development machine and has not been evaluated on a physical edge device.

## Output

Running the project generates:

- Classification accuracy
- Precision, Recall and F1-score
- Confusion Matrix
- OOS rejection statistics
- Evaluation reports for all 14 supported commands
- `results/clean_evaluation_report.txt`
- `results/clean_confusion_matrix.png`
- `results/noisy_evaluation_report.txt`
- `results/noisy_confusion_matrix.png`

---

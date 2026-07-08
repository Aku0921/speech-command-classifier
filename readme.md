# Lightweight Speech Command Classifier

A lightweight semantic command classifier designed for offline edge deployment. The system classifies predefined voice commands based on their semantic meaning using Sentence Transformers and a cosine similarity-based classifier.

---

## Features

- Semantic command classification
- Sentence embedding generation using all-MiniLM-L6-v2
- Cosine similarity-based prediction
- Unknown command rejection using confidence threshold
- Evaluation using Accuracy, Precision, Recall, F1-score and Confusion Matrix
- ASR-inspired noise simulation for robustness testing
- ONNX model export for edge deployment
- INT8 dynamic quantization
- ONNX prediction verification against the PyTorch model
- Offline inference
- Lightweight architecture for edge devices

---

## Current Pipeline

```text
Text Input
      ↓
ASR Noise Simulation (Evaluation only)
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

## Model Deployment Pipeline

```text
PyTorch Sentence Transformer
            ↓
Export to ONNX
            ↓
INT8 Dynamic Quantization
            ↓
ONNX Runtime Verification
```
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
│   └── verify_onnx.py
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

### Milestone 3 – Model Compression and Edge Export

#### Objective

Compress the embedding model for efficient edge deployment and export it to ONNX format.

#### Steps Performed

- Benchmarked the original PyTorch embedding model.
- Exported the Sentence Transformer to ONNX format.
- Applied INT8 dynamic quantization.
- Compared original and quantized model sizes.
- Verified that the exported ONNX model produces identical predictions to the original PyTorch model.

#### Results

| Metric | Value |
|---------|------:|
| Original ONNX Size | 86.18 MB |
| Quantized INT8 ONNX Size | 21.92 MB |
| Model Size Reduction | 74.56% |
| Average PyTorch Latency | 9.14 ms |
| ONNX Prediction Match | 100% (40 / 40 samples) |

#### Files Generated

```text
models/
└── onnx/
    ├── model.onnx
    └── model_int8.onnx
```

#### Benchmark

Average PyTorch inference latency:
```

9.14 ms

```

### Upcoming Milestones

- [ ] M4 - Extension Commands
- [ ] M5 - End-to-End Integration

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the classifier:

```bash
python src/main.py
```

Benchmark the PyTorch model:

```bash
python src/benchmark.py
```

Export the model to ONNX:

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
python src/verify_onnx.py
```

---

## Limitations

- The classifier supports a predefined set of commands.
- The ONNX export is applied only to the embedding model. The cosine similarity classifier remains implemented in Python.
- The project is benchmarked on a CPU development machine and has not been evaluated on a physical edge device.

## Output

Running the project generates:

- Classification accuracy
- Precision, Recall and F1-score
- Confusion Matrix
- `results/clean_evaluation_report.txt`
- `results/clean_confusion_matrix.png`
- `results/noisy_evaluation_report.txt`
- `results/noisy_confusion_matrix.png`

---
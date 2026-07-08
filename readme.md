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
- ONNX and INT8 model verification against the original PyTorch model
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
│   └── verify_model.py
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
- Verified prediction consistency between the PyTorch, ONNX, and INT8 models.

#### Results

| Metric | Value |
|---------|------:|
| Original ONNX Size | 86.18 MB |
| Quantized INT8 ONNX Size | 21.92 MB |
| Model Size Reduction | 74.56% |
| Average PyTorch Latency | 9.14 ms |
| PyTorch vs ONNX Prediction Match | 100% (40 / 40) |
| PyTorch vs INT8 Prediction Match | 95% (38 / 40) |

The exported ONNX model produced identical predictions to the original PyTorch model on all 40 test samples. After INT8 dynamic quantization, the compressed model matched 38 out of 40 predictions (95%), demonstrating a small accuracy trade-off in exchange for a 74.56% reduction in model size

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

## Model Artifacts

The repository includes the final INT8-quantized ONNX model (`model_int8.onnx`), which is the deployment artifact generated during Milestone 3.

The intermediate FP32 ONNX model (`model.onnx`) is not included to reduce repository size. It can be regenerated at any time using:

```bash
python src/export_onnx.py
```
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
- `results/clean_evaluation_report.txt`
- `results/clean_confusion_matrix.png`
- `results/noisy_evaluation_report.txt`
- `results/noisy_confusion_matrix.png`

---
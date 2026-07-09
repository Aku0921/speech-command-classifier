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

### Milestone 3 – Model Compression and Edge Export ✅

- [x] Benchmarked the original PyTorch embedding model.
- [x] Exported the Sentence Transformer to ONNX format.
- [x] Applied INT8 dynamic quantization.
- [x] Compared original and quantized model sizes.
- [x] Verified prediction consistency between the PyTorch, ONNX, and INT8 models.

### Milestone 4 – Extension Commands and Extensibility ✅

#### Objective

Extend the classifier from 10 to 14 semantic commands without modifying the classifier architecture.

#### Commands Added

- Increase Brightness
- Decrease Brightness
- Start Vehicle
- Stop Vehicle

#### Changes Required

The extension only required updating the training and testing datasets with examples for the four new commands.

No modifications were required on any other files.

This demonstrates that the architecture is data-driven and supports new commands without requiring changes to the core implementation.

#### Evaluation

The extended classifier was evaluated on all 14 commands using both clean and noisy test sets.

##### Clean Test Results

| Metric | Value |
|---------|------:|
| Accuracy | 96.15% |

##### Noisy Test Results

| Metric | Value |
|---------|------:|
| Accuracy | 88.46% |

#### Interference Analysis

The addition of the new commands did not introduce noticeable confusion with semantically similar existing commands.

The classification errors were primarily associated with the existing music playback commands under noisy conditions.

#### Extensibility Assessment

The classifier architecture scales well because command definitions are stored entirely in the dataset.

Adding new commands only requires:

1. Adding labelled examples to the training dataset.
2. Adding evaluation samples to the test dataset.
3. Running the existing pipeline.

No changes to the embedding model, classifier implementation, or deployment pipeline were necessary.

As the number of supported commands grows, semantically similar commands may become increasingly difficult to distinguish. Future improvements could include additional training examples, threshold tuning, or a supervised classifier.

### Upcoming Milestones

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
- Evaluation reports for all 14 supported commands
- `results/clean_evaluation_report.txt`
- `results/clean_confusion_matrix.png`
- `results/noisy_evaluation_report.txt`
- `results/noisy_confusion_matrix.png`

---
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

## Project Structure

```text
speech-command-classifier/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── src/
│   ├── classifier.py
│   ├── dataset.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── noise_generator.py
│   └── main.py
│
├── results/
│   ├── clean_confusion_matrix.png
│   └── clean_evaluation_report.txt
│   ├── noisy_confusion_matrix.png
│   └── noisy_evaluation_report.txt
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

### Upcoming Milestones

- [ ] M3 - Model Compression & Edge Export
- [ ] M4 - Extension Commands
- [ ] M5 - End-to-End Integration

---

## How to Run

```bash
pip install -r requirements.txt
python src/main.py
```
---

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
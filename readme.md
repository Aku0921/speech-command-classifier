# Lightweight Speech Command Classifier

A lightweight semantic command classifier designed for offline edge deployment. The system classifies predefined voice commands based on their semantic meaning using Sentence Transformers and a cosine similarity-based classifier.

---

## Features

- Semantic command classification
- Sentence embedding generation using all-MiniLM-L6-v2
- Cosine similarity-based prediction
- Unknown command rejection using confidence threshold
- Evaluation using Accuracy, Precision, Recall, F1-score and Confusion Matrix
- Offline inference
- Lightweight architecture for edge devices

---

## Current Pipeline

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
│   └── main.py
│
├── results/
│   ├── confusion_matrix.png
│   └── evaluation_report.txt
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

- [x] Dataset creation
- [x] Dataset loading
- [x] Sentence embeddings
- [x] Cosine similarity classifier
- [x] Threshold-based OOS rejection
- [x] Evaluation
- [x] Confusion matrix generation

### Upcoming Milestones

- [ ] M2 - Noise Robustness & ASR Simulation
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
- `results/evaluation_report.txt`
- `results/confusion_matrix.png`

---
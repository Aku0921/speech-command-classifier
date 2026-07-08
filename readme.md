# Lightweight Speech Command Classifier

A lightweight semantic command classifier designed for offline edge deployment. The system classifies predefined voice commands based on their semantic meaning using Sentence Transformers and a cosine similarity-based classifier.

---

## Features

- Semantic command classification
- Cosine similarity-based prediction
- Unknown command rejection using confidence threshold
- Offline inference
- Lightweight architecture for edge devices

---

## Project Structure

```text
speech-command-classifier/
│
├── data/
│
├── src/
│
├── results/
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

- [x] Dataset creation
- [x] Dataset loading module
- [x] Sentence embedding generation
- [ ] Cosine similarity classifier
- [ ] Unknown command rejection
- [ ] Evaluation
- [ ] Documentation

---

## How to Run

```bash
pip install -r requirements.txt
python src/main.py

import argparse
from classifier import CosineSimilarityClassifier
from dataset import load_datasets
from embeddings import (
    generate_embeddings,
    load_embedding_model,
)
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Speech Command Classifier"
    )
    parser.add_argument(
        "text",
        type=str,
        help="Input command to classify.",
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "int8"],
        default="pytorch",
        help="Embedding backend to use.",
    )
    args = parser.parse_args()
    # Load datasets
    train_df, _ = load_datasets()
    # Load embedding model
    model = load_embedding_model(args.backend)
    # Generate training embeddings
    train_embeddings = generate_embeddings(
        model,
        train_df["text"].tolist(),
    )
    # Train classifier
    classifier = CosineSimilarityClassifier()
    classifier.fit(
        train_embeddings,
        train_df["label"].tolist(),
    )
    # Generate embedding for input
    input_embedding = generate_embeddings(
        model,
        [args.text],
    )[0]
    # Predict
    prediction, similarity = classifier.predict(
        input_embedding
    )
    status = (
        "REJECTED (Out-of-Scope)"
        if prediction == "unknown"
        else "ACCEPTED"
    )
    print("=" * 60)
    print("Speech Command Classifier")
    print("=" * 60)
    print(f"Backend    : {args.backend.upper()}")
    print(f"Input      : {args.text}")
    print(f"Prediction : {prediction}")
    print(f"Similarity : {similarity:.4f}")
    print(f"Status     : {status}")
if __name__ == "__main__":
    main()
from dataset import load_datasets
from embeddings import (
    generate_embeddings,
    load_embedding_model,
)
from classifier import CosineSimilarityClassifier
from evaluation import evaluate
def main() -> None:
    # Load datasets
    train_df, test_df = load_datasets()
    # Load embedding model
    model = load_embedding_model()
    # Generate embeddings
    train_embeddings = generate_embeddings(
        model,
        train_df["text"].tolist(),
    )
    test_embeddings = generate_embeddings(
        model,
        test_df["text"].tolist(),
    )
    # Initialize classifier
    classifier = CosineSimilarityClassifier()
    # Store training embeddings and labels
    classifier.fit(
        train_embeddings,
        train_df["label"].tolist(),
    )
    # Evaluate classifier
    evaluate(
        classifier,
        test_embeddings,
        test_df["label"].tolist(),
    )
if __name__ == "__main__":
    main()
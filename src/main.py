from dataset import load_datasets
from embeddings import (
    generate_embeddings,
    load_embedding_model,
)
from classifier import CosineSimilarityClassifier


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

    # Create classifier
    classifier = CosineSimilarityClassifier()

    # Store training data
    classifier.fit(
        train_embeddings,
        train_df["label"].tolist(),
    )

    # Predict the first test sample
    prediction, similarity = classifier.predict(
        test_embeddings[0]
    )

    print("Test Sentence:")
    print(test_df["text"].iloc[0])

    print("\nActual Label:")
    print(test_df["label"].iloc[0])

    print("\nPredicted Label:")
    print(prediction)

    print("\nSimilarity Score:")
    print(f"{similarity:.4f}")


if __name__ == "__main__":
    main()
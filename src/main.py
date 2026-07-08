from dataset import load_datasets
from embeddings import (
    generate_embeddings,
    load_embedding_model,
)
from classifier import CosineSimilarityClassifier
from evaluation import evaluate
from noise_generator import generate_noisy_sentence
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
    # Generate clean test embeddings
    test_embeddings = generate_embeddings(
        model,
        test_df["text"].tolist(),
    )
    # Generate noisy test sentences
    noisy_test_sentences = [
        generate_noisy_sentence(sentence)
        for sentence in test_df["text"]
    ]
    # Generate noisy test embeddings
    noisy_test_embeddings = generate_embeddings(
        model,
        noisy_test_sentences,
    )
    # Initialize classifier
    classifier = CosineSimilarityClassifier()
    # Store training embeddings and labels
    classifier.fit(
        train_embeddings,
        train_df["label"].tolist(),
    )
    # Evaluate classifier
    print("=" * 60)
    print("Evaluation on Clean Test Set")
    print("=" * 60)
    evaluate(
        classifier,
        test_embeddings,
        test_df["label"].tolist(),
        report_name="clean",
    )
    print("\n" + "=" * 60)
    print("Evaluation on Noisy Test Set")
    print("=" * 60)
    evaluate(
        classifier,
        noisy_test_embeddings,
        test_df["label"].tolist(),
        report_name="noisy",
    )
if __name__ == "__main__":
    main()
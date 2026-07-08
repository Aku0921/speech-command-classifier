from dataset import load_datasets
from embeddings import (
    generate_embeddings,
    load_embedding_model,
)


def main() -> None:

    train_df, test_df = load_datasets()

    model = load_embedding_model()

    train_embeddings = generate_embeddings(
        model,
        train_df["text"].tolist(),
    )

    print(train_embeddings.shape)


if __name__ == "__main__":
    main()
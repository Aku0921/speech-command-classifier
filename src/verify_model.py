from dataset import load_datasets
from embeddings import (
    load_embedding_model,
    generate_embeddings,
)
from classifier import CosineSimilarityClassifier
def main() -> None:
    print("=" * 60)
    print("Verifying ONNX Model")
    print("=" * 60)
    train_df, test_df = load_datasets()
    # Load both models
    pytorch_model = load_embedding_model("pytorch")
    onnx_model = load_embedding_model("onnx")
    int8_model = load_embedding_model("int8")
    # Generate train embeddings using PyTorch
    train_embeddings = generate_embeddings(
        pytorch_model,
        train_df["text"].tolist(),
    )
    classifier = CosineSimilarityClassifier()
    classifier.fit(
        train_embeddings,
        train_df["label"].tolist(),
    )
    # Generate test embeddings
    pytorch_test_embeddings = generate_embeddings(
        pytorch_model,
        test_df["text"].tolist(),
    )
    onnx_test_embeddings = generate_embeddings(
        onnx_model,
        test_df["text"].tolist(),
    )
    int8_test_embeddings = generate_embeddings(
        int8_model,
        test_df["text"].tolist(),
    )
    print("\nComparing predictions...\n")
    total = len(test_df)
    onnx_matches = 0
    int8_matches = 0
    for sentence, pt_embedding, onnx_embedding, int8_embedding in zip(
        test_df["text"],
        pytorch_test_embeddings,
        onnx_test_embeddings,
        int8_test_embeddings,
    ):
        pt_prediction, _ = classifier.predict(pt_embedding)
        onnx_prediction, _ = classifier.predict(onnx_embedding)
        int8_prediction, _ = classifier.predict(int8_embedding)
        if pt_prediction == onnx_prediction:
            onnx_matches += 1
        else:
            print(f"ONNX mismatch for: {sentence}")
            print(f"PyTorch : {pt_prediction}")
            print(f"ONNX    : {onnx_prediction}")
            print("-" * 60)
        if pt_prediction == int8_prediction:
            int8_matches += 1
        else:
            print(f"INT8 mismatch for: {sentence}")
            print(f"PyTorch : {pt_prediction}")
            print(f"INT8    : {int8_prediction}")
            print("-" * 60)
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print("\nPyTorch vs ONNX")
    print(f"Matching Predictions : {onnx_matches}/{total}")
    print(f"Prediction Match     : {(onnx_matches / total) * 100:.2f}%")
    print("\nPyTorch vs INT8")
    print(f"Matching Predictions : {int8_matches}/{total}")
    print(f"Prediction Match     : {(int8_matches / total) * 100:.2f}%")
    if onnx_matches == total and int8_matches == total:
        print("\nSUCCESS: Both ONNX and INT8 models produce identical predictions.")
    else:
        print("\nWARNING: Prediction differences detected.")
if __name__ == "__main__":
    main()
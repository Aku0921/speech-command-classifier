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
    print("\nComparing predictions...\n")
    total = len(test_df)
    matches = 0
    for sentence, pt_embedding, onnx_embedding in zip(
        test_df["text"],
        pytorch_test_embeddings,
        onnx_test_embeddings,
    ):
        pt_prediction, _ = classifier.predict(pt_embedding)
        onnx_prediction, _ = classifier.predict(onnx_embedding)
        match = pt_prediction == onnx_prediction
        if not match:
            print(f"Input      : {sentence}")
            print(f"PyTorch    : {pt_prediction}")
            print(f"ONNX       : {onnx_prediction}")
            print("-" * 60)
        if match:    
            matches += 1
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    print(f"Total Samples        : {total}")
    print(f"Matching Predictions : {matches}")
    print(f"Prediction Match     : {(matches / total) * 100:.2f}%")
    if matches == total:
        print("\nSUCCESS: ONNX model produces identical predictions.")
    else:
        print("\nWARNING: Prediction differences detected.")
if __name__ == "__main__":
    main()
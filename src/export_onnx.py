from pathlib import Path
from sentence_transformers import SentenceTransformer
def main() -> None:
    # Directory where the exported model will be stored
    output_dir = Path("models")
    # Load the model using the ONNX backend.
    # export=True forces the model to be exported to ONNX.
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        backend="onnx",
        model_kwargs={"export": True},
    )
    # Save the exported model
    model.save(str(output_dir))
    print(f"\nONNX model exported successfully to: {output_dir.resolve()}")
if __name__ == "__main__":
    main()  
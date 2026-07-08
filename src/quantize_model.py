from pathlib import Path
from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model,
)
def main() -> None:
    # Directory containing the exported ONNX model
    model_dir = Path("models")
    # Load the ONNX model
    model = SentenceTransformer(
        str(model_dir),
        backend="onnx",
    )
    # Export a dynamically quantized INT8 model
    export_dynamic_quantized_onnx_model(
        model=model,
        quantization_config="avx2",
        model_name_or_path=str(model_dir),
        file_suffix="int8",
    )
    print("\nINT8 quantized model created successfully.")
if __name__ == "__main__":
    main()
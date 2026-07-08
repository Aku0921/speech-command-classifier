from pathlib import Path
def get_size_mb(file_path: Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)
def main():
    original = Path("models/onnx/model.onnx")
    quantized = Path("models/onnx/model_int8.onnx")
    original_size = get_size_mb(original)
    quantized_size = get_size_mb(quantized)
    reduction = (
        (original_size - quantized_size)
        / original_size
        * 100
    )
    print("=" * 50)
    print("ONNX Model Size Comparison")
    print("=" * 50)
    print(f"Original ONNX : {original_size:.2f} MB")
    print(f"INT8 ONNX     : {quantized_size:.2f} MB")
    print(f"Reduction     : {reduction:.2f}%")
if __name__ == "__main__":
    main()
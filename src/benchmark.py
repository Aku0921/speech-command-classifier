import argparse
import statistics
import time
from embeddings import (
    load_embedding_model,
    generate_embeddings,
)
NUM_RUNS = 100
WARMUP_RUNS = 5
def benchmark_embedding_model(backend: str) -> None:
    print("=" * 50)
    print(f"{backend.upper()} Model Benchmark")
    print("=" * 50)
    model = load_embedding_model(backend)
    sentence = ["turn on the lights"]
    # Warm-up
    for _ in range(WARMUP_RUNS):
        generate_embeddings(model, sentence)
    timings = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        generate_embeddings(model, sentence)
        end = time.perf_counter()
        timings.append((end - start) * 1000)
    print(f"Average Latency : {statistics.mean(timings):.2f} ms")
    print(f"Minimum Latency : {min(timings):.2f} ms")
    print(f"Maximum Latency : {max(timings):.2f} ms")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx", "int8"],
        default="pytorch",
    )
    args = parser.parse_args()
    benchmark_embedding_model(args.backend)
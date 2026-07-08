import time
import statistics
from embeddings import (
    load_embedding_model,
    generate_embeddings,
)
# Number of benchmark iterations
NUM_RUNS = 100
# Number of warm-up runs
WARMUP_RUNS = 5
def benchmark_embedding_model() -> None:
    print("=" * 50)
    print("PyTorch Model Benchmark")
    print("=" * 50)
    # Load model
    model = load_embedding_model()
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
    average = statistics.mean(timings)
    minimum = min(timings)
    maximum = max(timings)
    print(f"Average Latency : {average:.2f} ms")
    print(f"Minimum Latency : {minimum:.2f} ms")
    print(f"Maximum Latency : {maximum:.2f} ms")
if __name__ == "__main__":
    benchmark_embedding_model()
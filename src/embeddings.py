from typing import List, Literal
import numpy as np
from sentence_transformers import SentenceTransformer
def load_embedding_model(
    backend: Literal["pytorch", "onnx", "int8"] = "pytorch",
) -> SentenceTransformer:
    """
    Load the embedding model.
    """
    if backend == "pytorch":
        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )
    elif backend == "onnx":
        return SentenceTransformer(
            "models",
            backend="onnx",
            model_kwargs={
                "file_name": "onnx/model.onnx"
            },
        )
    elif backend == "int8":
        return SentenceTransformer(
            "models",
            backend="onnx",
            model_kwargs={
                "file_name": "onnx/model_int8.onnx"
            },
        )
    raise ValueError(
        f"Unsupported backend: {backend}"
    )
def generate_embeddings(
    model: SentenceTransformer,
    sentences: List[str],
) -> np.ndarray:
    """
    Generate embeddings for a list of sentences.
    """
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
    )
    return embeddings
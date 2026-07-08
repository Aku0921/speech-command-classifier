from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
def load_embedding_model() -> SentenceTransformer:
    """
    Loading the sentence embedding model.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")
def generate_embeddings(
    model: SentenceTransformer,
    sentences: List[str],
) -> np.ndarray:
    """
    Generate embeddings for a list of sentences.
    """
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True
    )
    return embeddings
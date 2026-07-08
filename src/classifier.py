from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class CosineSimilarityClassifier:
    """
    Classifier based on cosine similarity.
    """
    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self.train_embeddings = None
        self.train_labels = None
    def fit(
        self,
        train_embeddings: np.ndarray,
        train_labels: List[str],
    ) -> None:
        """
        Store the training embeddings and labels.
        """
        self.train_embeddings = train_embeddings
        self.train_labels = train_labels
    def predict(
        self,
        test_embedding: np.ndarray,
    ) -> tuple[str, float]:
        """
        Predict the label and similarity score for a single embedding.
        """
        if self.train_embeddings is None or self.train_labels is None:
            raise ValueError(
                "Classifier has not been fitted."
            )
        similarities = cosine_similarity(
            test_embedding.reshape(1, -1),
            self.train_embeddings,
        )
        best_match_index = np.argmax(similarities)
        best_similarity = similarities[0][best_match_index]
        if best_similarity >= self.threshold:
            return (
                self.train_labels[best_match_index],
                float(best_similarity),
            )
        return (
            "unknown",
            float(best_similarity),
        )
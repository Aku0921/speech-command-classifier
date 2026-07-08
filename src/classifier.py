from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
class CosineSimilarityClassifier:
    """
    Classifier based on cosine similarity.
    """
    def __init__(self) -> None:
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
    ) -> str:
        """
        Predict the label for a single embedding.
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
        return self.train_labels[best_match_index]
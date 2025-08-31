"""Abstract base classifier interface for news classification."""

from abc import ABC, abstractmethod
from typing import Any

from shared.data_types import (
    BatchResult,
    FailedBatchResult,
    NewsArticle,
    NewsArticleWithLabel,
)


class BaseClassifier(ABC):
    """Abstract base class for all news classifiers."""

    def __init__(
        self,
        save_results: bool = True,
        results_dir: str = "cls_results",
    ):
        """
        Initialize base classifier.

        Args:
            save_results: Whether to save results to disk
            results_dir: Directory for saving results
        """
        self.save_results = save_results
        self.results_dir = results_dir

    @abstractmethod
    async def classify_articles(
        self, articles: list[NewsArticleWithLabel | NewsArticle], classes: list[str]
    ) -> list[BatchResult | FailedBatchResult]:
        """
        Classify articles and return results.

        Args:
            articles: List of articles to classify
            classes: List of valid class names

        Returns:
            List of BatchResult or FailedBatchResult objects
        """
        pass

    @abstractmethod
    def get_classifier_info(self) -> dict[str, Any]:
        """
        Get information about the classifier configuration.

        Returns:
            Dictionary with classifier information
        """
        pass

    def supports_training(self) -> bool:
        """
        Whether this classifier supports/requires training.

        Returns:
            True if classifier can be trained, False otherwise
        """
        return False

    async def train(self, train_data: list[NewsArticleWithLabel]) -> None:
        """
        Train the classifier if supported.

        Args:
            train_data: Training data

        Raises:
            NotImplementedError: If classifier doesn't support training
        """
        if not self.supports_training():
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support training"
            )

    def is_ready(self) -> bool:
        """
        Check if classifier is ready for classification.

        Returns:
            True if ready, False otherwise
        """
        return True

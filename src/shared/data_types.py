"""Data types for news classification system."""

import uuid
from dataclasses import dataclass
from enum import Enum


class NewsCategory(Enum):
    """News classification categories."""

    BUSINESS = "business"
    ENTERTAINMENT = "entertainment"
    POLITICS = "politics"
    SPORT = "sport"
    TECH = "tech"
    NA = "na"

    @classmethod
    def get_valid_classes(cls) -> list[str]:
        """Get list of valid classification classes (excluding NA)."""
        return [e.value for e in cls if e != cls.NA]


@dataclass
class NewsArticle:
    """News article with article ID and text."""

    article_id: int
    text: str


@dataclass
class NewsArticleWithLabel(NewsArticle):
    """Training article with article ID, text, and label."""

    label: str


@dataclass
class ClassificationResult:
    """Result from a single classification."""

    article_id: int
    prediction: str
    latency: float
    tokens_used: int
    cost: float
    true_label: str | None = None  # For train samples


@dataclass
class ClassifiedArticle:
    """Article with classification result."""

    article: NewsArticleWithLabel | NewsArticle
    prediction: str
    confidence: float | None = None


@dataclass
class ClassificationRequest:
    """Batch request containing multiple articles to classify."""

    articles: list[NewsArticleWithLabel | NewsArticle]
    request_id: str | None = None

    def __post_init__(self):
        """Generate request ID if not provided."""
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())[:8]


@dataclass
class BatchResult:
    """Result from batch classification."""

    request_id: str
    classified_articles: list[ClassifiedArticle]
    total_latency: float
    total_cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    batch_size: int

    @property
    def avg_latency_per_article(self) -> float:
        """Average latency per article in the batch."""
        return self.total_latency / self.batch_size if self.batch_size > 0 else 0.0

    @property
    def cost_per_article(self) -> float:
        """Cost per article in the batch."""
        return self.total_cost / self.batch_size if self.batch_size > 0 else 0.0

    @property
    def accuracy(self) -> float | None:
        """Accuracy if true labels are available."""
        labeled_articles = [
            ca
            for ca in self.classified_articles
            if isinstance(ca.article, NewsArticleWithLabel)
        ]

        if not labeled_articles:
            return None

        correct = sum(
            1
            for ca in labeled_articles
            if ca.prediction.lower() == ca.article.label.lower()
        )
        return correct / len(labeled_articles)

    @property
    def is_failed(self) -> bool:
        """Always returns False for successful batches."""
        return False


@dataclass
class FailedBatchResult:
    """Result for a failed batch classification."""

    request_id: str
    articles: list[NewsArticleWithLabel | NewsArticle]
    error: str
    batch_size: int

    @property
    def is_failed(self) -> bool:
        """Always returns True for failed batches."""
        return True

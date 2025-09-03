"""Route builder for creating semantic routes from training data."""

import os
import random
from collections import defaultdict
from typing import Any

from redisvl.extensions.router import Route, SemanticRouter
from redisvl.utils.vectorize import HFTextVectorizer, OpenAITextVectorizer

from shared.data_types import NewsArticleWithLabel
from utils.config_loader import VectorizerConfig
from utils.cost_calculator import EmbeddingCostCalculator
from utils.logger import get_logger

# Disable tokenizer parallelism to prevent deadlocks with async/multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RouteBuilder:
    """Builder for creating semantic routes from training data."""

    def __init__(
        self,
        vectorizer_config: VectorizerConfig,
        samples_per_class: int = 15,
        initial_threshold: float = 0.75,
        max_text_length: int = 500,
    ):
        """
        Initialize route builder.

        Args:
            vectorizer_config: Vectorizer configuration
            samples_per_class: Number of reference samples per category
            initial_threshold: Initial distance threshold for routes
            max_text_length: Maximum text length for route references
        """
        self.vectorizer_config = vectorizer_config
        self.samples_per_class = samples_per_class
        self.initial_threshold = initial_threshold
        self.max_text_length = max_text_length
        self.logger = get_logger(f"{__name__}.RouteBuilder")

        # Cost tracking
        self.total_embedding_cost = 0.0
        self.total_tokens = 0
        self.embedding_calls = 0

        # Initialize vectorizer
        try:
            self.vectorizer = self._create_vectorizer(vectorizer_config)
            self.logger.info(
                f"Initialized {vectorizer_config.type} vectorizer with model: {vectorizer_config.model}"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize vectorizer: {e}")
            raise

    def _sample_articles_by_class(
        self, train_data: list[NewsArticleWithLabel]
    ) -> dict[str, list[NewsArticleWithLabel]]:
        """
        Group and sample articles by class for route creation.

        Args:
            train_data: List of training articles

        Returns:
            Dictionary mapping class names to sampled articles
        """
        # Group articles by class
        articles_by_class = defaultdict(list)
        for article in train_data:
            articles_by_class[article.label.lower()].append(article)

        # Sample articles for each class
        sampled_articles = {}
        for class_name, articles in articles_by_class.items():
            if len(articles) <= self.samples_per_class:
                sampled_articles[class_name] = articles
            else:
                # Randomly sample articles
                sampled_articles[class_name] = random.sample(
                    articles, self.samples_per_class
                )

            self.logger.info(
                f"Class '{class_name}': {len(articles)} total, {len(sampled_articles[class_name])} sampled"
            )

        return sampled_articles

    def _prepare_reference_texts(
        self, articles: list[NewsArticleWithLabel]
    ) -> list[str]:
        """
        Prepare reference texts from articles.

        Args:
            articles: List of articles

        Returns:
            List of reference text strings
        """
        references = []
        for article in articles:
            # Truncate text if too long (legacy - should be removed)
            text = article.text[: self.max_text_length]
            if len(article.text) > self.max_text_length:
                text += "..."

            # Clean text (remove excessive whitespace)
            text = " ".join(text.split())
            references.append(text)

        return references

    def build_routes(self, train_data: list[NewsArticleWithLabel]) -> list[Route]:
        """
        Build semantic routes from training data.

        Args:
            train_data: List of training articles

        Returns:
            List of Route objects for semantic router
        """
        self.logger.info(f"Building routes from {len(train_data)} training articles")

        # Sample articles by class
        articles_by_class = self._sample_articles_by_class(train_data)
        routes = []
        for class_name, articles in articles_by_class.items():
            # Prepare reference texts
            references = self._prepare_reference_texts(articles)

            # Track embedding cost for route creation
            self._track_embedding_cost(references)

            # Create route
            route = Route(
                name=class_name,
                references=references,
                distance_threshold=self.initial_threshold,
                metadata={"category": class_name, "samples_count": len(articles)},
            )

            routes.append(route)
            self.logger.info(
                f"Created route '{class_name}' with {len(references)} references"
            )

        self.logger.info(f"Built {len(routes)} semantic routes")
        return routes

    def create_semantic_router(
        self,
        redis_url: str,
        train_data: list[NewsArticleWithLabel],
        router_name: str = "news-classification-router",
        overwrite: bool = True,
    ) -> SemanticRouter:
        """Create a complete semantic router from training data.

        Args:
            redis_url: Str representing the redis url for connection
            train_data: List of training articles
            router_name: Name for the semantic router

        Returns:
            Configured SemanticRouter instance
        """
        self.logger.info(f"Creating semantic router '{router_name}'")

        # Reset cost tracking for this router creation
        self.total_embedding_cost = 0.0
        self.total_tokens = 0
        self.embedding_calls = 0

        # Build routes
        routes = self.build_routes(train_data)

        # Create semantic router
        try:
            router = SemanticRouter(
                name=router_name,
                vectorizer=self.vectorizer,
                routes=routes,
                redis_url=redis_url,
                overwrite=overwrite,
            )

            self.logger.info(f"Created semantic router with {len(routes)} routes")
            return router

        except Exception as e:
            self.logger.error(f"Failed to create semantic router: {e}")
            raise

    def get_route_summary(self, routes: list[Route]) -> dict[str, Any]:
        """
        Get summary information about created routes.

        Args:
            routes: List of routes

        Returns:
            Summary dictionary
        """
        summary = {
            "total_routes": len(routes),
            "embedding_model": self.vectorizer_config.model,
            "samples_per_class": self.samples_per_class,
            "initial_threshold": self.initial_threshold,
            "routes": {},
        }

        for route in routes:
            summary["routes"][route.name] = {
                "references_count": len(route.references),
                "distance_threshold": route.distance_threshold,
                "metadata": route.metadata,
            }

        return summary

    def _create_vectorizer(
        self, config: VectorizerConfig
    ) -> HFTextVectorizer | OpenAITextVectorizer:
        """
        Create vectorizer based on configuration.

        Args:
            config: Vectorizer configuration

        Returns:
            Configured vectorizer instance
        """
        if config.type.lower() == "openai":
            api_config = {}
            if config.api_key_env:
                api_key = os.getenv(config.api_key_env)
                if not api_key:
                    raise ValueError(
                        f"OpenAI API key not found in environment variable: {config.api_key_env}"
                    )
                api_config["api_key"] = api_key

            return OpenAITextVectorizer(model=config.model, api_config=api_config)
        elif config.type.lower() == "huggingface":
            return HFTextVectorizer(model=config.model)
        else:
            raise ValueError(f"Unsupported vectorizer type: {config.type}")

    def _track_embedding_cost(self, texts: list[str]) -> None:
        """
        Track embedding cost for OpenAI models.

        Args:
            texts: List of texts being embedded
        """
        if (
            self.vectorizer_config.type.lower() == "openai"
            and self.vectorizer_config.track_usage
        ):
            cost_info = EmbeddingCostCalculator.calculate_batch_embedding_cost(
                texts, self.vectorizer_config.model
            )
            self.total_tokens += cost_info["total_tokens"]
            self.total_embedding_cost += cost_info["total_cost"]
            self.embedding_calls += len(texts)

    def get_cost_info(self) -> dict[str, Any]:
        """
        Get cost and usage information.

        Returns:
            Dictionary with cost breakdown
        """
        return {
            "vectorizer_type": self.vectorizer_config.type,
            "model": self.vectorizer_config.model,
            "total_embedding_calls": self.embedding_calls,
            "total_tokens": self.total_tokens,
            "total_embedding_cost": self.total_embedding_cost,
            "avg_cost_per_call": self.total_embedding_cost / self.embedding_calls
            if self.embedding_calls > 0
            else 0.0,
        }

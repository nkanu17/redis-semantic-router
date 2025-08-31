"""Route builder for creating semantic routes from training data."""

import os
import random
from collections import defaultdict
from typing import Any

from redisvl.extensions.router import Route, SemanticRouter
from redisvl.utils.vectorize import HFTextVectorizer

from shared.data_types import NewsArticleWithLabel
from utils.logger import get_logger

# Disable tokenizer parallelism to prevent deadlocks with async/multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RouteBuilder:
    """Builder for creating semantic routes from training data."""

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        samples_per_class: int = 15,
        initial_threshold: float = 0.75,
        max_text_length: int = 500,
    ):
        """
        Initialize route builder.

        Args:
            embedding_model: HuggingFace embedding model name
            samples_per_class: Number of reference samples per category
            initial_threshold: Initial distance threshold for routes
            max_text_length: Maximum text length for route references
        """
        self.embedding_model = embedding_model
        self.samples_per_class = samples_per_class
        self.initial_threshold = initial_threshold
        self.max_text_length = max_text_length
        self.logger = get_logger(f"{__name__}.RouteBuilder")

        # Initialize vectorizer
        try:
            self.vectorizer = HFTextVectorizer(model=embedding_model)
            self.logger.info(f"Initialized vectorizer with model: {embedding_model}")
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
        # TODO: I am still not sure if truncation is necessary, research.
        # TODO: On one hand, a sample is probably enough and could cut down noise, on the other hand
        # TODO: If I use an llm embedding it may not make that much of a difference.
        references = []
        for article in articles:
            # Truncate text if too long
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
        # TODO: Do I need an 'other' class? Training does not have it. Might increase noise.
        routes = []
        for class_name, articles in articles_by_class.items():
            # Prepare reference texts
            references = self._prepare_reference_texts(articles)

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

        # Build routes
        routes = self.build_routes(train_data)

        # Create semantic router
        try:
            router = SemanticRouter(
                name=router_name,
                vectorizer=self.vectorizer,
                routes=routes,
                redis_url=redis_url,
                overwrite=True,
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
            "embedding_model": self.embedding_model,
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

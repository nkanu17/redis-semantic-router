"""Redis-based semantic classification using vector similarity."""

import time
from typing import Any

from redis_retrieval_optimizer.threshold_optimization import RouterThresholdOptimizer
from redisvl.extensions.router import SemanticRouter

from semantic_router.route_builder import RouteBuilder
from shared.base_classifier import BaseClassifier
from shared.data_types import (
    BatchResult,
    ClassifiedArticle,
    FailedBatchResult,
    NewsArticle,
    NewsArticleWithLabel,
    NewsCategory,
)
from shared.metrics import calculate_batch_metrics
from shared.results_storage import ResultsStorage
from utils.config_loader import VectorizerConfig
from utils.cost_calculator import EmbeddingCostCalculator
from utils.logger import get_logger
from utils.redis_client import RedisConfig


class RedisSemanticClassifier(BaseClassifier):
    """Redis-based semantic classifier using vector similarity search."""

    def __init__(
        self,
        redis_config: RedisConfig,
        vectorizer_config: VectorizerConfig,
        samples_per_class: int = 15,
        initial_threshold: float = 0.75,
        router_name: str = "news-classification-router",
        save_results: bool = True,
        results_dir: str = "redis_cls_results",
        pipeline_config: dict[str, Any] | None = None,
        overwrite_existing: bool = True,
    ):
        """Initialize Redis semantic classifier.

        Args:
            redis_config: Redis connection configuration
            vectorizer_config: Vectorizer configuration (HF or OpenAI)
            samples_per_class: Number of reference samples per class
            initial_threshold: Initial distance threshold for matching
            router_name: Name for the semantic router
            save_results: Whether to save results to disk
            results_dir: Directory for saving results
        """
        super().__init__(save_results=save_results, results_dir=results_dir)

        self.redis_config = redis_config
        self.vectorizer_config = vectorizer_config
        self.samples_per_class = samples_per_class
        self.initial_threshold = initial_threshold
        self.router_name = router_name
        self.overwrite_existing = overwrite_existing
        self.logger = get_logger(f"{__name__}.RedisSemanticClassifier")

        # Cost tracking for classification phase
        self.classification_cost = 0.0
        self.classification_tokens = 0

        # Cost tracking for training phase
        self.training_cost = 0.0
        self.training_tokens = 0

        # Initialize results storage
        if self.save_results:
            self.results_storage = ResultsStorage(results_dir, pipeline_config)
        else:
            self.results_storage = None

        # Initialize router (will be set during training)
        self.router: SemanticRouter | None = None
        self.is_trained = False

    async def train(self, train_data: list[NewsArticleWithLabel]) -> None:
        """
        Train the semantic router with training data.

        Args:
            train_data: List of training articles
        """
        self.logger.info(f"Training semantic router with {len(train_data)} articles")

        try:
            # Check Redis connection
            if not self.redis_config.health_check():
                raise ConnectionError("Redis connection failed")

            # Skip existing router check during training since we always overwrite anyway
            # Only check for existing router during classification, not training

            # Build routes from training data
            route_builder = RouteBuilder(
                vectorizer_config=self.vectorizer_config,
                samples_per_class=self.samples_per_class,
                initial_threshold=self.initial_threshold,
            )

            # Create semantic router
            self.router = route_builder.create_semantic_router(
                redis_url=self.redis_config.get_url(),
                train_data=train_data,
                router_name=self.router_name,
                overwrite=self.overwrite_existing,
            )

            # Get training cost information from route builder
            cost_info = route_builder.get_cost_info()
            self.training_cost = cost_info["total_embedding_cost"]
            self.training_tokens = cost_info["total_tokens"]

            # Get route summary for logging
            routes = route_builder.build_routes(train_data)
            summary = route_builder.get_route_summary(routes)

            self.logger.info(
                f"Router training completed: {summary['total_routes']} routes created"
            )
            for route_name, route_info in summary["routes"].items():
                self.logger.info(
                    f"Route '{route_name}': {route_info['references_count']} references, "
                    f"threshold: {route_info['distance_threshold']}"
                )

            self.is_trained = True

        except Exception as e:
            self.logger.error(f"Failed to train semantic router: {e}")
            raise

    def classify_article(
        self, article: NewsArticleWithLabel | NewsArticle
    ) -> tuple[str, float]:
        """
        Classify a single article using semantic routing.

        Args:
            article: Article to classify

        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if not self.is_trained or self.router is None:
            raise RuntimeError("Router must be trained before classification")

        try:
            # Track classification cost for OpenAI
            if (
                self.vectorizer_config.type.lower() == "openai"
                and self.vectorizer_config.track_usage
            ):
                tokens, cost = EmbeddingCostCalculator.calculate_openai_embedding_cost(
                    article.text, self.vectorizer_config.model
                )
                self.classification_tokens += tokens
                self.classification_cost += cost

            # Route the article text
            route_match = self.router(article.text)

            if route_match is not None:
                category = route_match.name
                if not category:
                    category = NewsCategory.NA.value
                    confidence = 0.0
                else:
                    # Convert distance to confidence (lower distance = higher confidence)
                    confidence = max(0.0, 1.0 - route_match.distance)
            else:
                # No route matched - use fallback
                category = NewsCategory.NA.value
                confidence = 0.0
                self.logger.warning(
                    f"No route matched for article {article.article_id}"
                )

            return category, confidence

        except Exception as e:
            self.logger.error(f"Error classifying article {article.article_id}: {e}")
            return NewsCategory.NA.value, 0.0

    def classify_articles(
        self, articles: list[NewsArticleWithLabel | NewsArticle], classes: list[str]
    ) -> list[BatchResult | FailedBatchResult]:
        """
        Classify multiple articles using semantic routing.

        Args:
            articles: List of articles to classify
            classes: List of valid class names (for fallback)

        Returns:
            List containing a single BatchResult or FailedBatchResult
        """
        if not self.is_trained or self.router is None:
            error = FailedBatchResult(
                request_id="redis_classification",
                articles=articles,
                error="Router not trained",
                batch_size=len(articles),
            )
            return [error]

        self.logger.info(
            f"Classifying {len(articles)} articles with Redis semantic router"
        )
        start_time = time.time()

        try:
            classified_articles = []
            total_confidence = 0.0

            for article in articles:
                predicted_category, confidence = self.classify_article(article)

                # Fallback to first valid class if prediction is NA or invalid
                if (
                    predicted_category == NewsCategory.NA.value
                    or predicted_category.lower() not in [c.lower() for c in classes]
                ):
                    predicted_category = classes[0]
                    confidence = 0.1  # Low confidence for fallback

                classified_article = ClassifiedArticle(
                    article=article,
                    prediction=predicted_category.lower(),
                    confidence=confidence,
                )
                classified_articles.append(classified_article)
                total_confidence += confidence

            end_time = time.time()

            # Create batch result with embedding costs
            batch_result = BatchResult(
                request_id="redis_classification",
                classified_articles=classified_articles,
                total_latency=end_time - start_time,
                total_cost=self.classification_cost,
                prompt_tokens=0,  # No LLM tokens used
                completion_tokens=0,
                total_tokens=self.classification_tokens,
                batch_size=len(articles),
            )

            # Save results if enabled
            if self.save_results and self.results_storage:
                try:
                    run_id = self.results_storage.generate_run_id()
                    run_id = self.results_storage.create_run_directory(run_id)

                    # Classifier configuration
                    classifier_config = {
                        "classifier_type": "redis_semantic",
                        "embedding_model": self.vectorizer_config.model,
                        "samples_per_class": self.samples_per_class,
                        "initial_threshold": self.initial_threshold,
                        "router_name": self.router_name,
                        "total_articles_processed": len(articles),
                        "avg_confidence": total_confidence / len(articles)
                        if articles
                        else 0.0,
                    }

                    # Save results
                    self.results_storage.save_classifications(run_id, [batch_result])

                    # Calculate metrics for saving

                    metrics = calculate_batch_metrics([batch_result])
                    self.results_storage.save_metrics(
                        run_id, metrics, classifier_config
                    )

                    self.logger.info(
                        f"Redis classification results saved to run: {run_id}"
                    )

                except Exception as e:
                    self.logger.error(
                        f"Failed to save Redis classification results: {e}"
                    )

            return [batch_result]

        except Exception as e:
            self.logger.error(f"Redis classification failed: {e}")
            failed_result = FailedBatchResult(
                request_id="redis_classification",
                articles=articles,
                error=str(e),
                batch_size=len(articles),
            )
            return [failed_result]

    def get_router_info(self) -> dict[str, Any]:
        """
        Get information about the trained router.

        Returns:
            Dictionary with router information
        """
        if not self.is_trained or self.router is None:
            return {"status": "not_trained"}

        try:
            info = {
                "status": "trained",
                "router_name": self.router_name,
                "vectorizer_type": self.vectorizer_config.type,
                "embedding_model": self.vectorizer_config.model,
                "total_routes": len(self.router.routes),
                "routes": {},
            }

            for route in self.router.routes:
                info["routes"][route.name] = {
                    "references_count": len(route.references),
                    "distance_threshold": route.distance_threshold,
                    "metadata": route.metadata,
                }

            return info

        except Exception as e:
            self.logger.error(f"Failed to get router info: {e}")
            return {"status": "error", "error": str(e)}

    def optimize_thresholds(self, validation_data: list[NewsArticleWithLabel]) -> None:
        """
        Optimize route thresholds using training data.

        Args:
            validation_data: List of training articles for threshold optimization
        """
        if not self.is_trained or self.router is None:
            raise RuntimeError("Router must be trained before threshold optimization")

        self.logger.info(
            f"Optimizing thresholds with {len(validation_data)} training samples"
        )

        try:
            # Prepare test data in the format expected by optimizer
            data = [
                {"query": article.text, "query_match": article.label.lower()}
                for article in validation_data
            ]

            # Optimize thresholds
            optimizer = RouterThresholdOptimizer(self.router, data)
            optimizer.optimize()
            # Ending thresholds: {'sport': 0.957575757575758, 'business': 0.44444444444444453, 'politics': 0.5636363636363642, 'tech': 0.77979797979798, 'entertainment': 0.5797979797979796}
            # Ending thresholds: {'sport': 0.03646655696728547, 'entertainment': 0.9252525252525253, 'tech': 0.5333333333333334, 'politics': 0.4646464646464646, 'business': 0.8727272727272728}
            self.logger.info("Threshold optimization completed")

        except Exception as e:
            self.logger.error(f"Failed to optimize thresholds: {e}")
            raise

    def apply_threshold_overrides(self, threshold_overrides: dict[str, float]) -> None:
        """
        Apply threshold overrides from configuration.

        Args:
            threshold_overrides: Dictionary mapping route names to threshold values
        """
        if not self.is_trained or self.router is None:
            raise RuntimeError(
                "Router must be trained before applying threshold overrides"
            )

        self.logger.info(
            f"Applying threshold overrides for {len(threshold_overrides)} routes"
        )

        try:
            for route in self.router.routes:
                if route.name in threshold_overrides:
                    old_threshold = route.distance_threshold
                    route.distance_threshold = threshold_overrides[route.name]
                    self.logger.info(
                        f"Route '{route.name}': {old_threshold:.3f} → {route.distance_threshold:.3f}"
                    )
                else:
                    self.logger.info(
                        f"Route '{route.name}': keeping threshold {route.distance_threshold:.3f}"
                    )

            self.logger.info("Threshold overrides applied successfully")

        except Exception as e:
            self.logger.error(f"Failed to apply threshold overrides: {e}")
            raise

    def supports_training(self) -> bool:
        """Redis classifier supports training."""
        return True

    def is_ready(self) -> bool:
        """Check if classifier is ready for classification."""
        return self.is_trained and self.router is not None

    def load_existing_router(self) -> bool:
        """Try to load existing router from Redis.

        Returns:
            True if router was loaded successfully, False otherwise
        """
        self.logger.info(f"Checking for existing router '{self.router_name}' in Redis")

        try:
            if not self.redis_config.health_check():
                self.logger.error(
                    "Redis connection failed, cannot load existing router"
                )
                return False

            self.router = SemanticRouter.from_existing(
                name=self.router_name, redis_url=self.redis_config.get_url()
            )

            # Check if router loaded properly
            if self.router and len(self.router.routes) > 0:
                self.is_trained = True
                self.logger.info(
                    f"Loaded existing router with {len(self.router.routes)} routes"
                )

                for route in self.router.routes:
                    self.logger.info(
                        f"  Route '{route.name}': {len(route.references)} references, "
                        f"threshold: {route.distance_threshold}"
                    )

                return True
            self.logger.warning("Could not find existing routes.")
            return False
        except Exception as e:
            self.logger.error(f"Failed to check for existing router: {e}")
            return False

    def get_classifier_info(self) -> dict[str, Any]:
        """
        Get information about the classifier configuration.

        Returns:
            Dictionary with classifier information
        """
        info = {
            "classifier_type": "redis_semantic",
            "vectorizer_type": self.vectorizer_config.type,
            "embedding_model": self.vectorizer_config.model,
            "samples_per_class": self.samples_per_class,
            "initial_threshold": self.initial_threshold,
            "router_name": self.router_name,
            "supports_training": self.supports_training(),
            "is_ready": self.is_ready(),
            "is_trained": self.is_trained,
            "training_cost": self.training_cost,
            "training_tokens": self.training_tokens,
            "classification_cost": self.classification_cost,
            "classification_tokens": self.classification_tokens,
        }

        if self.router is not None:
            info.update(
                {
                    "total_routes": len(self.router.routes),
                    "redis_url": self.redis_config.get_url(),
                }
            )

        return info

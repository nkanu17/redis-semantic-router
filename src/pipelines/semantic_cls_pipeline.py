"""Semantic router classification pipeline."""

from typing import Any

from semantic_router.redis_router import RedisSemanticClassifier
from shared.data_types import BatchResult, FailedBatchResult, NewsCategory
from shared.metrics import calculate_batch_metrics
from utils.config_loader import ConfigLoader
from utils.data_loader import NewsDataLoader
from utils.logger import get_logger
from utils.redis_client import RedisConfig


class SemanticClassificationPipeline:
    """Pipeline for classification using existing trained semantic router."""

    def __init__(self, config_path: str = "../config/pipeline_config.yaml"):
        """
        Initialize semantic classification pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.logger = get_logger(f"{__name__}.SemanticClassificationPipeline")

    async def run(self) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
        """
        Run semantic router classification pipeline on test data using config settings.

        Returns:
            Tuple of (results, metrics)
        """
        return await self.run_classification(use_test_data=True)

    async def run_on_training_data(
        self,
    ) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
        """
        Run semantic router classification pipeline on training data using config settings.

        Returns:
            Tuple of (results, metrics)
        """
        return await self.run_classification(use_test_data=False)

    async def run_classification(
        self, use_test_data: bool = True
    ) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
        """
        Internal method to run classification on either test or training data.

        Args:
            use_test_data: If True, use test data; if False, use training data

        Returns:
            Tuple of (results, metrics)
        """
        self.logger.info("=== SEMANTIC ROUTER CLASSIFICATION PIPELINE ===")
        self.logger.info(f"Router name: {self.config.semantic_router.router_name}")

        # Initialize data loader
        data_loader = NewsDataLoader(
            data_dir=self.config.data.dataset_path,
            train_file=self.config.data.train_file,
            validation_file=self.config.data.validation_file,
        )

        # Load data
        if use_test_data:
            data = data_loader.load_test_samples()  # validation_data.csv
            data_type = "validation"
        else:
            data = data_loader.load_train_samples()  # train_data.csv
            data_type = "training"

        classes = NewsCategory.get_valid_classes()

        self.logger.info(f"Classifying {len(data)} {data_type} samples")
        self.logger.info(f"Classes: {classes}")

        # Setup Redis configuration
        redis_config = RedisConfig(
            host=self.config.semantic_router.redis.host,
            port=self.config.semantic_router.redis.port,
            db=self.config.semantic_router.redis.db,
            socket_timeout=self.config.semantic_router.redis.socket_timeout,
            socket_connect_timeout=self.config.semantic_router.redis.socket_connect_timeout,
            retry_on_timeout=self.config.semantic_router.redis.retry_on_timeout,
            decode_responses=self.config.semantic_router.redis.decode_responses,
        )

        # Create classifier
        classifier = RedisSemanticClassifier(
            redis_config=redis_config,
            embedding_model=self.config.semantic_router.embedding_model,
            samples_per_class=self.config.semantic_router.route_config.samples_per_class,
            initial_threshold=self.config.semantic_router.route_config.initial_threshold,
            router_name=self.config.semantic_router.router_name,
            save_results=self.config.semantic_router.save_results,
            results_dir=self.config.semantic_router.results_dir,
        )

        # Load existing router
        if not classifier.load_existing_router():
            raise RuntimeError(
                f"No trained router '{self.config.semantic_router.router_name}' found in Redis. "
                "Run 'python main.py train_router' first."
            )

        # Apply threshold overrides if provided
        if self.config.semantic_router.route_config.threshold_overrides:
            self.logger.info("=== APPLYING THRESHOLD OVERRIDES ===")
            classifier.apply_threshold_overrides(
                self.config.semantic_router.route_config.threshold_overrides
            )

        # Run classification
        results = classifier.classify_articles(data, classes)

        # Calculate metrics
        metrics = calculate_batch_metrics(results)
        metrics["pipeline_name"] = "semantic_classification"
        metrics["classifier_info"] = classifier.get_classifier_info()

        self.logger.info("=== SEMANTIC CLASSIFICATION COMPLETED ===")
        return results, metrics

"""Semantic router training pipeline."""

from typing import Any

from semantic_routing.redis_router import RedisSemanticClassifier
from shared.data_types import NewsCategory
from utils.config_loader import ConfigLoader
from utils.data_loader import NewsDataLoader
from utils.logger import get_logger
from utils.redis_client import RedisConfig


class SemanticTrainingPipeline:
    """Pipeline for training semantic router."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize semantic training pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.logger = get_logger(f"{__name__}.SemanticTrainingPipeline")

    async def run(self, force_retrain: bool = False) -> dict[str, Any]:
        """
        Run semantic router training pipeline using config settings.

        Args:
            force_retrain: Force retraining even if router exists

        Returns:
            Training summary information
        """

        self.logger.info("=== SEMANTIC ROUTER TRAINING PIPELINE ===")
        self.logger.info(
            f"Embedding model: {self.config.semantic_router.embedding_model}"
        )
        self.logger.info(
            f"Samples per class: {self.config.semantic_router.route_config.samples_per_class}"
        )
        self.logger.info(f"Router name: {self.config.semantic_router.router_name}")

        # Initialize data loader
        data_loader = NewsDataLoader(
            data_dir=self.config.data.dataset_path,
            train_file=self.config.data.train_file,
            validation_file=self.config.data.validation_file,
        )

        # Load training data
        train_data = data_loader.load_train_samples()
        validation_data = data_loader.load_test_samples()
        classes = NewsCategory.get_valid_classes()

        self.logger.info(f"Training with {len(train_data)} samples")
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
            save_results=False,  # Don't save during training
            results_dir=self.config.semantic_router.results_dir,
        )

        # Train the router
        await classifier.train(train_data, force_retrain=force_retrain)

        # Optimize thresholds if enabled
        if self.config.semantic_router.route_config.optimize_thresholds:
            self.logger.info("=== OPTIMIZING THRESHOLDS ===")

            # Use the same training data for threshold optimization
            self.logger.info(
                f"Using {len(validation_data)} samples for threshold optimization"
            )
            classifier.optimize_thresholds(validation_data)

        # Get training summary
        training_summary = classifier.get_classifier_info()
        training_summary["training_samples"] = len(train_data)
        training_summary["classes"] = classes

        self.logger.info("=== SEMANTIC ROUTER TRAINING COMPLETED ===")
        self.logger.info(f"Router: {training_summary['router_name']}")
        self.logger.info(f"Total routes: {training_summary.get('total_routes', 'N/A')}")
        self.logger.info(f"Embedding model: {training_summary['embedding_model']}")

        return training_summary

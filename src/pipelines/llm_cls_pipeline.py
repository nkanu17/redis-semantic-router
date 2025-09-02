"""LLM classification pipeline."""

from typing import Any

from llm_classifier.llm_classifier import LLMClassifier
from shared.data_types import BatchResult, FailedBatchResult, NewsCategory
from shared.metrics import calculate_batch_metrics
from utils.config_loader import ConfigLoader
from utils.data_loader import NewsDataLoader
from utils.logger import get_logger


class LLMClassificationPipeline:
    """Pipeline for running LLM-based classification."""

    def __init__(self, config_path: str = "../config/pipeline_config.yaml"):
        """
        Initialize LLM classification pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.logger = get_logger(f"{__name__}.LLMClassificationPipeline")

    async def run(self) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
        """
        Run LLM classification pipeline on test data using config settings.

        Returns:
            Tuple of (results, metrics)
        """
        return await self.run_classification(use_test_data=True)

    async def run_on_training_data(
        self,
    ) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
        """
        Run LLM classification pipeline on training data using config settings.

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
        self.logger.info("=== LLM CLASSIFICATION PIPELINE ===")
        self.logger.info(f"Model: {self.config.llm_classifier.model_name}")
        self.logger.info(f"Batch size: {self.config.llm_classifier.batch_size}")
        self.logger.info(f"Max concurrent: {self.config.llm_classifier.max_concurrent}")

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

        # Create classifier
        classifier = LLMClassifier(
            model_name=self.config.llm_classifier.model_name,
            max_concurrent=self.config.llm_classifier.max_concurrent,
            batch_size=self.config.llm_classifier.batch_size,
            max_retries=self.config.llm_classifier.max_retries,
            save_results=self.config.llm_classifier.save_results,
            results_dir=self.config.llm_classifier.results_dir,
            temperature=self.config.llm_classifier.temperature,
            max_tokens=self.config.llm_classifier.max_tokens,
            pipeline_config={
                "llm_classifier": self.config_loader.raw_config.get(
                    "llm_classifier", {}
                ),
                "data": self.config_loader.raw_config.get("data", {}),
            },
        )

        # Run classification
        results = await classifier.classify_articles(data, classes)

        # Calculate metrics
        metrics = calculate_batch_metrics(results)
        metrics["pipeline_name"] = "llm_classification"
        metrics["classifier_info"] = classifier.get_classifier_info()

        self.logger.info("=== LLM CLASSIFICATION COMPLETED ===")
        return results, metrics

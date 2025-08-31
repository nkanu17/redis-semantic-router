"""Results storage utility for saving classification outputs and metrics."""

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from shared.data_types import BatchResult, FailedBatchResult, NewsArticleWithLabel
from utils.logger import get_logger


class ResultsStorage:
    """Utility class for saving classification results to disk."""

    def __init__(self, base_dir: str = "cls_results"):
        """
        Initialize results storage.

        Args:
            base_dir: Base directory for storing results
        """
        self.base_dir = Path(base_dir)
        self.logger = get_logger(f"{__name__}.ResultsStorage")

        # Ensure base directory exists
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create base directory {base_dir}: {e}")

    def generate_run_id(self) -> str:
        """Generate unique run ID with timestamp and random suffix."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{timestamp}_{str(uuid.uuid4())}"

    def create_run_directory(self, run_id: str | None = None) -> str:
        """
        Create directory for a classification run.

        Args:
            run_id: Optional run ID. If None, generates one automatically.

        Returns:
            The run ID used

        Raises:
            Exception: If directory creation fails
        """
        if run_id is None:
            run_id = self.generate_run_id()

        run_dir = self.base_dir / run_id
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created run directory: {run_dir}")
            return run_id
        except Exception as e:
            self.logger.error(f"Failed to create run directory {run_dir}: {e}")
            raise

    def save_metrics(
        self,
        run_id: str,
        metrics: dict[str, Any],
        classifier_config: dict[str, Any] | None = None,
    ) -> None:
        """
        Save performance metrics and run metadata to JSON files.

        Args:
            run_id: Run identifier
            metrics: Performance metrics dictionary
            classifier_config: Classifier configuration parameters
        """
        run_dir = self.base_dir / run_id

        if not run_dir.exists():
            self.logger.warning(f"Run directory {run_dir} doesn't exist. Creating it.")
            run_dir.mkdir(parents=True, exist_ok=True)

        # Save comprehensive metrics
        metrics_file = run_dir / "metrics.json"
        run_info_file = run_dir / "run_info.json"

        try:
            # Prepare metrics data
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "performance_metrics": {},
                "cost_metrics": {},
                "latency_metrics": {},
                "classification_metrics": {},
            }

            # Organize metrics by category
            performance_keys = [
                "total_articles",
                "successful_articles",
                "failed_articles",
                "total_batches",
                "successful_batches",
                "failed_batches",
                "success_rate",
                "accuracy",
                "correct_predictions",
                "total_predictions",
            ]

            cost_keys = [
                "total_cost",
                "cost_per_article",
                "total_tokens",
                "total_prompt_tokens",
                "total_completion_tokens",
                "avg_tokens_per_article",
                "avg_prompt_tokens_per_article",
                "avg_completion_tokens_per_article",
            ]

            latency_keys = [
                "total_latency",
                "avg_batch_latency",
                "avg_latency_per_article",
                "total_processing_time",
            ]

            classification_keys = [
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "precision_weighted",
                "recall_weighted",
                "f1_weighted",
                "per_class_metrics",
                "classification_report",
            ]

            # Distribute metrics into categories
            for key, value in metrics.items():
                if key in performance_keys:
                    metrics_data["performance_metrics"][key] = value
                elif key in cost_keys:
                    metrics_data["cost_metrics"][key] = value
                elif key in latency_keys:
                    metrics_data["latency_metrics"][key] = value
                elif key in classification_keys:
                    metrics_data["classification_metrics"][key] = value
                else:
                    # Put remaining metrics in performance
                    metrics_data["performance_metrics"][key] = value

            # Save metrics
            with open(metrics_file, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)

            # Save run info separately
            run_info = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "classifier_config": classifier_config or {},
                "files_generated": {
                    "metrics": "metrics.json",
                    "classifications": "classifications.csv",
                    "run_info": "run_info.json",
                },
            }

            with open(run_info_file, "w", encoding="utf-8") as f:
                json.dump(run_info, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved metrics to {metrics_file}")
            self.logger.info(f"Saved run info to {run_info_file}")

        except Exception as e:
            self.logger.error(f"Failed to save metrics for run {run_id}: {e}")

    def save_classifications(
        self, run_id: str, results: list[BatchResult | FailedBatchResult]
    ) -> None:
        """
        Save article-level classification results to CSV.

        Args:
            run_id: Run identifier
            results: List of batch results (successful and failed)
        """
        run_dir = self.base_dir / run_id

        if not run_dir.exists():
            self.logger.warning(f"Run directory {run_dir} doesn't exist. Creating it.")
            run_dir.mkdir(parents=True, exist_ok=True)

        classifications_file = run_dir / "classifications.csv"

        try:
            with open(
                classifications_file, "w", newline="", encoding="utf-8"
            ) as csvfile:
                fieldnames = [
                    "batch_id",
                    "article_id",
                    "prediction",
                    "true_label",
                    "correct",
                    "batch_status",
                    "article_text_preview",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                # Process successful results
                for batch_result in results:
                    if isinstance(batch_result, BatchResult):
                        # Successful batch
                        for classified_article in batch_result.classified_articles:
                            article = classified_article.article
                            prediction = classified_article.prediction

                            # Get true label if available
                            true_label = None
                            correct = None
                            if isinstance(article, NewsArticleWithLabel):
                                true_label = article.label.lower()
                                correct = prediction.lower() == true_label.lower()

                            # Truncate article text for preview
                            text_preview = (
                                article.text[:100] + "..."
                                if len(article.text) > 100
                                else article.text
                            )
                            text_preview = text_preview.replace("\n", " ").replace(
                                "\r", " "
                            )

                            writer.writerow(
                                {
                                    "batch_id": batch_result.request_id,
                                    "article_id": article.article_id,
                                    "prediction": prediction,
                                    "true_label": true_label,
                                    "correct": correct,
                                    "batch_status": "success",
                                    "article_text_preview": text_preview,
                                }
                            )

                    elif isinstance(batch_result, FailedBatchResult):
                        # Failed batch - add entries with empty predictions
                        for article in batch_result.articles:
                            true_label = None
                            if isinstance(article, NewsArticleWithLabel):
                                true_label = article.label.lower()

                            text_preview = (
                                article.text[:100] + "..."
                                if len(article.text) > 100
                                else article.text
                            )
                            text_preview = text_preview.replace("\n", " ").replace(
                                "\r", " "
                            )

                            writer.writerow(
                                {
                                    "batch_id": batch_result.request_id,
                                    "article_id": article.article_id,
                                    "prediction": None,
                                    "true_label": true_label,
                                    "correct": None,
                                    "batch_status": f"failed: {batch_result.error}",
                                    "article_text_preview": text_preview,
                                }
                            )

            self.logger.info(f"Saved classifications to {classifications_file}")

        except Exception as e:
            self.logger.error(f"Failed to save classifications for run {run_id}: {e}")

    def get_run_summary(self, run_id: str) -> dict[str, Any] | None:
        """
        Load and return summary of a completed run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary with run summary or None if not found
        """
        run_dir = self.base_dir / run_id

        if not run_dir.exists():
            self.logger.error(f"Run directory {run_dir} not found")
            return None

        try:
            summary = {}

            # Load run info
            run_info_file = run_dir / "run_info.json"
            if run_info_file.exists():
                with open(run_info_file, "r", encoding="utf-8") as f:
                    summary["run_info"] = json.load(f)

            # Load metrics
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    summary["metrics"] = json.load(f)

            # Check if classifications file exists
            classifications_file = run_dir / "classifications.csv"
            summary["has_classifications"] = classifications_file.exists()
            summary["classifications_file"] = (
                str(classifications_file) if classifications_file.exists() else None
            )

            return summary

        except Exception as e:
            self.logger.error(f"Failed to load run summary for {run_id}: {e}")
            return None

    def list_runs(self) -> list[str]:
        """
        List all available run IDs.

        Returns:
            List of run ID strings, sorted by creation time (newest first)
        """
        if not self.base_dir.exists():
            return []

        try:
            run_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
            # Sort by modification time, newest first
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return [d.name for d in run_dirs]
        except Exception as e:
            self.logger.error(f"Failed to list runs: {e}")
            return []

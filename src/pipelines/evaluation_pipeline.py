"""Evaluation and comparison pipeline."""

from typing import Any

from shared.results_storage import ResultsStorage
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


class EvaluationPipeline:
    """Pipeline for evaluating and comparing classification results."""

    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize evaluation pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.logger = get_logger(f"{__name__}.EvaluationPipeline")

    def compare_results(
        self,
        baseline_dir: str | None = None,
        semantic_dir: str | None = None,
        baseline_run: str | None = None,
        semantic_run: str | None = None,
        save_results: bool = True,
    ) -> dict[str, Any]:
        """
        Compare baseline and semantic router results.

        Args:
            baseline_dir: Baseline results directory (default from config)
            semantic_dir: Semantic results directory (default from config)
            baseline_run: Specific baseline run ID (default: latest)
            semantic_run: Specific semantic run ID (default: latest)
            save_results: Whether to save comparison results

        Returns:
            Comparison results dictionary
        """
        # Use config defaults if not specified
        baseline_dir = baseline_dir or self.config.llm_classifier.results_dir
        semantic_dir = semantic_dir or self.config.semantic_router.results_dir

        self.logger.info("=== EVALUATION PIPELINE ===")
        self.logger.info(f"Comparing {baseline_dir} vs {semantic_dir}")

        # Load metrics from both runs
        baseline_metrics = self._load_latest_metrics(baseline_dir, baseline_run)
        semantic_metrics = self._load_latest_metrics(semantic_dir, semantic_run)

        if not baseline_metrics or not semantic_metrics:
            raise ValueError("Could not load metrics for comparison")

        # Perform comparison
        comparison = self._calculate_comparison(baseline_metrics, semantic_metrics)

        # Log comparison results
        self._log_comparison_results(comparison)

        # Save comparison results if requested
        if save_results:
            self._save_comparison_results(
                comparison, baseline_metrics, semantic_metrics
            )

        return comparison

    def list_available_runs(self) -> dict[str, list[str]]:
        """
        List all available runs in result directories.

        Returns:
            Dictionary mapping directory names to run lists
        """
        available_runs = {}

        # Check baseline results
        try:
            baseline_storage = ResultsStorage(self.config.llm_classifier.results_dir)
            available_runs["baseline"] = baseline_storage.list_runs()
        except Exception as e:
            self.logger.warning(f"Could not list baseline runs: {e}")
            available_runs["baseline"] = []

        # Check semantic router results
        try:
            semantic_storage = ResultsStorage(self.config.semantic_router.results_dir)
            available_runs["semantic"] = semantic_storage.list_runs()
        except Exception as e:
            self.logger.warning(f"Could not list semantic runs: {e}")
            available_runs["semantic"] = []

        # Log summary
        self.logger.info("=== AVAILABLE RUNS ===")
        for result_type, runs in available_runs.items():
            self.logger.info(f"{result_type}: {len(runs)} runs")
            if runs:
                self.logger.info(f"  Latest: {runs[0]}")

        return available_runs

    def _load_latest_metrics(
        self, results_dir: str, run_id: str | None = None
    ) -> dict[str, Any] | None:
        """Load metrics from specified or latest run."""
        try:
            storage = ResultsStorage(results_dir)

            if run_id is None:
                runs = storage.list_runs()
                if not runs:
                    self.logger.error(f"No runs found in {results_dir}")
                    return None
                run_id = runs[0]  # Latest run

            summary = storage.get_run_summary(run_id)
            if summary and "metrics" in summary:
                self.logger.info(f"Loaded metrics from {results_dir}/{run_id}")
                return summary["metrics"]
            else:
                self.logger.error(f"No metrics found in {results_dir}/{run_id}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to load metrics from {results_dir}: {e}")
            return None

    def _calculate_comparison(
        self, baseline_metrics: dict[str, Any], semantic_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate comparison between baseline and semantic metrics."""
        comparison = {}

        for metric in self.config.comparison.metrics_to_track:
            baseline_val = baseline_metrics.get(metric)
            semantic_val = semantic_metrics.get(metric)

            if baseline_val is not None and semantic_val is not None:
                if isinstance(baseline_val, (int, float)) and isinstance(
                    semantic_val, (int, float)
                ):
                    improvement = (
                        semantic_val / baseline_val
                        if baseline_val != 0
                        else float("inf")
                    )
                    comparison[metric] = {
                        "baseline": baseline_val,
                        "semantic": semantic_val,
                        "improvement_ratio": improvement,
                        "improvement_pct": (improvement - 1) * 100,
                    }

        return comparison


    def _log_comparison_results(self, comparison: dict[str, Any]) -> None:
        """Log comparison results in readable format."""
        self.logger.info("=== METRICS COMPARISON ===")

        for metric, values in comparison.items():
            if isinstance(values, dict) and "baseline" in values:
                baseline_val = values["baseline"]
                semantic_val = values["semantic"]
                improvement = values["improvement_ratio"]

                if "latency" in metric.lower():
                    speedup = (
                        baseline_val / semantic_val
                        if semantic_val != 0
                        else float("inf")
                    )
                    self.logger.info(
                        f"{metric}: {baseline_val:.3f} → {semantic_val:.3f} ({speedup:.1f}x faster)"
                    )
                elif "cost" in metric.lower():
                    savings = (1 - improvement) * 100 if improvement < 1 else 0
                    self.logger.info(
                        f"{metric}: ${baseline_val:.6f} → ${semantic_val:.6f} ({savings:.1f}% savings)"
                    )
                else:
                    self.logger.info(
                        f"{metric}: {baseline_val:.3f} → {semantic_val:.3f} ({improvement:.2f}x)"
                    )


        # Log summary
        if "summary" in comparison:
            summary = comparison["summary"]
            self.logger.info(
                f"Targets achieved: {summary['targets_achieved']}/{summary['total_targets']}"
            )
            self.logger.info(f"Overall success rate: {summary['success_rate']:.1%}")

    def _save_comparison_results(
        self,
        comparison: dict[str, Any],
        baseline_metrics: dict[str, Any],
        semantic_metrics: dict[str, Any],
    ) -> None:
        """Save comparison results to disk."""
        try:
            storage = ResultsStorage(self.config.comparison.results_dir)
            run_id = storage.generate_run_id()
            run_id = storage.create_run_directory(run_id)

            comparison_data = {
                "baseline_metrics": baseline_metrics,
                "semantic_metrics": semantic_metrics,
                "comparison": comparison,
                "config": {
                    "data": self.config.data.__dict__,
                    "comparison": self.config.comparison.__dict__,
                },
            }

            storage.save_metrics(
                run_id, comparison_data, {"comparison_type": "baseline_vs_semantic"}
            )
            self.logger.info(f"Comparison results saved to: {run_id}")

        except Exception as e:
            self.logger.error(f"Failed to save comparison results: {e}")

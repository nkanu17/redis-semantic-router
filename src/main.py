"""Main orchestrator for news classification pipelines."""

import argparse
import asyncio
import sys
from typing import Any

from pipelines.evaluation_pipeline import EvaluationPipeline
from pipelines.llm_cls_pipeline import LLMClassificationPipeline
from pipelines.semantic_cls_pipeline import SemanticClassificationPipeline
from pipelines.semantic_training_pipeline import SemanticTrainingPipeline
from shared.data_types import BatchResult, FailedBatchResult
from utils.logger import get_logger, log_classification_report, log_metrics
from utils.redis_client import RedisConfig


async def llm_cls(
    args: argparse.Namespace,
) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
    """Run baseline LLM classification."""
    logger = get_logger("llm_classifier_cmd")

    try:
        pipeline = LLMClassificationPipeline(args.config)

        # Use training articles if flag is set
        if args.train_articles:
            results, metrics = await pipeline.run_on_training_data()
        else:
            results, metrics = await pipeline.run()

        log_metrics(logger, metrics, "LLM CLASSIFIER")
        log_classification_report(logger, metrics, "LLM CLASSIFIER")
        return results, metrics

    except Exception as e:
        logger.error(f"Baseline classification failed: {e}")
        raise


async def semantic_router_trainer(args: argparse.Namespace) -> dict[str, Any]:
    """Train semantic router."""
    logger = get_logger("train_cmd")

    try:
        pipeline = SemanticTrainingPipeline(args.config)
        summary = await pipeline.run(force_retrain=args.force_retrain)

        logger.info("Training completed successfully")
        return summary

    except Exception as e:
        logger.error(f"Semantic router training failed: {e}")
        raise


async def semantic_router_cls(
    args: argparse.Namespace,
) -> tuple[list[BatchResult | FailedBatchResult], dict[str, Any]]:
    """Run semantic router classification."""
    logger = get_logger("classify_cmd")

    try:
        pipeline = SemanticClassificationPipeline(args.config)

        # Use training articles if flag is set
        if args.train_articles:
            results, metrics = await pipeline.run_on_training_data()
        else:
            results, metrics = await pipeline.run()

        log_metrics(logger, metrics, "SEMANTIC ROUTER")
        log_classification_report(logger, metrics, "SEMANTIC ROUTER")
        return results, metrics

    except Exception as e:
        logger.error(f"Semantic classification failed: {e}")
        raise


def evaluate(args: argparse.Namespace) -> dict[str, Any] | None:
    """Compare baseline vs semantic classification results."""
    logger = get_logger("evaluate_cmd")

    try:
        pipeline = EvaluationPipeline(args.config)
        comparison = pipeline.compare_results(save_results=True)
        return comparison

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


# async def run_all_command(args):
#     """Run complete end-to-end pipeline."""
#     logger = get_logger("run_all_cmd")
#
#     try:
#         logger.info("=== FULL END-TO-END PIPELINE ===")
#
#         # Step 1: Train semantic router
#         logger.info("Step 1: Training semantic router...")
#         await train_command(args)
#
#         # Step 2: Run semantic classification
#         logger.info("Step 2: Running semantic classification...")
#         await classify_command(args)
#
#         # Step 3: Run baseline classification
#         logger.info("Step 3: Running baseline classification...")
#         await baseline_command(args)
#
#         # Step 4: Compare results
#         logger.info("Step 4: Evaluating results...")
#         evaluate_command(args)
#
#         logger.info("=== PIPELINE COMPLETED ===")
#
#     except Exception as e:
#         logger.error(f"Full pipeline failed: {e}")
#         raise


async def clear_routes(_: argparse.Namespace) -> None:
    """Clear all routes from Redis."""
    logger = get_logger("clear_routes_cmd")

    try:
        redis_config = RedisConfig()
        if redis_config.health_check():
            # Clear all semantic router data
            client = redis_config.get_client()
            client.flushdb()
            logger.info("All routes cleared from Redis")
        else:
            logger.error("Redis connection failed - cannot clear routes")

    except Exception as e:
        logger.error(f"Failed to clear routes: {e}")
        raise


def get_status(args: argparse.Namespace) -> None:
    """Show system status."""
    logger = get_logger("status_cmd")

    try:
        redis_config = RedisConfig()

        logger.info("=== SYSTEM STATUS ===")

        if redis_config.health_check():
            logger.info("Redis: Connected")
            info = redis_config.get_info()
            logger.info(f"   Version: {info.get('redis_version', 'Unknown')}")
            logger.info(f"   Memory: {info.get('used_memory_human', 'Unknown')}")
        else:
            logger.info("Redis: Disconnected")

        # Check available results
        eval_pipeline = EvaluationPipeline(args.config)
        available_runs = eval_pipeline.list_available_runs()

        logger.info("=== AVAILABLE RESULTS ===")
        for result_type, runs in available_runs.items():
            logger.info(f"{result_type}: {len(runs)} runs")

    except Exception as e:
        logger.error(f"Status check failed: {e}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Semantic routing pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            python main.py status                    # Check system status
            python main.py train_router              # Train semantic routes
            python main.py llm_classifier            # Run baseline LLM classification  
            python main.py semantic_router           # Run cls with semantic router
            python main.py evaluate                  # Compare baseline vs semantic
            # python main.py run-all                   # Full end-to-end pipeline
            python main.py clear-routes              # Clear all routes from Redis
        """,
    )

    parser.add_argument(
        "command",
        choices=[
        "llm_classifier",
            "train_router",
            "semantic_router",
            "evaluate",
            # "run-all",
            "clear-routes",
            "status",
        ],
        help="Pipeline command to run",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../config/pipeline_config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if routes exist",
    )

    parser.add_argument(
        "--train-articles",
        action="store_true",
        help="Use training articles instead of test articles",
    )

    args = parser.parse_args()

    try:
        if args.command == "llm_classifier":
            asyncio.run(llm_cls(args))
        elif args.command == "train_router":
            asyncio.run(semantic_router_trainer(args))
        elif args.command == "semantic_router":
            asyncio.run(semantic_router_cls(args))
        elif args.command == "evaluate":
            evaluate(args)
        # elif args.command == "run-all":
        #     asyncio.run(run_all_command(args))
        elif args.command == "clear-routes":
            asyncio.run(clear_routes(args))
        elif args.command == "status":
            get_status(args)

    except KeyboardInterrupt:
        logger = get_logger("main")
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger = get_logger("main")
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

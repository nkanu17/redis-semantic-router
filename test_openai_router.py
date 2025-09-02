#!/usr/bin/env python3
"""Test script for OpenAI semantic router with validation."""

import asyncio
import sys
import time
from typing import Any

# Add src to path for imports
sys.path.append("src")

from semantic_router.redis_router import RedisSemanticClassifier
from shared.data_types import NewsCategory
from shared.metrics import calculate_batch_metrics
from utils.config_loader import ConfigLoader
from utils.data_loader import NewsDataLoader
from utils.redis_client import RedisConfig


async def test_openai_router_with_config() -> dict[str, Any]:
    """Test OpenAI semantic router using current configuration."""
    print("Testing OpenAI semantic router with current config...")

    try:
        # Load config
        config_loader = ConfigLoader("config/openai_test_config.yaml")
        config = config_loader.load_config()

        # Initialize data loader with correct local paths
        data_loader = NewsDataLoader(
            data_dir="data",
            train_file="train_data.csv",
            validation_file="validation_data.csv",
        )

        # Load data
        train_data = data_loader.load_train_samples()
        validation_data = data_loader.load_test_samples()
        classes = NewsCategory.get_valid_classes()

        print(
            f"Loaded {len(train_data)} training samples, {len(validation_data)} validation samples"
        )
        print(f"Classes: {classes}")

        # Setup Redis configuration
        redis_config = RedisConfig(
            host=config.semantic_router.redis.host,
            port=config.semantic_router.redis.port,
            db=config.semantic_router.redis.db,
            socket_timeout=config.semantic_router.redis.socket_timeout,
            socket_connect_timeout=config.semantic_router.redis.socket_connect_timeout,
            retry_on_timeout=config.semantic_router.redis.retry_on_timeout,
            decode_responses=config.semantic_router.redis.decode_responses,
        )

        # Create classifier with test-specific router name
        classifier = RedisSemanticClassifier(
            redis_config=redis_config,
            vectorizer_config=config.semantic_router.vectorizer,
            samples_per_class=config.semantic_router.route_config.samples_per_class,
            initial_threshold=config.semantic_router.route_config.initial_threshold,
            router_name="news-cls-openai-test",
            save_results=False,
            results_dir=config.semantic_router.results_dir,
            overwrite_existing=config.semantic_router.route_config.overwrite_existing,
        )

        # Train the router
        print("\n=== TRAINING ROUTER ===")
        await classifier.train(train_data)

        # Optimize thresholds if configured
        # if config.semantic_router.route_config.optimize_thresholds:
        #     print("=== OPTIMIZING THRESHOLDS ===")
        #     classifier.optimize_thresholds(validation_data)

        # Apply threshold overrides if configured
        # if config.semantic_router.route_config.threshold_overrides:
        #     print("=== APPLYING THRESHOLD OVERRIDES ===")
        #     classifier.apply_threshold_overrides(
        #         config.semantic_router.route_config.threshold_overrides
        #     )

        # Test on validation data
        print("\n=== TESTING ON VALIDATION DATA ===")
        start_time = time.time()

        batch_results = classifier.classify_articles(validation_data, classes)

        end_time = time.time()

        # Calculate metrics
        metrics = calculate_batch_metrics(batch_results)

        # Get classifier info for cost tracking
        classifier_info = classifier.get_classifier_info()

        # Print results
        print("\n=== RESULTS ===")
        print(
            f"Vectorizer: {classifier_info['vectorizer_type']} - {classifier_info['embedding_model']}"
        )
        print(f"Samples per class: {classifier_info['samples_per_class']}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Macro: {metrics['f1_macro']:.3f}")
        print(f"F1 Weighted: {metrics['f1_weighted']:.3f}")
        print(f"Total time: {end_time - start_time:.2f}s")
        print(f"Avg per article: {(end_time - start_time) / len(validation_data):.3f}s")

        if classifier_info["vectorizer_type"] == "openai":
            print(f"Training cost: ${classifier_info['training_cost']:.6f}")
            print(f"Classification cost: ${classifier_info['classification_cost']:.6f}")
            print(
                f"Total cost: ${classifier_info['training_cost'] + classifier_info['classification_cost']:.6f}"
            )

        return {
            "success": True,
            "metrics": metrics,
            "classifier_info": classifier_info,
            "total_time": end_time - start_time,
        }

    except Exception as e:
        print(f"Test failed: {e}")
        return {"success": False, "error": str(e)}


async def main():
    """Main test function."""
    print("OpenAI Semantic Router Test with Current Configuration")
    print("=" * 60)

    # Run test
    results = await test_openai_router_with_config()

    if results["success"]:
        print("\nTest completed successfully!")
    else:
        print(f"\nTest failed: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

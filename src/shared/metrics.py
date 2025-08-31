"""Shared metrics calculation utilities."""

from typing import Any

from sklearn.metrics import classification_report

from shared.data_types import BatchResult, FailedBatchResult, NewsArticleWithLabel


def calculate_batch_metrics(
    results: list[BatchResult | FailedBatchResult],
) -> dict[str, Any]:
    """Calculate aggregate metrics from batch results (successful and failed)."""
    if not results:
        return {}

    # Separate successful and failed results
    successful_results = [r for r in results if isinstance(r, BatchResult)]
    failed_results = [r for r in results if isinstance(r, FailedBatchResult)]

    total_articles = sum(r.batch_size for r in results)
    successful_articles = sum(r.batch_size for r in successful_results)
    failed_articles = sum(r.batch_size for r in failed_results)

    # Metrics for successful batches only
    total_latency = (
        sum(r.total_latency for r in successful_results) if successful_results else 0
    )
    total_cost = (
        sum(r.total_cost for r in successful_results) if successful_results else 0
    )
    total_prompt_tokens = (
        sum(r.prompt_tokens for r in successful_results) if successful_results else 0
    )
    total_completion_tokens = (
        sum(r.completion_tokens for r in successful_results)
        if successful_results
        else 0
    )
    total_tokens = (
        sum(r.total_tokens for r in successful_results) if successful_results else 0
    )

    # Calculate accuracy across successful batches only
    all_classified_articles = []
    for result in successful_results:
        all_classified_articles.extend(result.classified_articles)

    labeled_articles = [
        ca
        for ca in all_classified_articles
        if isinstance(ca.article, NewsArticleWithLabel)
    ]

    # Initialize classification metrics
    accuracy = None
    classification_metrics = {}

    if labeled_articles:
        # Extract true labels and predictions
        y_true = [ca.article.label.lower() for ca in labeled_articles]
        y_pred = [ca.prediction.lower() for ca in labeled_articles]

        # Calculate basic accuracy
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        accuracy = correct / len(labeled_articles)

        # Get unique labels for classification report
        unique_labels = sorted(list(set(y_true + y_pred)))

        try:
            # Calculate comprehensive metrics using sklearn
            classification_report_dict = classification_report(
                y_true, y_pred, labels=unique_labels, output_dict=True, zero_division=0
            )

            # Extract commonly used metrics from classification report for easy access
            macro_avg = classification_report_dict.get("macro avg", {})
            weighted_avg = classification_report_dict.get("weighted avg", {})

            classification_metrics = {
                "precision_macro": macro_avg.get("precision", 0.0),
                "recall_macro": macro_avg.get("recall", 0.0),
                "f1_macro": macro_avg.get("f1-score", 0.0),
                "precision_weighted": weighted_avg.get("precision", 0.0),
                "recall_weighted": weighted_avg.get("recall", 0.0),
                "f1_weighted": weighted_avg.get("f1-score", 0.0),
                "classification_report": classification_report_dict,
            }

        except Exception as e:
            # Fallback in case sklearn metrics fail
            classification_metrics = {
                "error": f"Could not calculate sklearn metrics: {str(e)}"
            }

    # Combine basic metrics with classification metrics
    metrics = {
        "total_articles": total_articles,
        "successful_articles": successful_articles,
        "failed_articles": failed_articles,
        "total_batches": len(results),
        "successful_batches": len(successful_results),
        "failed_batches": len(failed_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "total_latency": total_latency,
        "avg_batch_latency": total_latency / len(successful_results)
        if successful_results
        else 0,
        "avg_latency_per_article": total_latency / successful_articles
        if successful_articles
        else 0,
        "latency_unit": "seconds",
        "total_cost": total_cost,
        "cost_per_article": total_cost / successful_articles
        if successful_articles
        else 0,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_prompt_tokens_per_article": total_prompt_tokens / successful_articles
        if successful_articles
        else 0,
        "avg_completion_tokens_per_article": total_completion_tokens
        / successful_articles
        if successful_articles
        else 0,
        "avg_tokens_per_article": total_tokens / successful_articles
        if successful_articles
        else 0,
        "accuracy": accuracy,
        "f1_macro": classification_metrics.get("f1_macro")
        if labeled_articles
        else None,
        "correct_predictions": sum(
            1
            for true, pred in zip(
                [ca.article.label.lower() for ca in labeled_articles],
                [ca.prediction.lower() for ca in labeled_articles],
            )
            if true == pred
        )
        if labeled_articles
        else None,
        "total_predictions": len(labeled_articles) if labeled_articles else None,
    }

    # Add classification metrics
    metrics.update(classification_metrics)

    return metrics

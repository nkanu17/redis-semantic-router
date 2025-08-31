"""Batch classification pipeline using async processing."""

import asyncio
import json
import random
import re
import time
from typing import Any

import litellm
from litellm import acompletion

from llm_classifier.prompts import fetch_prompt
from shared.base_classifier import BaseClassifier
from shared.data_types import (
    BatchResult,
    ClassificationRequest,
    ClassifiedArticle,
    FailedBatchResult,
    NewsArticle,
    NewsArticleWithLabel,
)
from shared.exceptions import RetryableError, classify_error
from shared.metrics import calculate_batch_metrics
from shared.results_storage import ResultsStorage
from utils.logger import (
    get_logger,
)


class LLMClassifier(BaseClassifier):
    """Batch async classifier with configurable batch sizes."""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        max_concurrent: int = 50,
        batch_size: int = 10,
        debug: bool = False,
        max_retries: int = 3,
        save_results: bool = True,
        results_dir: str = "cls_results",
        temperature: float = 0.0,
        max_tokens: int = 50000,
    ):
        """
        Initialize batch classifier.

        Args:
            model_name: Model to use
            max_concurrent: Maximum concurrent requests
            batch_size: Number of articles per batch request
            debug: Enable debug logging
            max_retries: Maximum retry attempts for failed API calls
            save_results: Whether to save results to disk
            results_dir: Directory for saving results
        """
        super().__init__(save_results=save_results, results_dir=results_dir)

        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.initial_retry_delay = 1.0  # Default value
        self.enable_jitter = True  # Default value
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = get_logger(f"{__name__}.LLMClassifier")

        # Initialize results storage if enabled
        if self.save_results:
            self.results_storage = ResultsStorage(results_dir)
        else:
            self.results_storage = None

        # Set litellm to not print debug info
        litellm.set_verbose = False
        litellm.suppress_debug_info = True
        litellm._async_logging = False

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable using proper exception classification."""
        classified_error = classify_error(error)
        return isinstance(classified_error, RetryableError)

    async def _call_llm(self, prompt: str, request_id: str) -> Any:
        """
        Call LLM with exponential backoff retry logic.

        Args:
            prompt: The prompt to send to the LLM
            request_id: Request ID for logging

        Returns:
            LLM response object

        Raises:
            Exception: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                response = await acompletion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response

            except Exception as e:
                last_error = e

                # Don't retry if it's the last attempt
                if attempt == self.max_retries:
                    break

                # Don't retry if it's not a retryable error
                if not self._is_retryable_error(e):
                    self.logger.error(
                        f"Non-retryable error for batch {request_id}: {e}"
                    )
                    break

                # Calculate delay with exponential backoff
                delay = self.initial_retry_delay * (2**attempt)

                # Add jitter
                jitter = delay * 0.2 * (random.random() - 0.5)  # ±10% jitter
                delay += jitter

                self.logger.warning(
                    f"LLM call failed for batch {request_id} (attempt {attempt + 1}/{self.max_retries + 1}): {e}. "
                    f"Retrying in {delay:.2f}s"
                )

                await asyncio.sleep(delay)

        # All retries exhausted
        self.logger.error(
            f"LLM call failed for batch {request_id} after {self.max_retries + 1} attempts: {last_error}"
        )
        raise last_error

    async def classify_batch_request(
        self, request: ClassificationRequest, classes: list[str]
    ) -> BatchResult:
        """
        Classify a single batch request.

        Args:
            request: Classification request with articles
            classes: List of possible classes

        Returns:
            BatchResult with classified articles and metrics
        """
        async with self.semaphore:
            start_time = time.time()

            try:
                prompt = fetch_prompt(request.articles, classes)

                response = await self._call_llm(prompt, request.request_id)

                end_time = time.time()

                # Parse batch predictions
                predictions = self._parse_batch_predictions(
                    response.choices[0].message.content, request.articles, classes
                )

                # Create classified articles
                classified_articles = []
                for article, prediction in zip(request.articles, predictions):
                    classified_articles.append(
                        ClassifiedArticle(
                            article=article, prediction=prediction.lower()
                        )
                    )

                # Use litellm's built-in cost calculation and detailed token tracking
                return BatchResult(
                    request_id=request.request_id,
                    classified_articles=classified_articles,
                    total_latency=end_time - start_time,
                    total_cost=litellm.completion_cost(response),
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                    batch_size=len(request.articles),
                )

            except Exception as e:
                self.logger.error(f"Error classifying batch {request.request_id}: {e}")

                # Create fallback results
                classified_articles = []
                for article in request.articles:
                    classified_articles.append(
                        ClassifiedArticle(
                            article=article, prediction=classes[0].lower()
                        )
                    )

                return BatchResult(
                    request_id=request.request_id,
                    classified_articles=classified_articles,
                    total_latency=time.time() - start_time,
                    total_cost=0.0,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    batch_size=len(request.articles),
                )

    def _parse_batch_predictions(
        self,
        response_text: str,
        articles: list[NewsArticleWithLabel | NewsArticle],
        classes: list[str],
    ) -> list[str]:
        """Parse batch JSON response and extract predictions."""

        try:
            # Try to parse as JSON first - look for individual objects or array
            if "{" in response_text:
                # Extract all JSON objects from response
                json_matches = re.findall(
                    r'\{"article_id":\s*\d+,\s*"category":\s*"[^"]+"\}', response_text
                )

                if json_matches:
                    # Parse individual JSON objects
                    classifications = []
                    for match in json_matches:
                        try:
                            parsed_obj = json.loads(match)
                            classifications.append(parsed_obj)
                        except json.JSONDecodeError:
                            continue

                    # Create prediction list in article order
                    predictions = []
                    for article in articles:
                        # Find matching classification by article_id
                        matching_classification = None
                        for cls in classifications:
                            if cls.get("article_id") == article.article_id:
                                matching_classification = cls
                                break

                        if matching_classification:
                            category = matching_classification.get(
                                "category", classes[0]
                            )
                            # Validate category
                            if category.lower() in [c.lower() for c in classes]:
                                predictions.append(category)
                            else:
                                predictions.append(classes[0])
                        else:
                            predictions.append(classes[0])

                    return predictions

                # Fallback: try old format with "classifications" array
                elif "classifications" in response_text:
                    json_match = re.search(
                        r'\{.*"classifications".*\}', response_text, re.DOTALL
                    )
                    if json_match:
                        parsed = json.loads(json_match.group())
                        classifications = parsed.get("classifications", [])

                        # Create prediction list in article order
                        predictions = []
                        for article in articles:
                            # Find matching classification by article_id
                            matching_classification = None
                            for cls in classifications:
                                if cls.get("article_id") == article.article_id:
                                    matching_classification = cls
                                    break

                            if matching_classification:
                                category = matching_classification.get(
                                    "category", classes[0]
                                )
                                # Validate category
                                if category.lower() in [c.lower() for c in classes]:
                                    predictions.append(category)
                                else:
                                    predictions.append(classes[0])
                            else:
                                predictions.append(classes[0])

                        return predictions

            # Fallback: return first class for all articles
            return [classes[0]] * len(articles)

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.logger.warning(
                f"JSON parsing failed: {e}. Using fallback predictions."
            )
            # Fallback to first class for all articles
            return [classes[0]] * len(articles)

    def _chunk_articles(
        self, articles: list[NewsArticleWithLabel | NewsArticle]
    ) -> list[ClassificationRequest]:
        """Split articles into batch requests."""
        requests = []

        for i in range(0, len(articles), self.batch_size):
            chunk = articles[i : i + self.batch_size]
            request = ClassificationRequest(articles=chunk)
            requests.append(request)

        return requests

    async def classify_articles(
        self, articles: list[NewsArticleWithLabel | NewsArticle], classes: list[str]
    ) -> list[BatchResult | FailedBatchResult]:
        """Classify articles using batch processing.

        Args:
            articles: List of articles to classify
            classes: List of possible classes

        Returns:
            Flattened list of BatchResults (successful) and FailedBatchResults (failed)
        """
        # Split into batch requests
        requests = self._chunk_articles(articles)

        self.logger.info(
            f"Processing {len(articles)} articles in {len(requests)} batches of size {self.batch_size}"
        )

        # Process all batches concurrently
        tasks = [
            asyncio.create_task(self.classify_batch_request(request, classes))
            for request in requests
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Give LiteLLM workers time to finish cleanup to prevent cancellation errors
        await asyncio.sleep(0.1)

        # Flatten results into single list with both successful and failed
        successful_results = []
        failed_results = []
        failed_count = 0

        for i, batch_result in enumerate(results):
            if isinstance(batch_result, BatchResult):
                successful_results.append(batch_result)
            else:
                # Create FailedBatchResult for failed batch
                failed_batch = FailedBatchResult(
                    request_id=requests[i].request_id,
                    articles=requests[i].articles,
                    error=str(batch_result),
                    batch_size=len(requests[i].articles),
                )
                failed_results.append(failed_batch)
                failed_count += 1
                self.logger.error(
                    f"Batch {requests[i].request_id} failed with exception: {batch_result}"
                )

        if failed_count > 0:
            self.logger.warning(
                f"{failed_count} batches failed out of {len(results)} total"
            )

        # Combine successful and failed results
        all_results = successful_results + failed_results

        # Save results if enabled
        if self.save_results and self.results_storage:
            try:
                run_id = self.results_storage.generate_run_id()
                run_id = self.results_storage.create_run_directory(run_id)

                # Get classifier configuration
                classifier_config = {
                    "model_name": self.model_name,
                    "batch_size": self.batch_size,
                    "max_concurrent": self.semaphore._value,
                    "max_retries": self.max_retries,
                    "total_articles_processed": len(articles),
                }

                # Save classifications first
                self.results_storage.save_classifications(run_id, all_results)

                # Calculate and save metrics
                metrics = calculate_batch_metrics(all_results)
                self.results_storage.save_metrics(run_id, metrics, classifier_config)

                self.logger.info(f"Results saved to run: {run_id}")

            except Exception as e:
                self.logger.error(f"Failed to save results: {e}")

        return all_results

    def get_classifier_info(self) -> dict[str, Any]:
        """
        Get information about the classifier configuration.

        Returns:
            Dictionary with classifier information
        """
        return {
            "classifier_type": "llm_batch_async",
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_concurrent": self.semaphore._value,
            "max_retries": self.max_retries,
            "supports_training": self.supports_training(),
            "is_ready": self.is_ready(),
        }

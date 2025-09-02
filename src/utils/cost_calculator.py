"""Cost calculation utilities for different embedding providers."""

from typing import Any

import tiktoken


class EmbeddingCostCalculator:
    """Calculator for embedding API costs and token usage."""

    # OpenAI embedding model pricing (per 1K tokens) as of 2025
    OPENAI_PRICING = {
        "text-embedding-3-small": 0.00002,
        "text-embedding-3-large": 0.00013,
        "text-embedding-ada-002": 0.0001,
    }

    @staticmethod
    def estimate_tokens(text: str, model: str = "text-embedding-3-small") -> int:
        """
        Estimate token count for text using tiktoken.

        Args:
            text: Input text to tokenize
            model: Embedding model name

        Returns:
            Estimated token count
        """
        try:
            # Use cl100k_base encoding for OpenAI embedding models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Fallback estimation if tiktoken fails
            return int(len(text.split()) * 1.3)

    @staticmethod
    def calculate_openai_embedding_cost(
        text: str, model: str = "text-embedding-3-small"
    ) -> tuple[int, float]:
        """
        Calculate tokens and cost for OpenAI embedding.

        Args:
            text: Input text to embed
            model: OpenAI embedding model name

        Returns:
            Tuple of (token_count, cost_in_dollars)
        """
        tokens = EmbeddingCostCalculator.estimate_tokens(text, model)

        # Get pricing for model
        price_per_1k = EmbeddingCostCalculator.OPENAI_PRICING.get(model, 0.00002)
        cost = (tokens / 1000) * price_per_1k

        return tokens, cost

    @staticmethod
    def calculate_batch_embedding_cost(
        texts: list[str], model: str = "text-embedding-3-small"
    ) -> dict[str, Any]:
        """
        Calculate total tokens and cost for batch of texts.

        Args:
            texts: List of texts to embed
            model: OpenAI embedding model name

        Returns:
            Dictionary with cost breakdown
        """
        total_tokens = 0
        total_cost = 0.0

        for text in texts:
            tokens, cost = EmbeddingCostCalculator.calculate_openai_embedding_cost(
                text, model
            )
            total_tokens += tokens
            total_cost += cost

        return {
            "total_texts": len(texts),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "cost_per_text": total_cost / len(texts) if texts else 0.0,
            "avg_tokens_per_text": total_tokens / len(texts) if texts else 0.0,
            "model": model,
            "price_per_1k_tokens": EmbeddingCostCalculator.OPENAI_PRICING.get(
                model, 0.00002
            ),
        }

    @staticmethod
    def get_supported_models() -> dict[str, dict[str, Any]]:
        """
        Get information about supported embedding models.

        Returns:
            Dictionary with model information
        """
        return {
            "openai": {
                "models": {
                    "text-embedding-3-small": {
                        "dimensions": 1536,
                        "price_per_1k_tokens": 0.00002,
                        "description": "Most capable, cost-effective embedding model",
                    },
                    "text-embedding-3-large": {
                        "dimensions": 3072,
                        "price_per_1k_tokens": 0.00013,
                        "description": "Highest performance embedding model",
                    },
                    "text-embedding-ada-002": {
                        "dimensions": 1536,
                        "price_per_1k_tokens": 0.0001,
                        "description": "Legacy embedding model",
                    },
                }
            },
            "huggingface": {
                "models": {
                    "sentence-transformers/all-MiniLM-L6-v2": {
                        "dimensions": 384,
                        "price_per_1k_tokens": 0.0,
                        "description": "Free, fast, and efficient",
                    },
                    "sentence-transformers/all-mpnet-base-v2": {
                        "dimensions": 768,
                        "price_per_1k_tokens": 0.0,
                        "description": "Free, higher quality embeddings",
                    },
                }
            },
        }

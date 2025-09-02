"""Configuration loader for pipeline settings."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from utils.logger import get_logger


@dataclass
class LLMClassifierConfig:
    """Configuration for LLM classifier."""

    model_name: str
    batch_size: int
    max_concurrent: int
    temperature: float
    max_tokens: int
    max_retries: int
    save_results: bool
    results_dir: str


@dataclass
class RedisConnectionConfig:
    """Configuration for Redis connection."""

    host: str
    port: int
    db: int
    socket_timeout: float
    socket_connect_timeout: float
    retry_on_timeout: bool
    decode_responses: bool


@dataclass
class VectorizerConfig:
    """Configuration for embedding vectorizer."""

    type: str  # "huggingface" or "openai"
    model: str
    api_key_env: str | None = None
    track_usage: bool = True


@dataclass
class RouteConfig:
    """Configuration for route building."""

    samples_per_class: int
    initial_threshold: float
    max_text_length: int
    optimize_thresholds: bool
    threshold_overrides: dict[str, float] | None = None
    overwrite_existing: bool = True


@dataclass
class SemanticRouterConfig:
    """Configuration for semantic router."""

    redis: RedisConnectionConfig
    vectorizer: VectorizerConfig
    route_config: RouteConfig
    router_name: str
    save_results: bool
    results_dir: str

    # Backward compatibility
    @property
    def embedding_model(self) -> str:
        """Get embedding model for backward compatibility."""
        return self.vectorizer.model


@dataclass
class DataConfig:
    """Configuration for data loading."""

    dataset_path: str
    train_file: str
    validation_file: str


@dataclass
class ComparisonConfig:
    """Configuration for comparison pipeline."""

    metrics_to_track: list[str]
    save_results: bool
    results_dir: str


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    llm_classifier: LLMClassifierConfig
    semantic_router: SemanticRouterConfig
    data: DataConfig
    comparison: ComparisonConfig


class ConfigLoader:
    """Utility for loading and validating pipeline configuration."""

    def __init__(self, config_path: str | Path = "../config/pipeline_config.yaml"):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.logger = get_logger(f"{__name__}.ConfigLoader")
        self.raw_config: dict[str, Any] | None = None

    def load_config(self) -> PipelineConfig:
        """
        Load configuration from YAML file.

        Returns:
            PipelineConfig object with all settings

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_config = yaml.safe_load(f)

            self.logger.info(f"Loaded configuration from {self.config_path}")

            # Store raw config for reproducibility
            self.raw_config = raw_config

            # Parse and validate configuration sections
            config = self._parse_config(raw_config)

            self.logger.info("Configuration validation successful")
            return config

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            raise

    def _parse_config(self, raw_config: dict[str, Any]) -> PipelineConfig:
        """Parse raw configuration into structured dataclasses."""

        # Parse LLM classifier config
        llm_classifier_raw = raw_config.get("llm_classifier", {})
        llm_classifier = LLMClassifierConfig(
            model_name=llm_classifier_raw.get("model_name", "claude-sonnet-4-20250514"),
            batch_size=llm_classifier_raw.get("batch_size", 10),
            max_concurrent=llm_classifier_raw.get("max_concurrent", 20),
            temperature=llm_classifier_raw.get("temperature", 0.0),
            max_tokens=llm_classifier_raw.get("max_tokens", 50000),
            max_retries=llm_classifier_raw.get("max_retries", 3),
            save_results=llm_classifier_raw.get("save_results", True),
            results_dir=llm_classifier_raw.get("results_dir", "results/llm_classifier"),
        )

        # Parse semantic router config
        semantic_raw = raw_config.get("semantic_router", {})

        redis_raw = semantic_raw.get("redis", {})
        redis_config = RedisConnectionConfig(
            host=redis_raw.get("host", "localhost"),
            port=redis_raw.get("port", 6379),
            db=redis_raw.get("db", 0),
            socket_timeout=redis_raw.get("socket_timeout", 30.0),
            socket_connect_timeout=redis_raw.get("socket_connect_timeout", 30.0),
            retry_on_timeout=redis_raw.get("retry_on_timeout", True),
            decode_responses=redis_raw.get("decode_responses", True),
        )

        # Parse vectorizer config with backward compatibility
        vectorizer_raw = semantic_raw.get("vectorizer", {})
        if not vectorizer_raw and "embedding_model" in semantic_raw:
            # Backward compatibility: convert old embedding_model to new format
            embedding_model = semantic_raw["embedding_model"]
            if embedding_model.startswith("text-embedding"):
                vectorizer_config = VectorizerConfig(
                    type="openai",
                    model=embedding_model,
                    api_key_env="OPENAI_API_KEY",
                    track_usage=True,
                )
            else:
                vectorizer_config = VectorizerConfig(
                    type="huggingface", model=embedding_model, track_usage=False
                )
        else:
            vectorizer_config = VectorizerConfig(
                type=vectorizer_raw.get("type", "huggingface"),
                model=vectorizer_raw.get(
                    "model", "sentence-transformers/all-MiniLM-L6-v2"
                ),
                api_key_env=vectorizer_raw.get("api_key_env", "OPENAI_API_KEY"),
                track_usage=vectorizer_raw.get("track_usage", True),
            )

        route_raw = semantic_raw.get("route_config", {})
        route_config = RouteConfig(
            samples_per_class=route_raw.get("samples_per_class", 15),
            initial_threshold=route_raw.get("initial_threshold", 0.75),
            max_text_length=route_raw.get("max_text_length", 500),
            optimize_thresholds=route_raw.get("optimize_thresholds", True),
            threshold_overrides=route_raw.get("threshold_overrides"),
            overwrite_existing=route_raw.get("overwrite_existing", True),
        )

        semantic_router = SemanticRouterConfig(
            redis=redis_config,
            vectorizer=vectorizer_config,
            route_config=route_config,
            router_name=semantic_raw.get("router_name", "news-classification-router"),
            save_results=semantic_raw.get("save_results", True),
            results_dir=semantic_raw.get("results_dir", "results/semantic_router"),
        )

        # Parse data config
        data_raw = raw_config.get("data", {})
        data = DataConfig(
            dataset_path=data_raw.get("dataset_path", "bbc-news-articles-labeled"),
            train_file=data_raw.get("train_file", "train_data.csv"),
            validation_file=data_raw.get("validation_file", "validation_data.csv"),
        )

        # Parse comparison config
        comparison_raw = raw_config.get("comparison", {})
        comparison = ComparisonConfig(
            metrics_to_track=comparison_raw.get(
                "metrics_to_track",
                ["accuracy", "f1_macro", "total_latency", "total_cost"],
            ),
            save_results=comparison_raw.get("save_results", True),
            results_dir=comparison_raw.get("results_dir", "results/comparison"),
        )

        return PipelineConfig(
            llm_classifier=llm_classifier,
            semantic_router=semantic_router,
            data=data,
            comparison=comparison,
        )

    def validate_config(self, config: PipelineConfig) -> bool:
        """
        Validate configuration settings.

        Args:
            config: Configuration to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Validate data path exists
            dataset_path = Path(config.data.dataset_path)
            if not dataset_path.exists():
                self.logger.error(f"Dataset path does not exist: {dataset_path}")
                return False

            # Validate data files exist
            train_file_path = dataset_path / config.data.train_file
            if not train_file_path.exists():
                self.logger.error(f"Training file does not exist: {train_file_path}")
                return False

            validation_file_path = dataset_path / config.data.validation_file
            if not validation_file_path.exists():
                self.logger.error(
                    f"Validation file does not exist: {validation_file_path}"
                )
                return False

            # Validate LLM classifier config
            if config.llm_classifier.batch_size <= 0:
                self.logger.error(
                    f"batch_size set to {config.llm_classifier.batch_size}, must be positive"
                )
                return False

            if config.llm_classifier.max_concurrent <= 0:
                self.logger.error(
                    f"max_concurrent set to {config.llm_classifier.max_concurrent}, must be positive"
                )
                return False

            # Validate semantic router config
            if config.semantic_router.route_config.samples_per_class <= 0:
                self.logger.error(
                    f"samples_per_class set to {config.semantic_router.route_config.samples_per_class}, must be positive"
                )
                return False

            if not (
                0.0 <= config.semantic_router.route_config.initial_threshold <= 1.0
            ):
                self.logger.error(
                    f"initial_threshold set to {config.semantic_router.route_config.initial_threshold} must be between 0.0 and 1.0"
                )
                return False

            self.logger.info("Configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

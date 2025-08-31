"""Redis connection and configuration management."""

from typing import Any

import redis

from utils.logger import get_logger


class RedisConfig:
    """Redis connection configuration and management."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        socket_timeout: float = 30.0,
        socket_connect_timeout: float = 30.0,
        retry_on_timeout: bool = True,
        decode_responses: bool = True,
    ):
        """
        Initialize Redis configuration.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password if required
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            retry_on_timeout: Whether to retry on timeout
            decode_responses: Whether to decode byte responses to strings
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.socket_timeout = socket_timeout
        self.socket_connect_timeout = socket_connect_timeout
        self.retry_on_timeout = retry_on_timeout
        self.decode_responses = decode_responses
        self.logger = get_logger(f"{__name__}.RedisConfig")

        self._client = None

    def get_client(self) -> redis.Redis:
        """
        Get Redis client, creating connection if needed.

        Returns:
            Redis client instance

        Raises:
            redis.ConnectionError: If connection fails
        """
        if self._client is None:
            try:
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    socket_timeout=self.socket_timeout,
                    socket_connect_timeout=self.socket_connect_timeout,
                    retry_on_timeout=self.retry_on_timeout,
                    decode_responses=self.decode_responses,
                )

                # Test connection
                self._client.ping()
                self.logger.info(f"Connected to Redis at {self.host}:{self.port}")

            except Exception as e:
                self.logger.error(f"Failed to connect to Redis: {e}")
                raise

        return self._client

    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            client = self.get_client()
            response = client.ping()
            return response is True
        except Exception as e:
            self.logger.error(f"Redis health check failed: {e}")
            return False

    def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            try:
                self._client.close()
                self.logger.info("Redis connection closed")
            except Exception as e:
                self.logger.error(f"Error closing Redis connection: {e}")
            finally:
                self._client = None

    def get_info(self) -> dict[str, Any]:
        """
        Get Redis server information.

        Returns:
            Dictionary with Redis server info
        """
        try:
            client = self.get_client()
            info = client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory_human": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "uptime_in_seconds": info.get("uptime_in_seconds"),
            }
        except Exception as e:
            self.logger.error(f"Failed to get Redis info: {e}")
            return {}

    def get_url(self) -> str:
        """
        Get Redis URL for connection.

        Returns:
            Redis URL string in format redis://[password@]host:port[/db]
        """
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        else:
            return f"redis://{self.host}:{self.port}/{self.db}"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "RedisConfig":
        """
        Create RedisConfig from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            RedisConfig instance
        """
        return cls(
            host=config.get("host", "localhost"),
            port=config.get("port", 6379),
            db=config.get("db", 0),
            password=config.get("password"),
            socket_timeout=config.get("socket_timeout", 30.0),
            socket_connect_timeout=config.get("socket_connect_timeout", 30.0),
            retry_on_timeout=config.get("retry_on_timeout", True),
            decode_responses=config.get("decode_responses", True),
        )

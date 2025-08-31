"""Custom exceptions for the news classification system."""


class ClassificationError(Exception):
    """Base exception for classification errors."""

    pass


class RetryableError(ClassificationError):
    """Exception that can be retried."""

    pass


class NonRetryableError(ClassificationError):
    """Exception that should not be retried."""

    pass


class RateLimitError(RetryableError):
    """Rate limiting error from API."""

    pass


class TimeoutError(RetryableError):
    """Timeout error from API or network."""

    pass


class ConnectionError(RetryableError):
    """Network or connection error."""

    pass


class ServerError(RetryableError):
    """Server-side error (5xx status codes)."""

    pass


class AuthenticationError(NonRetryableError):
    """Authentication or authorization error."""

    pass


class BadRequestError(NonRetryableError):
    """Bad request error (4xx status codes)."""

    pass


class ParseError(NonRetryableError):
    """Error parsing LLM response."""

    pass


def classify_error(error: Exception) -> Exception:
    """
    Classify a generic exception into our custom exception hierarchy.

    Args:
        error: The original exception

    Returns:
        Classified exception (retryable or non-retryable)
    """
    error_str = str(error).lower()

    # Rate limiting
    if "rate limit" in error_str or "429" in error_str:
        return RateLimitError(str(error))

    # Timeout errors
    if "timeout" in error_str or "timed out" in error_str:
        return TimeoutError(str(error))

    # Connection errors
    if "connection" in error_str or "network" in error_str:
        return ConnectionError(str(error))

    # Server errors (5xx)
    if any(code in error_str for code in ["500", "502", "503", "504"]):
        return ServerError(str(error))

    # Authentication errors
    if any(code in error_str for code in ["401", "403"]):
        return AuthenticationError(str(error))

    # Bad request errors
    if any(code in error_str for code in ["400", "404"]):
        return BadRequestError(str(error))

    # JSON/parsing errors
    if "json" in error_str or "parse" in error_str:
        return ParseError(str(error))

    # Default to retryable for unknown errors
    return RetryableError(str(error))

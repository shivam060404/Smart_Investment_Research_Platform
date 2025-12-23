import re
from typing import Optional

from fastapi import HTTPException, status

from app.models.responses import ErrorCode


# Valid ticker pattern: 1-10 alphanumeric characters, may include dots and hyphens
# Examples: AAPL, BRK.A, BRK-B, GOOGL
TICKER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")

# Pattern to detect potential injection attempts
INJECTION_PATTERN = re.compile(r"[<>\"'`;\\|&$(){}[\]]")

# Reserved/invalid ticker patterns
RESERVED_PATTERNS = re.compile(r"^(null|undefined|none|true|false|nan|inf)$", re.IGNORECASE)

# Maximum lengths for various inputs
MAX_TICKER_LENGTH = 10
MAX_QUERY_LENGTH = 500


class TickerValidationError(ValueError):
    """Exception raised for invalid ticker symbols."""
    
    def __init__(self, ticker: str, reason: str):
        self.ticker = ticker
        self.reason = reason
        super().__init__(f"Invalid ticker '{ticker}': {reason}")


def validate_ticker(ticker: str) -> str:
    """Validate and normalize a stock ticker symbol.
    
    Performs comprehensive validation including:
    - Empty/whitespace check
    - Length validation
    - Character pattern validation
    - Injection attack prevention
    - Reserved word detection
    
    Args:
        ticker: Stock ticker symbol to validate.
        
    Returns:
        Normalized (uppercase, trimmed) ticker symbol.
        
    Raises:
        TickerValidationError: If ticker is invalid.
    """
    if ticker is None:
        raise TickerValidationError("", "Ticker symbol cannot be None")
    
    cleaned = ticker.strip()
    
    if not cleaned:
        raise TickerValidationError(ticker, "Ticker symbol cannot be empty")
    
    if len(cleaned) > MAX_TICKER_LENGTH:
        raise TickerValidationError(
            ticker, 
            f"Ticker symbol exceeds maximum length of {MAX_TICKER_LENGTH} characters"
        )
    
    if INJECTION_PATTERN.search(cleaned):
        raise TickerValidationError(
            ticker,
            "Ticker symbol contains invalid characters"
        )
    
    if RESERVED_PATTERNS.match(cleaned):
        raise TickerValidationError(
            ticker,
            "Ticker symbol uses a reserved word"
        )
    
    normalized = cleaned.upper()
    
    if not TICKER_PATTERN.match(normalized):
        raise TickerValidationError(
            ticker,
            "Ticker symbol must start with a letter and contain only letters, numbers, dots, or hyphens"
        )
    
    return normalized



def validate_ticker_http(ticker: str) -> str:
    """Validate ticker and raise HTTPException on failure.
    
    Wrapper around validate_ticker that converts TickerValidationError
    to HTTPException for use in API routes.
    
    Args:
        ticker: Stock ticker symbol to validate.
        
    Returns:
        Normalized ticker symbol.
        
    Raises:
        HTTPException: If ticker is invalid.
    """
    try:
        return validate_ticker(ticker)
    except TickerValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": ErrorCode.INVALID_TICKER.value,
                "message": str(e),
            }
        )


def sanitize_string(value: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Sanitize a string input by removing potentially dangerous characters.
    
    Args:
        value: String to sanitize.
        max_length: Maximum allowed length.
        
    Returns:
        Sanitized string.
        
    Raises:
        ValueError: If input exceeds max length after sanitization.
    """
    if value is None:
        return ""
    
    cleaned = value.strip()
    cleaned = cleaned.replace("\x00", "")
    cleaned = "".join(
        char for char in cleaned 
        if char >= " " or char in "\n\t\r"
    )
    
    if len(cleaned) > max_length:
        raise ValueError(f"Input exceeds maximum length of {max_length} characters")
    
    return cleaned


def sanitize_html(value: str) -> str:
    """Escape HTML special characters to prevent XSS.
    
    Args:
        value: String to escape.
        
    Returns:
        HTML-escaped string.
    """
    if value is None:
        return ""
    
    replacements = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#x27;",
    }
    
    result = value
    for char, entity in replacements.items():
        result = result.replace(char, entity)
    
    return result


def validate_date_range(start_date, end_date) -> None:
    """Validate that start_date is before end_date.
    
    Args:
        start_date: Start date.
        end_date: End date.
        
    Raises:
        ValueError: If start_date is not before end_date.
    """
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")


def validate_positive_number(value: float, field_name: str = "value") -> float:
    """Validate that a number is positive.
    
    Args:
        value: Number to validate.
        field_name: Name of the field for error messages.
        
    Returns:
        The validated value.
        
    Raises:
        ValueError: If value is not positive.
    """
    if value <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return value


def validate_percentage(value: float, field_name: str = "value") -> float:
    """Validate that a number is a valid percentage (0-100).
    
    Args:
        value: Number to validate.
        field_name: Name of the field for error messages.
        
    Returns:
        The validated value.
        
    Raises:
        ValueError: If value is not between 0 and 100.
    """
    if not 0 <= value <= 100:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return value


def validate_weight(value: float, field_name: str = "weight") -> float:
    """Validate that a weight is between 0 and 1.
    
    Args:
        value: Weight to validate.
        field_name: Name of the field for error messages.
        
    Returns:
        The validated value.
        
    Raises:
        ValueError: If value is not between 0 and 1.
    """
    if not 0 <= value <= 1:
        raise ValueError(f"{field_name} must be between 0 and 1")
    return value


def ticker_validator(v: str) -> str:
    """Pydantic field validator for ticker symbols.
    
    Use with @field_validator decorator in Pydantic models.
    
    Args:
        v: Ticker value to validate.
        
    Returns:
        Normalized ticker symbol.
    """
    return validate_ticker(v)


def sanitized_string_validator(v: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Pydantic field validator for sanitized strings.
    
    Args:
        v: String value to validate.
        max_length: Maximum allowed length.
        
    Returns:
        Sanitized string.
    """
    return sanitize_string(v, max_length)

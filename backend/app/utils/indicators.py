from typing import Optional
from app.models.stock import MACDData, BollingerData, TechnicalIndicators


def calculate_sma(prices: list[float], period: int) -> Optional[float]:
    """Calculate Simple Moving Average.
    
    Args:
        prices: List of closing prices (most recent last)
        period: Number of periods for the average
        
    Returns:
        SMA value or None if insufficient data
    """
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_rsi(prices: list[float], period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index (RSI).
    
    Uses the standard Wilder's smoothing method.
    
    Args:
        prices: List of closing prices (most recent last)
        period: RSI period (default 14)
        
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    changes = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [max(0, change) for change in changes]
    losses = [abs(min(0, change)) for change in changes]
    
    # Calculate initial average gain/loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Apply Wilder's smoothing for remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return round(rsi, 2)


def calculate_ema(prices: list[float], period: int) -> Optional[float]:
    """Calculate Exponential Moving Average.
    
    Args:
        prices: List of closing prices (most recent last)
        period: EMA period
        
    Returns:
        EMA value or None if insufficient data
    """
    if len(prices) < period:
        return None
    
    multiplier = 2 / (period + 1)
    
    # Start with SMA for the first EMA value
    ema = sum(prices[:period]) / period
    
    # Calculate EMA for remaining prices
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    
    return ema


def calculate_macd(
    prices: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Optional[MACDData]:
    """Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: List of closing prices (most recent last)
        fast_period: Fast EMA period (default 12)
        slow_period: Slow EMA period (default 26)
        signal_period: Signal line EMA period (default 9)
        
    Returns:
        MACDData with macd_line, signal_line, and histogram, or None if insufficient data
    """
    min_required = slow_period + signal_period
    if len(prices) < min_required:
        return None
    
    # Calculate MACD line values for signal line calculation
    macd_values = []
    for i in range(slow_period, len(prices) + 1):
        subset = prices[:i]
        fast_ema = calculate_ema(subset, fast_period)
        slow_ema = calculate_ema(subset, slow_period)
        if fast_ema is not None and slow_ema is not None:
            macd_values.append(fast_ema - slow_ema)
    
    if len(macd_values) < signal_period:
        return None
    
    # Current MACD line
    macd_line = macd_values[-1]
    
    # Signal line is EMA of MACD values
    signal_multiplier = 2 / (signal_period + 1)
    signal_line = sum(macd_values[:signal_period]) / signal_period
    for val in macd_values[signal_period:]:
        signal_line = (val - signal_line) * signal_multiplier + signal_line
    
    # Histogram
    histogram = macd_line - signal_line
    
    return MACDData(
        macd_line=round(macd_line, 4),
        signal_line=round(signal_line, 4),
        histogram=round(histogram, 4)
    )


def calculate_bollinger_bands(
    prices: list[float],
    period: int = 20,
    num_std: float = 2.0
) -> Optional[BollingerData]:
    """Calculate Bollinger Bands.
    
    Args:
        prices: List of closing prices (most recent last)
        period: SMA period (default 20)
        num_std: Number of standard deviations (default 2.0)
        
    Returns:
        BollingerData with upper, middle, and lower bands, or None if insufficient data
    """
    if len(prices) < period:
        return None
    
    # Middle band is SMA
    recent_prices = prices[-period:]
    middle_band = sum(recent_prices) / period
    
    # Calculate standard deviation
    variance = sum((p - middle_band) ** 2 for p in recent_prices) / period
    std_dev = variance ** 0.5
    
    # Upper and lower bands
    upper_band = middle_band + (num_std * std_dev)
    lower_band = middle_band - (num_std * std_dev)
    
    return BollingerData(
        upper_band=round(upper_band, 2),
        middle_band=round(middle_band, 2),
        lower_band=round(lower_band, 2)
    )


def determine_trend(
    current_price: float,
    sma_50: Optional[float],
    sma_200: Optional[float],
    rsi: Optional[float],
    macd: Optional[MACDData]
) -> str:
    """Determine current trend direction based on indicators.
    
    Args:
        current_price: Current stock price
        sma_50: 50-day SMA
        sma_200: 200-day SMA
        rsi: RSI value
        macd: MACD data
        
    Returns:
        Trend direction: 'bullish', 'bearish', or 'neutral'
    """
    bullish_signals = 0
    bearish_signals = 0
    
    # Price vs SMAs
    if sma_50 is not None:
        if current_price > sma_50:
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    if sma_200 is not None:
        if current_price > sma_200:
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    # Golden/Death cross
    if sma_50 is not None and sma_200 is not None:
        if sma_50 > sma_200:
            bullish_signals += 1
        else:
            bearish_signals += 1
    
    # RSI
    if rsi is not None:
        if rsi > 50:
            bullish_signals += 1
        elif rsi < 50:
            bearish_signals += 1
    
    # MACD
    if macd is not None:
        if macd.histogram > 0:
            bullish_signals += 1
        elif macd.histogram < 0:
            bearish_signals += 1
    
    # Determine trend
    if bullish_signals > bearish_signals + 1:
        return "bullish"
    elif bearish_signals > bullish_signals + 1:
        return "bearish"
    return "neutral"


def calculate_all_indicators(
    prices: list[float],
    current_price: Optional[float] = None
) -> Optional[TechnicalIndicators]:
    """Calculate all technical indicators from price data.
    
    Args:
        prices: List of closing prices (most recent last), needs at least 200 data points
        current_price: Current price (defaults to last price in list)
        
    Returns:
        TechnicalIndicators object or None if insufficient data
    """
    if len(prices) < 200:
        return None
    
    if current_price is None:
        current_price = prices[-1]
    
    sma_50 = calculate_sma(prices, 50)
    sma_200 = calculate_sma(prices, 200)
    rsi_14 = calculate_rsi(prices, 14)
    macd = calculate_macd(prices)
    bollinger = calculate_bollinger_bands(prices)
    
    if any(v is None for v in [sma_50, sma_200, rsi_14, macd, bollinger]):
        return None
    
    trend = determine_trend(current_price, sma_50, sma_200, rsi_14, macd)
    
    return TechnicalIndicators(
        sma_50=round(sma_50, 2),
        sma_200=round(sma_200, 2),
        rsi_14=rsi_14,
        macd=macd,
        bollinger_bands=bollinger,
        current_trend=trend
    )

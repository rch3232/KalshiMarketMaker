import argparse
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor, Future
import yaml
from dotenv import load_dotenv
import os
import time
from typing import Dict, List, Set

from mm import KalshiTradingAPI, AvellanedaMarketMaker

import re

# Patterns to detect parlay/combo markets in title/subtitle
PARLAY_KEYWORDS = re.compile(r'\b(parlay|combo|multi-leg|accumulator)\b', re.IGNORECASE)
# Pattern to detect "X and Y" connecting different outcomes (e.g., "Team A wins and Team B wins")
MULTI_OUTCOME_PATTERN = re.compile(r'\b\w+\s+(?:wins?|scores?|over|under)\s+and\s+\w+\s+(?:wins?|scores?|over|under)\b', re.IGNORECASE)


def is_parlay_or_combo_market(market: Dict) -> tuple[bool, str]:
    """Check if a market is a parlay/combo that should be skipped.

    Returns:
        tuple of (should_skip: bool, reason: str)
    """
    ticker = market.get('ticker', '')
    title = market.get('title', '')
    subtitle = market.get('subtitle', '')

    # Check 1: is_combo metadata flag - only skip if explicitly True
    if market.get('is_combo') is True:
        return True, "is_combo=True"

    # Check 2: Ticker format - more than one comma indicates parlay, or exceeds 60 characters
    comma_count = ticker.count(',')
    if comma_count > 1:
        return True, f"ticker contains {comma_count} commas (parlay indicator): {ticker}"
    if len(ticker) > 60:
        return True, f"ticker exceeds 60 chars: {ticker}"

    # Check 3: market_type must be 'binary'
    market_type = market.get('market_type', '')
    if market_type != 'binary':
        return True, f"market_type is '{market_type}', not 'binary'"

    # Check 4: Title/subtitle contains parlay keywords
    combined_text = f"{title} {subtitle}"
    if PARLAY_KEYWORDS.search(combined_text):
        return True, f"parlay keyword in title/subtitle: {combined_text[:50]}"

    # Check 5: Title/subtitle contains multi-outcome pattern (e.g., "X wins and Y wins")
    if MULTI_OUTCOME_PATTERN.search(combined_text):
        return True, f"multi-outcome pattern in title/subtitle: {combined_text[:50]}"

    # Check 6: Multiple " and " conjunctions suggesting combined bets
    # Count occurrences of " and " that might indicate multiple legs
    and_count = combined_text.lower().count(' and ')
    if and_count >= 2:
        return True, f"multiple 'and' conjunctions ({and_count}): {combined_text[:50]}"

    return False, ""


def cleanup_logger(logger: logging.Logger):
    """Properly close and remove all handlers from a logger to prevent resource leaks."""
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

# Global logger for the runner
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
runner_logger = logging.getLogger("Runner")


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_api(api_key: str, private_key: str, base_url: str, market_ticker: str, logger: logging.Logger):
    return KalshiTradingAPI(
        api_key=api_key,
        private_key=private_key,
        market_ticker=market_ticker,
        base_url=base_url,
        logger=logger,
    )


def create_market_maker(mm_config: Dict, api: KalshiTradingAPI, logger: logging.Logger):
    """Create an AvellanedaMarketMaker with Dual-Quote flipping strategy.

    Parameters are calibrated for binary probability markets (0.00-1.00 scale):
    - gamma: Risk aversion scaled down (0.01-0.05) for decimal probabilities
    - sigma: Volatility estimate for binary outcomes
    - min/max_spread: Hard caps to prevent boundary-stuck quotes
    - flip_skew_factor: Asymmetric urgency for inventory flipping
    """
    return AvellanedaMarketMaker(
        logger=logger,
        api=api,
        gamma=mm_config.get('gamma', 0.02),
        k=mm_config.get('k', 1.5),
        sigma=mm_config.get('sigma', 0.10),
        T=mm_config.get('T', 3600),
        max_position=mm_config.get('max_position', 5),
        order_expiration=mm_config.get('order_expiration', 300),
        min_spread=mm_config.get('min_spread', 0.02),
        max_spread=mm_config.get('max_spread', 0.10),
        position_limit_buffer=mm_config.get('position_limit_buffer', 0.1),
        flip_skew_factor=mm_config.get('flip_skew_factor', 0.03),
    )


def run_market_for_duration(
    market_ticker: str,
    api_key: str,
    private_key: str,
    base_url: str,
    mm_config: Dict,
    dt: float,
    duration: int
):
    """Run market maker for a specific market for the given duration."""
    logger = logging.getLogger(f"MM_{market_ticker}")
    logger.setLevel(logging.INFO)

    # Clean up any existing handlers first to prevent accumulation
    cleanup_logger(logger)

    # Use RotatingFileHandler to limit log file sizes (5MB max, keep 2 backups)
    fh = RotatingFileHandler(
        f"{market_ticker}.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=2
    )
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Starting market maker for {market_ticker}")
    api = None

    try:
        api = create_api(api_key, private_key, base_url, market_ticker, logger)

        # Override T with the duration for this run cycle
        config_with_duration = mm_config.copy()
        config_with_duration['T'] = duration

        market_maker = create_market_maker(config_with_duration, api, logger)
        market_maker.run(dt)

    except Exception as e:
        logger.error(f"Error running market maker for {market_ticker}: {e}")
    finally:
        if api:
            try:
                api.logout()
            except:
                pass
        # Clean up logger handlers to release file descriptors
        cleanup_logger(logger)

    logger.info(f"Market maker for {market_ticker} finished")


def test_api_connection(api_key: str, private_key: str, base_url: str) -> bool:
    """Test API connection and credentials before starting market makers."""
    logger = logging.getLogger("ConnectionTest")
    logger.info("=" * 60)
    logger.info("TESTING API CONNECTION")
    logger.info("=" * 60)

    try:
        test_api = KalshiTradingAPI(
            api_key=api_key,
            private_key=private_key,
            market_ticker="CONNECTION_TEST",
            base_url=base_url,
            logger=logger
        )
        return test_api.test_connection()
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        return False


def fetch_active_markets(series_list: List[str], api_key: str, private_key: str, base_url: str) -> List[str]:
    """Fetch all active market tickers for the given series list."""
    logger = logging.getLogger("MarketFetcher")
    active_tickers = []

    # Create a temporary API connection to fetch markets
    # We use a dummy ticker since we just need to be authenticated
    try:
        temp_api = KalshiTradingAPI(
            api_key=api_key,
            private_key=private_key,
            market_ticker="DUMMY",
            base_url=base_url,
            logger=logger
        )

        skipped_count = 0
        for series in series_list:
            try:
                markets = temp_api.get_active_markets_by_series(series)
                for market in markets:
                    ticker = market.get('ticker')
                    if not ticker:
                        continue

                    # Filter out parlay/combo markets
                    is_parlay, reason = is_parlay_or_combo_market(market)
                    if is_parlay:
                        logger.info(f"Skipping parlay/combo market: {ticker} - {reason}")
                        skipped_count += 1
                        continue

                    active_tickers.append(ticker)
                    print(f'Valid Market Found: {ticker}')
                    logger.info(f"Found active market: {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch markets for series {series}: {e}")

        logger.info(f"Total markets found: {len(active_tickers)} (skipped {skipped_count} parlay/combo markets)")

        temp_api.logout()

    except Exception as e:
        logger.error(f"Failed to connect to Kalshi API: {e}")

    return active_tickers


def fetch_active_markets_by_category(category: str, api_key: str, private_key: str, base_url: str) -> List[str]:
    """Fetch all active market tickers for a category (e.g., 'Sports').

    This function fetches ALL sports markets at once without needing to know
    specific series tickers (like KXNFLGAME, KXNBAGAME, etc.) in advance.

    Args:
        category: The market category to fetch (e.g., "Sports")
        api_key: Kalshi API key
        private_key: Kalshi private key (bytes or string)
        base_url: Kalshi API base URL

    Returns:
        List of active market ticker strings
    """
    logger = logging.getLogger("MarketFetcher")
    active_tickers = []

    try:
        temp_api = KalshiTradingAPI(
            api_key=api_key,
            private_key=private_key,
            market_ticker="DUMMY",
            base_url=base_url,
            logger=logger
        )

        markets = temp_api.get_active_markets_by_category(category)
        skipped_count = 0
        for market in markets:
            ticker = market.get('ticker')
            if not ticker:
                continue

            # Filter out parlay/combo markets
            is_parlay, reason = is_parlay_or_combo_market(market)
            if is_parlay:
                logger.info(f"Skipping parlay/combo market: {ticker} - {reason}")
                skipped_count += 1
                continue

            active_tickers.append(ticker)
            print(f'Valid Market Found: {ticker}')
            # Log with additional context about the market
            title = market.get('title', 'Unknown')
            subtitle = market.get('subtitle', '')
            logger.info(f"Found active market: {ticker} - {title} {subtitle}".strip())

        temp_api.logout()
        logger.info(f"Total markets found in category '{category}': {len(active_tickers)} (skipped {skipped_count} parlay/combo markets)")

    except Exception as e:
        logger.error(f"Failed to fetch markets for category {category}: {e}")

    return active_tickers


def run_dynamic_strategies(config: Dict):
    """
    Main loop that:
    1. Fetches active markets by category (e.g., 'Sports') or by series list
    2. Starts market makers for new markets
    3. Handles market expiration gracefully
    4. Refreshes market list periodically

    The bot can discover markets in two ways:
    - By category (recommended): Set 'category: Sports' to find ALL sports markets automatically
    - By series (legacy): Set 'series: [KXNFLGAME, KXNBAGAME, ...]' for specific series only
    """
    api_key = os.getenv("KALSHI_API_KEY")
    private_key_env = os.getenv("KALSHI_PRIVATE_KEY")
    base_url = os.getenv("KALSHI_BASE_URL")

    if not all([api_key, private_key_env, base_url]):
        runner_logger.error("Missing required environment variables: KALSHI_API_KEY, KALSHI_PRIVATE_KEY, KALSHI_BASE_URL")
        return

    # Check if private_key_env is a file path or the actual key
    if private_key_env.startswith('/') and os.path.isfile(private_key_env):
        runner_logger.info(f"Reading private key from file: {private_key_env}")
        with open(private_key_env, 'rb') as f:
            private_key = f.read()  # bytes, as per official Kalshi SDK
    else:
        private_key = private_key_env
        # Handle newlines in private key (environment variables often escape them)
        private_key = private_key.replace('\\n', '\n')

    # Test connection before proceeding
    if not test_api_connection(api_key, private_key, base_url):
        runner_logger.error("API connection test failed. Please check your credentials.")
        runner_logger.error("Ensure KALSHI_API_KEY and KALSHI_PRIVATE_KEY match and are for the correct environment.")
        runner_logger.error(f"Current KALSHI_BASE_URL: {base_url}")
        return

    # Extract configuration
    # New approach: use 'category' (e.g., "Sports") to fetch ALL sports markets
    # Legacy approach: use 'series' list for specific series tickers
    category = config.get('category')
    series_list = config.get('series', [])
    mm_config = config.get('market_maker', {})
    dt = config.get('dt', 2.0)
    refresh_interval = config.get('refresh_interval', 300)  # 5 minutes default
    market_duration = config.get('market_duration', 3600)  # 1 hour per market cycle
    max_concurrent_markets = config.get('max_concurrent_markets', 10)

    runner_logger.info(f"Starting dynamic market maker")
    if category:
        runner_logger.info(f"Market discovery mode: CATEGORY ('{category}')")
        runner_logger.info(f"  Will automatically find ALL markets in the '{category}' category")
    else:
        runner_logger.info(f"Market discovery mode: SERIES (legacy)")
        runner_logger.info(f"  Series to trade: {series_list}")
    runner_logger.info(f"Refresh interval: {refresh_interval}s")
    runner_logger.info(f"Market duration per cycle: {market_duration}s")
    runner_logger.info(f"Max concurrent markets: {max_concurrent_markets}")

    active_futures: Dict[str, Future] = {}

    with ThreadPoolExecutor(max_workers=max_concurrent_markets) as executor:
        while True:
            try:
                # Fetch current active markets using category or series approach
                runner_logger.info("Fetching active markets...")
                if category:
                    active_tickers = fetch_active_markets_by_category(category, api_key, private_key, base_url)
                else:
                    active_tickers = fetch_active_markets(series_list, api_key, private_key, base_url)
                runner_logger.info(f"Found {len(active_tickers)} active markets")

                # Clean up completed futures and check for exceptions
                completed = [ticker for ticker, future in active_futures.items() if future.done()]
                for ticker in completed:
                    future = active_futures[ticker]
                    try:
                        # Check for exceptions (this will raise if the future had an error)
                        future.result(timeout=0)
                        runner_logger.info(f"Market maker for {ticker} completed successfully")
                    except Exception as e:
                        runner_logger.warning(f"Market maker for {ticker} completed with error: {e}")
                    del active_futures[ticker]

                # Start market makers for new markets (up to max_concurrent)
                for ticker in active_tickers:
                    if ticker not in active_futures and len(active_futures) < max_concurrent_markets:
                        runner_logger.info(f"Starting market maker for {ticker}")
                        future = executor.submit(
                            run_market_for_duration,
                            ticker,
                            api_key,
                            private_key,
                            base_url,
                            mm_config,
                            dt,
                            market_duration
                        )
                        active_futures[ticker] = future

                runner_logger.info(f"Currently running {len(active_futures)} market makers")
                runner_logger.info(f"Active markets: {list(active_futures.keys())}")

                # Wait before next refresh
                time.sleep(refresh_interval)

            except KeyboardInterrupt:
                runner_logger.info("Shutting down...")
                break
            except Exception as e:
                runner_logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait a bit before retrying


def run_static_strategy(config_name: str, config: Dict):
    """Run a single static strategy (original behavior)."""
    logger = logging.getLogger(f"Strategy_{config_name}")
    logger.setLevel(config.get('log_level', 'INFO'))

    # Clean up any existing handlers first to prevent accumulation
    cleanup_logger(logger)

    # Use RotatingFileHandler to limit log file sizes (5MB max, keep 2 backups)
    fh = RotatingFileHandler(
        f"{config_name}.log",
        maxBytes=5*1024*1024,  # 5MB
        backupCount=2
    )
    fh.setLevel(config.get('log_level', 'INFO'))
    ch = logging.StreamHandler()
    ch.setLevel(config.get('log_level', 'INFO'))
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Starting strategy: {config_name}")

    # Handle private key - check if it's a file path or the actual key
    private_key_env = os.getenv("KALSHI_PRIVATE_KEY")
    if private_key_env:
        if private_key_env.startswith('/') and os.path.isfile(private_key_env):
            logger.info(f"Reading private key from file: {private_key_env}")
            with open(private_key_env, 'rb') as f:
                private_key = f.read()  # bytes, as per official Kalshi SDK
        else:
            private_key = private_key_env.replace('\\n', '\n')
    else:
        private_key = None

    api = KalshiTradingAPI(
        api_key=os.getenv("KALSHI_API_KEY"),
        private_key=private_key,
        market_ticker=config['api']['market_ticker'],
        base_url=os.getenv("KALSHI_BASE_URL"),
        logger=logger,
    )

    market_maker = create_market_maker(config['market_maker'], api, logger)

    try:
        market_maker.run(config.get('dt', 1.0))
    except KeyboardInterrupt:
        logger.info("Market maker stopped by user")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
    finally:
        api.logout()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kalshi Market Making Algorithm")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, choices=['static', 'dynamic'], default='dynamic',
                        help="Mode: 'static' for fixed markets, 'dynamic' for auto-fetching daily games")
    args = parser.parse_args()

    load_dotenv()
    configs = load_config(args.config)

    if args.mode == 'dynamic':
        # Dynamic mode: expects a single config with 'series' list
        runner_logger.info("Running in DYNAMIC mode - will auto-fetch active markets")
        run_dynamic_strategies(configs)
    else:
        # Static mode: original behavior with fixed market tickers
        runner_logger.info("Running in STATIC mode - using fixed market tickers")
        print("Starting the following strategies:")
        for config_name in configs:
            print(f"- {config_name}")

        with ThreadPoolExecutor(max_workers=len(configs)) as executor:
            for config_name, config in configs.items():
                executor.submit(run_static_strategy, config_name, config)

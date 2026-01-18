import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, Future
import yaml
from dotenv import load_dotenv
import os
import time
from typing import Dict, List, Set

from mm import KalshiTradingAPI, AvellanedaMarketMaker

# Global logger for the runner
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
runner_logger = logging.getLogger("Runner")


def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def create_api(email: str, password: str, base_url: str, market_ticker: str, logger: logging.Logger):
    return KalshiTradingAPI(
        email=email,
        password=password,
        market_ticker=market_ticker,
        base_url=base_url,
        logger=logger,
    )


def create_market_maker(mm_config: Dict, api: KalshiTradingAPI, logger: logging.Logger):
    return AvellanedaMarketMaker(
        logger=logger,
        api=api,
        gamma=mm_config.get('gamma', 0.1),
        k=mm_config.get('k', 1.5),
        sigma=mm_config.get('sigma', 0.5),
        T=mm_config.get('T', 3600),
        max_position=mm_config.get('max_position', 100),
        order_expiration=mm_config.get('order_expiration', 300),
        min_spread=mm_config.get('min_spread', 0.01),
        position_limit_buffer=mm_config.get('position_limit_buffer', 0.1),
        inventory_skew_factor=mm_config.get('inventory_skew_factor', 0.01),
        trade_side=mm_config.get('trade_side', 'yes')
    )


def run_market_for_duration(
    market_ticker: str,
    email: str,
    password: str,
    base_url: str,
    mm_config: Dict,
    dt: float,
    duration: int
):
    """Run market maker for a specific market for the given duration."""
    logger = logging.getLogger(f"MM_{market_ticker}")
    logger.setLevel(logging.INFO)

    # Add handlers if not already present
    if not logger.handlers:
        fh = logging.FileHandler(f"{market_ticker}.log")
        fh.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Starting market maker for {market_ticker}")

    try:
        api = create_api(email, password, base_url, market_ticker, logger)

        # Override T with the duration for this run cycle
        config_with_duration = mm_config.copy()
        config_with_duration['T'] = duration

        market_maker = create_market_maker(config_with_duration, api, logger)
        market_maker.run(dt)

    except Exception as e:
        logger.error(f"Error running market maker for {market_ticker}: {e}")
    finally:
        try:
            api.logout()
        except:
            pass

    logger.info(f"Market maker for {market_ticker} finished")


def fetch_active_markets(series_list: List[str], email: str, password: str, base_url: str) -> List[str]:
    """Fetch all active market tickers for the given series list."""
    logger = logging.getLogger("MarketFetcher")
    active_tickers = []

    # Create a temporary API connection to fetch markets
    # We use a dummy ticker since we just need to be authenticated
    try:
        temp_api = KalshiTradingAPI(
            email=email,
            password=password,
            market_ticker="DUMMY",
            base_url=base_url,
            logger=logger
        )

        for series in series_list:
            try:
                markets = temp_api.get_active_markets_by_series(series)
                for market in markets:
                    ticker = market.get('ticker')
                    if ticker:
                        active_tickers.append(ticker)
                        logger.info(f"Found active market: {ticker}")
            except Exception as e:
                logger.error(f"Failed to fetch markets for series {series}: {e}")

        temp_api.logout()

    except Exception as e:
        logger.error(f"Failed to connect to Kalshi API: {e}")

    return active_tickers


def run_dynamic_strategies(config: Dict):
    """
    Main loop that:
    1. Fetches active markets for configured series
    2. Starts market makers for new markets
    3. Handles market expiration gracefully
    4. Refreshes market list periodically
    """
    email = os.getenv("KALSHI_EMAIL")
    password = os.getenv("KALSHI_PASSWORD")
    base_url = os.getenv("KALSHI_BASE_URL")

    if not all([email, password, base_url]):
        runner_logger.error("Missing required environment variables: KALSHI_EMAIL, KALSHI_PASSWORD, KALSHI_BASE_URL")
        return

    # Extract configuration
    series_list = config.get('series', [])
    mm_config = config.get('market_maker', {})
    dt = config.get('dt', 2.0)
    refresh_interval = config.get('refresh_interval', 300)  # 5 minutes default
    market_duration = config.get('market_duration', 3600)  # 1 hour per market cycle
    max_concurrent_markets = config.get('max_concurrent_markets', 10)

    runner_logger.info(f"Starting dynamic market maker")
    runner_logger.info(f"Series to trade: {series_list}")
    runner_logger.info(f"Refresh interval: {refresh_interval}s")
    runner_logger.info(f"Market duration per cycle: {market_duration}s")
    runner_logger.info(f"Max concurrent markets: {max_concurrent_markets}")

    active_futures: Dict[str, Future] = {}

    with ThreadPoolExecutor(max_workers=max_concurrent_markets) as executor:
        while True:
            try:
                # Fetch current active markets
                runner_logger.info("Fetching active markets...")
                active_tickers = fetch_active_markets(series_list, email, password, base_url)
                runner_logger.info(f"Found {len(active_tickers)} active markets")

                # Clean up completed futures
                completed = [ticker for ticker, future in active_futures.items() if future.done()]
                for ticker in completed:
                    del active_futures[ticker]
                    runner_logger.info(f"Market maker for {ticker} completed")

                # Start market makers for new markets (up to max_concurrent)
                for ticker in active_tickers:
                    if ticker not in active_futures and len(active_futures) < max_concurrent_markets:
                        runner_logger.info(f"Starting market maker for {ticker}")
                        future = executor.submit(
                            run_market_for_duration,
                            ticker,
                            email,
                            password,
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

    if not logger.handlers:
        fh = logging.FileHandler(f"{config_name}.log")
        fh.setLevel(config.get('log_level', 'INFO'))
        ch = logging.StreamHandler()
        ch.setLevel(config.get('log_level', 'INFO'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info(f"Starting strategy: {config_name}")

    api = KalshiTradingAPI(
        email=os.getenv("KALSHI_EMAIL"),
        password=os.getenv("KALSHI_PASSWORD"),
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

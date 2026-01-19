import abc
import time
import re
import base64
import requests
import json
import threading
from typing import Dict, List, Tuple
import logging
import uuid
import math
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend


# Global rate limiter shared across all API instances (per Kalshi SDK recommendation)
# Kalshi requires minimum 100ms between API calls
_rate_limit_lock = threading.Lock()
_last_api_call_time = 0
RATE_LIMIT_MS = 100  # milliseconds between calls

# Clock drift correction - offset in milliseconds to add to local time
# Positive = local clock is behind server, Negative = local clock is ahead
_time_offset_ms = 0
_time_offset_lock = threading.Lock()


class AbstractTradingAPI(abc.ABC):
    @abc.abstractmethod
    def get_price(self) -> float:
        pass

    @abc.abstractmethod
    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        pass

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abc.abstractmethod
    def get_position(self) -> int:
        pass

    @abc.abstractmethod
    def get_orders(self) -> List[Dict]:
        pass


class KalshiTradingAPI(AbstractTradingAPI):
    """Kalshi Trading API with direct RSA-PSS signature implementation."""

    def __init__(
        self,
        api_key: str,
        private_key: str,
        market_ticker: str,
        base_url: str,
        logger: logging.Logger,
    ):
        self.api_key = api_key
        self.market_ticker = market_ticker
        self.logger = logger

        # Set up base URL - strip /trade-api/v2 if provided, we'll add it in requests
        # Demo: https://demo-api.kalshi.co
        # Production: https://api.elections.kalshi.com
        if base_url:
            self.host = base_url.rstrip('/').replace('/trade-api/v2', '')
        else:
            self.host = "https://api.elections.kalshi.com"
        self.logger.info(f"Using API host: {self.host}")

        # Load the private key - handle both bytes (from file) and string (from env var)
        if isinstance(private_key, bytes):
            # Direct bytes from file (official Kalshi SDK approach)
            key_data = private_key
        else:
            # String from environment variable - normalize and encode
            key_str = self._normalize_pem_key(private_key, logger)
            key_data = key_str.encode('utf-8')

        self.private_key = serialization.load_pem_private_key(
            key_data,
            password=None,
            backend=default_backend()
        )
        self.logger.info(f"RSA private key loaded successfully")

        # Use a session for connection pooling to reduce memory overhead
        self.session = requests.Session()
        # Configure connection pool size (default is 10, we keep it small for memory)
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=5)
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)

        self.logger.info(f"API initialized for market: {market_ticker}")

    def test_connection(self) -> bool:
        """Test the API connection and credentials. Returns True if successful."""
        self.logger.info("Testing API connection...")
        self.logger.info(f"  API Key: {self.api_key[:8]}...{self.api_key[-4:]}")
        self.logger.info(f"  Host: {self.host}")

        try:
            # Try to get account balance - a simple authenticated endpoint
            response = self._make_request("GET", "/portfolio/balance")
            balance = response.get("balance", 0) / 100  # Convert cents to dollars
            self.logger.info(f"  Connection successful! Account balance: ${balance:.2f}")
            return True
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"  Connection test FAILED: {error_msg}")

            # Provide helpful diagnostics
            if "INCORRECT_API_KEY_SIGNATURE" in error_msg:
                self.logger.error("  DIAGNOSTIC: Signature mismatch. Possible causes:")
                self.logger.error("    1. Private key doesn't match this API key")
                self.logger.error("    2. API key was regenerated (invalidates old private key)")
                self.logger.error("    3. Environment mismatch (demo keys on prod, or vice versa)")
                self.logger.error(f"    4. Check KALSHI_BASE_URL matches your API key environment")
                self.logger.error(f"       Demo: https://demo-api.kalshi.co")
                self.logger.error(f"       Prod: https://api.elections.kalshi.com")
            elif "INVALID_API_KEY" in error_msg:
                self.logger.error("  DIAGNOSTIC: API key not recognized. Check KALSHI_API_KEY")
            elif "authentication_error" in error_msg:
                self.logger.error("  DIAGNOSTIC: General auth error. Verify credentials.")

            return False

    @staticmethod
    def _normalize_pem_key(key: str, logger: logging.Logger) -> str:
        """Normalize a PEM key that may have formatting issues."""
        # Handle escaped newlines
        key = key.replace('\\n', '\n')
        key = key.replace('\\r', '')
        key = key.replace('\r', '')

        lines = key.strip().split('\n')
        if (len(lines) > 3 and
            lines[0].startswith('-----BEGIN') and
            lines[-1].startswith('-----END')):
            logger.info(f"PEM key properly formatted ({len(lines)} lines)")
            return key.strip() + '\n'

        # Try to reconstruct malformed key
        logger.info("Attempting to reconstruct PEM key...")
        match = re.match(r'(-----BEGIN [A-Z ]+-----)(.+)(-----END [A-Z ]+-----)',
                        key.replace(' ', '').replace('\n', ''))
        if match:
            header, data, footer = match.groups()
            data_lines = [data[i:i+64] for i in range(0, len(data), 64)]
            key = header + '\n' + '\n'.join(data_lines) + '\n' + footer + '\n'
            logger.info("PEM key reconstructed")
        return key

    def _sign_request(self, method: str, path: str, timestamp_ms: int) -> str:
        """Generate RSA-PSS signature for Kalshi API request."""
        # Message format: timestamp_ms + method + path (per Kalshi docs)
        message = f"{timestamp_ms}{method}{path}"
        self.logger.debug(f"Signing message: {message}")

        # Use PSS.DIGEST_LENGTH constant as per official Kalshi SDK
        signature = self.private_key.sign(
            message.encode('utf-8'),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )

        return base64.b64encode(signature).decode('utf-8')

    def _rate_limit(self):
        """Enforce rate limiting across all API instances (100ms between calls)."""
        global _last_api_call_time
        with _rate_limit_lock:
            current_time_ms = int(time.time() * 1000)
            elapsed = current_time_ms - _last_api_call_time
            if elapsed < RATE_LIMIT_MS:
                sleep_time = (RATE_LIMIT_MS - elapsed) / 1000.0
                time.sleep(sleep_time)
            _last_api_call_time = int(time.time() * 1000)

    def _get_corrected_timestamp_ms(self) -> int:
        """Get current timestamp in milliseconds, corrected for clock drift."""
        global _time_offset_ms
        with _time_offset_lock:
            return int(time.time() * 1000) + _time_offset_ms

    def _update_time_offset(self, response):
        """Update clock drift offset based on server's Date header."""
        global _time_offset_ms
        server_date = response.headers.get('Date')
        if server_date:
            try:
                from email.utils import parsedate_to_datetime
                server_time = parsedate_to_datetime(server_date)
                server_ms = int(server_time.timestamp() * 1000)
                local_ms = int(time.time() * 1000)
                drift = server_ms - local_ms

                with _time_offset_lock:
                    # Only update if drift is significant (> 500ms)
                    if abs(drift) > 500 and abs(drift - _time_offset_ms) > 100:
                        direction = "server ahead of local" if drift > 0 else "local ahead of server"
                        self.logger.warning(f"Clock drift detected: {drift}ms ({direction})")
                        _time_offset_ms = drift
            except Exception as e:
                self.logger.debug(f"Could not parse server Date header: {e}")

    def _make_request(self, method: str, endpoint: str, data: dict = None) -> dict:
        """Make an authenticated request to the Kalshi API."""
        # Rate limit to avoid exceeding Kalshi API limits
        self._rate_limit()

        # Build full URL: host + /trade-api/v2 + endpoint
        url = f"{self.host}/trade-api/v2{endpoint}"

        # For signing: MUST include /trade-api/v2, strip query params
        # Per Kalshi docs: URL /trade-api/v2/markets?limit=100 signs as /trade-api/v2/markets
        path_for_signing = f"/trade-api/v2{endpoint.split('?')[0]}"

        # Generate timestamp in milliseconds, corrected for any detected clock drift
        timestamp_ms = self._get_corrected_timestamp_ms()

        # Build the message string for signing (for debug logging)
        msg_string = f"{timestamp_ms}{method.upper()}{path_for_signing}"

        self.logger.debug(f"URL: {url}")
        self.logger.debug(f"Signing message: {msg_string}")
        self.logger.debug(f"Timestamp (ms): {timestamp_ms}, Method: {method.upper()}, Path: {path_for_signing}")
        signature = self._sign_request(method.upper(), path_for_signing, timestamp_ms)

        headers = {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
        }

        response = None
        try:
            if method.upper() == "GET":
                response = self.session.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = self.session.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, headers=headers, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # Update clock drift offset from server response
            self._update_time_offset(response)

            if response.status_code == 401:
                # Log detailed debug info for auth failures
                self.logger.error(f"Authentication failed: {response.text}")
                self.logger.error(f"DEBUG - Timestamp sent: {timestamp_ms}")
                self.logger.error(f"DEBUG - Message signed: {msg_string}")
                self.logger.error(f"DEBUG - Path: {path_for_signing}")
                self.logger.error(f"DEBUG - API Key (first 8 chars): {self.api_key[:8]}...")
                raise Exception(f"Authentication error: {response.text}")

            response.raise_for_status()
            return response.json() if response.text else {}

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            raise

    def logout(self):
        """Clean up session resources."""
        try:
            self.session.close()
        except Exception:
            pass
        self.logger.info("Session ended (connection pool closed)")

    def get_position(self) -> int:
        self.logger.info("Retrieving position...")
        try:
            response = self._make_request("GET", f"/portfolio/positions?ticker={self.market_ticker}")
            positions = response.get("market_positions", [])

            total_position = 0
            for position in positions:
                if position.get("ticker") == self.market_ticker:
                    total_position += position.get("position", 0)

            self.logger.info(f"Current position: {total_position}")
            return total_position
        except Exception as e:
            self.logger.error(f"Failed to get position: {e}")
            raise

    def get_position_with_cost(self) -> Dict:
        """Get position with cost basis information.

        Returns a dict with:
        - position: int (positive = long YES, negative = long NO)
        - yes_cost: float (average cost of YES position in dollars, 0 if no YES position)
        - no_cost: float (average cost of NO position in dollars, 0 if no NO position)
        """
        self.logger.info("Retrieving position with cost basis...")
        try:
            response = self._make_request("GET", f"/portfolio/positions?ticker={self.market_ticker}")
            positions = response.get("market_positions", [])

            result = {"position": 0, "yes_cost": 0.0, "no_cost": 0.0}

            for pos in positions:
                if pos.get("ticker") == self.market_ticker:
                    result["position"] += pos.get("position", 0)

                    # Kalshi API returns costs in cents
                    # market_exposure is total cost, resting_orders_count for pending
                    # For cost basis, we use the realized_pnl info or calculate from exposure
                    # The API may return: total_traded, fees_paid, etc.
                    # Most reliable: use market_exposure / abs(position) if available

                    # Check for YES position cost
                    yes_qty = pos.get("yes_position", 0) or pos.get("position", 0)
                    if yes_qty > 0:
                        # market_exposure is total dollars spent (in cents)
                        exposure_cents = pos.get("market_exposure", 0)
                        if exposure_cents and yes_qty > 0:
                            result["yes_cost"] = round(exposure_cents / 100 / yes_qty, 2)
                        # Fallback: check for average_buy_price if available
                        avg_price = pos.get("average_buy_price")
                        if avg_price:
                            result["yes_cost"] = round(avg_price / 100, 2)

                    # Check for NO position cost (position < 0 means long NO)
                    # First try explicit no_position field, then derive from negative position
                    no_qty = pos.get("no_position", 0)
                    position_val = pos.get("position", 0)
                    if not no_qty and position_val < 0:
                        # Negative position means long NO
                        no_qty = abs(position_val)
                    if no_qty and no_qty > 0:
                        exposure_cents = pos.get("market_exposure", 0)
                        if exposure_cents and no_qty > 0:
                            result["no_cost"] = round(exposure_cents / 100 / no_qty, 2)
                        # Fallback: check for average_buy_price (used for the current position side)
                        avg_price = pos.get("average_buy_price")
                        if avg_price and position_val < 0:
                            result["no_cost"] = round(avg_price / 100, 2)

            self.logger.info(f"Position: {result['position']}, YES cost: ${result['yes_cost']:.2f}, "
                           f"NO cost: ${result['no_cost']:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get position with cost: {e}")
            raise

    def get_price(self) -> Dict[str, float]:
        self.logger.info("Retrieving market data...")
        try:
            response = self._make_request("GET", f"/markets/{self.market_ticker}")
            market = response.get("market", {})

            # Get raw bid/ask values (in cents)
            yes_bid_raw = market.get("yes_bid", 0) or 0
            yes_ask_raw = market.get("yes_ask", 0) or 0
            no_bid_raw = market.get("no_bid", 0) or 0
            no_ask_raw = market.get("no_ask", 0) or 0

            # Log raw values for diagnostics
            self.logger.debug(f"Raw market data: yes_bid={yes_bid_raw}, yes_ask={yes_ask_raw}, "
                             f"no_bid={no_bid_raw}, no_ask={no_ask_raw}")

            # Convert to decimal (0-1 scale)
            yes_bid = float(yes_bid_raw) / 100
            yes_ask = float(yes_ask_raw) / 100
            no_bid = float(no_bid_raw) / 100
            no_ask = float(no_ask_raw) / 100

            # Calculate implied YES prices from NO side (for cross-validation)
            # In Kalshi: YES_price + NO_price ≈ $1
            # So: YES_bid ≈ 1 - NO_ask and YES_ask ≈ 1 - NO_bid
            implied_yes_bid = round(1 - no_ask, 2) if no_ask > 0 else 0
            implied_yes_ask = round(1 - no_bid, 2) if no_bid > 0 else 0

            # Calculate mid-price with improved fallback logic
            if yes_bid > 0 and yes_ask > 0:
                # Normal case: both bid and ask exist
                yes_mid_price = round((yes_bid + yes_ask) / 2, 2)
            elif yes_bid > 0 or yes_ask > 0:
                # One-sided YES book - use available data plus NO-side cross-validation
                if yes_bid > 0 and implied_yes_ask > 0:
                    # Have YES bid and implied ask from NO side
                    yes_mid_price = round((yes_bid + implied_yes_ask) / 2, 2)
                    self.logger.warning(f"No YES asks, using NO-implied mid: ${yes_mid_price:.2f}")
                elif yes_ask > 0 and implied_yes_bid > 0:
                    # Have YES ask and implied bid from NO side
                    yes_mid_price = round((implied_yes_bid + yes_ask) / 2, 2)
                    self.logger.warning(f"No YES bids, using NO-implied mid: ${yes_mid_price:.2f}")
                elif yes_ask > 0:
                    # Only YES ask exists - use proportional buffer (10% of price)
                    buffer = max(0.01, yes_ask * 0.10)
                    yes_mid_price = round(yes_ask - buffer, 2)
                    self.logger.warning(f"No YES bids or NO data, using ask-based mid: ${yes_mid_price:.2f}")
                else:
                    # Only YES bid exists - use proportional buffer (10% of remaining)
                    buffer = max(0.01, (1 - yes_bid) * 0.10)
                    yes_mid_price = round(yes_bid + buffer, 2)
                    self.logger.warning(f"No YES asks or NO data, using bid-based mid: ${yes_mid_price:.2f}")
            elif implied_yes_bid > 0 or implied_yes_ask > 0:
                # No YES data but have NO data - derive from NO side
                if implied_yes_bid > 0 and implied_yes_ask > 0:
                    yes_mid_price = round((implied_yes_bid + implied_yes_ask) / 2, 2)
                    self.logger.warning(f"No YES data, using NO-derived mid: ${yes_mid_price:.2f}")
                elif implied_yes_bid > 0:
                    buffer = max(0.01, (1 - implied_yes_bid) * 0.10)
                    yes_mid_price = round(implied_yes_bid + buffer, 2)
                    self.logger.warning(f"Only NO ask exists, using derived mid: ${yes_mid_price:.2f}")
                else:
                    buffer = max(0.01, implied_yes_ask * 0.10)
                    yes_mid_price = round(implied_yes_ask - buffer, 2)
                    self.logger.warning(f"Only NO bid exists, using derived mid: ${yes_mid_price:.2f}")
            else:
                # No bid or ask on either side - try last_price or default to 0.50
                last_price = market.get("last_price", 0) or 0
                if last_price > 0:
                    yes_mid_price = round(float(last_price) / 100, 2)
                    self.logger.warning(f"No bid/ask data, using last_price: ${yes_mid_price:.2f}")
                else:
                    yes_mid_price = 0.50
                    self.logger.warning(f"No market data available, defaulting to ${yes_mid_price:.2f}")

            # Ensure mid-price is within valid bounds (expanded to allow near-boundary pricing)
            yes_mid_price = max(0.01, min(0.99, yes_mid_price))
            no_mid_price = round(1 - yes_mid_price, 2)

            self.logger.info(f"Market mid-prices: YES=${yes_mid_price:.2f}, NO=${no_mid_price:.2f}")
            self.logger.info(f"Market bid/ask: YES bid=${yes_bid:.2f}, YES ask=${yes_ask:.2f}")
            return {
                "yes": yes_mid_price,
                "no": no_mid_price,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "no_bid": no_bid,
                "no_ask": no_ask,
            }
        except Exception as e:
            self.logger.error(f"Failed to get price: {e}")
            raise

    def place_order(self, action: str, side: str, price: float, quantity: int, expiration_ts: int = None) -> str:
        self.logger.info(f"Placing {action} order for {side} side at price ${price:.2f} with quantity {quantity}...")
        try:
            price_cents = int(price * 100)

            order_data = {
                "ticker": self.market_ticker,
                "action": action.lower(),
                "type": "limit",
                "side": side,
                "count": quantity,
                "client_order_id": str(uuid.uuid4()),
            }

            if side == "yes":
                order_data["yes_price"] = price_cents
            else:
                order_data["no_price"] = price_cents

            if expiration_ts is not None:
                order_data["expiration_ts"] = expiration_ts

            response = self._make_request("POST", "/portfolio/orders", order_data)
            order_id = response.get("order", {}).get("order_id")
            self.logger.info(f"Placed {action} order, order ID: {order_id}")
            return str(order_id)
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    def cancel_order(self, order_id: int) -> bool:
        self.logger.info(f"Canceling order with ID {order_id}...")
        try:
            response = self._make_request("DELETE", f"/portfolio/orders/{order_id}")
            success = response.get("reduced_by", 0) > 0
            self.logger.info(f"Canceled order with ID {order_id}, success: {success}")
            return success
        except Exception as e:
            self.logger.error(f"Failed to cancel order: {e}")
            raise

    def get_orders(self) -> List[Dict]:
        self.logger.info("Retrieving orders...")
        try:
            response = self._make_request("GET", f"/portfolio/orders?ticker={self.market_ticker}&status=resting")
            orders = response.get("orders", [])
            self.logger.info(f"Retrieved {len(orders)} open orders")
            return orders
        except Exception as e:
            self.logger.error(f"Failed to get orders: {e}")
            raise

    def get_market_info(self) -> Dict:
        """Get market metadata including min_tick_size.

        Returns a dict with:
        - min_tick_size: float (e.g., 0.01 for 1 cent, 0.005 for half-cent markets)
        - ticker: str
        - status: str
        - Other market metadata
        """
        self.logger.info("Retrieving market info...")
        try:
            response = self._make_request("GET", f"/markets/{self.market_ticker}")
            market = response.get("market", {})

            # min_tick_size is returned as cents in the API, convert to dollars
            # e.g., 1 (cent) -> 0.01, 0.5 (half cent) -> 0.005
            tick_cents = market.get("tick_size", 1) or 1
            min_tick_size = tick_cents / 100.0

            result = {
                "min_tick_size": min_tick_size,
                "ticker": market.get("ticker", self.market_ticker),
                "status": market.get("status", "unknown"),
                "title": market.get("title", ""),
            }
            self.logger.info(f"Market info: tick_size=${min_tick_size:.4f} ({tick_cents}¢)")
            return result
        except Exception as e:
            self.logger.error(f"Failed to get market info: {e}")
            raise

    def decrease_order(self, order_id: str, reduce_by: int) -> bool:
        """Decrease an order's quantity without losing queue priority.

        Uses POST /portfolio/orders/{order_id}/decrease to reduce order size
        while maintaining time priority in the order book.

        Args:
            order_id: The order ID to decrease
            reduce_by: Number of contracts to reduce by

        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Decreasing order {order_id} by {reduce_by} contracts...")
        try:
            data = {"reduce_by": reduce_by}
            response = self._make_request("POST", f"/portfolio/orders/{order_id}/decrease", data)
            reduced = response.get("order", {}).get("remaining_count", 0)
            self.logger.info(f"Order {order_id} decreased, remaining: {reduced}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to decrease order {order_id}: {e}")
            return False

    def place_order_subpenny(
        self,
        action: str,
        side: str,
        price: float,
        quantity: int,
        expiration_ts: int = None,
        use_dollars: bool = False
    ) -> str:
        """Place an order with sub-penny support using yes_price_dollars/no_price_dollars.

        For markets with min_tick_size < 0.01 (e.g., 0.005 for half-cent markets),
        use yes_price_dollars/no_price_dollars as strings for precise pricing.

        Args:
            action: 'buy' or 'sell'
            side: 'yes' or 'no'
            price: Price in dollars (e.g., 0.455 for 45.5 cents)
            quantity: Number of contracts
            expiration_ts: Optional expiration timestamp
            use_dollars: If True, use *_price_dollars fields for sub-penny precision

        Returns:
            Order ID as string
        """
        price_str = f"{price:.3f}"
        self.logger.info(f"Placing {action} order for {side} side at ${price_str} with quantity {quantity}...")
        try:
            order_data = {
                "ticker": self.market_ticker,
                "action": action.lower(),
                "type": "limit",
                "side": side,
                "count": quantity,
                "client_order_id": str(uuid.uuid4()),
            }

            if use_dollars:
                # Use *_price_dollars for sub-penny precision (string format)
                if side == "yes":
                    order_data["yes_price_dollars"] = price_str
                else:
                    order_data["no_price_dollars"] = price_str
            else:
                # Standard integer cents pricing
                price_cents = int(round(price * 100))
                if side == "yes":
                    order_data["yes_price"] = price_cents
                else:
                    order_data["no_price"] = price_cents

            if expiration_ts is not None:
                order_data["expiration_ts"] = expiration_ts

            response = self._make_request("POST", "/portfolio/orders", order_data)
            order_id = response.get("order", {}).get("order_id")
            self.logger.info(f"Placed {action} order, order ID: {order_id}")
            return str(order_id)
        except Exception as e:
            self.logger.error(f"Failed to place order: {e}")
            raise

    def get_active_markets_by_series(self, series_ticker: str) -> List[Dict]:
        """Get all open markets for a series."""
        self.logger.info(f"Fetching markets for series: {series_ticker}")
        try:
            response = self._make_request("GET", f"/markets?series_ticker={series_ticker}&status=open")
            markets = response.get("markets", [])
            self.logger.info(f"Found {len(markets)} open markets in series {series_ticker}")
            return markets
        except Exception as e:
            self.logger.error(f"Failed to fetch markets for series {series_ticker}: {e}")
            raise

    def get_active_markets_by_category(self, category: str = "Sports", max_markets: int = 500) -> List[Dict]:
        """Get open markets for a category (e.g., 'Sports').

        This method fetches sports markets without needing to know specific series
        tickers in advance. It handles pagination with a configurable limit.

        Uses mve_filter=exclude to filter out multivariate (combo/parlay) markets
        at the API level, which is more efficient than filtering after fetch.

        Args:
            category: The market category to fetch (default: "Sports")
            max_markets: Maximum number of markets to fetch to prevent unbounded memory
                        usage (default: 500). Set to 0 for unlimited.

        Returns:
            List of market dictionaries containing ticker, title, and other market info
        """
        self.logger.info(f"Fetching markets for category: {category} (max: {max_markets if max_markets else 'unlimited'})")
        all_markets = []
        cursor = None

        try:
            while True:
                # Build endpoint with pagination support
                # mve_filter=exclude tells Kalshi to exclude multivariate (parlay/combo) markets
                endpoint = f"/markets?category={category}&status=open&limit=200&mve_filter=exclude"
                if cursor:
                    endpoint += f"&cursor={cursor}"

                response = self._make_request("GET", endpoint)
                markets = response.get("markets", [])
                all_markets.extend(markets)

                # Check if we've hit the max limit
                if max_markets and len(all_markets) >= max_markets:
                    all_markets = all_markets[:max_markets]
                    self.logger.info(f"Reached max_markets limit ({max_markets}), stopping pagination")
                    break

                # Check for pagination cursor
                cursor = response.get("cursor")
                if not cursor or not markets:
                    break

                self.logger.info(f"Fetched {len(all_markets)} markets so far, continuing pagination...")

            self.logger.info(f"Found {len(all_markets)} total open markets in category {category}")
            return all_markets

        except Exception as e:
            self.logger.error(f"Failed to fetch markets for category {category}: {e}")
            raise


class AvellanedaMarketMaker:
    """Market maker using Avellaneda-Stoikov strategy with Desired State Engine.

    This implementation is calibrated for binary probability markets (0.00 to 1.00 scale)
    and uses a dual-quote strategy that places BUY orders on both YES and NO sides
    to capture spread from whichever direction the market moves.

    Key features:
    - Desired State Engine: Reconciles orders instead of cancel-and-replace
    - Sub-penny support: Uses yes_price_dollars for half-cent markets
    - Queue priority preservation: Uses decrease endpoint, 10s MIN_AGE filter
    - Tick-aware pricing: 2¢ spread for 1¢ ticks, 1¢ spread for 0.5¢ ticks
    - Decimal-scale aligned parameters (gamma scaled to 0.01-0.05 range)
    - Auto-exit: if one side fills but opposite doesn't within timeout, exits at small profit
    """

    # Minimum order age before modification (preserve queue priority)
    MIN_ORDER_AGE = 10.0  # seconds

    def __init__(
        self,
        logger: logging.Logger,
        api: AbstractTradingAPI,
        gamma: float = 0.02,           # Risk aversion - scaled down for 0-1 scale
        k: float = 1.5,                # Liquidity parameter
        sigma: float = 0.10,           # Volatility estimate for binary outcomes
        T: float = 3600,               # Time horizon in seconds
        max_position: int = 5,         # Max contracts per side
        order_expiration: int = 300,   # Order TTL in seconds
        min_spread: float = 0.02,      # Minimum spread ($0.02) - for 1¢ tick markets
        max_spread: float = 0.10,      # Maximum spread ($0.10)
        position_limit_buffer: float = 0.1,
        flip_skew_factor: float = 0.03,  # Aggressive skew for inventory flipping
        exit_timeout: float = 30.0,    # Seconds before auto-exit kicks in
        exit_profit_target: float = 0.02,  # Target profit when exiting ($0.02)
    ):
        self.logger = logger
        self.api = api
        self.gamma = gamma
        self.k = k
        self.sigma = sigma
        self.T = T
        self.max_position = max_position
        self.order_expiration = order_expiration
        self.min_spread = min_spread
        self.max_spread = max_spread
        self.position_limit_buffer = position_limit_buffer
        self.flip_skew_factor = flip_skew_factor
        self.exit_timeout = exit_timeout
        self.exit_profit_target = exit_profit_target

        self.t = 0
        self.active_yes_order_id = None
        self.active_no_order_id = None

        # Market tick size tracking (fetched once at startup)
        self.min_tick_size = None  # Will be set on first iteration
        self.use_subpenny = False  # True if min_tick_size < 0.01

        # Order tracking for Desired State Engine
        # Format: {'order_id': str, 'price': float, 'side': str, 'created_at': float, 'count': int}
        self.tracked_orders = {}  # keyed by side ('yes' or 'no')

        # Fill tracking for auto-exit feature
        # Tracks the last known position to detect fills
        self.last_position = None
        # Tracks pending fills that need to be exited if opposite side doesn't fill
        # Format: {'side': 'yes'|'no', 'entry_price': float, 'fill_time': float, 'quantity': int}
        self.pending_exit = None
        # Track if we have an active exit order
        self.active_exit_order_id = None
        # Track if we're in dual exit mode (selling owned side + bidding on opposite)
        self.in_dual_exit_mode = False
        # Track if we've completed a pair trade (hold both YES and NO until expiry)
        self.holding_pair = False

    def compute_reservation_price(self, mid_price: float, q: int, t: float) -> float:
        """Compute reservation price with inventory adjustment.

        The reservation price is the market maker's indifference price given
        current inventory. With positive inventory (long), the MM wants to
        sell, so reservation price is lowered.

        Formula: r(t) = S(t) - q * gamma * sigma^2 * (T - t)

        For binary markets with small gamma (0.01-0.05), this provides
        gentle pressure to reduce inventory without extreme price swings.
        """
        time_factor = max(0.001, self.T - t)
        # Scale the time factor to be reasonable for our horizon
        normalized_time = time_factor / self.T
        r = mid_price - q * self.gamma * (self.sigma ** 2) * normalized_time
        return r

    def compute_optimal_spread(self, t: float) -> float:
        """Compute optimal spread based on Avellaneda-Stoikov model.

        Formula: delta(t) = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)

        The spread is clamped to [min_spread, max_spread] to ensure:
        - We always maintain a minimum spread to cover fees
        - We never have spreads so wide they're unfillable
        """
        time_factor = max(0.001, self.T - t)
        normalized_time = time_factor / self.T

        # Time-dependent component (shrinks as T approaches)
        time_spread = self.gamma * (self.sigma ** 2) * normalized_time

        # Liquidity component (constant based on risk/liquidity tradeoff)
        liquidity_spread = (2 / self.gamma) * math.log(1 + self.gamma / self.k)

        raw_spread = time_spread + liquidity_spread

        # Clamp spread to [min_spread, max_spread]
        spread = max(self.min_spread, min(self.max_spread, raw_spread))

        self.logger.debug(f"Spread calc: time={time_spread:.4f}, liq={liquidity_spread:.4f}, "
                         f"raw={raw_spread:.4f}, clamped={spread:.4f}")
        return spread

    def compute_dual_quotes(
        self,
        yes_mid: float,
        q: int,
        t: float,
        market_yes_bid: float = 0,
        market_yes_ask: float = 0,
        market_no_bid: float = 0,
        market_no_ask: float = 0,
    ) -> Tuple[float, float]:
        """Compute bid prices for both YES and NO contracts (Dual-Quote strategy).

        Returns:
            Tuple of (yes_bid_price, no_bid_price)

        The dual-quote strategy places BUY orders on both sides:
        - Buy YES at yes_bid_price
        - Buy NO at no_bid_price

        When long YES (q > 0), buying NO is equivalent to selling YES,
        so we make the NO bid more aggressive to encourage fills that
        reduce our YES exposure.

        Asymmetric Inventory Urgency:
        - If q > 0 (long YES): NO bid is closer to mid (aggressive), YES bid is farther
        - If q < 0 (long NO): YES bid is closer to mid (aggressive), NO bid is farther

        Market bid/ask values are used to ensure our bids are competitive - we
        bid at least at the current market bid level to have a chance of getting filled.
        """
        r = self.compute_reservation_price(yes_mid, q, t)
        spread = self.compute_optimal_spread(t)
        half_spread = spread / 2

        # Base quotes around reservation price
        base_yes_bid = r - half_spread
        base_yes_ask = r + half_spread  # Used to derive NO bid

        # Apply asymmetric inventory urgency (flip skew)
        if q > 0:
            # Long YES: want to flip by buying NO (which closes YES position)
            # Make NO bid aggressive (closer to mid), YES bid defensive (farther from mid)
            flip_adjustment = q * self.flip_skew_factor
            yes_bid = base_yes_bid - flip_adjustment  # Push YES bid down (less aggressive)
            # NO bid derived from YES ask, pushed up (more aggressive to fill)
            no_bid = (1 - base_yes_ask) + flip_adjustment
        elif q < 0:
            # Long NO (short YES): want to flip by buying YES
            # Make YES bid aggressive, NO bid defensive
            flip_adjustment = abs(q) * self.flip_skew_factor
            yes_bid = base_yes_bid + flip_adjustment  # Push YES bid up (more aggressive)
            no_bid = (1 - base_yes_ask) - flip_adjustment  # Push NO bid down (less aggressive)
        else:
            # Neutral inventory: symmetric quotes
            yes_bid = base_yes_bid
            no_bid = 1 - base_yes_ask

        # Round to cents
        yes_bid = round(yes_bid, 2)
        no_bid = round(no_bid, 2)

        # Ensure competitive pricing: bids must be at least at current market bid
        # to join the queue and have a chance of getting filled.
        # Bidding below the market bid means our order sits behind existing bidders
        # and will likely never fill.
        if market_yes_bid > 0 and yes_bid < market_yes_bid:
            self.logger.info(
                f"YES bid ${yes_bid:.2f} below market bid ${market_yes_bid:.2f}, "
                f"raising to market bid"
            )
            yes_bid = market_yes_bid

        if market_no_bid > 0 and no_bid < market_no_bid:
            self.logger.info(
                f"NO bid ${no_bid:.2f} below market bid ${market_no_bid:.2f}, "
                f"raising to market bid"
            )
            no_bid = market_no_bid

        # Apply boundary recalculation if quotes hit extremes
        yes_bid, no_bid = self._apply_boundary_recalculation(
            yes_mid, yes_bid, no_bid, market_yes_bid, market_no_bid
        )

        return yes_bid, no_bid

    def _apply_boundary_recalculation(
        self,
        yes_mid: float,
        yes_bid: float,
        no_bid: float,
        market_yes_bid: float = 0,
        market_no_bid: float = 0,
    ) -> Tuple[float, float]:
        """Apply hard adaptive caps - recalculate if quotes hit boundaries.

        If the calculated quote hits 0.01 or 0.99, force recalculation
        to the current market bid to ensure competitive, fillable orders.
        """
        no_mid = 1 - yes_mid

        # Check YES bid against boundaries
        if yes_bid <= 0.01 or yes_bid >= 0.99:
            # Use market bid if available, otherwise fall back to mid-based calculation
            if market_yes_bid > 0:
                yes_bid = market_yes_bid
            else:
                yes_bid = round(yes_mid - self.min_spread / 2, 2)
            self.logger.info(f"YES bid hit boundary, recalculated to ${yes_bid:.2f}")

        # Check NO bid against boundaries
        if no_bid <= 0.01 or no_bid >= 0.99:
            # Use market bid if available, otherwise fall back to mid-based calculation
            if market_no_bid > 0:
                no_bid = market_no_bid
            else:
                no_bid = round(no_mid - self.min_spread / 2, 2)
            self.logger.info(f"NO bid hit boundary, recalculated to ${no_bid:.2f}")

        # Final clamp to valid range [0.02, 0.98] to stay off the book edges
        yes_bid = max(0.02, min(0.98, yes_bid))
        no_bid = max(0.02, min(0.98, no_bid))

        return yes_bid, no_bid

    def _fetch_tick_size(self) -> None:
        """Fetch and cache the market's min_tick_size on first iteration."""
        if self.min_tick_size is not None:
            return  # Already fetched

        try:
            market_info = self.api.get_market_info()
            self.min_tick_size = market_info.get("min_tick_size", 0.01)
            self.use_subpenny = self.min_tick_size < 0.01

            if self.use_subpenny:
                self.logger.info(f"SUB-PENNY MARKET: tick_size=${self.min_tick_size:.4f}, "
                               f"using 1¢ spread with half-cent precision")
            else:
                self.logger.info(f"STANDARD MARKET: tick_size=${self.min_tick_size:.4f}, "
                               f"using 2¢ spread")
        except Exception as e:
            self.logger.warning(f"Failed to fetch tick size, defaulting to 1¢: {e}")
            self.min_tick_size = 0.01
            self.use_subpenny = False

    def compute_tick_aware_quotes(self, yes_mid: float, q: int, t: float,
                                   market_yes_bid: float = 0, market_no_bid: float = 0) -> Tuple[float, float]:
        """Compute bid prices based on market tick size.

        Pricing Strategy:
        - If min_tick_size is 0.01 (1 cent): Set bid 1 penny below mid (2 cent spread)
        - If min_tick_size < 0.01 (half cent): Set bid 0.5 cents below mid (1 cent spread)

        Returns:
            Tuple of (yes_bid_price, no_bid_price)
        """
        no_mid = 1 - yes_mid

        if self.use_subpenny:
            # Half-cent market: bid 0.5¢ below mid for 1¢ spread
            half_spread = 0.005  # $0.005 = 0.5 cents

            # Round to nearest half-cent
            yes_bid = round((yes_mid - half_spread) * 200) / 200  # Round to 0.005
            no_bid = round((no_mid - half_spread) * 200) / 200

            self.logger.debug(f"Sub-penny quotes: YES mid=${yes_mid:.3f} -> bid=${yes_bid:.3f}, "
                            f"NO mid=${no_mid:.3f} -> bid=${no_bid:.3f}")
        else:
            # Standard 1-cent market: bid 1¢ below mid for 2¢ spread
            half_spread = 0.01  # $0.01 = 1 cent

            # Round to nearest cent
            yes_bid = round(yes_mid - half_spread, 2)
            no_bid = round(no_mid - half_spread, 2)

            self.logger.debug(f"Standard quotes: YES mid=${yes_mid:.2f} -> bid=${yes_bid:.2f}, "
                            f"NO mid=${no_mid:.2f} -> bid=${no_bid:.2f}")

        # Apply inventory skew for position management
        if q != 0:
            flip_adj = abs(q) * self.flip_skew_factor
            if q > 0:
                # Long YES: make NO bid more aggressive
                no_bid = no_bid + flip_adj
                yes_bid = yes_bid - flip_adj
            else:
                # Long NO: make YES bid more aggressive
                yes_bid = yes_bid + flip_adj
                no_bid = no_bid - flip_adj

            # Re-round after adjustment
            if self.use_subpenny:
                yes_bid = round(yes_bid * 200) / 200
                no_bid = round(no_bid * 200) / 200
            else:
                yes_bid = round(yes_bid, 2)
                no_bid = round(no_bid, 2)

        # Ensure we're at least at market bid (competitive pricing)
        if market_yes_bid > 0 and yes_bid < market_yes_bid:
            self.logger.info(f"YES bid ${yes_bid:.3f} below market ${market_yes_bid:.3f}, raising")
            yes_bid = market_yes_bid

        if market_no_bid > 0 and no_bid < market_no_bid:
            self.logger.info(f"NO bid ${no_bid:.3f} below market ${market_no_bid:.3f}, raising")
            no_bid = market_no_bid

        # Final clamp
        yes_bid = max(0.02, min(0.98, yes_bid))
        no_bid = max(0.02, min(0.98, no_bid))

        return yes_bid, no_bid

    def _price_matches(self, price1: float, price2: float) -> bool:
        """Check if two prices match within tick tolerance."""
        tolerance = self.min_tick_size if self.min_tick_size else 0.01
        return abs(price1 - price2) < tolerance / 2

    def _sync_tracked_orders(self) -> None:
        """Sync local order tracking with actual orders from API.

        Updates self.tracked_orders to reflect current resting orders.
        """
        try:
            api_orders = self.api.get_orders()
            current_time = time.time()

            # Build a map of current orders by side
            api_orders_by_side = {'yes': None, 'no': None}
            for order in api_orders:
                side = order.get('side', '')
                if side in ('yes', 'no') and order.get('action') == 'buy':
                    # Get price - check both cents and dollars formats
                    if side == 'yes':
                        price = order.get('yes_price', 0) / 100.0
                        if 'yes_price_dollars' in order:
                            price = float(order['yes_price_dollars'])
                    else:
                        price = order.get('no_price', 0) / 100.0
                        if 'no_price_dollars' in order:
                            price = float(order['no_price_dollars'])

                    api_orders_by_side[side] = {
                        'order_id': order.get('order_id'),
                        'price': price,
                        'side': side,
                        'count': order.get('remaining_count', 1),
                        'created_at': order.get('created_time', current_time),
                    }

            # Update tracked orders
            for side in ('yes', 'no'):
                api_order = api_orders_by_side[side]
                tracked = self.tracked_orders.get(side)

                if api_order is None:
                    # Order no longer exists
                    if tracked:
                        self.logger.debug(f"Order for {side} side no longer exists, clearing tracking")
                    self.tracked_orders[side] = None
                elif tracked is None or tracked['order_id'] != api_order['order_id']:
                    # New order or different order - update tracking
                    # Parse created_time if it's a string
                    created_at = api_order['created_at']
                    if isinstance(created_at, str):
                        try:
                            from datetime import datetime
                            dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            created_at = dt.timestamp()
                        except:
                            created_at = current_time
                    self.tracked_orders[side] = {
                        'order_id': api_order['order_id'],
                        'price': api_order['price'],
                        'side': side,
                        'count': api_order['count'],
                        'created_at': created_at,
                    }
                else:
                    # Same order - update count (may have partially filled)
                    tracked['count'] = api_order['count']

        except Exception as e:
            self.logger.warning(f"Failed to sync tracked orders: {e}")

    def reconcile_orders(self, desired_yes_price: float, desired_no_price: float,
                         desired_count: int = 1, position: int = 0) -> None:
        """Reconcile current orders with desired state (Desired State Engine).

        This is the core of the Desired State Engine:
        1. Compare desired price to active orders
        2. If price matches: Do NOT cancel (preserve queue priority)
           - If count is too high, use decrease endpoint
        3. If price is different AND order age > MIN_AGE: Cancel and replace
        4. If price is different AND order age < MIN_AGE: Wait (don't jump out of line)

        Args:
            desired_yes_price: Target YES bid price
            desired_no_price: Target NO bid price
            desired_count: Desired order quantity (default 1)
            position: Current inventory position
        """
        current_time = time.time()
        position_limit = int(self.max_position * (1 - self.position_limit_buffer))
        expiration_ts = int(current_time) + self.order_expiration

        # Sync our tracking with actual API orders
        self._sync_tracked_orders()

        # Process YES side
        if position < position_limit:
            self._reconcile_side('yes', desired_yes_price, desired_count, expiration_ts, current_time)
        else:
            # At position limit - cancel any YES orders
            self._cancel_side_if_exists('yes')

        # Process NO side
        if position > -position_limit:
            self._reconcile_side('no', desired_no_price, desired_count, expiration_ts, current_time)
        else:
            # At position limit - cancel any NO orders
            self._cancel_side_if_exists('no')

    def _reconcile_side(self, side: str, desired_price: float, desired_count: int,
                        expiration_ts: int, current_time: float) -> None:
        """Reconcile orders for a single side (YES or NO).

        Args:
            side: 'yes' or 'no'
            desired_price: Target price for this side
            desired_count: Desired quantity
            expiration_ts: Order expiration timestamp
            current_time: Current time for age calculations
        """
        tracked = self.tracked_orders.get(side)

        if tracked is None:
            # No existing order - place new one
            self.logger.info(f"No existing {side.upper()} order, placing at ${desired_price:.3f}")
            self._place_order(side, desired_price, desired_count, expiration_ts)
            return

        # Check if price matches
        if self._price_matches(tracked['price'], desired_price):
            # Price matches - preserve queue priority
            self.logger.debug(f"{side.upper()} order at ${tracked['price']:.3f} matches desired, keeping")

            # Check if we need to decrease quantity
            if tracked['count'] > desired_count:
                reduce_by = tracked['count'] - desired_count
                self.logger.info(f"{side.upper()} order has {tracked['count']} contracts, "
                               f"reducing by {reduce_by} to {desired_count}")
                self.api.decrease_order(tracked['order_id'], reduce_by)
            return

        # Price doesn't match - check order age
        order_age = current_time - tracked['created_at']
        if order_age < self.MIN_ORDER_AGE:
            # Order too young - don't cancel yet (preserve queue priority)
            remaining = self.MIN_ORDER_AGE - order_age
            self.logger.info(f"{side.upper()} order age {order_age:.1f}s < {self.MIN_ORDER_AGE}s MIN_AGE, "
                           f"waiting {remaining:.1f}s before repricing "
                           f"(current: ${tracked['price']:.3f}, desired: ${desired_price:.3f})")
            return

        # Order is old enough and price changed - cancel and replace
        self.logger.info(f"{side.upper()} price changed: ${tracked['price']:.3f} -> ${desired_price:.3f} "
                        f"(age: {order_age:.1f}s), canceling and replacing")
        try:
            self.api.cancel_order(tracked['order_id'])
        except Exception as e:
            self.logger.warning(f"Failed to cancel {side} order: {e}")

        self._place_order(side, desired_price, desired_count, expiration_ts)

    def _place_order(self, side: str, price: float, count: int, expiration_ts: int) -> None:
        """Place an order and track it locally."""
        try:
            if self.use_subpenny:
                order_id = self.api.place_order_subpenny(
                    "buy", side, price, count, expiration_ts, use_dollars=True
                )
            else:
                order_id = self.api.place_order("buy", side, price, count, expiration_ts)

            # Track the new order locally
            self.tracked_orders[side] = {
                'order_id': order_id,
                'price': price,
                'side': side,
                'count': count,
                'created_at': time.time(),
            }

            # Update legacy tracking
            if side == 'yes':
                self.active_yes_order_id = order_id
            else:
                self.active_no_order_id = order_id

        except Exception as e:
            self.logger.error(f"Failed to place {side} order at ${price:.3f}: {e}")

    def _cancel_side_if_exists(self, side: str) -> None:
        """Cancel an order for a side if it exists."""
        tracked = self.tracked_orders.get(side)
        if tracked:
            try:
                self.api.cancel_order(tracked['order_id'])
                self.tracked_orders[side] = None
                self.logger.info(f"Canceled {side.upper()} order (at position limit)")
            except Exception as e:
                self.logger.warning(f"Failed to cancel {side} order: {e}")

    def _is_exit_order_active(self) -> bool:
        """Check if our exit sell order is still active (not filled).

        Used in dual exit mode to determine if position went to 0 because:
        - Exit order filled (we sold our position) -> return False
        - Opposite side bid filled (pair trade completed) -> return True
        """
        if not self.active_exit_order_id:
            return False

        try:
            orders = self.api.get_orders()
            for order in orders:
                if order.get('order_id') == self.active_exit_order_id:
                    return True  # Exit order still resting, not filled
            return False  # Exit order not in resting orders, must have filled
        except Exception as e:
            self.logger.warning(f"Could not check exit order status: {e}")
            return False  # Assume filled on error

    def detect_fill_and_track(self, current_position: int, yes_bid: float, no_bid: float) -> bool:
        """Detect if a fill occurred and track it for potential auto-exit.

        Returns True if a new fill was detected that requires tracking.
        """
        if self.last_position is None:
            # First iteration - record position
            self.last_position = current_position

            # If we're starting with an existing position, set up exit tracking
            # so we properly manage selling above cost
            if current_position != 0 and self.pending_exit is None:
                fill_side = 'yes' if current_position > 0 else 'no'
                # Use bid price as placeholder - actual cost will be fetched from API when exiting
                entry_price = yes_bid if current_position > 0 else no_bid
                self.pending_exit = {
                    'side': fill_side,
                    'entry_price': entry_price,
                    'fill_time': time.time(),  # Start timeout from now
                    'quantity': abs(current_position),
                }
                self.logger.info(f"STARTUP: Detected existing {fill_side.upper()} position ({current_position}), "
                               f"setting up exit tracking (actual cost will be fetched from API)")
            return False

        position_delta = current_position - self.last_position

        if position_delta == 0:
            # No fill occurred
            return False

        # A fill occurred - determine which side
        if position_delta > 0:
            # Position increased: we bought YES (or NO side order expired/cancelled)
            # This means we're now long YES and need the NO side to fill to flatten
            fill_side = 'yes'
            # Entry price is our YES bid price
            entry_price = yes_bid
            self.logger.info(f"FILL DETECTED: Bought {position_delta} YES @ ~${entry_price:.2f}, "
                           f"position: {self.last_position} -> {current_position}")
        else:
            # Position decreased: we bought NO (or YES side order expired/cancelled)
            # This means we're now long NO and need the YES side to fill to flatten
            fill_side = 'no'
            # Entry price is our NO bid price
            entry_price = no_bid
            self.logger.info(f"FILL DETECTED: Bought {abs(position_delta)} NO @ ~${entry_price:.2f}, "
                           f"position: {self.last_position} -> {current_position}")

        # Check if this fill neutralized a pending exit (opposite side filled)
        if self.pending_exit is not None:
            pending_side = self.pending_exit['side']
            if (pending_side == 'yes' and position_delta < 0) or \
               (pending_side == 'no' and position_delta > 0):
                # Position is flattening - but why?
                # In dual exit mode, need to distinguish between:
                # 1. Our sell order filled (exit complete)
                # 2. Opposite side bid filled (pair trade complete)
                if self.in_dual_exit_mode and current_position == 0:
                    # Check if exit order is still active
                    if self._is_exit_order_active():
                        # Exit order NOT filled, so opposite side bid must have filled
                        # This means we've completed a pair trade - hold both to expiry!
                        self.logger.info(f"PAIR TRADE COMPLETED! Bought opposite side while holding {pending_side.upper()}")
                        self.logger.info(f"Holding both YES and NO until expiry for guaranteed spread profit")
                        self.holding_pair = True
                        # Cancel the exit sell order since we're holding
                        try:
                            self.api.cancel_order(self.active_exit_order_id)
                            self.logger.info(f"Canceled exit sell order - now holding pair to expiry")
                        except Exception as e:
                            self.logger.warning(f"Could not cancel exit order: {e}")
                        self.pending_exit = None
                        self.active_exit_order_id = None
                        self.in_dual_exit_mode = False
                    else:
                        # Exit order filled - normal exit
                        self.logger.info(f"EXIT COMPLETE: Sold {pending_side.upper()} position")
                        self.pending_exit = None
                        self.active_exit_order_id = None
                        self.in_dual_exit_mode = False
                else:
                    # Not in dual exit mode or position not yet 0
                    self.logger.info(f"POSITION FLATTENING: Opposite side filled, clearing pending exit")
                    self.pending_exit = None
                    self.active_exit_order_id = None
                    self.in_dual_exit_mode = False

        # If position is now 0 and we're not holding a pair, clear any pending state
        if current_position == 0 and not self.holding_pair:
            if self.pending_exit is not None:
                self.logger.info("Position is now flat, clearing pending exit")
            self.pending_exit = None
            self.active_exit_order_id = None
            self.in_dual_exit_mode = False
        elif self.pending_exit is None and not self.holding_pair:
            # New fill that creates a position - track for potential exit
            # (Skip if we're holding a pair - no exit needed)
            self.pending_exit = {
                'side': fill_side,
                'entry_price': entry_price,
                'fill_time': time.time(),
                'quantity': abs(position_delta),
            }
            self.logger.info(f"TRACKING FOR AUTO-EXIT: {fill_side} @ ${entry_price:.2f}, "
                           f"will exit in {self.exit_timeout}s if opposite doesn't fill")

        self.last_position = current_position
        return True

    def check_and_place_exit_order(self, current_position: int, yes_mid: float = 0.50) -> str:
        """Check if we need to place an exit order and do so if timeout exceeded.

        DUAL EXIT MODE: After timeout, we run two strategies in parallel:
        1. Try to SELL the owned side (exit at profit)
        2. Continue bidding on opposite side to complete pair trade

        If the opposite side fills while we have a sell order out, we've completed
        a pair trade (hold both YES and NO). Cancel all orders and hold to expiry
        for guaranteed spread profit.

        The exit price is calculated as max(cost_basis + 0.01, mid_price) to:
        1. Always ensure at least 1 cent profit above our actual cost
        2. Take advantage of favorable market conditions when mid > cost + 1 cent

        Returns:
            'none' - No exit needed or timeout not reached
            'dual_exit' - Exit order placed, continue bidding on opposite side
            'holding' - Pair trade completed, holding both sides
        """
        # If already holding a pair, don't do anything
        if self.holding_pair:
            return 'holding'

        if self.pending_exit is None:
            self.in_dual_exit_mode = False
            return 'none'

        if current_position == 0:
            # Position already flat, no exit needed
            self.pending_exit = None
            self.active_exit_order_id = None
            self.in_dual_exit_mode = False
            return 'none'

        elapsed = time.time() - self.pending_exit['fill_time']
        if elapsed < self.exit_timeout:
            # Not yet timed out, continue normal market making
            remaining = self.exit_timeout - elapsed
            self.logger.debug(f"Pending exit: {remaining:.1f}s remaining before auto-exit")
            return 'none'

        # Timeout exceeded - enter DUAL EXIT MODE
        # Place exit order but CONTINUE bidding on opposite side
        pending_side = self.pending_exit['side']

        # Get actual cost basis from the API
        try:
            position_info = self.api.get_position_with_cost()
            if pending_side == 'yes':
                cost_basis = position_info.get('yes_cost', 0)
            else:
                cost_basis = position_info.get('no_cost', 0)
        except Exception as e:
            self.logger.warning(f"Could not get cost basis from API: {e}, using tracked entry price")
            cost_basis = 0

        # Fall back to tracked entry_price if cost basis unavailable
        if cost_basis <= 0:
            cost_basis = self.pending_exit['entry_price']
            self.logger.info(f"Using tracked entry price as cost basis: ${cost_basis:.2f}")
        else:
            self.logger.info(f"Got actual cost basis from API: ${cost_basis:.2f}")

        # Calculate exit price: max(cost + 1 cent, mid price)
        min_exit_price = cost_basis + 0.01  # At least 1 cent above cost

        if pending_side == 'yes':
            mid_for_exit = yes_mid
        else:
            mid_for_exit = 1 - yes_mid

        exit_price = round(max(min_exit_price, mid_for_exit), 2)
        exit_price = max(0.02, min(0.98, exit_price))

        self.logger.info(f"Exit price calculation: cost=${cost_basis:.2f}, min_exit=${min_exit_price:.2f}, "
                        f"mid=${mid_for_exit:.2f}, final=${exit_price:.2f}")

        # Store cost basis for pair trade bid calculation
        self.pending_exit['cost_basis'] = cost_basis

        # If we already have an active exit order at the right price, don't replace it
        if self.active_exit_order_id and self.in_dual_exit_mode:
            self.logger.debug(f"DUAL EXIT: Already have exit order active, continuing to bid opposite side")
            return 'dual_exit'

        # Place exit order - cancel only the same-side BUY order (not the opposite side)
        if pending_side == 'yes' and current_position > 0:
            self.logger.info(f"DUAL EXIT: Placing SELL YES @ ${exit_price:.2f} "
                           f"(cost=${cost_basis:.2f}), continuing NO bids")
            try:
                # Cancel YES BUY order only (keep NO BUY order active)
                self._cancel_side_if_exists('yes')
                expiration_ts = int(time.time()) + self.order_expiration
                self.active_exit_order_id = self.api.place_order(
                    "sell", "yes", exit_price, abs(current_position), expiration_ts
                )
                self.in_dual_exit_mode = True
                return 'dual_exit'
            except Exception as e:
                self.logger.error(f"Failed to place YES exit order: {e}")
                return 'none'
        elif pending_side == 'no' and current_position < 0:
            self.logger.info(f"DUAL EXIT: Placing SELL NO @ ${exit_price:.2f} "
                           f"(cost=${cost_basis:.2f}), continuing YES bids")
            try:
                # Cancel NO BUY order only (keep YES BUY order active)
                self._cancel_side_if_exists('no')
                expiration_ts = int(time.time()) + self.order_expiration
                self.active_exit_order_id = self.api.place_order(
                    "sell", "no", exit_price, abs(current_position), expiration_ts
                )
                self.in_dual_exit_mode = True
                return 'dual_exit'
            except Exception as e:
                self.logger.error(f"Failed to place NO exit order: {e}")
                return 'none'
        else:
            # Position flipped or something unexpected - clear pending
            self.logger.warning(f"Position mismatch: pending_side={pending_side}, "
                              f"current_position={current_position}, clearing pending exit")
            self.pending_exit = None
            self.active_exit_order_id = None
            self.in_dual_exit_mode = False
            return 'none'

    def cancel_existing_orders(self):
        """Cancel all existing orders to ensure fresh quotes at BBO."""
        try:
            orders = self.api.get_orders()
            if orders:
                self.logger.info(f"Canceling {len(orders)} existing orders")
            for order in orders:
                order_id = order.get("order_id")
                if order_id:
                    try:
                        self.api.cancel_order(order_id)
                    except Exception as e:
                        self.logger.warning(f"Failed to cancel order {order_id}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to get orders for cancellation: {e}")

    def run_iteration(self, dt: float):
        """Run one iteration of the market making loop (Desired State Engine).

        Each iteration:
        1. Fetches market tick size (once) to determine pricing precision
        2. Fetches current market prices and inventory
        3. Computes tick-aware quotes based on market type:
           - 1¢ tick markets: 2¢ spread (bid 1¢ below mid)
           - 0.5¢ tick markets: 1¢ spread (bid 0.5¢ below mid)
        4. Reconciles orders instead of cancel-and-replace:
           - If price matches: Keep order (preserve queue priority)
           - If count too high: Use decrease endpoint
           - If price changed AND age > 10s: Cancel and replace
           - If price changed AND age < 10s: Wait (don't jump out of line)
        5. Auto-exit logic for one-sided fills
        """
        try:
            # Fetch tick size once on first iteration
            self._fetch_tick_size()

            # Get current market state
            price_data = self.api.get_price()
            yes_mid = price_data["yes"]
            no_mid = price_data["no"]
            market_yes_bid = price_data["yes_bid"]
            market_no_bid = price_data["no_bid"]
            q = self.api.get_position()

            # Compute tick-aware quotes based on market type
            yes_bid, no_bid = self.compute_tick_aware_quotes(
                yes_mid, q, self.t,
                market_yes_bid=market_yes_bid,
                market_no_bid=market_no_bid,
            )

            # Detect fills and track for auto-exit
            self.detect_fill_and_track(q, yes_bid, no_bid)

            # Check if we need to exit (timeout exceeded) or handle dual exit mode
            exit_status = self.check_and_place_exit_order(q, yes_mid)

            if exit_status == 'holding':
                # Pair trade completed - cancel all orders and hold to expiry
                self.logger.info("HOLDING PAIR: Both YES and NO filled, holding to expiry for guaranteed spread")
                self.cancel_existing_orders()
                self.t += dt
                return

            # Apply cost basis constraint when holding position
            # In dual exit mode, use stored cost_basis; otherwise get from API
            min_profit_margin = 0.01 if not self.use_subpenny else 0.005

            if self.pending_exit is not None and q != 0:
                # Get cost basis - prefer stored value in dual exit mode
                cost_basis = self.pending_exit.get('cost_basis', 0)
                if cost_basis <= 0:
                    try:
                        position_info = self.api.get_position_with_cost()
                        if q > 0:
                            cost_basis = position_info.get('yes_cost', 0)
                        else:
                            cost_basis = position_info.get('no_cost', 0)
                    except Exception as e:
                        self.logger.warning(f"Could not get cost basis for constraint check: {e}")
                        cost_basis = self.pending_exit.get('entry_price', 0)

                if cost_basis > 0:
                    if q > 0:  # Long YES, constraining NO bid
                        max_no_bid = 1.00 - cost_basis - min_profit_margin
                        if self.use_subpenny:
                            max_no_bid = round(max_no_bid * 200) / 200
                        else:
                            max_no_bid = round(max_no_bid, 2)
                        if no_bid > max_no_bid:
                            self.logger.info(f"COST BASIS CONSTRAINT: NO bid ${no_bid:.3f} exceeds "
                                           f"max ${max_no_bid:.3f} (YES cost=${cost_basis:.3f}), capping")
                            no_bid = max(0.02, max_no_bid)
                    elif q < 0:  # Long NO, constraining YES bid
                        max_yes_bid = 1.00 - cost_basis - min_profit_margin
                        if self.use_subpenny:
                            max_yes_bid = round(max_yes_bid * 200) / 200
                        else:
                            max_yes_bid = round(max_yes_bid, 2)
                        if yes_bid > max_yes_bid:
                            self.logger.info(f"COST BASIS CONSTRAINT: YES bid ${yes_bid:.3f} exceeds "
                                           f"max ${max_yes_bid:.3f} (NO cost=${cost_basis:.3f}), capping")
                            yes_bid = max(0.02, max_yes_bid)

            # Log market state
            if self.use_subpenny:
                self.logger.info(f"Market: YES mid=${yes_mid:.3f}, NO mid=${no_mid:.3f} (sub-penny)")
                if exit_status == 'dual_exit':
                    # In dual exit, show which side we're bidding on
                    opposite_side = 'NO' if self.pending_exit and self.pending_exit['side'] == 'yes' else 'YES'
                    bid_price = no_bid if opposite_side == 'NO' else yes_bid
                    self.logger.info(f"Position: {q} | DUAL EXIT: Selling owned + BUY {opposite_side} @${bid_price:.3f}")
                else:
                    self.logger.info(f"Position: {q} | Desired: BUY YES @${yes_bid:.3f}, BUY NO @${no_bid:.3f}")
            else:
                self.logger.info(f"Market: YES mid=${yes_mid:.2f}, NO mid=${no_mid:.2f}")
                if exit_status == 'dual_exit':
                    opposite_side = 'NO' if self.pending_exit and self.pending_exit['side'] == 'yes' else 'YES'
                    bid_price = no_bid if opposite_side == 'NO' else yes_bid
                    self.logger.info(f"Position: {q} | DUAL EXIT: Selling owned + BUY {opposite_side} @${bid_price:.2f}")
                else:
                    self.logger.info(f"Position: {q} | Desired: BUY YES @${yes_bid:.2f}, BUY NO @${no_bid:.2f}")

            # DESIRED STATE ENGINE: Reconcile orders instead of cancel-and-replace
            # In dual exit mode, only reconcile the opposite side (don't touch the sell order)
            if exit_status == 'dual_exit' and self.pending_exit:
                # Only reconcile opposite side - the owned side has a SELL order, not a BUY
                if self.pending_exit['side'] == 'yes':
                    # Long YES, selling YES, only bid on NO
                    self._reconcile_side('no', no_bid, 1, int(time.time()) + self.order_expiration, time.time())
                else:
                    # Long NO, selling NO, only bid on YES
                    self._reconcile_side('yes', yes_bid, 1, int(time.time()) + self.order_expiration, time.time())
            else:
                # Normal mode: reconcile both sides
                self.reconcile_orders(yes_bid, no_bid, desired_count=1, position=q)

            self.t += dt

        except Exception as e:
            self.logger.error(f"Error in market maker loop: {e}")
            raise

    def run(self, dt: float):
        """Run the market maker loop until T is reached."""
        self.logger.info(f"Starting Desired State Engine market maker (T={self.T}s, dt={dt}s)")
        self.logger.info(f"Parameters: gamma={self.gamma}, sigma={self.sigma}, "
                        f"flip_skew={self.flip_skew_factor}")
        self.logger.info(f"Queue priority: MIN_ORDER_AGE={self.MIN_ORDER_AGE}s (won't reprice younger orders)")
        self.logger.info(f"Auto-exit: timeout={self.exit_timeout}s, profit_target=${self.exit_profit_target:.2f}")
        self.logger.info(f"Tick-aware pricing: 2¢ spread for 1¢ markets, 1¢ spread for 0.5¢ markets")
        self.logger.info(f"DUAL EXIT MODE: After timeout, sells owned side + continues bidding opposite for pair trade")
        while self.t < self.T:
            self.run_iteration(dt)
            time.sleep(dt)

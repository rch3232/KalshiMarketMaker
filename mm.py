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

            # Calculate mid-price with fallback handling for illiquid markets
            if yes_bid > 0 and yes_ask > 0:
                # Normal case: both bid and ask exist
                yes_mid_price = round((yes_bid + yes_ask) / 2, 2)
            elif yes_ask > 0:
                # Only ask exists (no bids) - use ask with buffer
                yes_mid_price = round(yes_ask - 0.05, 2)
                self.logger.warning(f"No YES bids, using ask-based mid: ${yes_mid_price:.2f}")
            elif yes_bid > 0:
                # Only bid exists (no asks) - use bid with buffer
                yes_mid_price = round(yes_bid + 0.05, 2)
                self.logger.warning(f"No YES asks, using bid-based mid: ${yes_mid_price:.2f}")
            else:
                # No bid or ask - try last_price or default to 0.50
                last_price = market.get("last_price", 0) or 0
                if last_price > 0:
                    yes_mid_price = round(float(last_price) / 100, 2)
                    self.logger.warning(f"No bid/ask data, using last_price: ${yes_mid_price:.2f}")
                else:
                    yes_mid_price = 0.50
                    self.logger.warning(f"No market data available, defaulting to ${yes_mid_price:.2f}")

            # Ensure mid-price is within valid bounds
            yes_mid_price = max(0.05, min(0.95, yes_mid_price))
            no_mid_price = round(1 - yes_mid_price, 2)

            self.logger.info(f"Market mid-prices: YES=${yes_mid_price:.2f}, NO=${no_mid_price:.2f}")
            return {"yes": yes_mid_price, "no": no_mid_price}
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
                endpoint = f"/markets?category={category}&status=open&limit=200"
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
    """Market maker using Avellaneda-Stoikov strategy with Dual-Quote flipping.

    This implementation is calibrated for binary probability markets (0.00 to 1.00 scale)
    and uses a dual-quote strategy that places BUY orders on both YES and NO sides
    to capture spread from whichever direction the market moves.

    Key features:
    - Decimal-scale aligned parameters (gamma scaled to 0.01-0.05 range)
    - Simultaneous YES/NO bidding to capture spread from either direction
    - Asymmetric inventory urgency for aggressive position flipping
    - Hard adaptive caps to prevent quotes from getting stuck at boundaries
    """

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
        min_spread: float = 0.02,      # Minimum spread ($0.02)
        max_spread: float = 0.10,      # Maximum spread ($0.10)
        position_limit_buffer: float = 0.1,
        flip_skew_factor: float = 0.03,  # Aggressive skew for inventory flipping
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

        self.t = 0
        self.active_yes_order_id = None
        self.active_no_order_id = None

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

    def compute_dual_quotes(self, yes_mid: float, q: int, t: float) -> Tuple[float, float]:
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

        # Apply boundary recalculation if quotes hit extremes
        yes_bid, no_bid = self._apply_boundary_recalculation(yes_mid, yes_bid, no_bid)

        return yes_bid, no_bid

    def _apply_boundary_recalculation(
        self, yes_mid: float, yes_bid: float, no_bid: float
    ) -> Tuple[float, float]:
        """Apply hard adaptive caps - recalculate if quotes hit boundaries.

        If the calculated quote hits 0.01 or 0.99, force recalculation
        at mid price +/- (min_spread / 2) to ensure competitive, fillable orders.
        """
        no_mid = 1 - yes_mid

        # Check YES bid against boundaries
        if yes_bid <= 0.01 or yes_bid >= 0.99:
            # Force YES bid to be competitive around mid
            yes_bid = round(yes_mid - self.min_spread / 2, 2)
            self.logger.info(f"YES bid hit boundary, recalculated to ${yes_bid:.2f}")

        # Check NO bid against boundaries
        if no_bid <= 0.01 or no_bid >= 0.99:
            # Force NO bid to be competitive around mid
            no_bid = round(no_mid - self.min_spread / 2, 2)
            self.logger.info(f"NO bid hit boundary, recalculated to ${no_bid:.2f}")

        # Final clamp to valid range [0.02, 0.98] to stay off the book edges
        yes_bid = max(0.02, min(0.98, yes_bid))
        no_bid = max(0.02, min(0.98, no_bid))

        return yes_bid, no_bid

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
        """Run one iteration of the market making loop.

        Each iteration:
        1. Fetches current market prices
        2. Gets current inventory position
        3. Cancels ALL existing orders (ensures we're always at BBO)
        4. Computes new dual quotes with inventory urgency
        5. Places BUY orders on both YES and NO sides
        """
        try:
            # Get current market state
            price_data = self.api.get_price()
            yes_mid = price_data["yes"]
            no_mid = price_data["no"]
            q = self.api.get_position()

            # Cancel all existing orders first (order management requirement)
            self.cancel_existing_orders()

            # Compute dual quotes with asymmetric inventory urgency
            yes_bid, no_bid = self.compute_dual_quotes(yes_mid, q, self.t)

            self.logger.info(f"Market: YES mid=${yes_mid:.2f}, NO mid=${no_mid:.2f}")
            self.logger.info(f"Position: {q} | Quotes: BUY YES @${yes_bid:.2f}, BUY NO @${no_bid:.2f}")

            position_limit = int(self.max_position * (1 - self.position_limit_buffer))
            expiration_ts = int(time.time()) + self.order_expiration

            # Place BUY YES order (if not at max long position)
            if q < position_limit:
                self.logger.info(f"Placing BUY YES at ${yes_bid:.2f}")
                try:
                    self.active_yes_order_id = self.api.place_order(
                        "buy", "yes", yes_bid, 1, expiration_ts
                    )
                except Exception as e:
                    self.logger.error(f"Failed to place YES order: {e}")

            # Place BUY NO order (effectively a sell YES when filled)
            # Only place if we have room in position or need to reduce YES exposure
            if q > -position_limit:
                self.logger.info(f"Placing BUY NO at ${no_bid:.2f}")
                try:
                    self.active_no_order_id = self.api.place_order(
                        "buy", "no", no_bid, 1, expiration_ts
                    )
                except Exception as e:
                    self.logger.error(f"Failed to place NO order: {e}")

            self.t += dt

        except Exception as e:
            self.logger.error(f"Error in market maker loop: {e}")
            raise

    def run(self, dt: float):
        """Run the market maker loop until T is reached."""
        self.logger.info(f"Starting Dual-Quote market maker (T={self.T}s, dt={dt}s)")
        self.logger.info(f"Parameters: gamma={self.gamma}, sigma={self.sigma}, "
                        f"spread=[{self.min_spread:.2f}, {self.max_spread:.2f}], "
                        f"flip_skew={self.flip_skew_factor}")
        while self.t < self.T:
            self.run_iteration(dt)
            time.sleep(dt)

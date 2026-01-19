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
                response = requests.get(url, headers=headers, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=30)
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
        """No logout needed for API key auth."""
        self.logger.info("Session ended (API key auth - no logout required)")

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

            yes_bid = float(market.get("yes_bid", 0)) / 100
            yes_ask = float(market.get("yes_ask", 0)) / 100
            no_bid = float(market.get("no_bid", 0)) / 100
            no_ask = float(market.get("no_ask", 0)) / 100

            yes_mid_price = round((yes_bid + yes_ask) / 2, 2)
            no_mid_price = round((no_bid + no_ask) / 2, 2)

            self.logger.info(f"Current yes mid-market price: ${yes_mid_price:.2f}")
            self.logger.info(f"Current no mid-market price: ${no_mid_price:.2f}")
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


class AvellanedaMarketMaker:
    """Market maker using Avellaneda-Stoikov strategy."""

    def __init__(
        self,
        logger: logging.Logger,
        api: AbstractTradingAPI,
        gamma: float = 0.1,
        k: float = 1.5,
        sigma: float = 0.5,
        T: float = 3600,
        max_position: int = 100,
        order_expiration: int = 300,
        min_spread: float = 0.01,
        position_limit_buffer: float = 0.1,
        inventory_skew_factor: float = 0.01,
        trade_side: str = "yes",
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
        self.position_limit_buffer = position_limit_buffer
        self.inventory_skew_factor = inventory_skew_factor
        self.trade_side = trade_side

        self.t = 0
        self.active_bid_id = None
        self.active_ask_id = None

    def compute_reservation_price(self, mid_price: float, q: int, t: float) -> float:
        """Compute reservation price with inventory adjustment."""
        time_factor = max(0.001, self.T - t)
        r = mid_price - q * self.gamma * (self.sigma ** 2) * time_factor
        return r

    def compute_optimal_spread(self, t: float) -> float:
        """Compute optimal spread based on Avellaneda-Stoikov model."""
        time_factor = max(0.001, self.T - t)
        spread = self.gamma * (self.sigma ** 2) * time_factor + (2 / self.gamma) * math.log(1 + self.gamma / self.k)
        return max(spread, self.min_spread)

    def compute_quotes(self, mid_price: float, q: int, t: float) -> Tuple[float, float]:
        """Compute bid and ask prices."""
        r = self.compute_reservation_price(mid_price, q, t)
        spread = self.compute_optimal_spread(t)

        inventory_skew = q * self.inventory_skew_factor
        bid_price = r - spread / 2 - inventory_skew
        ask_price = r + spread / 2 - inventory_skew

        bid_price = max(0.01, min(0.99, round(bid_price, 2)))
        ask_price = max(0.01, min(0.99, round(ask_price, 2)))

        if ask_price <= bid_price:
            mid = (bid_price + ask_price) / 2
            bid_price = round(mid - self.min_spread / 2, 2)
            ask_price = round(mid + self.min_spread / 2, 2)

        return bid_price, ask_price

    def cancel_existing_orders(self):
        """Cancel all existing orders."""
        try:
            orders = self.api.get_orders()
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
        """Run one iteration of the market making loop."""
        try:
            price_data = self.api.get_price()
            mid_price = price_data[self.trade_side]
            q = self.api.get_position()

            self.cancel_existing_orders()

            bid_price, ask_price = self.compute_quotes(mid_price, q, self.t)

            self.logger.info(f"Mid price: ${mid_price:.2f}, Position: {q}")
            self.logger.info(f"Computed bid: ${bid_price:.2f}, ask: ${ask_price:.2f}")

            position_limit = int(self.max_position * (1 - self.position_limit_buffer))
            expiration_ts = int(time.time()) + self.order_expiration

            if q < position_limit:
                self.logger.info(f"Placing bid at ${bid_price:.2f}")
                self.api.place_order("buy", self.trade_side, bid_price, 1, expiration_ts)

            if q > -position_limit:
                self.logger.info(f"Placing ask at ${ask_price:.2f}")
                self.api.place_order("sell", self.trade_side, ask_price, 1, expiration_ts)

            self.t += dt

        except Exception as e:
            self.logger.error(f"Error in market maker loop: {e}")
            raise

    def run(self, dt: float):
        """Run the market maker loop until T is reached."""
        self.logger.info(f"Starting market maker loop (T={self.T}s, dt={dt}s)")
        while self.t < self.T:
            self.run_iteration(dt)
            time.sleep(dt)

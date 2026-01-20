#!/usr/bin/env python3
"""Diagnostic script to see what markets are being fetched and why they're filtered."""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check env vars
api_key = os.getenv("KALSHI_API_KEY")
private_key_env = os.getenv("KALSHI_PRIVATE_KEY")
base_url = os.getenv("KALSHI_BASE_URL")

if not all([api_key, private_key_env, base_url]):
    print("ERROR: Missing environment variables")
    print(f"  KALSHI_API_KEY: {'SET' if api_key else 'MISSING'}")
    print(f"  KALSHI_PRIVATE_KEY: {'SET' if private_key_env else 'MISSING'}")
    print(f"  KALSHI_BASE_URL: {'SET' if base_url else 'MISSING'}")
    sys.exit(1)

# Handle private key
if private_key_env.startswith('/') and os.path.isfile(private_key_env):
    with open(private_key_env, 'rb') as f:
        private_key = f.read()
else:
    private_key = private_key_env.replace('\\n', '\n')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Diagnose")

from mm import KalshiTradingAPI

# Create API
api = KalshiTradingAPI(
    api_key=api_key,
    private_key=private_key,
    market_ticker="DUMMY",
    base_url=base_url,
    logger=logger
)

print("\n" + "="*80)
print("FETCHING SPORTS MARKETS")
print("="*80)

markets = api.get_active_markets_by_category("Sports")
print(f"\nTotal markets fetched: {len(markets)}")

# Analyze why markets are filtered
empty_orderbook = []
wide_spread = []
low_volume = []
longshot_low = []
longshot_high = []
passed = []

min_volume = 3500
max_spread_cents = 15
price_floor = 20
price_ceiling = 80

for m in markets:
    ticker = m.get('ticker', 'UNKNOWN')
    title = m.get('title', '')[:50]
    yes_bid = m.get('yes_bid', 0) or 0
    yes_ask = m.get('yes_ask', 0) or 0
    volume = m.get('volume', 0) or m.get('volume_24h', 0) or 0

    # Check filters
    if yes_bid == 0 or yes_ask == 0:
        empty_orderbook.append((ticker, title, yes_bid, yes_ask, volume))
        continue

    spread = yes_ask - yes_bid
    if spread > max_spread_cents:
        wide_spread.append((ticker, title, yes_bid, yes_ask, spread, volume))
        continue

    if volume < min_volume:
        low_volume.append((ticker, title, yes_bid, yes_ask, volume))
        continue

    mid_price = (yes_bid + yes_ask) / 2
    if mid_price < price_floor:
        longshot_low.append((ticker, title, mid_price, volume))
        continue
    if mid_price > price_ceiling:
        longshot_high.append((ticker, title, mid_price, volume))
        continue

    passed.append((ticker, title, yes_bid, yes_ask, spread, volume))

print(f"\n{'='*80}")
print("FILTER BREAKDOWN")
print(f"{'='*80}")
print(f"Empty order book (yes_bid=0 or yes_ask=0): {len(empty_orderbook)}")
print(f"Wide spread (>{max_spread_cents}¢): {len(wide_spread)}")
print(f"Low volume (<{min_volume}): {len(low_volume)}")
print(f"Longshot LOW (<{price_floor}¢): {len(longshot_low)}")
print(f"Longshot HIGH (>{price_ceiling}¢): {len(longshot_high)}")
print(f"PASSED ALL FILTERS: {len(passed)}")

# Show samples from each category
if empty_orderbook:
    print(f"\n--- SAMPLE Empty Order Books (first 10) ---")
    for ticker, title, bid, ask, vol in empty_orderbook[:10]:
        print(f"  {ticker}: bid={bid}, ask={ask}, vol={vol} | {title}")

if wide_spread:
    print(f"\n--- SAMPLE Wide Spreads (first 10) ---")
    for ticker, title, bid, ask, spread, vol in wide_spread[:10]:
        print(f"  {ticker}: spread={spread}¢ (bid={bid}, ask={ask}), vol={vol} | {title}")

if low_volume:
    print(f"\n--- SAMPLE Low Volume (first 10) ---")
    for ticker, title, bid, ask, vol in low_volume[:10]:
        print(f"  {ticker}: vol={vol}, bid={bid}, ask={ask} | {title}")

if longshot_low:
    print(f"\n--- SAMPLE Longshot LOW (first 10) ---")
    for ticker, title, mid, vol in longshot_low[:10]:
        print(f"  {ticker}: mid={mid}¢, vol={vol} | {title}")

if longshot_high:
    print(f"\n--- SAMPLE Longshot HIGH (first 10) ---")
    for ticker, title, mid, vol in longshot_high[:10]:
        print(f"  {ticker}: mid={mid}¢, vol={vol} | {title}")

if passed:
    print(f"\n--- MARKETS THAT PASS ALL FILTERS (first 20) ---")
    for ticker, title, bid, ask, spread, vol in passed[:20]:
        print(f"  {ticker}: bid={bid}¢, ask={ask}¢, spread={spread}¢, vol={vol} | {title}")

api.logout()
print("\nDone!")

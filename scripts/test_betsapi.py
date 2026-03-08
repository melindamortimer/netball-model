"""Quick script to test BetsAPI endpoints.

Usage:
    poetry run python scripts/test_betsapi.py
"""
import os
import ssl

import certifi
import httpx
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.environ.get("BETSAPI_TOKEN", "")
BASE = "https://api.b365api.com"

ca = os.environ.get("SSL_CERT_FILE", certifi.where())
ctx = ssl.create_default_context(cafile=ca)
client = httpx.Client(verify=ctx, timeout=30)


def call(path: str, **params):
    params["token"] = TOKEN
    url = f"{BASE}/{path}"
    print(f"\n>>> GET {path}")
    print(f"    params: {params}")
    r = client.get(url, params=params)
    data = r.json()
    print(f"    status: {r.status_code}")
    print(f"    success: {data.get('success')}")
    results = data.get("results", [])
    if isinstance(results, list):
        print(f"    results: {len(results)} items")
        for item in results[:3]:
            print(f"      {item}")
    elif isinstance(results, dict):
        print(f"    results: {results}")
    else:
        print(f"    response: {data}")
    return data


# 1. Test odds summary for a known event
print("=" * 60)
print("Test: v2/event/odds/summary")
call("v2/event/odds/summary", event_id=232751)

# 2. Find netball ended events (try sport_id 13)
print("\n" + "=" * 60)
print("Test: v1/events/ended (netball, sport_id=13, SSN league)")
call("v1/events/ended", sport_id=13, league_id=23816, day="20240401")

# 3. Try bet365 upcoming for netball
print("\n" + "=" * 60)
print("Test: v1/bet365/upcoming (sport_id=13)")
call("v1/bet365/upcoming", sport_id=13)

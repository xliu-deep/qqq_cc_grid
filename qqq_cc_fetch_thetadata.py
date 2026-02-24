"""
QQQ Covered Call — ThetaData MCP Fetcher
=========================================
Connects to the local ThetaData MCP SSE server (port 25503) and fetches
real EOD greeks data for all (date, expiration) pairs identified by the
dry-run script.  Populates qqq_cc_grid_cache.json.

Usage:
    python3 qqq_cc_fetch_thetadata.py
"""

import os
import sys
import json
import time
import http.client
from collections import defaultdict
from typing import Dict, List, Tuple

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_FILE = os.path.join(WORK_DIR, "qqq_cc_needed_pairs.json")
CACHE_FILE = os.path.join(WORK_DIR, "qqq_cc_grid_cache.json")
MCP_HOST = "127.0.0.1"
MCP_PORT = 25503
SYMBOL = "QQQ"
SAVE_EVERY = 10
CALL_DELAY = 0.25


class MCPSession:
    """Minimal MCP client over SSE transport using raw http.client."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.endpoint: str = ""
        self._sse_conn: http.client.HTTPConnection = None
        self._sse_resp = None

    def connect(self):
        self._sse_conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
        self._sse_conn.request(
            "GET", "/mcp/sse",
            headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        )
        self._sse_resp = self._sse_conn.getresponse()
        if self._sse_resp.status != 200:
            raise ConnectionError(f"SSE returned {self._sse_resp.status}")

        event_type, data = self._read_event(timeout=10)
        if event_type != "endpoint":
            raise ConnectionError(f"Expected 'endpoint' event, got '{event_type}'")
        self.endpoint = data.strip()

    def initialize(self):
        init = {
            "jsonrpc": "2.0", "id": "init",
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "qqq-cc-fetcher", "version": "1.0"},
            },
        }
        self._post(init)
        self._read_event(timeout=10)

        notif = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self._post(notif)

    def call_tool(self, name: str, args: dict, call_id: str = "t",
                  timeout: int = 120) -> dict:
        rpc = {
            "jsonrpc": "2.0", "id": call_id,
            "method": "tools/call",
            "params": {"name": name, "arguments": args},
        }
        self._post(rpc)
        start = time.time()
        while time.time() - start < timeout:
            event_type, data = self._read_event(timeout=timeout)
            if not data or not data.strip():
                continue
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == call_id:
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"No response for call_id={call_id}")

    def close(self):
        try:
            self._sse_conn.close()
        except Exception:
            pass

    # ── internal ──

    def _post(self, body: dict):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=120)
        raw = json.dumps(body)
        conn.request("POST", self.endpoint, body=raw,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()
        if resp.status not in (200, 202, 204):
            raise ConnectionError(f"POST returned {resp.status}")

    def _read_event(self, timeout=120) -> Tuple[str, str]:
        """Read one complete SSE event (ends with blank line)."""
        buf = bytearray()
        start = time.time()
        prev = 0
        while time.time() - start < timeout:
            b = self._sse_resp.read(1)
            if not b:
                break
            buf.append(b[0])
            cur = b[0]
            if cur == 0x0A and prev == 0x0A:
                break
            prev = cur

        text = buf.decode("utf-8", errors="replace")
        event_type = ""
        data_lines = []
        for line in text.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        return event_type, "\n".join(data_lines)


def _repair_json(text: str) -> str:
    """Repair truncated ThetaData JSON by closing open brackets in order."""
    text = text.rstrip()
    idx = text.rfind("}")
    if idx < 0:
        return text
    text = text[: idx + 1]

    in_string = False
    escape = False
    stack = []
    matching = {"{": "}", "[": "]"}
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in matching:
            stack.append(matching[ch])
        elif ch in ("}", "]"):
            if stack and stack[-1] == ch:
                stack.pop()

    stack.reverse()
    return text + "".join(stack)


def parse_mcp_greeks(result: dict) -> Dict[str, list]:
    """Parse MCP tool response → {date_str: [chain_row, ...]}."""
    chains: Dict[str, list] = defaultdict(list)

    content = result.get("content", [])
    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item["text"])
        elif isinstance(item, str):
            parts.append(item)
    if not parts:
        return chains

    raw = "".join(parts)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        repaired = _repair_json(raw)
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return chains

    for block in data.get("response", []):
        contract = block.get("contract", {})
        strike_raw = contract.get("strike", 0)
        strike = strike_raw / 1000.0 if strike_raw > 5000 else strike_raw
        if contract.get("right", "") not in ("CALL", "C"):
            continue

        for row in block.get("data", []):
            delta = row.get("delta", 0)
            bid = row.get("bid", 0)
            ask = row.get("ask", 0)
            close_px = row.get("close", 0)
            iv = row.get("implied_vol", 0)
            underlying = row.get("underlying_price", 0)
            mid = (bid + ask) / 2 if bid > 0 and ask > 0 else close_px

            ts = row.get("underlying_timestamp", "") or row.get("timestamp", "")
            date_str = ts[:10] if ts else ""
            if not date_str or date_str < "2000":
                continue
            if not (0.01 < delta < 0.90 and mid > 0.01 and strike > 0):
                continue

            chains[date_str].append({
                "strike": strike,
                "delta": round(delta, 6),
                "bid": bid, "ask": ask,
                "mid": round(mid, 4),
                "close": close_px,
                "iv": round(iv, 6),
                "underlying": underlying,
            })

    return chains


def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


BACKUP_CACHE = "/tmp/qqq_cc_grid_cache.json"

def save_cache(cache: dict):
    n_chains = sum(1 for k in cache if k.startswith("chain_"))
    print(f"\n[Save] Writing {n_chains} chains ({len(cache)} keys)...")
    for path in [BACKUP_CACHE, CACHE_FILE]:
        try:
            with open(path, "w") as f:
                json.dump(cache, f)
            sz = os.path.getsize(path)
            print(f"[Save] {path} — {sz:,} bytes")
        except Exception as e:
            print(f"[Save] FAILED {path}: {e}")


def main():
    print("=" * 65)
    print("  QQQ Covered Call — ThetaData MCP Batch Fetcher")
    print("=" * 65)

    with open(PAIRS_FILE) as f:
        pairs_data = json.load(f)
    all_pairs = pairs_data["pairs"]
    print(f"[Pairs]  {len(all_pairs)} (date, expiration) pairs from dry-run")

    cache = load_cache()
    existing = sum(1 for k in cache if k.startswith("chain_"))
    print(f"[Cache]  {existing} chains already cached")

    fetch_list = []
    for p in all_pairs:
        key = f"chain_{SYMBOL}_{p['expiration']}_{p['query_date']}"
        if key not in cache:
            fetch_list.append((p["query_date"], p["expiration"]))

    print(f"[Plan]   {existing} cached, {len(fetch_list)} to fetch "
          f"({len(fetch_list)} API calls)")

    if not fetch_list:
        print("\n[Done] All data already in cache!")
        return

    estimated_min = len(fetch_list) * 1.3 / 60
    print(f"[ETA]    ~{estimated_min:.0f} minutes")

    # ── Connect ──
    print(f"\n[MCP] Connecting to {MCP_HOST}:{MCP_PORT}...")
    session = MCPSession(MCP_HOST, MCP_PORT)
    try:
        session.connect()
        print(f"[MCP] SSE connected, endpoint: {session.endpoint}")
        session.initialize()
        print("[MCP] Initialized successfully")
    except Exception as e:
        print(f"[Error] {e}")
        return

    # ── Fetch loop (single-date queries) ──
    t_start = time.time()
    fetched = 0
    errors = 0
    consecutive_errors = 0
    save_ctr = 0
    n_total = len(fetch_list)

    for i, (qdate, exp) in enumerate(fetch_list):
        elapsed = time.time() - t_start
        rate = (i / elapsed * 60) if elapsed > 0 and i > 0 else 0
        eta = ((n_total - i) / (rate / 60)) / 60 if rate > 0 else 0

        sys.stdout.write(
            f"\r[{i+1:4d}/{n_total}] {qdate}/{exp} "
            f"[{rate:.0f}/min, ETA {eta:.0f}m] "
        )
        sys.stdout.flush()

        try:
            result = session.call_tool("option_history_greeks_eod", {
                "symbol": SYMBOL,
                "expiration": exp,
                "start_date": qdate,
                "end_date": qdate,
                "right": "C",
            }, call_id=str(i), timeout=60)

            chains = parse_mcp_greeks(result)
            chain = chains.get(qdate, [])
            if chain:
                cache[f"chain_{SYMBOL}_{exp}_{qdate}"] = chain
                fetched += 1
                consecutive_errors = 0
                sys.stdout.write(f"✓ {len(chain)} strikes  \n")
            else:
                all_dates = list(chains.keys())
                if all_dates:
                    for d, c in chains.items():
                        cache[f"chain_{SYMBOL}_{exp}_{d}"] = c
                        fetched += 1
                    sys.stdout.write(f"✓ found dates {all_dates}  \n")
                else:
                    sys.stdout.write("✓ empty  \n")
                consecutive_errors = 0

        except Exception as e:
            errors += 1
            consecutive_errors += 1
            sys.stdout.write(f"✗ {e}\n")
            if consecutive_errors >= 5:
                print("\n[Reconnecting after 5 consecutive errors...]")
                try:
                    session.close()
                    time.sleep(3)
                    session = MCPSession(MCP_HOST, MCP_PORT)
                    session.connect()
                    session.initialize()
                    print("[Reconnected]")
                    consecutive_errors = 0
                except Exception as re:
                    print(f"[Reconnect failed: {re}] Aborting.")
                    break

        save_ctr += 1
        if save_ctr >= SAVE_EVERY:
            save_cache(cache)
            save_ctr = 0

        time.sleep(CALL_DELAY)

    save_cache(cache)
    session.close()

    total_chains = sum(1 for k in cache if k.startswith("chain_"))
    elapsed_total = (time.time() - t_start) / 60

    print(f"\n{'=' * 65}")
    print(f"  Completed in {elapsed_total:.1f} minutes")
    print(f"  Chains in cache:   {total_chains}")
    print(f"  New pairs fetched: {fetched}")
    print(f"  Errors:            {errors}")
    print(f"  Coverage:          {total_chains}/{len(all_pairs)} "
          f"({total_chains/len(all_pairs)*100:.1f}%)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

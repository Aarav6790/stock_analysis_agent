import argparse
import datetime as dt
import json
import math
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any


GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
DEFAULT_NEWS_LIMIT = 6


def compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Model did not return a valid JSON object.")


def wait_seconds_from_rate_limit(error_body: str) -> float:
    try:
        parsed = json.loads(error_body)
        message = parsed.get("error", {}).get("message", "")
    except json.JSONDecodeError:
        message = error_body
    match = re.search(r"try again in ([0-9.]+)s", message)
    if match:
        return float(match.group(1)) + 1.0
    return 20.0


def call_groq_json(
    step_name: str,
    system_prompt: str,
    payload: dict[str, Any],
    output_keys: list[str],
    max_completion_tokens: int = 700,
    max_retries: int = 4,
) -> tuple[dict[str, Any], dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY. Run: export GROQ_API_KEY='your_key_here'")

    user_prompt = (
        "Input JSON:\n"
        f"{compact_json(payload)}\n\n"
        "Return JSON only. Required top-level keys:\n"
        f"{compact_json(output_keys)}"
    )
    request_body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_completion_tokens": max_completion_tokens,
        "stream": False,
    }

    url = f"{GROQ_BASE_URL}/chat/completions"
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            url,
            data=compact_json(request_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "college-stock-agent/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return extract_json_object(content), {
                "step": step_name,
                "model": body.get("model"),
                "usage": body.get("usage"),
                "finish_reason": body["choices"][0].get("finish_reason"),
            }
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429 and attempt < max_retries:
                wait_for = wait_seconds_from_rate_limit(error_body)
                print(f"Groq rate limit hit in {step_name}. Waiting {wait_for:.1f}s and retrying...")
                time.sleep(wait_for)
                continue
            raise RuntimeError(f"Groq API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Groq API request failed: {exc}") from exc

    raise RuntimeError(f"Groq request failed after retries in {step_name}.")


def http_json(url: str, timeout: int = 20) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "college-stock-agent/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_text(url: str, timeout: int = 20) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "college-stock-agent/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def round_number(value: Any, digits: int = 4) -> float | None:
    try:
        return None if value is None else round(float(value), digits)
    except (TypeError, ValueError):
        return None


def trace_step(
    state: dict[str, Any],
    number: int,
    name: str,
    kind: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    verbose: bool,
) -> None:
    state["trace"].append(
        {
            "step": number,
            "name": name,
            "kind": kind,
            "input": input_data,
            "output": output_data,
            "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    if verbose:
        print(f"\nStep {number}: {name} ({kind})")
        print(pretty_json(output_data))
# ---------------------------------------------------------------------------
# Numeric field normalisation
#
# Llama (and most LLMs) default to 0-1 floats for "confidence" and similar
# fields even when the prompt says 0-100. We fix this post-call rather than
# hoping the model complies, since it affects every downstream step.
# ---------------------------------------------------------------------------

def _normalise_confidence_fields(output: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """Convert 0-1 floats to 0-100 integers for any confidence-style field."""
    for field in fields:
        val = output.get(field)
        if val is None:
            continue
        try:
            v = float(val)
            if 0.0 <= v <= 1.0:
                output[field] = round(v * 100)
            else:
                output[field] = max(0, min(100, round(v)))
        except (TypeError, ValueError):
            output[field] = 50
    return output


def _normalise_reconciliation_score(output: dict[str, Any]) -> dict[str, Any]:
    """Reconciliation score is -100 to +100. Normalise 0-1 floats to that scale."""
    val = output.get("reconciliation_score")
    if val is None:
        return output
    try:
        v = float(val)
        if -1.0 <= v <= 1.0:
            output["reconciliation_score"] = round(v * 100)
        else:
            output["reconciliation_score"] = max(-100, min(100, round(v)))
    except (TypeError, ValueError):
        output["reconciliation_score"] = 0
    return output


def _normalise_penalty(output: dict[str, Any]) -> dict[str, Any]:
    """Confidence penalty should be an integer number of points (0-50 range)."""
    val = output.get("suggested_confidence_penalty")
    if val is None:
        return output
    try:
        v = float(val)
        if 0.0 <= v <= 1.0:
            output["suggested_confidence_penalty"] = round(v * 30)
        else:
            output["suggested_confidence_penalty"] = max(0, min(50, round(v)))
    except (TypeError, ValueError):
        output["suggested_confidence_penalty"] = 5
    return output


def _coerce_string_fields(output: dict[str, Any], fields: list[str]) -> dict[str, Any]:
    """If the model returned a dict/list for a field that should be a string, serialise it."""
    for field in fields:
        val = output.get(field)
        if val is not None and not isinstance(val, str):
            if isinstance(val, dict):
                output[field] = "; ".join(f"{k}: {v}" for k, v in val.items())
            elif isinstance(val, list):
                output[field] = "; ".join(str(x) for x in val)
            else:
                output[field] = str(val)
    return output



# ---------------------------------------------------------------------------
# Step 1 — LLM: parse and validate user request
# ---------------------------------------------------------------------------

def step_1_parse_request(state: dict[str, Any], verbose: bool) -> None:
    system = (
        "You are Step 1 of a stock/crypto agent. Parse the user's request into JSON. "
        "Do not analyze the investment. If it is not a public stock/ETF or crypto, set is_valid false. "
        "Use null for unknown fields. JSON only."
    )
    payload = {"raw_user_input": state["user_input"], "today": dt.date.today().isoformat()}
    keys = [
        "is_valid",
        "canonical_ticker",
        "asset_type",
        "company_or_asset_name",
        "market_data_symbol",
        "coingecko_id",
        "news_query",
        "horizon",
        "validation_notes",
    ]
    output, metadata = call_groq_json("step_1_parse_request", system, payload, keys, 450)
    output.setdefault("is_valid", False)
    output.setdefault("validation_notes", [])
    state["parsed_request"] = output
    state["llm_metadata"].append(metadata)
    trace_step(state, 1, "LLM parse and validate request", "llm", payload, output, verbose)


# ---------------------------------------------------------------------------
# Step 2 — Tool: fetch market data (also preserves historical close series)
# ---------------------------------------------------------------------------

def fetch_stock_market_data(symbol: str) -> dict[str, Any]:
    result = {
        "ok": False,
        "provider": "Yahoo Finance chart endpoint",
        "symbol": symbol,
        "currency": None,
        "price": None,
        "previous_close": None,
        "change_percent_1d": None,
        "volume": None,
        "market_cap": None,
        "fifty_two_week_low": None,
        "fifty_two_week_high": None,
        # Full series kept for the technicals tool in Step 2b
        "_closes": [],
        "_volumes": [],
        "source_urls": [],
        "errors": [],
    }
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}?"
        + urllib.parse.urlencode({"range": "1y", "interval": "1d"})
    )
    result["source_urls"].append(url)
    try:
        data = http_json(url)
        chart = (data.get("chart", {}).get("result") or [None])[0]
        if not chart:
            result["errors"].append("Yahoo returned no chart data.")
            return result
        meta = chart.get("meta", {})
        quote = (chart.get("indicators", {}).get("quote") or [{}])[0]
        closes = [float(x) for x in quote.get("close", []) if x is not None]
        volumes = [int(x) for x in quote.get("volume", []) if x is not None]
        highs = [float(x) for x in quote.get("high", []) if x is not None]
        lows = [float(x) for x in quote.get("low", []) if x is not None]
        price = round_number(meta.get("regularMarketPrice") or (closes[-1] if closes else None))
        previous = round_number(meta.get("previousClose") or (closes[-2] if len(closes) > 1 else None))
        change = round_number(((price - previous) / previous) * 100) if price and previous else None
        result.update(
            {
                "ok": True,
                "currency": meta.get("currency"),
                "price": price,
                "previous_close": previous,
                "change_percent_1d": change,
                "volume": volumes[-1] if volumes else None,
                "fifty_two_week_low": round_number(min(lows)) if lows else None,
                "fifty_two_week_high": round_number(max(highs)) if highs else None,
                "_closes": closes,
                "_volumes": volumes,
            }
        )
    except Exception as exc:
        result["errors"].append(str(exc))
    return result


def fetch_crypto_market_data(parsed: dict[str, Any]) -> dict[str, Any]:
    coin_id = parsed.get("coingecko_id") or parsed.get("market_data_symbol")
    result = {
        "ok": False,
        "provider": "CoinGecko public API",
        "coingecko_id": coin_id,
        "currency": "USD",
        "price": None,
        "change_percent_24h": None,
        "change_percent_7d": None,
        "volume": None,
        "market_cap": None,
        # Crypto historical closes fetched separately for technicals
        "_closes": [],
        "_volumes": [],
        "source_urls": [],
        "errors": [],
    }
    if not coin_id:
        result["errors"].append("No CoinGecko id available from Step 1.")
        return result

    # Current market snapshot
    snapshot_url = (
        "https://api.coingecko.com/api/v3/coins/markets?"
        + urllib.parse.urlencode(
            {
                "vs_currency": "usd",
                "ids": coin_id,
                "price_change_percentage": "24h,7d",
                "sparkline": "false",
            }
        )
    )
    result["source_urls"].append(snapshot_url)
    try:
        data = http_json(snapshot_url)
        if not data:
            result["errors"].append("CoinGecko returned no market data.")
            return result
        coin = data[0]
        result.update(
            {
                "ok": True,
                "symbol": coin.get("symbol"),
                "asset_name": coin.get("name"),
                "price": round_number(coin.get("current_price")),
                "change_percent_24h": round_number(coin.get("price_change_percentage_24h")),
                "change_percent_7d": round_number(coin.get("price_change_percentage_7d_in_currency")),
                "volume": coin.get("total_volume"),
                "market_cap": coin.get("market_cap"),
            }
        )
    except Exception as exc:
        result["errors"].append(str(exc))
        return result

    # 90-day OHLC series for technicals (free endpoint, daily candles)
    ohlc_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=90"
    result["source_urls"].append(ohlc_url)
    try:
        ohlc = http_json(ohlc_url)
        # Each entry: [timestamp, open, high, low, close]
        closes = [round_number(row[4]) for row in ohlc if len(row) >= 5]
        result["_closes"] = [c for c in closes if c is not None]
        # CoinGecko OHLC doesn't include volume per candle; leave _volumes empty
    except Exception as exc:
        result["errors"].append(f"OHLC fetch failed (technicals may be partial): {exc}")

    return result


def step_2_fetch_market_data(state: dict[str, Any], verbose: bool) -> None:
    parsed = state["parsed_request"]
    symbol = parsed.get("market_data_symbol") or parsed.get("canonical_ticker")
    if parsed.get("asset_type") == "crypto":
        output = fetch_crypto_market_data(parsed)
    else:
        output = fetch_stock_market_data(str(symbol).upper())
    state["market_data"] = output
    # Expose only the public-facing fields to the trace (strip the _ prefixed series)
    trace_output = {k: v for k, v in output.items() if not k.startswith("_")}
    trace_step(state, 2, "Tool fetch market data", "tool", {"parsed_request": parsed}, trace_output, verbose)


# ---------------------------------------------------------------------------
# Step 2b — Tool: compute technical indicators from the historical series
#
# This is a pure-Python computation tool. No LLM call, no network request.
# It takes the raw close/volume series stored in state["market_data"] and
# produces a structured dict of well-known indicators that the LLM in Step 5
# can reason about directly, rather than being handed a bare price number.
# ---------------------------------------------------------------------------

def _sma(series: list[float], period: int) -> float | None:
    if len(series) < period:
        return None
    return round(sum(series[-period:]) / period, 4)


def _ema(series: list[float], period: int) -> float | None:
    """Exponential moving average of the last `period` values in the series."""
    if len(series) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(series[:period]) / period
    for price in series[period:]:
        ema = price * k + ema * (1 - k)
    return round(ema, 4)


def _rsi(closes: list[float], period: int = 14) -> float | None:
    """Wilder's RSI. Returns None when there is insufficient data."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)


def _macd(closes: list[float]) -> dict[str, float | None]:
    """Standard MACD: EMA(12) - EMA(26), signal = EMA(9) of MACD line."""
    if len(closes) < 26:
        return {"macd_line": None, "signal_line": None, "histogram": None}

    k12, k26, k9 = 2 / 13, 2 / 27, 2 / 10

    def rolling_ema(series: list[float], k: float, warmup: int) -> list[float]:
        ema = sum(series[:warmup]) / warmup
        result = [ema]
        for price in series[warmup:]:
            ema = price * k + ema * (1 - k)
            result.append(ema)
        return result

    ema12 = rolling_ema(closes, k12, 12)
    ema26 = rolling_ema(closes, k26, 26)
    # Align: ema12 is len(closes)-11 long, ema26 is len(closes)-25 long
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[offset + i] - ema26[i] for i in range(len(ema26))]

    if len(macd_line) < 9:
        return {"macd_line": round(macd_line[-1], 4), "signal_line": None, "histogram": None}

    signal_ema = rolling_ema(macd_line, k9, 9)
    macd_val = round(macd_line[-1], 4)
    signal_val = round(signal_ema[-1], 4)
    return {
        "macd_line": macd_val,
        "signal_line": signal_val,
        "histogram": round(macd_val - signal_val, 4),
    }


def _bollinger_bands(closes: list[float], period: int = 20) -> dict[str, float | None]:
    """Bollinger Bands: middle = SMA(20), upper/lower = ±2σ."""
    if len(closes) < period:
        return {"upper": None, "middle": None, "lower": None, "band_width_pct": None}
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = round(middle + 2 * std, 4)
    lower = round(middle - 2 * std, 4)
    middle = round(middle, 4)
    band_width_pct = round(((upper - lower) / middle) * 100, 2) if middle else None
    return {"upper": upper, "middle": middle, "lower": lower, "band_width_pct": band_width_pct}


def _volume_ratio(volumes: list[float], short: int = 5, long: int = 20) -> float | None:
    """Ratio of recent average volume to longer-term average. >1 = rising interest."""
    if len(volumes) < long:
        return None
    avg_short = sum(volumes[-short:]) / short
    avg_long = sum(volumes[-long:]) / long
    return round(avg_short / avg_long, 3) if avg_long else None


def compute_technicals(closes: list[float], volumes: list[float]) -> dict[str, Any]:
    """
    Pure-Python technical indicator tool.
    Receives the historical close and volume series from Step 2 and returns a
    structured dict of indicators. The LLM in Step 5 receives this dict as
    concrete facts rather than being asked to estimate technicals itself.
    """
    if not closes or len(closes) < 2:
        return {"ok": False, "error": "Insufficient price history for technical analysis.", "indicators": {}}

    price = closes[-1]
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    ema12 = _ema(closes, 12)

    indicators: dict[str, Any] = {
        # Trend
        "sma_20": sma20,
        "sma_50": sma50,
        "ema_12": ema12,
        "price_vs_sma20_pct": round(((price - sma20) / sma20) * 100, 2) if sma20 else None,
        "price_vs_sma50_pct": round(((price - sma50) / sma50) * 100, 2) if sma50 else None,
        "sma20_above_sma50": (sma20 > sma50) if (sma20 and sma50) else None,
        # Momentum
        "rsi_14": _rsi(closes, 14),
        "macd": _macd(closes),
        # Volatility
        "bollinger": _bollinger_bands(closes, 20),
        # Volume
        "volume_ratio_5d_20d": _volume_ratio(volumes, 5, 20) if volumes else None,
    }

    # Plain-English signals derived from the numbers — these reduce the
    # burden on the LLM to interpret raw values and make Step 5 more reliable.
    signals: list[str] = []
    rsi = indicators["rsi_14"]
    if rsi is not None:
        if rsi >= 70:
            signals.append(f"RSI {rsi} — overbought territory, momentum may be exhausted.")
        elif rsi <= 30:
            signals.append(f"RSI {rsi} — oversold territory, potential reversal setup.")
        else:
            signals.append(f"RSI {rsi} — neutral momentum range.")

    macd = indicators["macd"]
    if macd.get("macd_line") is not None and macd.get("signal_line") is not None:
        if macd["histogram"] and macd["histogram"] > 0:
            signals.append("MACD histogram positive — bullish momentum building.")
        elif macd["histogram"] and macd["histogram"] < 0:
            signals.append("MACD histogram negative — bearish momentum building.")

    bb = indicators["bollinger"]
    if bb.get("band_width_pct") is not None:
        if bb["band_width_pct"] < 5:
            signals.append(f"Bollinger Band width {bb['band_width_pct']}% — price compression, breakout likely soon.")
        elif bb["band_width_pct"] > 20:
            signals.append(f"Bollinger Band width {bb['band_width_pct']}% — elevated volatility.")

    if indicators.get("sma20_above_sma50") is True:
        signals.append("20-day SMA above 50-day SMA — medium-term uptrend structure.")
    elif indicators.get("sma20_above_sma50") is False:
        signals.append("20-day SMA below 50-day SMA — medium-term downtrend structure.")

    vr = indicators.get("volume_ratio_5d_20d")
    if vr is not None:
        if vr > 1.5:
            signals.append(f"Volume ratio {vr} — recent volume significantly above average (strong participation).")
        elif vr < 0.7:
            signals.append(f"Volume ratio {vr} — recent volume below average (weak conviction).")

    return {"ok": True, "data_points": len(closes), "indicators": indicators, "signals": signals}


def step_2b_compute_technicals(state: dict[str, Any], verbose: bool) -> None:
    market = state["market_data"]
    closes = market.get("_closes", [])
    volumes = market.get("_volumes", [])
    output = compute_technicals(closes, volumes)
    state["technicals"] = output
    trace_step(
        state,
        "2b",
        "Tool compute technical indicators",
        "tool",
        {"data_points": len(closes), "volume_points": len(volumes)},
        output,
        verbose,
    )


# ---------------------------------------------------------------------------
# Step 3 — Tool: fetch recent news headlines
# ---------------------------------------------------------------------------

def step_3_fetch_headlines(state: dict[str, Any], verbose: bool, limit: int) -> None:
    parsed = state["parsed_request"]
    query = parsed.get("news_query") or parsed.get("company_or_asset_name") or parsed.get("canonical_ticker")
    result = {
        "ok": False,
        "provider": "Google News RSS",
        "query": query,
        "items": [],
        "source_urls": [],
        "errors": [],
    }
    if not query:
        result["errors"].append("No news query available from Step 1.")
    else:
        url = (
            "https://news.google.com/rss/search?"
            + urllib.parse.urlencode({"q": f"{query} when:14d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
        )
        result["source_urls"].append(url)
        try:
            root = ET.fromstring(http_text(url))
            for item in root.findall(".//item")[:limit]:
                result["items"].append(
                    {
                        "title": (item.findtext("title") or "").strip(),
                        "published": (item.findtext("pubDate") or "").strip(),
                        "source": (item.findtext("source") or "").strip() or None,
                    }
                )
            result["ok"] = bool(result["items"])
            if not result["ok"]:
                result["errors"].append("No headlines found.")
        except Exception as exc:
            result["errors"].append(str(exc))
    state["news_headlines"] = result
    trace_step(state, 3, "Tool fetch recent headlines", "tool", {"query": query, "limit": limit}, result, verbose)


# ---------------------------------------------------------------------------
# Step 4 — LLM: sentiment analysis of headlines
# ---------------------------------------------------------------------------

def compact_headlines(news: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": news.get("ok"),
        "query": news.get("query"),
        "errors": news.get("errors", []),
        "titles": [item.get("title") for item in news.get("items", [])],
    }


def compact_market(market: dict[str, Any]) -> dict[str, Any]:
    keep = [
        "ok", "provider", "symbol", "coingecko_id", "price", "currency",
        "change_percent_1d", "change_percent_24h", "change_percent_7d",
        "volume", "market_cap", "fifty_two_week_low", "fifty_two_week_high", "errors",
    ]
    return {key: market.get(key) for key in keep if key in market}


def step_4_analyze_sentiment(state: dict[str, Any], verbose: bool) -> None:
    system = (
        "You are Step 4. Classify the supplied headlines for the parsed asset as bullish, "
        "bearish, neutral, or mixed. Use only the given headline text. JSON only."
    )
    payload = {
        "parsed_request_from_step_1": state["parsed_request"],
        "headlines_from_step_3": compact_headlines(state["news_headlines"]),
    }
    keys = ["overall_label", "score", "confidence", "themes", "headline_judgements", "warnings"]
    output, metadata = call_groq_json("step_4_analyze_sentiment", system, payload, keys, 650)
    state["sentiment_analysis"] = output
    state["llm_metadata"].append(metadata)
    trace_step(state, 4, "LLM analyze headline sentiment", "llm", payload, output, verbose)


# ---------------------------------------------------------------------------
# Step 5 — LLM: reconcile technicals, market data, and sentiment
#
# The key design change from the original: this step now receives the
# pre-computed technical indicators from Step 2b as concrete numbers with
# plain-English signals already attached. The LLM's job is genuine
# multi-signal reconciliation — deciding whether the three sources
# (technicals, market data, sentiment) agree or contradict — rather than
# interpreting a bare price figure. This is the step that most benefits
# from being a separate LLM call.
# ---------------------------------------------------------------------------

def step_5_reconcile_signals(state: dict[str, Any], verbose: bool) -> None:
    system = (
        "You are Step 5 of an analysis chain. You receive three independent signal sources: "
        "(A) raw market data from Step 2, (B) computed technical indicators from Step 2b, "
        "(C) news sentiment from Step 4. "
        "Your task is to reconcile them. Do NOT issue a final BUY/HOLD/SELL. Instead, identify: "
        "whether the signals agree or contradict each other; which signal is most credible given "
        "the data quality; what the balance of evidence suggests about risk and opportunity; "
        "and what a prudent analyst would want to verify before making a call. "
        "Be specific — reference actual indicator values, not vague generalities. "
        "STRICT TYPE RULES: "
        "technical_reading and sentiment_reading must be plain-English STRING sentences, not objects. "
        "conflict_notes must be a plain-English STRING. "
        "data_quality_notes must be a plain-English STRING. "
        "key_risks and key_opportunities must be JSON arrays of short strings. "
        "reconciliation_score must be an INTEGER from -100 (strongly bearish) to +100 (strongly bullish). "
        "confidence must be an INTEGER from 0 to 100. "
        "JSON only."
    )
    payload = {
        "parsed_request_from_step_1": state["parsed_request"],
        "market_data_from_step_2": compact_market(state["market_data"]),
        "technicals_from_step_2b": state["technicals"],
        "sentiment_from_step_4": state["sentiment_analysis"],
    }
    keys = [
        "signal_agreement",       # string: "aligned_bullish" | "aligned_bearish" | "mixed" | "contradictory"
        "technical_reading",      # string: plain-English summary of what the technicals say
        "sentiment_reading",      # string: plain-English summary of what sentiment says
        "conflict_notes",         # string: any contradictions worth flagging to the final step
        "key_risks",              # array of strings
        "key_opportunities",      # array of strings
        "data_quality_notes",     # string: flags when tool data was missing or partial
        "reconciliation_score",   # integer: -100 (very bearish) to +100 (very bullish)
        "confidence",             # integer: 0-100
    ]
    output, metadata = call_groq_json("step_5_reconcile_signals", system, payload, keys, 900)
    # Normalise numeric fields: model sometimes returns 0-1 floats despite instructions
    output = _normalise_confidence_fields(output, ["confidence"])
    output = _normalise_reconciliation_score(output)
    state["reconciled_signals"] = output
    state["llm_metadata"].append(metadata)
    trace_step(state, 5, "LLM reconcile technical, market, and sentiment signals", "llm", payload, output, verbose)


# ---------------------------------------------------------------------------
# Step 6 — LLM: draft final recommendation
# ---------------------------------------------------------------------------

def step_6_draft_recommendation(state: dict[str, Any], verbose: bool) -> None:
    system = (
        "You are Step 6 of an analysis chain. Based on the reconciled signals from Step 5, "
        "produce a draft investment recommendation. "
        "Recommendation must be one of: BUY, HOLD, SELL, or INSUFFICIENT_DATA. "
        "Lower confidence when tool data is missing or signals contradict. "
        "This is a draft — it will be critiqued in the next step. "
        "STRICT TYPE RULES: "
        "confidence must be an INTEGER from 0 to 100 (e.g. 62, not 0.62). "
        "thesis must be a STRING of 2-4 sentences explaining the reasoning — not a list. "
        "evidence, catalysts, risks, data_gaps must be arrays of plain-English strings. "
        "JSON only."
    )
    payload = {
        "parsed_request_from_step_1": state["parsed_request"],
        "market_summary_from_step_2": compact_market(state["market_data"]),
        "technicals_from_step_2b": state["technicals"],
        "sentiment_from_step_4": state["sentiment_analysis"],
        "reconciled_signals_from_step_5": state["reconciled_signals"],
    }
    keys = ["ticker", "asset_name", "recommendation", "confidence", "thesis", "evidence", "catalysts", "risks", "data_gaps", "disclaimer"]
    output, metadata = call_groq_json("step_6_draft_recommendation", system, payload, keys, 800)
    output = _normalise_confidence_fields(output, ["confidence"])
    output = _coerce_string_fields(output, ["thesis", "critique_response"])
    state["draft_report"] = output
    state["llm_metadata"].append(metadata)
    trace_step(state, 6, "LLM draft recommendation", "llm", payload, output, verbose)


# ---------------------------------------------------------------------------
# Step 7 — LLM: confidence audit
#
# This step audits whether the draft's stated confidence is actually justified
# by the evidence. It is NOT forced to argue the opposite case — if signals
# are strongly aligned, it should say so and apply little or no penalty.
# The penalty is proportional to real gaps: contradicting signals, thin data,
# ignored risks. Vague market uncertainty does not count. A single prompt
# cannot audit its own confidence, but a chained call can do it honestly.
# ---------------------------------------------------------------------------

def step_7_critique_recommendation(state: dict[str, Any], verbose: bool) -> None:
    draft = state["draft_report"]
    draft_conf = draft.get("confidence", 50)
    system = (
        f"You are Step 7, a Ruthless risk auditor reviewing a draft recommendation. "
        f"The draft says {draft.get('recommendation')} with confidence {draft_conf}/100. "
        # "Your job is NOT to argue for the opposite position. "
        "Your job is to assess whether the stated confidence is justified by the evidence. "
        "Ignore any external bias or previous conclusions; use only your own independent reasoning "
        "to provide a clean, unbiased review based strictly on the provided data. "
        "Ask: are there real data gaps, contradicting signals, or thin evidence that should "
        "reduce confidence? If the signals are strongly aligned and the evidence is solid, "
        "say so — a minor critique with a low penalty (0-5 pts) is a valid and correct output. "
        "Only apply a large penalty when you find specific, concrete problems: a signal the "
        "draft over-weighted relative to its actual strength, a material risk it ignored, a "
        "data gap that would change the call if filled, or a realistic scenario where the "
        "thesis breaks. Vague market uncertainty does not justify a penalty. "
        "Be specific — cite actual indicator values and reconciliation scores from the input. "
        "STRICT TYPE RULES: "
        "counter_thesis must be a plain-English STRING of 2-4 sentences summarising your audit finding. "
        "overweighted_signals, underweighted_risks, data_gaps_that_matter, scenarios_where_draft_fails "
        "must all be arrays of short plain-English strings — empty arrays are valid if nothing applies. "
        "critique_severity must be exactly one of: minor, moderate, major. "
        "suggested_confidence_penalty must be an INTEGER number of points to deduct (0-40); "
        "use 0-5 for well-supported drafts, 10-20 for moderate gaps, 25-40 only for major flaws. "
        "JSON only."
    )
    payload = {
        "draft_recommendation_from_step_6": draft,
        "technicals_from_step_2b": state["technicals"],
        "reconciled_signals_from_step_5": state["reconciled_signals"],
    }
    keys = [
        "draft_recommendation",          # string: echo back what you're critiquing
        "counter_thesis",                # string: the main argument against the draft
        "overweighted_signals",          # array: what the draft relied on too heavily
        "underweighted_risks",           # array: risks the draft glossed over
        "data_gaps_that_matter",         # array: missing information that would change the call
        "scenarios_where_draft_fails",   # array: concrete situations that invalidate the thesis
        "critique_severity",             # string: "minor" | "moderate" | "major"
        "suggested_confidence_penalty",  # integer: points to subtract from draft confidence
    ]
    output, metadata = call_groq_json("step_7_critique_recommendation", system, payload, keys, 800)
    output = _normalise_penalty(output)
    output = _coerce_string_fields(output, ["counter_thesis"])
    state["critique"] = output
    state["llm_metadata"].append(metadata)
    trace_step(state, 7, "LLM self-critique draft recommendation", "llm", payload, output, verbose)


# ---------------------------------------------------------------------------
# Step 8 — LLM: final report reconciling draft and critique
# ---------------------------------------------------------------------------

def step_8_final_report(state: dict[str, Any], verbose: bool) -> None:
    draft = state["draft_report"]
    critique = state["critique"]
    raw_penalty = critique.get("suggested_confidence_penalty", 0) or 0
    # Compute the actual adjusted confidence ceiling the model must stay under
    draft_conf = draft.get("confidence", 50)
    try:
        adjusted_ceiling = max(0, int(draft_conf) - int(raw_penalty))
    except (TypeError, ValueError):
        adjusted_ceiling = 50
    system = (
        "You are Step 8, the final step of an analysis chain. You have a draft recommendation "
        "and a critical counter-argument. Your job is to produce the definitive report. "
        "Read the critique carefully. If it identifies a major flaw that genuinely invalidates "
        "the draft thesis, revise the recommendation accordingly. If the draft holds up despite "
        "the critique, confirm it — but your confidence must not exceed "
        f"{adjusted_ceiling} (draft confidence {draft_conf} minus penalty {raw_penalty}). "
        "In critique_response, write 3-5 sentences explaining: did the critique change your view, "
        "why or why not, and what would need to be true for the opposite recommendation to be correct. "
        "Recommendation must be one of: BUY, HOLD, SELL, or INSUFFICIENT_DATA. "
        "STRICT TYPE RULES: "
        "confidence must be an INTEGER from 0 to 100. "
        "thesis and critique_response must be plain-English STRINGS, not lists or objects. "
        "evidence, catalysts, risks, data_gaps must be arrays of plain-English strings. "
        "JSON only."
    )
    payload = {
        "draft_from_step_6": draft,
        "critique_from_step_7": critique,
        "reconciled_signals_from_step_5": state["reconciled_signals"],
    }
    keys = [
        "ticker", "asset_name", "recommendation", "confidence",
        "thesis", "evidence", "catalysts", "risks", "data_gaps",
        "critique_response",   # 3-5 sentences: did the draft hold up and why?
        "disclaimer",
    ]
    output, metadata = call_groq_json("step_8_final_report", system, payload, keys, 900)
    output = _normalise_confidence_fields(output, ["confidence"])
    output = _coerce_string_fields(output, ["thesis", "critique_response"])
    state["final_report"] = output
    state["llm_metadata"].append(metadata)
    trace_step(state, 8, "LLM finalize report after critique", "llm", payload, output, verbose)


# ---------------------------------------------------------------------------
# Invalid input fast-exit
# ---------------------------------------------------------------------------

def invalid_input_report(state: dict[str, Any], verbose: bool) -> None:
    parsed = state.get("parsed_request", {})
    output = {
        "ticker": parsed.get("canonical_ticker"),
        "asset_name": parsed.get("company_or_asset_name"),
        "recommendation": "INSUFFICIENT_DATA",
        "confidence": 0,
        "thesis": "The input was not a clear public stock, ETF, or crypto request.",
        "evidence": parsed.get("validation_notes", []),
        "catalysts": [],
        "risks": ["Cannot analyze an invalid or ambiguous asset request."],
        "data_gaps": ["No reliable market-data symbol was produced by Step 1."],
        "critique_response": "N/A — chain exited early due to invalid input.",
        "disclaimer": "Educational analysis only, not financial advice.",
    }
    state["final_report"] = output
    trace_step(state, 2, "Stop safely for malformed input", "control", {"parsed_request": parsed}, output, verbose)


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _fmt_pct(val: Any, decimals: int = 2) -> str:
    """Format a percentage value cleanly, handling None."""
    if val is None:
        return "N/A"
    try:
        return f"{float(val):+.{decimals}f}%"
    except (TypeError, ValueError):
        return str(val)


def _fmt_price(val: Any, currency: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        suffix = f" {currency}" if currency else ""
        return f"${v:,.2f}{suffix}" if not currency or currency == "USD" else f"{v:,.2f} {currency}"
    except (TypeError, ValueError):
        return str(val)


def _fmt_large(val: Any) -> str:
    """Format large numbers (volume, market cap) with B/M suffixes."""
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.2f}B"
        if v >= 1e6:
            return f"${v/1e6:.2f}M"
        return f"{v:,.0f}"
    except (TypeError, ValueError):
        return str(val)


def _bullet_list(items: Any) -> str:
    if not items:
        return "- None reported"
    if isinstance(items, str):
        return f"- {items}"
    if isinstance(items, dict):
        return "\n".join(f"- **{k}:** {v}" for k, v in items.items())
    return "\n".join(f"- {item}" for item in items)


def _str_field(val: Any) -> str:
    """Ensure a field renders as clean prose, not a raw dict/list repr."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return "; ".join(f"{k}: {v}" for k, v in val.items())
    if isinstance(val, list):
        return "; ".join(str(x) for x in val)
    return str(val)


def _rsi_label(rsi: Any) -> str:
    if rsi is None:
        return ""
    try:
        v = float(rsi)
        if v >= 70:
            return " ⚠️ overbought"
        if v <= 30:
            return " ⚠️ oversold"
        return " (neutral)"
    except (TypeError, ValueError):
        return ""


def _rec_badge(rec: str | None) -> str:
    badges = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "HOLD": "🟡 HOLD", "INSUFFICIENT_DATA": "⚪ INSUFFICIENT DATA"}
    return badges.get(rec or "", rec or "N/A")


def markdown_report(state: dict[str, Any]) -> str:
    final = state["final_report"]
    market = state.get("market_data", {})
    sentiment = state.get("sentiment_analysis", {})
    technicals = state.get("technicals", {})
    reconciled = state.get("reconciled_signals", {})
    critique = state.get("critique", {})
    draft = state.get("draft_report", {})

    change = market.get("change_percent_1d") or market.get("change_percent_24h")
    currency = market.get("currency") or "USD"

    indicators = technicals.get("indicators", {})
    macd = indicators.get("macd", {}) or {}
    bb = indicators.get("bollinger", {}) or {}

    # Sentiment themes: handle list or string
    themes_raw = sentiment.get("themes", [])
    if isinstance(themes_raw, list):
        themes_str = ", ".join(themes_raw) if themes_raw else "N/A"
    else:
        themes_str = str(themes_raw) if themes_raw else "N/A"

    # Sentiment score: display as 0-1 float (that's what step 4 returns)
    sent_score = sentiment.get("score")
    sent_score_str = f"{float(sent_score):.2f}" if sent_score is not None else "N/A"

    rec = final.get("recommendation")
    conf = final.get("confidence")
    conf_str = f"{conf}/100" if conf is not None else "N/A"

    recon_score = reconciled.get("reconciliation_score")
    recon_str = f"{recon_score:+d}" if isinstance(recon_score, int) else str(recon_score or "N/A")

    severity_emoji = {"minor": "🟢", "moderate": "🟡", "major": "🔴"}.get(
        str(critique.get("critique_severity", "")).lower(), "⚪"
    )

    return f"""# Stock / Crypto Analysis Report

| Field | Value |
|---|---|
| **Ticker** | {final.get("ticker", "N/A")} |
| **Asset** | {final.get("asset_name", "N/A")} |
| **Recommendation** | {_rec_badge(rec)} |
| **Confidence** | {conf_str} |
| **Analysis date** | {state.get("created_at", "")[:10]} |

---

## Investment Thesis

{_str_field(final.get("thesis"))}

## Critique Response

{_str_field(final.get("critique_response"))}

---

## Market Snapshot

| Metric | Value |
|---|---|
| Price | {_fmt_price(market.get("price"), currency)} |
| 1-day change | {_fmt_pct(change)} |
| Volume | {_fmt_large(market.get("volume"))} |
| Market cap | {_fmt_large(market.get("market_cap"))} |
| 52-week low | {_fmt_price(market.get("fifty_two_week_low"), currency)} |
| 52-week high | {_fmt_price(market.get("fifty_two_week_high"), currency)} |

---

## Technical Indicators

| Indicator | Value | Reading |
|---|---|---|
| RSI (14) | {indicators.get("rsi_14", "N/A")} | {_rsi_label(indicators.get("rsi_14")).strip()} |
| MACD line | {macd.get("macd_line", "N/A")} | Histogram: {macd.get("histogram", "N/A")} |
| Signal line | {macd.get("signal_line", "N/A")} | {"Bullish" if (macd.get("histogram") or 0) > 0 else "Bearish"} momentum |
| SMA 20 | {indicators.get("sma_20", "N/A")} | Price {_fmt_pct(indicators.get("price_vs_sma20_pct"), 1)} vs SMA20 |
| SMA 50 | {indicators.get("sma_50", "N/A")} | Price {_fmt_pct(indicators.get("price_vs_sma50_pct"), 1)} vs SMA50 |
| Bollinger width | {bb.get("band_width_pct", "N/A")}% | Upper: {bb.get("upper", "N/A")} / Lower: {bb.get("lower", "N/A")} |
| Volume ratio (5d/20d) | {indicators.get("volume_ratio_5d_20d", "N/A")} | {"Above avg" if (indicators.get("volume_ratio_5d_20d") or 0) > 1 else "Below avg"} |

### What the technicals say

{_bullet_list(technicals.get("signals", []))}

---

## News Sentiment

- **Overall label:** {sentiment.get("overall_label", "N/A").upper()}
- **Sentiment score:** {sent_score_str} (−1.0 = very bearish, +1.0 = very bullish)
- **Key themes:** {themes_str}

---

## Signal Reconciliation *(Step 5)*

**Agreement across signals:** `{reconciled.get("signal_agreement", "N/A")}`  
**Reconciliation score:** {recon_str} / range −100 to +100

**Technicals reading:**
{_str_field(reconciled.get("technical_reading"))}

**Sentiment reading:**
{_str_field(reconciled.get("sentiment_reading"))}

**Conflicts and tensions:**
{_str_field(reconciled.get("conflict_notes"))}

**Data quality notes:**
{_str_field(reconciled.get("data_quality_notes"))}

---

## Step 7 Critique of Draft

**Draft recommendation reviewed:** {_rec_badge(draft.get("recommendation"))} (confidence {draft.get("confidence", "N/A")}/100)  
**Critique severity:** {severity_emoji} {str(critique.get("critique_severity", "N/A")).upper()}  
**Confidence penalty applied:** −{critique.get("suggested_confidence_penalty", 0)} pts

**Counter-thesis:**
{_str_field(critique.get("counter_thesis"))}

**Signals the draft over-weighted:**
{_bullet_list(critique.get("overweighted_signals"))}

**Risks the draft under-weighted:**
{_bullet_list(critique.get("underweighted_risks"))}

**Scenarios where the thesis breaks:**
{_bullet_list(critique.get("scenarios_where_draft_fails"))}

---

## Final Evidence & Risk Assessment

### Supporting evidence
{_bullet_list(final.get("evidence"))}

### Catalysts to watch
{_bullet_list(final.get("catalysts"))}

### Key risks
{_bullet_list(final.get("risks"))}

### Data gaps
{_bullet_list(final.get("data_gaps"))}

---

*{final.get("disclaimer", "Educational analysis only — not financial advice.")}*
"""


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "analysis"


def write_outputs(state: dict[str, Any], output_dir: str) -> dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    ticker = state["final_report"].get("ticker") or state.get("parsed_request", {}).get("canonical_ticker") or "analysis"
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{safe_filename(str(ticker))}_{stamp}"
    json_path = os.path.abspath(os.path.join(output_dir, f"report_{base}.json"))
    md_path = os.path.abspath(os.path.join(output_dir, f"report_{base}.md"))
    trace_path = os.path.abspath(os.path.join(output_dir, f"trace_{base}.json"))
    with open(json_path, "w", encoding="utf-8") as file:
        file.write(pretty_json(state["final_report"]))
    with open(md_path, "w", encoding="utf-8") as file:
        file.write(markdown_report(state))
    with open(trace_path, "w", encoding="utf-8") as file:
        file.write(pretty_json(state))
    return {"json_report": json_path, "markdown_report": md_path, "trace": trace_path}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_agent(user_input: str, output_dir: str, news_limit: int, verbose: bool) -> dict[str, Any]:
    state: dict[str, Any] = {
        "user_input": user_input,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": GROQ_MODEL,
        "trace": [],
        "llm_metadata": [],
    }

    # Step 1: Parse and validate the user's ticker/asset request
    step_1_parse_request(state, verbose)

    if not state["parsed_request"].get("is_valid"):
        invalid_input_report(state, verbose)
    else:
        # Step 2: Fetch live market data (also stores historical close series in state)
        step_2_fetch_market_data(state, verbose)

        # Step 2b: Compute technical indicators from the historical series — pure tool, no LLM
        step_2b_compute_technicals(state, verbose)

        # Step 3: Fetch recent news headlines
        step_3_fetch_headlines(state, verbose, news_limit)

        # Step 4: Classify headline sentiment — LLM reads text, not numbers
        step_4_analyze_sentiment(state, verbose)

        # Step 5: Reconcile technicals + market data + sentiment — the hard reasoning step
        step_5_reconcile_signals(state, verbose)

        # Step 6: Draft a recommendation from the reconciled picture
        step_6_draft_recommendation(state, verbose)

        # Step 7: Steelman the opposite case — critique the draft
        step_7_critique_recommendation(state, verbose)

        # Step 8: Final report that responds to the critique and locks in the call
        step_8_final_report(state, verbose)

    state["output_files"] = write_outputs(state, output_dir)
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Groq Llama stock/crypto analysis agent with technicals and self-critique.")
    parser.add_argument("query", nargs="+", help='Example: "Analyze AAPL short term"')
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--news-limit", type=int, default=DEFAULT_NEWS_LIMIT)
    parser.add_argument("--api-key", help="Groq API key (overrides GROQ_API_KEY env var)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.api_key:
        os.environ["GROQ_API_KEY"] = args.api_key

    try:
        state = run_agent(" ".join(args.query), args.output_dir, args.news_limit, args.verbose)
    except Exception as exc:
        print(f"Agent failed: {exc}")
        return 1

    final = state["final_report"]
    print("\nFinal recommendation:", final.get("recommendation"))
    print("Confidence:", f"{final.get('confidence')}/100")
    print("Critique severity:", state.get("critique", {}).get("critique_severity", "n/a"))
    print("Markdown report:", state["output_files"]["markdown_report"])
    print("JSON report:", state["output_files"]["json_report"])
    print("Trace:", state["output_files"]["trace"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
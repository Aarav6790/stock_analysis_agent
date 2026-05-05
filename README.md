# Stock / Crypto Analysis Agent

A multi-step LLM agent that produces a structured investment analysis report for any public
stock, ETF, or cryptocurrency. Built for the *Build a Multi-Step LLM Agent* programming
assignment using the [Groq](https://groq.com) API with `llama-3.3-70b-versatile`.

> **Note on API naming:** The assignment brief says "Grok API". This project uses
> **Groq** (groq.com), which hosts open-source LLMs including Llama 3.3. If the
> assignment intended xAI's Grok model, the only change required is swapping the
> `GROQ_BASE_URL` and `GROQ_MODEL` environment variables.

---

## What the agent does

The agent accepts a plain-English request (e.g. `"Analyze AAPL short term"`) and works
through eight sequential steps to produce a structured investment analysis report:

| Step | Kind | What it does |
|------|------|--------------|
| 1 | LLM | Parse and validate the user's ticker/asset request |
| 2 | Tool | Fetch live market data from Yahoo Finance (stocks) or CoinGecko (crypto) |
| 2b | Tool | Compute technical indicators (RSI, MACD, Bollinger Bands, SMAs) from the price series |
| 3 | Tool | Fetch recent news headlines from Google News RSS |
| 4 | LLM | Classify headline sentiment (bullish / bearish / neutral / mixed) |
| 5 | LLM | Reconcile technicals, market data, and sentiment — identify agreements and conflicts |
| 6 | LLM | Draft an initial BUY / HOLD / SELL recommendation with supporting evidence |
| 7 | LLM | Steelman the opposite case — critique the draft and identify every weakness |
| 8 | LLM | Produce the final report, explicitly responding to the critique |

No step can be removed without breaking the chain. Step 5 cannot run without the tool
outputs from Steps 2, 2b, and 3. Step 7 cannot argue against a position that Step 6
has not yet taken. Step 8 cannot reconcile a draft and critique that Step 6 and 7 have
not produced.

---

## Chain design rationale

The split between Steps 6, 7, and 8 is the core design decision. A single prompt cannot
genuinely critique its own output — asking one call to produce a recommendation *and* argue
against it produces hedged, uncommitted output. Separating the draft (Step 6) from the
adversarial critique (Step 7) from the final reconciliation (Step 8) forces each LLM call
to do one thing well. Step 7 is explicitly instructed to argue for the *opposite* position,
which surfaces risks the draft glossed over and produces a more honest final confidence score.

Step 5 exists because technicals, price action, and sentiment often point in different
directions. Asking the model to reconcile three signal sources before drafting anything means
Step 6 starts from a picture of the evidence rather than raw numbers.

---

## Dependencies

No third-party packages are required. The agent uses only Python standard library modules:

```
urllib, json, xml.etree.ElementTree, argparse, math, re, time, datetime, os
```

Requires Python 3.10 or later (uses `float | None` union syntax).

---

## Setup

**1. Get a Groq API key**

Sign up at [console.groq.com](https://console.groq.com) and create a free API key.

**2. Export the key**

```bash
export GROQ_API_KEY="gsk_your_key_here"
```

**3. (Optional) Override model or base URL**

```bash
export GROQ_MODEL="llama-3.3-70b-versatile"          # default
export GROQ_API_BASE_URL="https://api.groq.com/openai/v1"  # default
```

---

## Running the agent

```bash
python main.py "Analyze AAPL"
python main.py "Should I buy Bitcoin right now?"
python main.py "What do you think about TSLA short term?"
python main.py --verbose "Analyze ETH"
python main.py --output-dir results --news-limit 10 "Analyze NVDA"
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `outputs/` | Directory to write report files |
| `--news-limit` | `6` | Number of news headlines to fetch |
| `--verbose` | off | Print each step's output to the terminal as it runs |

---

## Inputs the agent handles

- **Stock tickers:** `AAPL`, `NVDA`, `TSLA`, `SPY`, etc.
- **Crypto:** `Bitcoin`, `ETH`, `Solana`, `BTC`, etc.
- **Natural language:** `"Should I buy Apple?"`, `"Analyze gold ETF"` — Step 1 resolves these to a canonical ticker

### What breaks the chain

- **Non-public assets:** private companies, commodities not traded as ETFs, fantasy tickers. Step 1 sets `is_valid: false` and the chain exits early with an `INSUFFICIENT_DATA` report.
- **Delisted or misspelled tickers:** Step 2 returns an error; the chain continues but technicals will be unavailable.
- **No news found:** Step 3 returns empty; Step 4 proceeds with a warning and lower confidence propagates forward.

---

## Outputs

Three files are written to `--output-dir` after every run:

| File | Description |
|------|-------------|
| `report_<TICKER>_<TIMESTAMP>.md` | Human-readable Markdown report |
| `report_<TICKER>_<TIMESTAMP>.json` | Structured JSON with all fields |
| `trace_<TICKER>_<TIMESTAMP>.json` | Full agent state: every step's input, output, and LLM metadata |

The trace file is what the demo questions refer to: it shows exactly what each step received
and what it returned.

---

## Tool integration

Three tool calls are made per run, none of which use an LLM:

| Step | Tool | Why external, not LLM |
|------|------|----------------------|
| 2 | Yahoo Finance chart API / CoinGecko markets API | LLMs cannot produce current price data reliably |
| 2b | Pure-Python implementation of RSI, MACD, Bollinger Bands, SMA | Computing indicators from a 250-point series requires arithmetic, not language generation |
| 3 | Google News RSS | LLMs cannot access real-time news; hallucinated headlines would invalidate the sentiment step |

The output of each tool enters the shared `state` dict as structured data and is passed
verbatim to the next LLM call. The LLMs never see raw price series — only the computed
summary values, which reduces the chance of arithmetic errors in the prompt context.

---

## Prompt design summary

Each LLM step receives:

1. A **system prompt** that defines the step's single responsibility and strict type rules for output fields
2. A **user prompt** containing the prior steps' outputs as a compact JSON payload
3. A `response_format: json_object` constraint so the model cannot return free text

Key prompt decisions:

- **Step 5** is explicitly told *not* to issue a recommendation — its only job is to describe what the signals say and where they conflict. This prevents it from short-circuiting the chain.
- **Step 7** is told to argue for the *opposite* of whatever Step 6 recommended, and is given the draft confidence so it can calibrate how hard to push back.
- **Step 8** receives a computed `adjusted_ceiling` (draft confidence minus penalty) so the final confidence is mechanically bounded by the critique's severity.

Full prompts are in `main.py` inside each `step_N_*` function.

---

## Known limitations

- **Sentiment sample is small.** Only 6 headlines (configurable with `--news-limit`) are analyzed. A trending story can dominate the sentiment score.
- **No fundamental data.** P/E ratio, earnings dates, and revenue figures are not fetched. The analysis is purely technical + sentiment.
- **Yahoo Finance rate limits.** Running many queries in quick succession may return HTTP 429. The Groq rate-limit retry logic handles Groq limits; Yahoo has no retry logic.
- **Market cap is often missing.** Yahoo's free chart endpoint does not reliably return `marketCap`. The report shows N/A and lists this as a data gap.
- **No position sizing or portfolio context.** The recommendation is directional only (BUY / HOLD / SELL). It has no knowledge of the user's holdings or risk tolerance.

---

## Example output summary (AAPL, 2026-05-05)

```
Final recommendation: BUY
Confidence:          55/100
Critique severity:   MODERATE
Confidence penalty:  −15 pts (applied by Step 7)
```

The moderate critique correctly identifies that the 1-day price drop conflicts with the
bullish technical indicators, and that the sentiment sample is limited. The final report
acknowledges these tensions rather than ignoring them.

# Prompt Design Appendix

The agent uses four Groq Llama 70B LLM calls. Each call receives compact JSON from the shared `state` dictionary and returns JSON only. The code also uses Groq JSON Object Mode with `response_format: {"type": "json_object"}`.

The user prompt wrapper is the same for all LLM steps:

```text
Input JSON:
{compact payload from state}

Return JSON only. Required top-level keys:
{list of required keys}
```

This wrapper is short on purpose. The earlier version used long schemas and passed too much state, which caused Groq token-per-minute errors. The current version keeps the chain explainable but uses fewer tokens.

## Step 1: Parse Request

System prompt:

```text
You are Step 1 of a stock/crypto agent. Parse the user's request into JSON. Do not analyze the investment. If it is not a public stock/ETF or crypto, set is_valid false. Use null for unknown fields. JSON only.
```

Required keys:

```json
[
  "is_valid",
  "canonical_ticker",
  "asset_type",
  "company_or_asset_name",
  "market_data_symbol",
  "coingecko_id",
  "news_query",
  "horizon",
  "validation_notes"
]
```

Why this prompt works: Step 2 cannot fetch data unless Step 1 produces a usable symbol and asset type. The prompt forbids investment analysis so this step stays focused on validation.

## Step 4: Sentiment Analysis

System prompt:

```text
You are Step 4. Classify the supplied headlines for the parsed asset as bullish, bearish, neutral, or mixed. Use only the given headline text. JSON only.
```

Required keys:

```json
[
  "overall_label",
  "score",
  "confidence",
  "themes",
  "headline_judgements",
  "warnings"
]
```

Why this prompt works: Step 4 depends on Step 1's parsed asset and Step 3's headline tool output. The prompt says to use only supplied headlines so the model does not invent news.

## Step 5: Combined Market And Sentiment Analysis

System prompt:

```text
You are Step 5. Combine market data from Step 2 and sentiment from Step 4. Discuss momentum, volume/liquidity, volatility, and whether sentiment confirms price action. Do not make a final BUY/HOLD/SELL recommendation. JSON only.
```

Required keys:

```json
[
  "momentum",
  "volume_signal",
  "volatility_risk",
  "sentiment_alignment",
  "opportunities",
  "risks",
  "analysis_score",
  "confidence_notes"
]
```

Why this prompt works: Step 5 is deliberately not the final report. It creates an intermediate analysis that Step 6 can depend on. This makes the chain stronger than four independent prompts.

## Step 6: Final Report

System prompt:

```text
You are Step 6. Produce the final educational stock/crypto report. Recommendation must be BUY, HOLD, SELL, or INSUFFICIENT_DATA. Lower confidence when tool data is missing. JSON only.
```

Required keys:

```json
[
  "ticker",
  "asset_name",
  "recommendation",
  "confidence",
  "thesis",
  "evidence",
  "catalysts",
  "risks",
  "data_gaps",
  "disclaimer"
]
```

Why this prompt works: the final report receives Step 5's analysis, Step 4's sentiment, and Step 2's compact market summary. It is forced to include gaps and risks so failures are visible instead of hidden.

## Prompt Iteration

The first implementation used longer prompts and passed the full shared state into the final LLM call. That was explainable, but it wasted tokens and caused Groq rate-limit errors on the on-demand tier. The revised version keeps the same six-step chain but passes compact data into later steps.

The other important change was adding `max_completion_tokens` to every Groq call. Without that cap, Groq counted a much larger possible completion size against the token-per-minute limit. With the cap, each step is smaller and the retry logic can recover when the limit is still reached.

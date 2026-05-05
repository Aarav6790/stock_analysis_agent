# Written Report

## Problem Statement

The agent solves a focused stock and crypto research task: given a natural-language request such as "Apple stock short term" or "Should I hold BTC next month?", it produces a structured educational BUY, HOLD, SELL, or INSUFFICIENT_DATA report. This problem benefits from multi-step chaining because the final recommendation depends on several intermediate products that should not be collapsed into one prompt. The system first needs to understand which asset the user means, then it needs current market data, then it needs current headlines, and only after that can it compare price action with sentiment. A single prompt would either guess at live data or mix parsing, retrieval, sentiment, and recommendation into one opaque response.

## Chain Design

The first step is a Groq-hosted Llama 70B LLM call that parses the raw user request into a structured asset object. It returns fields such as `canonical_ticker`, `asset_type`, `coingecko_id`, `news_query`, and `horizon`. This is separate because every later tool call depends on having a clean symbol and a clear asset type. If this step marks the input invalid, the chain stops gracefully and writes an INSUFFICIENT_DATA report rather than pretending that a malformed query is analyzable.

The second step is a tool call that fetches market data. For stocks it calls Yahoo Finance's chart endpoint, and for crypto it calls CoinGecko. The output is stored as `market_data` with prices, change percentages, volume, market cap when available, source URLs, and errors if the call fails. The third step is another tool call that fetches recent headlines through Google News RSS. Its output is stored as `news_headlines`. These two tool steps are separate because price action and media sentiment are different kinds of evidence, and a failure in one should not erase the other.

The fourth step is the second Groq-hosted Llama 70B LLM call. It receives the parsed request from Step 1 and the headline array from Step 3, then classifies each headline as bullish, bearish, or neutral. It also returns a numeric sentiment score, key themes, notable signals, and data-quality warnings. This step cannot run meaningfully before Step 3 because it needs raw current headlines, and it needs Step 1 so that it knows which asset the headlines should be interpreted against.

The fifth step is the third Groq-hosted Llama 70B LLM call. It receives Step 4's sentiment JSON and Step 2's market data JSON. It analyzes momentum, volume, volatility, and whether sentiment agrees or conflicts with price action. It deliberately does not make the final recommendation, because its role is to produce an intermediate analytical judgment. The sixth step is the fourth Groq-hosted Llama 70B LLM call. It receives Step 5's analysis and the earlier state, then writes the final structured report. This makes the dependency chain traceable: the final recommendation depends on Step 5, Step 5 depends on Step 4 and Step 2, and Step 4 depends on Step 3 and Step 1.

## Tool Integration

The agent uses free public data sources instead of asking the LLM to invent current numbers. Yahoo Finance and CoinGecko provide price, change, volume, market-cap, and range data. Google News RSS provides recent headline text. The tool outputs enter the shared state as JSON and are passed directly into later LLM prompts. Each tool response includes an `ok` flag, a provider name, source URLs, and an error list. If a tool fails or returns no results, that failure becomes part of the model input so the final report can lower confidence and name the missing evidence.

## Limitations

The largest limitation is data reliability. The tool layer uses public endpoints with no paid service-level guarantee, so rate limits or response-format changes can break market or headline retrieval. The agent also relies on the LLM in Step 1 to normalize ambiguous inputs. A query like "meta" could mean Meta Platforms stock, a general concept, or a token name, so the parser may need to mark it ambiguous. The sentiment step only sees headlines, not full articles, earnings transcripts, filings, or analyst models, so its conclusions are shallow by design. The recommendation is educational and should not be treated as financial advice.

## Reflection

If I had more time, I would add a stronger validation layer after Step 1 by checking candidate tickers against an exchange symbol database before making market calls. I would also cache tool responses for demos, because live APIs are useful but fragile in classroom evaluation settings. During development, I used an LLM to help turn the rough chain design into implementation structure and documentation. The main correction was making the chain stricter than the original sketch: each LLM step now receives the previous LLM output explicitly, every step writes to one shared state object, and tool failures are represented as state rather than hidden behind exceptions. After testing with Groq's on-demand token limits, I also simplified the prompts, limited headlines, and added retry logic for HTTP 429 rate-limit errors.

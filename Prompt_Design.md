Prompt Design

All LLM steps use:

- Strict JSON outputs for reliable chaining. JSON structures are easier for LLMs to consistently generate and easier for downstream steps to parse deterministically without ambiguity.

- Type constraints (for example, confidence = integer 0–100).

- Positional framing (“You are Step X...”) to keep each model focused on one task.

- Selective context injection so each step only receives the data it needs.

---

## Step 1 — Request Parsing

**Full Prompt:**

> “You are Step 1 of a stock/crypto agent. Parse the user's request into JSON. Do not analyze the investment. If it is not a public stock/ETF or crypto, set is_valid false. Use null for unknown fields. JSON only.”

**Why:**
Parses unstructured natural-language user input into clean structured metadata such as ticker symbols, asset type, and market identifiers. This ensures downstream APIs and later reasoning stages receive reliable structured inputs for accurate financial analysis.

---

## Step 4 — Sentiment Analysis

**Full Prompt:**

> “You are Step 4. Classify the supplied headlines for the parsed asset as bullish, bearish, neutral, or mixed. Use only the given headline text. JSON only.”

**Why:**
This stage simply classifies recent news headlines into sentiment labels such as bullish, bearish, neutral, or mixed. The generated sentiment label is later combined with technical indicators during the reconciliation stage, which cleanly separates qualitative sentiment analysis from numerical market analysis.

---

## Step 5 — Signal Reconciliation

**Full Prompt:**

> “You are Step 5 of an analysis chain. You receive three independent signal sources: (A) raw market data from Step 2, (B) computed technical indicators from Step 2b, and (C) news sentiment from Step 4. Your task is to reconcile them. Do NOT issue a final BUY/HOLD/SELL recommendation. Identify whether the signals agree or contradict each other, which signal is most credible, what the balance of evidence suggests, and what risks still need verification. Be specific and reference actual indicator values. reconciliation_score must be an INTEGER from −100 to +100. confidence must be an INTEGER from 0 to 100. JSON only.”

**Why:**
Combines inputs from multiple independent sources such as technical indicators, market data, and sentiment labels, then reasons over them together to generate a balanced interpretation of the asset. This creates a more informed foundation for the later draft recommendation stage instead of relying on any single signal source alone.

---

## Step 6 — Draft Recommendation

**Full Prompt:**

> “You are Step 6 of an analysis chain. Based on the reconciled signals from Step 5, produce a draft investment recommendation. Recommendation must be one of: BUY, HOLD, SELL, or INSUFFICIENT_DATA. Lower confidence when tool data is missing or signals contradict. confidence must be an INTEGER from 0 to 100. thesis must be a STRING explaining the reasoning. evidence, catalysts, risks, and data_gaps must be arrays of plain-English strings. JSON only.”

**Why:**
Separating Step 5 and Step 6 reduces confirmation bias in the model’s reasoning process. LLMs generate text sequentially, so if the model begins drafting a BUY or SELL recommendation too early, it may subconsciously ignore conflicting evidence to keep its response internally consistent. By explicitly preventing Step 5 from issuing a final recommendation, the model is forced to objectively reconcile technical indicators, market data, and sentiment signals first. This produces a more balanced evidence summary before Step 6 converts that reasoning into a draft investment recommendation.

---

## Step 7 — Risk Auditor

### Final Prompt

> “You are Step 7, a Ruthless Risk Auditor reviewing a draft recommendation. The draft says BUY/HOLD/SELL with a stated confidence score. Your job is NOT to argue for the opposite position. Your job is to assess whether the stated confidence is justified by the evidence. Ignore any external bias or previous conclusions and use only the provided data. Ask: are there real data gaps, contradicting signals, or thin evidence that should reduce confidence? If the signals are strongly aligned and the evidence is solid, say so — a minor critique with a low penalty (0–5 points) is completely valid. Only apply a large penalty when there are specific and concrete problems: a signal that was over-weighted, a material risk that was ignored, missing information that could change the recommendation, or realistic scenarios where the thesis fails. Vague market uncertainty does not justify a penalty. Be specific and reference actual indicator values and reconciliation scores. counter_thesis must be a plain-English STRING. overweighted_signals, underweighted_risks, data_gaps_that_matter, and scenarios_where_draft_fails must all be arrays of short strings. critique_severity must be minor, moderate, or major. suggested_confidence_penalty must be an INTEGER from 0–40. JSON only.”

**Why:**
This stage acts as a verification and auditing layer for the entire pipeline. The model reviews the draft recommendation produced by earlier steps and checks whether the conclusions are actually supported by the technical indicators, sentiment signals, and reconciled evidence. If the reasoning from previous steps appears inconsistent, overconfident, or unsupported by the data, the auditor applies a confidence penalty to reduce the reliability of the final recommendation.

### Initial Failed Prompt

> “You are Step 7, a critical analyst reviewing a draft recommendation. Your job is to argue for the opposite position.”

**Problem:**
LLMs are naturally susceptible to sycophancy and tend to follow the behavioral framing given in the prompt. Because the model was explicitly instructed to argue against the draft recommendation, it began manufacturing weak or artificial critiques even when the evidence strongly supported the original signal. This often resulted in fake contradictions and unnecessarily lowered confidence scores for otherwise strong recommendations.

### Improvement

The final version only applies penalties when there are real contradictions, weak evidence, or missing data.

---

## Step 8 — Final Synthesis

**Full Prompt:**

> “You are Step 8, the final step of an analysis chain. You have a draft recommendation and a critical counter-argument. Your job is to produce the definitive report. Read the critique carefully. If it identifies a major flaw that genuinely invalidates the draft thesis, revise the recommendation accordingly. If the draft holds up despite the critique, confirm it — but reduce confidence appropriately based on the critique penalty. In critique_response, explain whether the critique changed your view, why or why not, and what would need to be true for the opposite recommendation to be correct. Recommendation must be one of: BUY, HOLD, SELL, or INSUFFICIENT_DATA. confidence must be an INTEGER from 0 to 100. thesis and critique_response must be plain-English STRINGS. evidence, catalysts, risks, and data_gaps must be arrays of plain-English strings. JSON only.”

**Why:** Creates a final synthesis layer that balances both the original thesis and the critique before generating the final recommendation. Keeping this as a separate step is important because LLMs can become biased toward their own previously generated outputs and may defend earlier reasoning instead of objectively reevaluating the evidence. By introducing an independent synthesis stage after the critique, the model is forced to reassess both perspectives before producing the final recommendation.

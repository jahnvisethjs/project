# Claude Code Instructions — Uncertainty-Aware Clarification Agent

## Project overview

Build an uncertainty-aware conversational agent that decides whether to ask a follow-up clarification question or answer directly, based on how much the LLM disagrees with itself across multiple responses to the same query.

This is a KRR (Knowledge Representation and Reasoning) course project at ASU. The LLM API is **Groq** (free tier). Use **Llama 3.1 70B** as the model.

---

## File structure

```
project/
├── sampler.py          # Multi-sample generator with structured output
├── scorer.py           # Disagreement analysis → uncertainty score
├── agent.py            # Main orchestrator — the conversation loop
├── evaluate.py         # Run experiments + compare baselines
├── test_queries.json   # 30-40 test queries with ground truth labels
├── config.py           # API keys, model name, thresholds, constants
├── requirements.txt    # Dependencies
├── results/            # Output folder for logs, plots, CSVs
└── notebooks/
    └── analysis.ipynb  # Plots and result analysis
```

---

## Module 1: `config.py`

Store all constants here:
- Groq API key (read from environment variable `GROQ_API_KEY`)
- Model name: `"llama-3.1-70b-versatile"`
- Temperature: `1.0` (for sampling diversity)
- Number of samples per query: `5`
- Uncertainty threshold: `0.4` (above this → ask, below → answer)
- Max follow-up rounds: `2` (prevent infinite asking)

---

## Module 2: `sampler.py`

### Function: `generate_samples(query: str, n: int = 5) -> list[dict]`

For a given user query, call the Groq API `n` times at high temperature. Each call uses a system prompt that forces the LLM to respond in **strict JSON** with these fields:

```json
{
  "inferred_intent": "what the model thinks the user actually wants",
  "assumptions": ["list of assumptions the model is making"],
  "missing_information": ["list of info that would help give a better answer"],
  "needs_followup": true/false,
  "suggested_question": "a follow-up question to ask the user (if needed)",
  "answer": "the model's best attempt at answering"
}
```

**System prompt to use** (put this in the API call):

```
You are a helpful assistant. Analyze the user's query and respond ONLY with a JSON object (no markdown, no extra text). The JSON must have these exact fields:

- "inferred_intent": a one-sentence description of what you think the user wants
- "assumptions": a list of assumptions you are making to answer this query
- "missing_information": a list of details that are missing from the query that would help you give a better answer
- "needs_followup": true if the query is too vague or ambiguous to answer well, false if you can answer confidently
- "suggested_question": if needs_followup is true, write one clear follow-up question to ask the user. If false, set to null.
- "answer": your best answer given the information available
```

**Important implementation details:**
- Add `time.sleep(2)` between API calls to respect Groq rate limits
- Parse the JSON response with error handling (sometimes the model adds markdown backticks — strip them)
- If JSON parsing fails for a sample, retry once, then skip that sample
- Return a list of parsed dicts

---

## Module 3: `scorer.py`

### Function: `compute_uncertainty(samples: list[dict]) -> dict`

Takes the list of structured samples from `sampler.py` and computes disagreement.

**Step 1: Compute field-level disagreement**

Use `sentence-transformers` (model: `all-MiniLM-L6-v2`) to embed text fields and compute cosine similarity.

- `intent_score`: Embed all `inferred_intent` strings → compute mean pairwise cosine similarity → disagreement = 1 - mean_similarity
- `assumption_score`: Concatenate each sample's assumptions list into one string → embed → same pairwise process
- `missing_info_score`: Same process with `missing_information` field
- `answer_score`: Same process with `answer` field
- `followup_agreement`: What fraction of samples said `needs_followup = true`? If all agree (all true or all false), agreement is high. Score = 1 - abs(mean - 0.5) * 2 ... actually simpler: just compute the variance of the boolean values.

**Step 2: Combine into overall uncertainty score**

```python
uncertainty = (
    0.25 * intent_score +
    0.25 * assumption_score +
    0.25 * missing_info_score +
    0.15 * answer_score +
    0.10 * followup_variance
)
```

The weights prioritize intent and assumption disagreement since those reveal true ambiguity.

**Return:**
```python
{
    "overall_uncertainty": float,  # 0.0 (confident) to 1.0 (very uncertain)
    "intent_disagreement": float,
    "assumption_disagreement": float,
    "missing_info_disagreement": float,
    "answer_disagreement": float,
    "followup_agreement": float,
    "decision": "ask" or "answer"  # based on threshold from config
}
```

---

## Module 4: `agent.py`

### The main orchestrator and interaction loop

This is the entry point. It has two modes:

**Mode 1: Interactive chat (for demo)**
```
python agent.py --interactive
```
- User types a query
- Agent runs sampler → scorer → decision
- If decision is "ask": prints the follow-up question, waits for user reply, then re-runs the pipeline with the combined context (original query + user's clarification)
- If decision is "answer": picks the best answer from the samples (the one whose intent has highest average similarity to all other intents — i.e., the most "consensus" answer)
- Max 2 follow-up rounds, then force an answer
- Print the uncertainty score and decision at each step (for visibility)

**Mode 2: Single query (for evaluation)**
```
python agent.py --query "I need a laptop"
```
- Runs the pipeline once, prints the result as JSON

**The conversation loop logic:**
```
def run_agent(query):
    context = query
    for round in range(max_rounds):
        samples = generate_samples(context)
        scores = compute_uncertainty(samples)

        if scores["decision"] == "answer" or round == max_rounds - 1:
            best_answer = select_best_answer(samples)
            return best_answer, scores, round + 1

        followup = select_best_question(samples)
        user_reply = get_user_input(followup)
        context = f"Original query: {query}\nClarification: {user_reply}"

    return best_answer, scores, max_rounds
```

**`select_best_answer(samples)`**: Pick the answer whose `inferred_intent` embedding is closest to the centroid of all intent embeddings. This picks the most "consensus" response.

**`select_best_question(samples)`**: Collect all `suggested_question` values from samples where `needs_followup=True`, pick the most common theme or just the first one.

---

## Module 5: `test_queries.json`

Create 35 test queries across these domains. Each query has a label.

```json
[
  {
    "id": 1,
    "query": "I need a laptop",
    "domain": "shopping",
    "expected_label": "ambiguous",
    "notes": "No budget, use case, or specs mentioned"
  },
  {
    "id": 2,
    "query": "I need a ThinkPad T14 with 16GB RAM and 512GB SSD",
    "domain": "shopping",
    "expected_label": "clear",
    "notes": "Fully specified request"
  }
]
```

**Include queries across these domains** (roughly 5-7 per domain):

1. **Shopping**: laptop, phone, headphones — vary from vague ("get me headphones") to specific ("Sony WH-1000XM5 in black")
2. **Travel**: flights, hotels — vary from "book me a flight to Portland" to "book a window seat on Delta flight 402 on June 15"
3. **Coding help**: "write code to store user data" vs "write a Python function that inserts a user dict into a PostgreSQL users table"
4. **General QA**: "tell me about Mercury" (planet or element?) vs "what is the surface temperature of Mercury the planet?"
5. **Food/recipes**: "make me dinner" vs "give me a recipe for chicken tikka masala for 4 people"
6. **Writing help**: "write an email" vs "write a thank-you email to my manager for approving my vacation request"

Label each as "ambiguous" or "clear".

---

## Module 6: `evaluate.py`

### Run full evaluation comparing 3 systems

**System 1: Your uncertainty-aware agent**
- Runs the full pipeline (sample → score → decide)

**System 2: Always-answer baseline**
- Calls the LLM once with the query, returns the answer directly, never asks follow-up

**System 3: Always-ask baseline**
- Always asks one follow-up question before answering (uses the same structured prompt, but ignores uncertainty and always asks)

**For each test query, log:**
```json
{
  "query_id": 1,
  "query": "...",
  "expected_label": "ambiguous",
  "system": "uncertainty_agent",
  "uncertainty_score": 0.72,
  "decision": "ask",
  "correct_decision": true,
  "num_turns": 2,
  "followup_asked": "What is your budget and primary use case?",
  "final_answer": "..."
}
```

**Metrics to compute (print as a table and save to CSV):**
- **Decision accuracy**: did the system correctly ask for ambiguous queries and answer for clear ones?
- **Unnecessary clarification rate**: % of clear queries where the system asked a follow-up anyway
- **Missed ambiguity rate**: % of ambiguous queries where the system answered without asking
- **Average turns to completion**: mean number of turns across all queries
- **Per-domain breakdown**: show metrics split by domain

**Generate these plots** (save as PNG in results/ folder):
1. Uncertainty score distribution: histogram showing scores for "clear" vs "ambiguous" queries (should show separation)
2. Bar chart: decision accuracy for all 3 systems
3. Bar chart: unnecessary clarification rate for all 3 systems
4. Bar chart: average turns for all 3 systems
5. Scatter plot: uncertainty score vs query length (to show it's not just based on length)

Use matplotlib for plots.

---

## Dependencies (`requirements.txt`)

```
groq
sentence-transformers
torch
numpy
scikit-learn
matplotlib
pandas
```

---

## How to run

```bash
# Install
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"

# Interactive demo
python agent.py --interactive

# Run full evaluation
python evaluate.py

# Results will be in results/ folder
```

---

## Important notes

- Always add `time.sleep(2)` between Groq API calls to avoid rate limiting
- Use `json.loads()` with fallback stripping of markdown backticks for parsing
- Cache API responses to avoid re-running expensive calls during debugging — save raw responses to a JSON file and load from cache if it exists
- Print progress during evaluation ("Processing query 5/35...")
- Handle edge cases: what if all 5 samples fail to parse? Return a default "answer directly" decision
- The sentence-transformers model should be loaded once globally, not per function call

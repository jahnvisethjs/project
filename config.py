"""Configuration constants for the Uncertainty-Aware Clarification Agent."""

import os

# Groq API settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "llama-3.3-70b-versatile"

# Sampling settings
TEMPERATURE = 1.0
NUM_SAMPLES = 5

# Decision settings
UNCERTAINTY_THRESHOLD = 0.3  # Dempster-Shafer belief threshold for asking
MAX_FOLLOWUP_ROUNDS = 2      # prevent infinite asking

# Rate limiting
API_SLEEP_SECONDS = 2

# Sentence-transformers model (loaded once globally)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------- KRR-Enhanced Scoring Weights ----------

# Evidence source weights for Dempster-Shafer mass assignment
# Each source contributes a mass function; these control the strength
# of each source's "vote" in the combination.
DS_WEIGHTS = {
    "followup_consensus": 0.40,   # Strongest signal: LLM's own assessment
    "epistemic_gap": 0.25,        # Missing information count (CWA)
    "null_answer_rate": 0.15,     # Proportion of null answers
    "embedding_disagreement": 0.20,  # Original cosine dissimilarity
}

# Legacy weights (kept for backward compatibility / comparison)
WEIGHTS = {
    "intent": 0.25,
    "assumption": 0.25,
    "missing_info": 0.25,
    "answer": 0.15,
    "followup_variance": 0.10,
}

# ---------- Epistemic Logic: Knowledge Gap Normalization ----------
# Queries with >= this many missing info items are considered maximally uncertain
EPISTEMIC_GAP_MAX = 4.0

# ---------- Default Reasoning: Domain Information Requirements ----------
# Under default logic, if required slots are in missing_information, the query
# is presumed ambiguous unless overridden by strong counter-evidence.
DOMAIN_REQUIREMENTS = {
    "shopping":   ["budget", "use case", "specifications", "brand"],
    "travel":     ["origin", "destination", "dates", "class"],
    "coding":     ["language", "framework", "task description", "error details"],
    "food":       ["cuisine", "dietary restrictions", "servings", "ingredients"],
    "writing":    ["recipient", "purpose", "tone", "document type"],
    "general_qa": ["specific entity", "specific question", "context"],
}

# System prompt for structured sampling
SYSTEM_PROMPT = """You are a helpful assistant. Analyze the user's query and respond ONLY with a JSON object (no markdown, no extra text). The JSON must have these exact fields:

- "inferred_intent": a one-sentence description of what you think the user wants
- "assumptions": a list of assumptions you are making to answer this query
- "missing_information": a list of details that are missing from the query that would help you give a better answer
- "needs_followup": true if the query is too vague or ambiguous to answer well, false if you can answer confidently
- "suggested_question": if needs_followup is true, write one clear follow-up question to ask the user. If false, set to null.
- "answer": your best answer given the information available"""

# Cache settings
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

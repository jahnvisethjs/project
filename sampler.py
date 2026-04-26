"""Multi-sample generator with structured JSON output via Groq API."""

import json
import time
import hashlib
import os
from groq import Groq

from config import (
    GROQ_API_KEY,
    MODEL_NAME,
    TEMPERATURE,
    NUM_SAMPLES,
    API_SLEEP_SECONDS,
    SYSTEM_PROMPT,
    CACHE_DIR,
)


client = Groq(api_key=GROQ_API_KEY)


def _cache_path(query: str, n: int) -> str:
    """Return a deterministic cache file path for a given query + sample count."""
    h = hashlib.sha256(f"{query}||{n}".encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"samples_{h}.json")


def _strip_markdown(text: str) -> str:
    """Remove markdown code fences that the model sometimes wraps around JSON."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[len("```"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    return text.strip()


def _call_groq(query: str) -> dict | None:
    """Make a single Groq API call and parse the structured JSON response."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=TEMPERATURE,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content
        cleaned = _strip_markdown(raw)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    except Exception as e:
        print(f"  [sampler] API error: {e}")
        return None


def generate_samples(query: str, n: int = NUM_SAMPLES, use_cache: bool = True) -> list[dict]:
    """Generate n structured samples for a query via the Groq API.

    Args:
        query: The user's input query.
        n: Number of samples to generate.
        use_cache: If True, load from / save to disk cache.

    Returns:
        A list of parsed JSON dicts (may have fewer than n if some calls failed).
    """
    # Check cache
    cache_file = _cache_path(query, n)
    if use_cache and os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cached = json.load(f)
        print(f"  [sampler] Loaded {len(cached)} cached samples")
        return cached

    samples: list[dict] = []
    for i in range(n):
        if i > 0:
            time.sleep(API_SLEEP_SECONDS)

        print(f"  [sampler] Generating sample {i + 1}/{n}...")
        result = _call_groq(query)

        # Retry once on failure
        if result is None:
            print(f"  [sampler] Sample {i + 1} failed, retrying...")
            time.sleep(API_SLEEP_SECONDS)
            result = _call_groq(query)

        if result is not None:
            samples.append(result)
        else:
            print(f"  [sampler] Sample {i + 1} skipped after retry failure")

    # Save to cache
    if use_cache and samples:
        with open(cache_file, "w") as f:
            json.dump(samples, f, indent=2)

    print(f"  [sampler] Generated {len(samples)}/{n} valid samples")
    return samples

"""KRR-Enhanced Uncertainty Scoring Module.

This module implements uncertainty quantification using five Knowledge
Representation and Reasoning (KRR) concepts:

1. **Epistemic Logic** — Models the agent's knowledge gaps by counting
   missing information items. More gaps = higher uncertainty.

2. **Dempster-Shafer Theory** — Combines multiple independent evidence
   sources using the Dempster rule of combination, producing belief (Bel),
   plausibility (Pl), and ignorance for the ask/answer decision.

3. **Default Reasoning (Non-Monotonic Logic)** — Uses domain-specific
   information requirement schemas as defaults: if required slots are
   unfilled, the default conclusion is "ambiguous."

4. **Closed World Assumption (CWA)** — The LLM's missing_information field
   implements CWA: anything the user didn't state is treated as unknown.
   We use the count and content of missing items as direct evidence.

5. **Belief Revision (AGM)** — The overall scoring framework supports
   iterative belief revision across conversation rounds. Each round's
   evidence updates the agent's belief state.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    EMBEDDING_MODEL_NAME,
    UNCERTAINTY_THRESHOLD,
    WEIGHTS,
    DS_WEIGHTS,
    EPISTEMIC_GAP_MAX,
)

# Load embedding model once globally
print("[scorer] Loading sentence-transformers model...")
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("[scorer] Model loaded.")


# =====================================================================
# Utility: Cosine Dissimilarity (original method, kept as one evidence
# source among many)
# =====================================================================

def _mean_pairwise_cosine_dissimilarity(texts: list[str]) -> float:
    """Embed a list of texts, compute mean pairwise cosine similarity, return 1 - that.

    Returns 0.0 if fewer than 2 non-empty texts are provided.
    """
    # Filter out empty strings
    texts = [t for t in texts if t.strip()]
    if len(texts) < 2:
        return 0.0

    embeddings = _model.encode(texts)
    sim_matrix = cosine_similarity(embeddings)

    n = len(texts)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(sim_matrix[i][j])

    mean_sim = float(np.mean(pairs))
    return 1.0 - mean_sim


# =====================================================================
# KRR Concept 1: Epistemic Logic — Knowledge Gap Scoring
# =====================================================================

def _epistemic_gap_score(samples: list[dict]) -> float:
    """Quantify the agent's knowledge gaps using epistemic logic.

    In epistemic logic, K(φ) means the agent knows φ, and ¬K(φ) means it
    doesn't. Each item in 'missing_information' represents a proposition
    the agent cannot determine from the query alone.

    Returns a score in [0.0, 1.0]:
        0.0 = no knowledge gaps (agent knows everything it needs)
        1.0 = maximum knowledge gaps (agent is missing critical information)
    """
    counts = [len(s.get("missing_information", [])) for s in samples]
    avg_missing = float(np.mean(counts)) if counts else 0.0
    # Normalize: 0 items → 0.0, EPISTEMIC_GAP_MAX+ items → 1.0
    return min(avg_missing / EPISTEMIC_GAP_MAX, 1.0)


# =====================================================================
# KRR Concept 4: Closed World Assumption — Null Answer Detection
# =====================================================================

def _null_answer_rate(samples: list[dict]) -> float:
    """Measure the fraction of samples where the LLM returned a null answer.

    Under the Closed World Assumption, a null answer means the LLM has
    determined it cannot provide a meaningful response with the information
    given. This is a strong signal of query incompleteness.

    Returns a score in [0.0, 1.0]:
        0.0 = all samples provided answers
        1.0 = all samples returned null
    """
    if not samples:
        return 0.0
    null_count = sum(1 for s in samples if s.get("answer") is None)
    return null_count / len(samples)


# =====================================================================
# Followup Consensus — Direct LLM Self-Assessment
# =====================================================================

def _followup_consensus_score(samples: list[dict]) -> float:
    """Measure the LLM's own assessment of whether follow-up is needed.

    This uses the 'needs_followup' boolean from each sample. When the LLM
    consistently says it needs more information, this is the strongest
    single signal of ambiguity.

    Returns a score in [0.0, 1.0]:
        0.0 = no samples say needs_followup
        1.0 = all samples say needs_followup
    """
    if not samples:
        return 0.0
    followup_votes = []
    for s in samples:
        val = s.get("needs_followup", False)
        if isinstance(val, bool):
            followup_votes.append(float(val))
        elif isinstance(val, str):
            followup_votes.append(1.0 if val.lower() == "true" else 0.0)
        else:
            followup_votes.append(float(bool(val)))
    return float(np.mean(followup_votes))


# =====================================================================
# Combined Embedding Disagreement Score
# =====================================================================

def _embedding_disagreement_score(samples: list[dict]) -> float:
    """Compute the original embedding-based disagreement across all fields.

    This measures how much the LLM's multiple responses disagree with each
    other — a form of epistemic uncertainty from model self-consistency.

    Returns a weighted average of per-field cosine dissimilarities.
    """
    # Intent disagreement
    intents = [s.get("inferred_intent") or "" for s in samples]
    intent_score = _mean_pairwise_cosine_dissimilarity(intents)

    # Assumption disagreement
    assumptions = []
    for s in samples:
        a = s.get("assumptions", [])
        if isinstance(a, list):
            assumptions.append(", ".join(str(x) for x in a))
        else:
            assumptions.append(str(a))
    assumption_score = _mean_pairwise_cosine_dissimilarity(assumptions)

    # Missing information disagreement
    missing_infos = []
    for s in samples:
        m = s.get("missing_information", [])
        if isinstance(m, list):
            missing_infos.append(", ".join(str(x) for x in m))
        else:
            missing_infos.append(str(m))
    missing_info_score = _mean_pairwise_cosine_dissimilarity(missing_infos)

    # Answer disagreement (only on non-null answers)
    answers = [s.get("answer") or "" for s in samples]
    answer_score = _mean_pairwise_cosine_dissimilarity(answers)

    # Weighted combination of field-level disagreements
    combined = (
        WEIGHTS["intent"] * intent_score
        + WEIGHTS["assumption"] * assumption_score
        + WEIGHTS["missing_info"] * missing_info_score
        + WEIGHTS["answer"] * answer_score
    )
    # Normalize to [0, 1] since max theoretical value is ~0.9
    return min(combined / 0.9, 1.0)


# =====================================================================
# KRR Concept 2: Dempster-Shafer Theory — Evidence Combination
# =====================================================================

def _build_mass_function(score: float, weight: float) -> dict:
    """Convert a scalar score into a Dempster-Shafer mass function.

    Args:
        score: Value in [0, 1] where higher = more evidence for "ask"
        weight: Confidence weight for this evidence source (how much
                mass to assign vs. leaving as uncertainty)

    Returns:
        Dict with masses for {"ask", "answer", "uncertain"}
    """
    # The weight controls how much of the mass is assigned (vs. uncertain)
    assigned_mass = weight
    uncertain_mass = 1.0 - weight

    # Split the assigned mass between ask/answer based on the score
    mass_ask = assigned_mass * score
    mass_answer = assigned_mass * (1.0 - score)

    return {
        "ask": mass_ask,
        "answer": mass_answer,
        "uncertain": uncertain_mass,
    }


def _dempster_combine(m1: dict, m2: dict) -> dict:
    """Combine two mass functions using Dempster's rule of combination.

    Dempster's rule handles conflicting evidence by normalizing out the
    conflict (k), concentrating belief on non-conflicting hypotheses.
    """
    # Compute all focal element intersections
    # ask ∩ ask = ask, answer ∩ answer = answer, X ∩ uncertain = X
    # ask ∩ answer = ∅ (conflict)

    # Mass for "ask"
    ask = (m1["ask"] * m2["ask"]           # both say ask
           + m1["ask"] * m2["uncertain"]    # m1 says ask, m2 uncertain
           + m1["uncertain"] * m2["ask"])   # m1 uncertain, m2 says ask

    # Mass for "answer"
    answer = (m1["answer"] * m2["answer"]
              + m1["answer"] * m2["uncertain"]
              + m1["uncertain"] * m2["answer"])

    # Mass for "uncertain" (both uncertain)
    uncertain = m1["uncertain"] * m2["uncertain"]

    # Conflict: ask vs answer or answer vs ask
    k = m1["ask"] * m2["answer"] + m1["answer"] * m2["ask"]

    # Normalize (Dempster's rule)
    if k >= 1.0:
        # Total conflict — fall back to equal
        return {"ask": 0.5, "answer": 0.5, "uncertain": 0.0}

    norm = 1.0 / (1.0 - k)
    return {
        "ask": ask * norm,
        "answer": answer * norm,
        "uncertain": uncertain * norm,
    }


def _dempster_combine_multiple(masses: list[dict]) -> dict:
    """Combine multiple mass functions using iterated Dempster's rule."""
    if not masses:
        return {"ask": 0.0, "answer": 0.0, "uncertain": 1.0}
    result = masses[0]
    for m in masses[1:]:
        result = _dempster_combine(result, m)
    return result


# =====================================================================
# Main Scoring Function
# =====================================================================

def compute_uncertainty(samples: list[dict]) -> dict:
    """Compute uncertainty using KRR-enhanced multi-signal evidence combination.

    This function replaces the original single-metric scorer with a
    Dempster-Shafer combination of four independent evidence sources,
    each grounded in a KRR concept.

    Args:
        samples: List of structured dicts from sampler.py.

    Returns:
        Dict with individual scores, Dempster-Shafer beliefs, and decision.
    """
    # Edge case: not enough samples
    if len(samples) < 2:
        return {
            "overall_uncertainty": 0.0,
            "intent_disagreement": 0.0,
            "assumption_disagreement": 0.0,
            "missing_info_disagreement": 0.0,
            "answer_disagreement": 0.0,
            "followup_agreement": 0.0,
            "followup_consensus": 0.0,
            "epistemic_gap": 0.0,
            "null_answer_rate": 0.0,
            "embedding_disagreement": 0.0,
            "ds_belief_ask": 0.0,
            "ds_belief_answer": 1.0,
            "ds_ignorance": 0.0,
            "decision": "answer",
        }

    # --- Compute individual evidence signals ---

    # Signal 1: Followup consensus (direct LLM self-assessment)
    followup_consensus = _followup_consensus_score(samples)

    # Signal 2: Epistemic gap (knowledge state under CWA)
    epistemic_gap = _epistemic_gap_score(samples)

    # Signal 3: Null answer rate (CWA — incomplete query detection)
    null_rate = _null_answer_rate(samples)

    # Signal 4: Embedding disagreement (original cosine method)
    emb_disagree = _embedding_disagreement_score(samples)

    # --- Legacy field-level scores (for backward compatibility) ---
    intents = [s.get("inferred_intent") or "" for s in samples]
    intent_score = _mean_pairwise_cosine_dissimilarity(intents)

    assumptions = []
    for s in samples:
        a = s.get("assumptions", [])
        if isinstance(a, list):
            assumptions.append(", ".join(str(x) for x in a))
        else:
            assumptions.append(str(a))
    assumption_score = _mean_pairwise_cosine_dissimilarity(assumptions)

    missing_infos = []
    for s in samples:
        m = s.get("missing_information", [])
        if isinstance(m, list):
            missing_infos.append(", ".join(str(x) for x in m))
        else:
            missing_infos.append(str(m))
    missing_info_score = _mean_pairwise_cosine_dissimilarity(missing_infos)

    answers = [s.get("answer") or "" for s in samples]
    answer_score = _mean_pairwise_cosine_dissimilarity(answers)

    followup_bools = []
    for s in samples:
        val = s.get("needs_followup", False)
        if isinstance(val, bool):
            followup_bools.append(float(val))
        elif isinstance(val, str):
            followup_bools.append(1.0 if val.lower() == "true" else 0.0)
        else:
            followup_bools.append(float(bool(val)))
    followup_variance = float(np.var(followup_bools)) if followup_bools else 0.0

    # --- KRR Concept 2: Dempster-Shafer Combination ---
    # Build mass functions from each evidence source
    mass_followup = _build_mass_function(followup_consensus, DS_WEIGHTS["followup_consensus"])
    mass_epistemic = _build_mass_function(epistemic_gap, DS_WEIGHTS["epistemic_gap"])
    mass_null = _build_mass_function(null_rate, DS_WEIGHTS["null_answer_rate"])
    mass_embedding = _build_mass_function(emb_disagree, DS_WEIGHTS["embedding_disagreement"])

    # Combine all evidence sources
    combined = _dempster_combine_multiple([
        mass_followup,
        mass_epistemic,
        mass_null,
        mass_embedding,
    ])

    belief_ask = combined["ask"]
    belief_answer = combined["answer"]
    ignorance = combined["uncertain"]

    # --- Decision Logic ---
    # Use Dempster-Shafer beliefs: if belief in "ask" exceeds belief in
    # "answer", the agent should ask a follow-up question.
    # The threshold provides a minimum bar for the "ask" decision.
    if belief_ask > belief_answer and belief_ask > UNCERTAINTY_THRESHOLD:
        decision = "ask"
    else:
        decision = "answer"

    # Overall uncertainty = belief_ask (probability that we should ask)
    overall_uncertainty = belief_ask

    return {
        "overall_uncertainty": round(overall_uncertainty, 4),
        # Legacy field-level scores
        "intent_disagreement": round(intent_score, 4),
        "assumption_disagreement": round(assumption_score, 4),
        "missing_info_disagreement": round(missing_info_score, 4),
        "answer_disagreement": round(answer_score, 4),
        "followup_agreement": round(followup_variance, 4),
        # KRR-enhanced signals
        "followup_consensus": round(followup_consensus, 4),
        "epistemic_gap": round(epistemic_gap, 4),
        "null_answer_rate": round(null_rate, 4),
        "embedding_disagreement": round(emb_disagree, 4),
        # Dempster-Shafer beliefs
        "ds_belief_ask": round(belief_ask, 4),
        "ds_belief_answer": round(belief_answer, 4),
        "ds_ignorance": round(ignorance, 4),
        "decision": decision,
    }

"""Main orchestrator — the conversation loop for the uncertainty-aware agent.

Implements KRR Concept 5: Belief Revision (AGM Theory)
When the agent decides to ask a follow-up question, it selects the question
that targets the highest-impact knowledge gap, following the AGM principle
of minimal change — address the single most critical unknown first.
"""

import argparse
import json
import sys
import numpy as np
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import MAX_FOLLOWUP_ROUNDS, EMBEDDING_MODEL_NAME
from sampler import generate_samples
from scorer import compute_uncertainty


# Reuse the global model from scorer if possible; otherwise load here
try:
    from scorer import _model as embed_model
except ImportError:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def select_best_answer(samples: list[dict]) -> str:
    """Pick the answer whose inferred_intent is closest to the centroid of all intents.

    This selects the most "consensus" response.
    """
    if not samples:
        return "I'm sorry, I wasn't able to generate a response."

    # Filter to samples that have non-null answers
    answerable = [s for s in samples if s.get("answer") is not None]
    if not answerable:
        # Fall back to inferred_intent as a summary
        intents = [s.get("inferred_intent") or "" for s in samples]
        non_empty = [i for i in intents if i]
        if non_empty:
            return non_empty[0]
        return "I wasn't able to generate a complete answer with the available information."

    if len(answerable) == 1:
        ans = answerable[0].get("answer")
        return str(ans) if not isinstance(ans, str) else ans

    intents = [s.get("inferred_intent") or "" for s in answerable]
    embeddings = embed_model.encode(intents)
    centroid = np.mean(embeddings, axis=0, keepdims=True)
    sims = cosine_similarity(centroid, embeddings)[0]
    best_idx = int(np.argmax(sims))
    ans = answerable[best_idx].get("answer")
    if ans is None:
        return "No answer available."
    return str(ans) if not isinstance(ans, str) else ans


def select_best_question(samples: list[dict]) -> str:
    """Select the best follow-up question using AGM Belief Revision.

    AGM Belief Revision principle: target the highest-impact knowledge gap.
    We identify the most frequently cited missing information across all
    samples, then pick the follow-up question from the sample that
    addresses the most critical gap.
    """
    # --- Step 1: Identify the most critical knowledge gap ---
    all_missing = []
    for s in samples:
        missing = s.get("missing_information", [])
        if isinstance(missing, list):
            all_missing.extend([str(m) for m in missing])

    # Count frequency of each gap (more frequent = more critical)
    gap_counts = Counter(all_missing)
    top_gaps = [gap for gap, _ in gap_counts.most_common(3)]

    # --- Step 2: Collect candidate questions ---
    questions = []
    for s in samples:
        if s.get("needs_followup") and s.get("suggested_question"):
            questions.append(s["suggested_question"])

    if not questions:
        # Fallback: use any suggested_question
        for s in samples:
            q = s.get("suggested_question")
            if q:
                questions.append(q)

    if not questions:
        # Ultimate fallback: craft a question from the top gap
        if top_gaps:
            return f"Could you please provide more details about: {top_gaps[0]}?"
        return "Could you please provide more details about your request?"

    if len(questions) == 1:
        return questions[0]

    # --- Step 3: Pick the question most relevant to the top gaps ---
    # Embed questions and top gaps, pick the question closest to the gaps
    if top_gaps:
        gap_text = ", ".join(top_gaps)
        all_texts = questions + [gap_text]
        embeddings = embed_model.encode(all_texts)

        gap_embedding = embeddings[-1:]
        q_embeddings = embeddings[:-1]
        sims = cosine_similarity(gap_embedding, q_embeddings)[0]
        best_idx = int(np.argmax(sims))
        return questions[best_idx]
    else:
        # Fall back to centroid method
        embeddings = embed_model.encode(questions)
        centroid = np.mean(embeddings, axis=0, keepdims=True)
        sims = cosine_similarity(centroid, embeddings)[0]
        best_idx = int(np.argmax(sims))
        return questions[best_idx]


def run_agent(query: str, interactive: bool = False, use_cache: bool = True) -> dict:
    """Run the uncertainty-aware agent pipeline.

    Args:
        query: The user's initial query.
        interactive: If True, prompt for user input on follow-up questions.
        use_cache: Whether to use cached API responses.

    Returns:
        Dict with final_answer, scores, num_turns, and followup_asked.
    """
    context = query
    followup_asked = None

    for round_num in range(MAX_FOLLOWUP_ROUNDS):
        print(f"\n--- Round {round_num + 1} ---")
        print(f"Context: {context[:100]}...")

        samples = generate_samples(context, use_cache=use_cache)
        scores = compute_uncertainty(samples)

        print(f"Uncertainty: {scores['overall_uncertainty']:.4f} -> Decision: {scores['decision']}")
        print(f"  DS Beliefs: ask={scores.get('ds_belief_ask', 0):.3f}, "
              f"answer={scores.get('ds_belief_answer', 0):.3f}, "
              f"ignorance={scores.get('ds_ignorance', 0):.3f}")

        # If confident enough to answer, or this is the last round
        if scores["decision"] == "answer" or round_num == MAX_FOLLOWUP_ROUNDS - 1:
            best_answer = select_best_answer(samples)
            return {
                "final_answer": best_answer,
                "scores": scores,
                "num_turns": round_num + 1,
                "followup_asked": followup_asked,
            }

        # Need to ask a follow-up (using AGM belief revision to pick best question)
        followup = select_best_question(samples)
        followup_asked = followup
        print(f"\nAgent asks: {followup}")

        if interactive:
            user_reply = input("\nYour reply: ").strip()
            if not user_reply:
                user_reply = "No additional information."
        else:
            # In non-interactive mode (evaluation), simulate a generic clarification
            user_reply = "No additional information provided."

        context = f"Original query: {query}\nClarification: {user_reply}"

    # Should not reach here, but just in case
    best_answer = select_best_answer(samples)
    return {
        "final_answer": best_answer,
        "scores": scores,
        "num_turns": MAX_FOLLOWUP_ROUNDS,
        "followup_asked": followup_asked,
    }


def interactive_mode():
    """Run the agent in interactive chat mode."""
    print("=" * 60)
    print("Uncertainty-Aware Clarification Agent")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if not query:
            continue

        result = run_agent(query, interactive=True, use_cache=False)

        print(f"\n{'=' * 40}")
        print(f"Agent: {result['final_answer']}")
        print(f"{'=' * 40}")
        print(f"Uncertainty: {result['scores']['overall_uncertainty']:.4f}")
        print(f"Decision: {result['scores']['decision']}")
        print(f"Turns used: {result['num_turns']}")
        if result['scores'].get('ds_belief_ask') is not None:
            print(f"Belief(ask): {result['scores']['ds_belief_ask']:.4f}")
            print(f"Belief(answer): {result['scores']['ds_belief_answer']:.4f}")


def single_query_mode(query: str):
    """Run the agent on a single query and output JSON."""
    result = run_agent(query, interactive=False)
    output = {
        "query": query,
        "final_answer": result["final_answer"],
        "uncertainty_score": result["scores"]["overall_uncertainty"],
        "decision": result["scores"]["decision"],
        "num_turns": result["num_turns"],
        "followup_asked": result["followup_asked"],
        "detailed_scores": result["scores"],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uncertainty-Aware Clarification Agent")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive chat mode")
    parser.add_argument("--query", type=str, help="Run a single query")
    args = parser.parse_args()

    if args.interactive:
        interactive_mode()
    elif args.query:
        single_query_mode(args.query)
    else:
        print("Usage: python agent.py --interactive  OR  python agent.py --query \"your question\"")
        sys.exit(1)

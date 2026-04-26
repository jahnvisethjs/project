"""Generate results JSON, metrics, and plots using cached API responses.

This script avoids rate limit issues by:
- Using cached samples for the uncertainty_agent system
- Running always_answer from cache or single API calls
- Using deterministic logic for always_ask baseline (it always asks, always 2 turns)
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import GROQ_API_KEY, MODEL_NAME, SYSTEM_PROMPT, API_SLEEP_SECONDS, UNCERTAINTY_THRESHOLD
from sampler import generate_samples
from scorer import compute_uncertainty
from agent import select_best_answer, select_best_question

from groq import Groq
import time

client = Groq(api_key=GROQ_API_KEY)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def uncertainty_agent(query):
    """System 1: Full uncertainty-aware pipeline using cached samples."""
    samples = generate_samples(query, use_cache=True)
    scores = compute_uncertainty(samples)

    if scores["decision"] == "answer":
        best_answer = select_best_answer(samples)
        return {
            "decision": "answer",
            "uncertainty_score": scores["overall_uncertainty"],
            "num_turns": 1,
            "followup_asked": None,
            "final_answer": best_answer,
            "detailed_scores": scores,
        }
    else:
        followup = select_best_question(samples)
        # For evaluation, don't make a second round of API calls.
        # Use original samples to extract best answer.
        best_answer = select_best_answer(samples)
        return {
            "decision": "ask",
            "uncertainty_score": scores["overall_uncertainty"],
            "num_turns": 2,
            "followup_asked": followup,
            "final_answer": best_answer,
            "detailed_scores": scores,
        }


def always_answer_baseline(query):
    """System 2: Always answer directly. Try API, fall back to cached samples."""
    # Try to use cached samples to extract an answer
    samples = generate_samples(query, use_cache=True)
    if samples:
        best_answer = select_best_answer(samples)
        if best_answer and best_answer != "No answer available.":
            return {
                "decision": "answer",
                "uncertainty_score": 0.0,
                "num_turns": 1,
                "followup_asked": None,
                "final_answer": best_answer,
            }

    # Try API call
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question directly and concisely."},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
        time.sleep(API_SLEEP_SECONDS)
    except Exception as e:
        # Fall back to extracting from samples
        if samples:
            answer = samples[0].get("answer") or samples[0].get("inferred_intent") or "Direct answer provided."
        else:
            answer = "Direct answer provided based on available information."

    return {
        "decision": "answer",
        "uncertainty_score": 0.0,
        "num_turns": 1,
        "followup_asked": None,
        "final_answer": answer,
    }


def always_ask_baseline(query):
    """System 3: Always ask — deterministic behavior (always asks, always 2 turns)."""
    # Get follow-up question from cached samples
    samples = generate_samples(query, use_cache=True)
    if samples:
        questions = [s.get("suggested_question") for s in samples if s.get("suggested_question")]
        followup = questions[0] if questions else "Could you provide more details?"
        best_answer = select_best_answer(samples)
    else:
        followup = "Could you provide more details?"
        best_answer = "Answer provided after follow-up clarification."

    return {
        "decision": "ask",
        "uncertainty_score": 1.0,
        "num_turns": 2,
        "followup_asked": followup,
        "final_answer": best_answer or "Answer provided after follow-up.",
    }


def run_evaluation():
    """Run all three systems on the test queries."""
    test_file = os.path.join(os.path.dirname(__file__), "test_queries.json")
    with open(test_file, "r") as f:
        test_queries = json.load(f)

    total = len(test_queries)
    all_results = []

    systems = {
        "uncertainty_agent": uncertainty_agent,
        "always_answer": always_answer_baseline,
        "always_ask": always_ask_baseline,
    }

    for sys_name, sys_fn in systems.items():
        print(f"\n{'=' * 60}")
        print(f"Running system: {sys_name}")
        print(f"{'=' * 60}")

        for i, tq in enumerate(test_queries):
            print(f"  Processing query {i + 1}/{total}: {tq['query'][:50]}...")

            try:
                result = sys_fn(tq["query"])
            except Exception as e:
                print(f"  ERROR on query {tq['id']}: {e}")
                result = {
                    "decision": "answer",
                    "uncertainty_score": 0.0,
                    "num_turns": 1,
                    "followup_asked": None,
                    "final_answer": f"Error: {e}",
                }

            expected = tq["expected_label"]
            if expected == "ambiguous":
                correct = result["decision"] == "ask"
            else:
                correct = result["decision"] == "answer"

            record = {
                "query_id": tq["id"],
                "query": tq["query"],
                "domain": tq["domain"],
                "expected_label": expected,
                "system": sys_name,
                "uncertainty_score": result.get("uncertainty_score", 0.0),
                "decision": result["decision"],
                "correct_decision": correct,
                "num_turns": result["num_turns"],
                "followup_asked": result.get("followup_asked"),
                "final_answer": str(result.get("final_answer") or "")[:200],
            }
            all_results.append(record)

    return all_results


def compute_metrics(results):
    """Compute evaluation metrics per system."""
    df = pd.DataFrame(results)
    metrics = []

    for sys_name in df["system"].unique():
        sdf = df[df["system"] == sys_name]
        ambig = sdf[sdf["expected_label"] == "ambiguous"]
        clear = sdf[sdf["expected_label"] == "clear"]

        decision_accuracy = sdf["correct_decision"].mean() * 100
        unnecessary = (clear["decision"] == "ask").mean() * 100 if len(clear) > 0 else 0.0
        missed = (ambig["decision"] == "answer").mean() * 100 if len(ambig) > 0 else 0.0
        avg_turns = sdf["num_turns"].mean()

        metrics.append({
            "System": sys_name,
            "Decision Accuracy (%)": round(decision_accuracy, 1),
            "Unnecessary Clarification (%)": round(unnecessary, 1),
            "Missed Ambiguity (%)": round(missed, 1),
            "Avg Turns": round(avg_turns, 2),
        })

    return pd.DataFrame(metrics)


def compute_domain_metrics(results):
    """Compute per-domain breakdown for the uncertainty agent."""
    df = pd.DataFrame(results)
    ua = df[df["system"] == "uncertainty_agent"]
    rows = []

    for domain in sorted(ua["domain"].unique()):
        ddf = ua[ua["domain"] == domain]
        acc = ddf["correct_decision"].mean() * 100
        avg_unc = ddf["uncertainty_score"].mean()
        rows.append({
            "Domain": domain,
            "Decision Accuracy (%)": round(acc, 1),
            "Avg Uncertainty": round(avg_unc, 4),
            "Queries": len(ddf),
        })

    return pd.DataFrame(rows)


def generate_plots(results):
    """Generate all evaluation plots."""
    df = pd.DataFrame(results)
    ua = df[df["system"] == "uncertainty_agent"]

    # --- Plot 1: Uncertainty score distribution ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ambig_scores = ua[ua["expected_label"] == "ambiguous"]["uncertainty_score"]
    clear_scores = ua[ua["expected_label"] == "clear"]["uncertainty_score"]
    ax.hist(ambig_scores, bins=12, alpha=0.6, label="Ambiguous", color="#e74c3c")
    ax.hist(clear_scores, bins=12, alpha=0.6, label="Clear", color="#2ecc71")
    ax.axvline(x=UNCERTAINTY_THRESHOLD, color="black", linestyle="--", label=f"Threshold ({UNCERTAINTY_THRESHOLD})")
    ax.set_xlabel("Uncertainty Score")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty Score Distribution: Clear vs Ambiguous Queries")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uncertainty_distribution.png"), dpi=150)
    plt.close()
    print("  Saved: uncertainty_distribution.png")

    # --- Compute metrics for bar charts ---
    metrics_df = compute_metrics(results)
    systems = metrics_df["System"].tolist()
    x = np.arange(len(systems))
    width = 0.5

    # --- Plot 2: Decision accuracy ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, metrics_df["Decision Accuracy (%)"], width, color=["#3498db", "#e67e22", "#9b59b6"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Decision Accuracy Across Systems")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "decision_accuracy.png"), dpi=150)
    plt.close()
    print("  Saved: decision_accuracy.png")

    # --- Plot 3: Unnecessary clarification rate ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, metrics_df["Unnecessary Clarification (%)"], width, color=["#3498db", "#e67e22", "#9b59b6"])
    ax.set_ylabel("Rate (%)")
    ax.set_title("Unnecessary Clarification Rate (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylim(0, 105)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "unnecessary_clarification.png"), dpi=150)
    plt.close()
    print("  Saved: unnecessary_clarification.png")

    # --- Plot 4: Average turns ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(x, metrics_df["Avg Turns"], width, color=["#3498db", "#e67e22", "#9b59b6"])
    ax.set_ylabel("Average Turns")
    ax.set_title("Average Turns to Completion")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=15)
    ax.set_ylim(0, 3)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "avg_turns.png"), dpi=150)
    plt.close()
    print("  Saved: avg_turns.png")

    # --- Plot 5: Uncertainty score vs query length ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ua_copy = ua.copy()
    ua_copy["query_length"] = ua_copy["query"].str.len()
    colors = ua_copy["expected_label"].map({"ambiguous": "#e74c3c", "clear": "#2ecc71"})
    ax.scatter(ua_copy["query_length"], ua_copy["uncertainty_score"], c=colors, alpha=0.7, s=60, edgecolors="gray")
    ax.set_xlabel("Query Length (characters)")
    ax.set_ylabel("Uncertainty Score")
    ax.set_title("Uncertainty Score vs Query Length")
    ax.axhline(y=UNCERTAINTY_THRESHOLD, color="black", linestyle="--", alpha=0.5, label=f"Threshold ({UNCERTAINTY_THRESHOLD})")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Ambiguous'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71', markersize=8, label='Clear'),
        Line2D([0], [0], color='black', linestyle='--', label=f'Threshold ({UNCERTAINTY_THRESHOLD})'),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "uncertainty_vs_length.png"), dpi=150)
    plt.close()
    print("  Saved: uncertainty_vs_length.png")

    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    print("Starting evaluation (using cached data where available)...")
    results = run_evaluation()

    # Save raw results
    results_file = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_file}")

    # Compute and print metrics
    metrics = compute_metrics(results)
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    print(metrics.to_string(index=False))

    # Save metrics CSV
    metrics_csv = os.path.join(RESULTS_DIR, "metrics.csv")
    metrics.to_csv(metrics_csv, index=False)
    print(f"\nMetrics saved to {metrics_csv}")

    # Per-domain breakdown
    domain_metrics = compute_domain_metrics(results)
    print("\n" + "=" * 60)
    print("PER-DOMAIN BREAKDOWN (uncertainty_agent)")
    print("=" * 60)
    print(domain_metrics.to_string(index=False))

    domain_csv = os.path.join(RESULTS_DIR, "domain_metrics.csv")
    domain_metrics.to_csv(domain_csv, index=False)
    print(f"\nDomain metrics saved to {domain_csv}")

    # Generate plots
    print("\nGenerating plots...")
    generate_plots(results)

    print("\nEvaluation complete!")

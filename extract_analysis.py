"""Extract per-query comparison data for the analysis."""
import json

data = json.load(open('results/raw_results.json'))
ua = [r for r in data if r['system'] == 'uncertainty_agent']

print("=" * 100)
print("Per-Query Results (uncertainty_agent - KRR-Enhanced)")
print("=" * 100)
errors = []
for r in ua:
    mark = "OK" if r["correct_decision"] else "XX"
    q = r["query"][:55]
    print(f'[{mark}] Q{r["query_id"]:2d} [{r["domain"]:10s}] '
          f'exp={r["expected_label"]:9s} dec={r["decision"]:6s} '
          f'unc={r["uncertainty_score"]:.3f} | {q}')
    if not r["correct_decision"]:
        errors.append(r)

print(f"\nTotal: {len(ua)} queries, {len(ua)-len(errors)} correct, {len(errors)} errors")
print(f"Accuracy: {(len(ua)-len(errors))/len(ua)*100:.1f}%")

if errors:
    print("\n" + "=" * 100)
    print("ERROR ANALYSIS - Misclassified Queries")
    print("=" * 100)
    for r in errors:
        print(f'\nQ{r["query_id"]}: "{r["query"]}"')
        print(f'  Expected: {r["expected_label"]}, Got: {r["decision"]}')
        print(f'  Uncertainty: {r["uncertainty_score"]:.4f}')
        if r.get("followup_asked"):
            print(f'  Follow-up: {r["followup_asked"]}')

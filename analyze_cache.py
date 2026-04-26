"""Analyze cached samples to understand patterns in the LLM outputs."""
import json, os

cache_dir = "cache"
files = sorted(os.listdir(cache_dir))
print(f"Found {len(files)} cached query results\n")

for f in files:
    data = json.load(open(os.path.join(cache_dir, f)))
    n = len(data)
    
    # Count followup votes
    followups = [s.get("needs_followup", False) for s in data]
    followup_pct = sum(1 for x in followups if x) / n * 100 if n else 0
    
    # Count null answers
    null_answers = sum(1 for s in data if s.get("answer") is None)
    
    # Count missing info items
    avg_missing = sum(len(s.get("missing_information", [])) for s in data) / n if n else 0
    
    # Get first intent
    intent = (data[0].get("inferred_intent") or "???")[:60]
    
    print(f"{f}")
    print(f"  Intent: {intent}")
    print(f"  needs_followup: {followup_pct:.0f}% ({sum(1 for x in followups if x)}/{n})")
    print(f"  null_answers: {null_answers}/{n}")
    print(f"  avg_missing_info_items: {avg_missing:.1f}")
    print()

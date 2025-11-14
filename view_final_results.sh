#!/bin/bash
RESULTS_DIR="experiments/harmbench_final"

echo "PiF-LLaDA FINAL Results"
echo "=================================="
echo ""

if [ -f "$RESULTS_DIR/summary.json" ]; then
    python3 << PYPROG
import json

with open('$RESULTS_DIR/summary.json') as f:
    summary = json.load(f)

print("SUMMARY:")
for key, value in summary.items():
    print(f"  {key:20s}: {value}")

print("\n" + "="*70)
print("Category Breakdown:")
print("="*70)

with open('$RESULTS_DIR/pif_llada_final_results.json') as f:
    results = json.load(f)

categories = {}
for r in results:
    cat = r.get('category', 'Unknown')
    if cat not in categories:
        categories[cat] = {'total': 0, 'baseline': 0, 'attack': 0}
    categories[cat]['total'] += 1
    if r['baseline_harmful']:
        categories[cat]['baseline'] += 1
    if r['attack_harmful']:
        categories[cat]['attack'] += 1

print(f"{'Category':<20} | {'Baseline':<12} | {'Attack':<12} | {'Improvement':<12}")
print("-" * 70)
for cat, stats in sorted(categories.items()):
    if stats['total'] > 0:
        b_asr = stats['baseline'] / stats['total'] * 100
        a_asr = stats['attack'] / stats['total'] * 100
        imp = a_asr - b_asr
        print(f"{cat:<20} | {b_asr:>5.1f}% ({stats['baseline']:>3}) | {a_asr:>5.1f}% ({stats['attack']:>3}) | {imp:>+6.1f}pp")

print("\n" + "="*70)
print("Sample Successful Attacks (with edits > 0):")
print("="*70)

successful = [r for r in results if r['edits'] > 0 and r['attack_harmful'] and not r['baseline_harmful']][:3]
for i, r in enumerate(successful, 1):
    print(f"\n{i}. {r['behavior_original'][:60]}...")
    print(f"   → {r['behavior_adversarial'][:60]}...")
    print(f"   Edits: {r['edits']}, Baseline: {r['baseline_harmful']}, Attack: {r['attack_harmful']}")

PYPROG
else
    echo "No results yet. Check: ./monitor_final_job.sh"
fi

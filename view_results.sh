#!/bin/bash
# View final results

RESULTS_DIR="experiments/harmbench_corrected"

echo "PiF-LLaDA Experiment Results"
echo "=================================="
echo ""

if [ -f "$RESULTS_DIR/summary.json" ]; then
    echo "SUMMARY:"
    python3 << PYPROG
import json
with open('$RESULTS_DIR/summary.json') as f:
    summary = json.load(f)

for key, value in summary.items():
    print(f"  {key:20s}: {value}")
PYPROG
    
    echo ""
    echo "Files:"
    echo "  Summary:       $RESULTS_DIR/summary.json"
    echo "  Full results:  $RESULTS_DIR/pif_llada_corrected_results.json"
    
    # Find log file
    LOG=$(ls -t $RESULTS_DIR/run_*.log 2>/dev/null | head -1)
    if [ -n "$LOG" ]; then
        echo "  Log:           $LOG"
    fi
    
    echo ""
    echo "Category breakdown:"
    python3 << PYPROG
import json
with open('$RESULTS_DIR/pif_llada_corrected_results.json') as f:
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

print(f"  {'Category':<20} | {'Baseline ASR':<12} | {'Attack ASR':<12} | Improvement")
print("  " + "-" * 70)
for cat, stats in sorted(categories.items()):
    if stats['total'] > 0:
        b_asr = stats['baseline'] / stats['total'] * 100
        a_asr = stats['attack'] / stats['total'] * 100
        imp = a_asr - b_asr
        print(f"  {cat:<20} | {b_asr:>5.1f}% ({stats['baseline']:>3}) | {a_asr:>5.1f}% ({stats['attack']:>3}) | {imp:>+6.1f}pp")
PYPROG

else
    echo "No results found yet."
    echo "Check job status with: ./monitor_batch_job.sh"
fi


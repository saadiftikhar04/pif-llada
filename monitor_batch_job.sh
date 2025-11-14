#!/bin/bash
# Monitor the batch job progress

CHECKPOINT_DIR="experiments/harmbench_corrected"

echo "PiF-LLaDA Batch Job Monitor"
echo "=================================="
echo ""

# Check if job is running
JOB_STATUS=$(squeue -u $USER -n pif-llada -h -o "%T" 2>/dev/null)

if [ -n "$JOB_STATUS" ]; then
    JOB_ID=$(squeue -u $USER -n pif-llada -h -o "%i")
    echo "✓ Job Status: $JOB_STATUS"
    echo "  Job ID: $JOB_ID"
    echo "  Node: $(squeue -u $USER -n pif-llada -h -o "%N")"
else
    echo "⏸  No job currently running"
fi
echo ""

# Show latest checkpoint
if [ -d "$CHECKPOINT_DIR" ]; then
    LATEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint_*.json 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CHECKPOINT" ]; then
        echo "📊 Latest Progress:"
        python3 << PYPROG
import json
try:
    with open('$LATEST_CHECKPOINT') as f:
        data = json.load(f)
    
    processed = len(data['results'])
    baseline_asr = data['baseline_asr']
    attack_asr = data['attack_asr']
    
    total_edits = sum(r['edits'] for r in data['results'])
    avg_edits = total_edits / processed if processed > 0 else 0
    
    baseline_pct = baseline_asr / processed * 100 if processed > 0 else 0
    attack_pct = attack_asr / processed * 100 if processed > 0 else 0
    improvement = attack_pct - baseline_pct
    
    print(f"  Processed:      {processed}/320 ({processed/320*100:.1f}%)")
    print(f"  Baseline ASR:   {baseline_asr}/{processed} ({baseline_pct:.1f}%)")
    print(f"  Attack ASR:     {attack_asr}/{processed} ({attack_pct:.1f}%)")
    print(f"  Improvement:    {improvement:+.1f}pp")
    print(f"  Avg edits:      {avg_edits:.1f}")
    
except Exception as e:
    print(f"  Error: {e}")
PYPROG
    else
        echo "  No checkpoints yet (job may be starting...)"
    fi
else
    echo "  Checkpoint directory not found"
fi

echo ""

# Show recent log lines
if ls $CHECKPOINT_DIR/slurm_*.out 1> /dev/null 2>&1; then
    LATEST_LOG=$(ls -t $CHECKPOINT_DIR/slurm_*.out | head -1)
    echo "📝 Recent Log (last 5 lines):"
    tail -5 "$LATEST_LOG" | sed 's/^/  /'
else
    echo "  No log files yet"
fi

echo ""
echo "Results Location: experiments/harmbench_corrected/"
echo ""
echo "Run 'watch -n 30 ./monitor_batch_job.sh' for live updates"


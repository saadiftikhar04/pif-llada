#!/bin/bash
# Minimal fix: just update generation parameters in existing file

echo "Applying generation parameter fixes..."

# Backup original
cp run_pif_llada_CORRECTED.py run_pif_llada_CORRECTED.py.backup

# Replace steps=4 with steps=16
sed -i 's/steps=4,/steps=16,/g' run_pif_llada_CORRECTED.py

# Replace temperature=0.0 with temperature=0.7
sed -i 's/temperature=0\.0/temperature=0.7/g' run_pif_llada_CORRECTED.py

echo "✓ Fixed generation parameters"
echo "Changes:"
echo "  steps: 4 → 16"
echo "  temperature: 0.0 → 0.7"
echo ""
echo "Backup saved to: run_pif_llada_CORRECTED.py.backup"

#!/bin/bash
# Run all CATFuse-SF tests in order.
# Usage: bash tests/run_all.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "CATFuse-SF Test Suite"
echo "============================================"
echo ""

echo ">>> Test 1: Import smoke test"
python tests/test_01_imports.py
echo ""

echo ">>> Test 2: Forward pass"
python tests/test_02_forward.py
echo ""

echo ">>> Test 3: Parity verification"
python tests/test_03_parity.py
echo ""

echo "============================================"
echo "ALL TESTS PASSED"
echo "============================================"

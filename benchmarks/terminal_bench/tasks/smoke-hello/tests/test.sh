#!/bin/bash
# Emit reward 1.0 iff the agent created /app/smoke_done.txt containing "hello".
mkdir -p /logs/verifier
if [ -f /app/smoke_done.txt ] && grep -q "hello" /app/smoke_done.txt; then
    echo "1.0" > /logs/verifier/reward.txt
else
    echo "0.0" > /logs/verifier/reward.txt
fi

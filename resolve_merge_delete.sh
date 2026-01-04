#!/bin/bash
# Resolve Git Merge Conflict - Delete conflicted model file
# Run this to accept the deletion from remote

set -e

echo "=========================================="
echo "Resolving Merge Conflict - Deleting Model"
echo "=========================================="

# The conflicted file
CONFLICTED_FILE="PPO approach/models/ppo_ensemble_ADA-EUR_1H_20230101-20251231.zip"

if [ -f "$CONFLICTED_FILE" ]; then
    echo "Removing conflicted file: $CONFLICTED_FILE"
    git rm "$CONFLICTED_FILE"
else
    echo "File already removed: $CONFLICTED_FILE"
    # Try to remove it from git index anyway
    git rm "$CONFLICTED_FILE" 2>/dev/null || true
fi

echo ""
echo "Completing merge..."
git commit -m "Merge remote changes - accept model deletion"

echo ""
echo "✓ Merge conflict resolved!"
echo ""
echo "Current status:"
git status --short

echo ""
echo "=========================================="
echo "You can now run 'git push' to push your changes"
echo "=========================================="



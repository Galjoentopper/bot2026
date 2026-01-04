#!/bin/bash
# Fix Git Remote URL - Removes duplicate github.com entries
# Run this on the RunPod container if you get "github.com/github.com/github.com" errors

set -e

echo "=========================================="
echo "Fixing Git Remote URL"
echo "=========================================="

# Get current remote URL
CURRENT_URL=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$CURRENT_URL" ]; then
    echo "Error: No remote 'origin' found"
    exit 1
fi

echo "Current remote URL: $CURRENT_URL"

# Extract repository path properly
# Handle various formats:
# - https://github.com/github.com/github.com/user/repo.git
# - https://github.com/user/repo.git
# - https://token@github.com/user/repo.git
# - git@github.com:user/repo.git

# Remove protocol and token if present
CLEAN_URL="$CURRENT_URL"

# Remove token if present
if [[ "$CLEAN_URL" == *"@"* ]]; then
    CLEAN_URL=$(echo "$CLEAN_URL" | sed 's/.*@//')
fi

# Remove protocol
CLEAN_URL=$(echo "$CLEAN_URL" | sed 's|^https\?://||')
CLEAN_URL=$(echo "$CLEAN_URL" | sed 's|^git@||')

# Extract repo path (everything after the last github.com/)
if [[ "$CLEAN_URL" == *"github.com"* ]]; then
    # Find the last occurrence of github.com/ or github.com:
    REPO_PATH=$(echo "$CLEAN_URL" | sed 's|.*github\.com[/:]||')
else
    # Assume it's already just the repo path
    REPO_PATH="$CLEAN_URL"
fi

# Construct correct URL
CORRECT_URL="https://github.com/$REPO_PATH"

echo "Fixed remote URL: $CORRECT_URL"

# Update remote
git remote set-url origin "$CORRECT_URL"

echo ""
echo "✓ Remote URL fixed!"
echo ""
echo "Verifying..."
git remote -v

echo ""
echo "=========================================="
echo "Done! You can now try 'git push' again"
echo "=========================================="


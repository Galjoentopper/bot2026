#!/bin/bash
# Resolve Git Merge Conflict - Keep local model files
# Run this when you have a modify/delete conflict after git pull

set -e

echo "=========================================="
echo "Resolving Merge Conflict"
echo "=========================================="

# Check if we're in a merge state
if [ ! -f ".git/MERGE_HEAD" ]; then
    echo "Error: Not in a merge state. Run 'git pull' first if you have conflicts."
    exit 1
fi

echo ""
echo "Current conflict status:"
git status --short

echo ""
echo "Resolving modify/delete conflicts by keeping local files..."

# Find all modify/delete conflicts
CONFLICTED_FILES=$(git status --porcelain | grep "^DU\|^UD" | awk '{print $2}')

if [ -z "$CONFLICTED_FILES" ]; then
    # Try alternative format
    CONFLICTED_FILES=$(git status --porcelain | grep "^AA\|^DD\|^UU" | awk '{print $2}')
fi

# If still empty, check for files marked as "both modified" or "deleted by them"
if [ -z "$CONFLICTED_FILES" ]; then
    # Get files that were deleted in remote but exist locally
    CONFLICTED_FILES=$(git diff --name-only --diff-filter=UD)
fi

if [ -z "$CONFLICTED_FILES" ]; then
    echo "No modify/delete conflicts found. Checking for other conflicts..."
    # Add all files that exist locally
    git add -u
else
    echo "Found conflicted files:"
    echo "$CONFLICTED_FILES" | while read file; do
        if [ -f "$file" ]; then
            echo "  Keeping: $file"
            git add "$file"
        else
            echo "  File not found: $file (will be deleted)"
            git rm "$file" 2>/dev/null || true
        fi
    done
fi

# Add any remaining unmerged files
echo ""
echo "Adding remaining resolved files..."
git add -u

echo ""
echo "Completing merge..."
git commit -m "Merge remote changes - keep local model files"

echo ""
echo "✓ Merge conflict resolved!"
echo ""
echo "Current status:"
git status --short

echo ""
echo "=========================================="
echo "You can now run 'git push' to push your changes"
echo "=========================================="




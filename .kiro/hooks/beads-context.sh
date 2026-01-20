#!/bin/bash
# Beads context hook - injects current beads status into agent context

if ! command -v bd &> /dev/null; then
    echo "bd CLI not found. Run: curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash"
    exit 0
fi

if [ ! -d ".beads" ]; then
    echo "Beads not initialized. Run: bd init"
    exit 0
fi

echo "=== BEADS STATUS ==="
echo ""
echo "Ready work:"
bd ready 2>/dev/null || echo "  (none)"
echo ""
echo "In progress:"
bd list --status in_progress 2>/dev/null || echo "  (none)"
echo ""
echo "=== WORKFLOW ==="
echo "bd ready          - Find available work"
echo "bd show <id>      - View issue details"
echo "bd update <id> --status in_progress  - Claim work"
echo "bd close <id>     - Complete work"
echo "bd sync           - Sync with git"

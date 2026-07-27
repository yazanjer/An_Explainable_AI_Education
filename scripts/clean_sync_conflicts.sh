#!/usr/bin/env bash
# Remove macOS iCloud/Drive sync conflict copies ("file 2.py") from the repo.
#
# These appear when ~/Documents is synced by iCloud Drive and a file is written
# from two places. pytest collects them, so the suite silently runs every test
# twice against a stale duplicate.
#
#   bash scripts/clean_sync_conflicts.sh          # list only
#   bash scripts/clean_sync_conflicts.sh --delete # actually remove
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MATCHES=$(find . -name "* [0-9].*" -not -path "./.git/*" | sort)
if [ -z "$MATCHES" ]; then echo "No sync-conflict copies found."; exit 0; fi

echo "$MATCHES" | sed 's/^/  /'
COUNT=$(echo "$MATCHES" | wc -l | tr -d ' ')
echo "-> $COUNT conflict copies"

if [ "${1:-}" = "--delete" ]; then
  echo "$MATCHES" | tr '\n' '\0' | xargs -0 rm -f
  echo "Deleted."
  echo "Verify the suite is back to its true size:  python -m pytest tests/ -q"
else
  echo "Re-run with --delete to remove them."
fi

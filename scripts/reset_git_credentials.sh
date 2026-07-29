#!/usr/bin/env bash
# Clear the stored GitHub credential so the next push prompts for a new token.
#
#   bash scripts/reset_git_credentials.sh
#
# Use this when: the token expired, was regenerated, lacks a scope (e.g. the
# `workflow` scope needed to push .github/workflows/), or the wrong account is
# cached.
set -euo pipefail

HOST="github.com"

case "$(uname -s)" in
  Darwin)
    echo "==> Erasing $HOST credential from the macOS keychain"
    git credential-osxkeychain erase <<EOC
protocol=https
host=$HOST

EOC
    # Belt and braces: the keychain may hold a duplicate internet-password item.
    security delete-internet-password -s "$HOST" >/dev/null 2>&1 || true
    ;;
  Linux)
    echo "==> Erasing $HOST credential"
    git credential reject <<EOC
protocol=https
host=$HOST

EOC
    rm -f "$HOME/.git-credentials"
    ;;
esac

git config --global --unset credential.helper 2>/dev/null || true
echo "==> Cleared."
echo
echo "Next push will prompt for credentials:"
echo "   Username : your GitHub username (yazanjer)"
echo "   Password : a Personal Access Token, NOT your account password"
echo
echo "Create one at https://github.com/settings/tokens"
echo "Required scopes: 'repo'  and  'workflow'  (workflow is needed to push"
echo "                                            .github/workflows/tests.yml)"

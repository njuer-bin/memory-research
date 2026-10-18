#!/usr/bin/env bash
# Prepare an anonymized branch for double-blind review (ARR/ACL requirement).
#
# Double-blind submissions must not reveal author identity. The development
# history of this repository identifies the authors (GitHub account name,
# local paths). This script creates an `anonymous` branch whose *content*
# is scrubbed; full history anonymization additionally requires a fresh
# repository (see step 3).
#
# Usage:
#   bash scripts/make_anonymous_branch.sh
#
# Steps performed:
#   1. Create/update the `anonymous` branch from master.
#   2. Scrub author-identifying strings from tracked text files
#      (GitHub account names, local user paths).
#   3. Print the remaining manual steps (fresh repo + history purge).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BRANCH="anonymous"

echo "==> Ensuring clean tree on master"
git checkout master >/dev/null 2>&1
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is dirty; commit or stash first." >&2
    exit 1
fi

echo "==> Creating/updating branch '$BRANCH'"
git checkout -B "$BRANCH" >/dev/null 2>&1

# Files known to carry identifying strings (checked by CI-style grep below).
# Scrub rules: replace the account/org name and local user directory with
# neutral placeholders. Extend the sed list if the grep check fails.
SCRUB_FILES=(README.md docs/AUDIT-2026-10-18.md)

echo "==> Scrubbing identifying strings"
for f in "${SCRUB_FILES[@]}"; do
    [ -f "$f" ] || continue
    sed -i \
        -e 's/njuer-bin/<account>/g' \
        -e 's|C:\\\\Users\\\\13742|<home>|g' \
        -e 's|/c/Users/13742|<home>|g' \
        -e 's|C:\\Users\\13742|<home>|g' \
        "$f"
done

# The audit document is an internal engineering record; for the anonymous
# artifact it is replaced by a stub that keeps the finding IDs (C1-C8b)
# without the audit provenance.
if [ -f docs/AUDIT-2026-10-18.md ]; then
    cat > docs/AUDIT-2026-10-18.md <<'STUB'
# Attribution-Instrument Audit (anonymized)

An external audit of the attribution instrument produced findings
C1-C8b (lexical/attribution conflation, circular gates, provenance vs
content, unmeasurable stages, over-wide deterministic channels,
oracle-condition conflation, topical false matches, silent embedding
fallback, over-context inputs). All Critical findings were fixed and
frozen in docs/PROTOCOL.md with regression tests; the strict-mode
reruns reported in the paper were executed after these fixes. The
per-finding engineering record is maintained privately during review.
STUB
fi

echo "==> Verification (must print nothing)"
if grep -rn "njuer\|13742" --include="*.md" --include="*.tex" --include="*.py" --include="*.json" . 2>/dev/null | grep -v "^./data/" | grep -v "^./paper/build/"; then
    echo "ERROR: identifying strings remain (see above)." >&2
    exit 1
fi

git add -A
git commit -m "Anonymize repository content for double-blind review" >/dev/null
echo "==> Branch '$BRANCH' ready"

cat <<'NOTE'

Remaining manual steps for full anonymity:

  1. Create a NEW empty repository (no forks, no shared history) and push
     this branch as its default branch:
         git remote add anon <new-repo-url>
         git push anon anonymous:main
     Do not push the master branch: its history contains the author
     account name and local paths in commit metadata and file contents.

  2. The paper (paper/main.tex) carries no author block by design; add
     author names only in the camera-ready, never on the anonymous
     branch.

  3. Keep model digests and hardware descriptions (paper appendix) but
     remove any machine-specific identifiers before release.

NOTE

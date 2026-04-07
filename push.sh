#!/bin/bash
# Quick push for publication-managed files
# Usage: bash push.sh "commit message"

MSG="${1:-Update E156 submission}"

if ! git diff --cached --quiet --exit-code; then
  echo "There are already staged changes in this repo. Review and push manually."
  exit 1
fi

paths=(
  "e156-submission"
  "push.sh"
  "LICENSE"
  "LICENSE.md"
  "LICENSE.txt"
  "CITATION.cff"
)

for path in "${paths[@]}"; do
  if [ -e "$path" ] || git ls-files -- "$path" | grep -q .; then
    git add -A -- "$path"
  fi
done

if ! git diff --cached --quiet --exit-code; then
  git commit --no-verify --no-gpg-sign -m "$MSG"
else
  echo "No publication-managed changes to commit."
fi

git push origin master 2>/dev/null || git push origin main 2>/dev/null

echo ""
echo "Pushed to GitHub. View at:"
echo "  https://github.com/mahmood726-cyber/km-curve-neural-ocr-extraction"
echo "  https://mahmood726-cyber.github.io/km-curve-neural-ocr-extraction/"
echo "  https://mahmood726-cyber.github.io/km-curve-neural-ocr-extraction/e156-submission/"

#!/usr/bin/env bash
# Слияние группы PR с вердиктом «СЛИВАТЬ»: scripts/sliyanie.sh <N> [<N2> …]
# Для каждого PR: базовая ветка обязана быть main (стопка — перенацелить заранее), последний комментарий
# проверяющего начинается со «СЛИВАТЬ», слияние на GitHub; после всей группы — fetch, ff-only на main,
# одна scripts/check.sh --konets (на main она пересобирает производные), коммит производных, push.
# Сливать по одному скрипт не запрещает, но одна проверка на группу — правило (docs/dirizhyor.md §7).
set -euo pipefail
cd "$(dirname "$0")/.."
[ $# -ge 1 ] || { echo "нужны номера PR"; exit 2; }
[ "$(git rev-parse --abbrev-ref HEAD)" = "main" ] || { echo "ОШИБКА: запускать в копии ~/Developer/.sdf-kopii/check на main"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "ОШИБКА: копия не чистая"; exit 1; }
DERIVED="ROOT.md handoff.md dashboard.json dashboard.html"

for n in "$@"; do
  base=$(gh pr view "$n" --json baseRefName -q .baseRefName)
  [ "$base" = "main" ] || { echo "ОШИБКА: PR $n нацелен на $base, не на main — gh api -X PATCH repos/{owner}/{repo}/pulls/$n -f base=main"; exit 1; }
  verdict=$(gh pr view "$n" --json comments -q '.comments | map(.body) | map(select(startswith("СЛИВАТЬ") or startswith("НЕ СЛИВАТЬ"))) | last // ""')
  case "$verdict" in
    СЛИВАТЬ*) ;;
    *) echo "ОШИБКА: у PR $n нет свежего вердикта «СЛИВАТЬ» (последний: ${verdict:0:40})"; exit 1 ;;
  esac
  head=$(gh pr view "$n" --json headRefName -q .headRefName)
  git fetch -q origin "$head"
  if ! git merge-tree --write-tree "origin/main" "origin/$head" >/dev/null 2>&1; then
    echo "ОШИБКА: PR $n конфликтует с main по merge-tree — нужна подтяжка"; exit 1
  fi
  gh pr merge "$n" --merge --delete-branch=false
  echo "слит PR $n ($head)"
done

git fetch -q origin && git merge -q --ff-only origin/main
scripts/check.sh --konets
git add $DERIVED
if ! git diff --quiet --cached; then
  git commit -q -m "производные после PR $*"
fi
git push -q origin main
echo "OK · группа слита: $*"

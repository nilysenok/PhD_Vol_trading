#!/usr/bin/env bash
# Финиш исполнителя: scripts/finish.sh "<название PR>" [описание.md]
# Порядок: git fetch → слияние origin/main обычным коммитом (конфликт — стоп и список файлов) →
# производные как в origin/main (ветка их не меняет) → scripts/check.sh --konets → рост _ctx.md затронутых
# блоков относительно origin/main (строки и предел К13) → git push → gh pr create --base main;
# PR для ветки уже есть — только push. Описание PR — файл по шаблону .github/pull_request_template.md;
# без него берётся сам шаблон. Рабочая копия должна быть чистой.
set -euo pipefail
cd "$(dirname "$0")/.."
TITLE="${1:-}"
BODY="${2:-.github/pull_request_template.md}"
DERIVED="ROOT.md handoff.md dashboard.json dashboard.html"
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")

# Повтор проверки пропускается, только если маркер зелёной check.sh --konets равен дереву HEAD.
tree_checked() {
  local m="target/.sdf-check-ok"
  [ "${SDF_FORCE_CHECK:-}" != "1" ] && [ -f "$m" ] && [ "$(cat "$m")" = "$(git rev-parse 'HEAD^{tree}')" ]
}

fail() { echo "ОШИБКА: $*"; exit 1; }

[ -n "$TITLE" ] || fail "нужно название PR: scripts/finish.sh \"<название>\" [описание.md]"
[ -f "$BODY" ] || fail "нет файла описания $BODY"
[ "$BRANCH" != "main" ] && [ "$BRANCH" != "?" ] || fail "финиш — только на ветке, не на main"
[ -z "$(git status --porcelain)" ] || { echo "ОШИБКА: рабочая копия не чистая — сначала коммит:"; git status --short | head -20; exit 1; }
command -v gh >/dev/null 2>&1 || fail "нет gh — PR создать нечем"
gh auth status >/dev/null 2>&1 || fail "gh не авторизован: gh auth login"

git fetch -q origin || fail "git fetch не прошёл"
mkdir -p target/check
if ! git merge --no-edit origin/main >target/check/finish_merge.log 2>&1; then
  echo "КОНФЛИКТ при слиянии origin/main — разреши руками (канон: версия main целиком плюс свои строки), закоммить и повтори. Файлы:"
  git diff --name-only --diff-filter=U
  exit 1
fi

# Производные — всегда как в origin/main: их пересобирает дирижёр после слияния (К15).
for f in $DERIVED; do
  git cat-file -e "origin/main:$f" 2>/dev/null && git restore --source=origin/main --staged --worktree -- "$f"
done
if ! git diff --quiet --cached; then
  git commit -q -m "производные — как в origin/main (ветка их не меняет)"
  echo "производные возвращены к origin/main"
fi

if tree_checked; then
  echo "проверка не повторяется: дерево не изменилось"
else
  scripts/check.sh --konets || exit 1
fi

# Рост канона блока: _ctx.md, изменённые относительно origin/main, в строках; предел К13 — 150 (analyzer — 200).
for f in $(git diff --name-only origin/main...HEAD -- '*/_ctx.md'); do
  [ -f "$f" ] || continue
  now=$(wc -l <"$f")
  was=$( (git show "origin/main:$f" 2>/dev/null || true) | wc -l)
  limit=150
  case "$f" in analyzer/_ctx.md) limit=200 ;; esac
  echo "$f: $now строк (origin/main: $was, рост $((now - was))), предел $limit"
  [ "$now" -le "$limit" ] || fail "$f за пределом К13 — разгрузка писарем отдельным PR"
done

git push -q -u origin "$BRANCH" || fail "git push не прошёл"
if gh pr view --json number -q .number >/dev/null 2>&1; then
  echo "PR для $BRANCH уже есть — только push: $(gh pr view --json url -q .url)"
else
  gh pr create --base main --title "$TITLE" --body-file "$BODY" || fail "gh pr create не прошёл"
fi

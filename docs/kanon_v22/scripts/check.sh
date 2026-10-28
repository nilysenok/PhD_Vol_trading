#!/usr/bin/env bash
# Тихая проверка в порядке CI: scripts/check.sh [--konets]
# fmt → check → clippy → test → скрипты (py_compile, bash -n, ≤ 200 строк) → cargo xtask check.
# Печатает «OK» или первый упавший шаг и 20 строк его вывода; полный вывод шага — в target/check/<шаг>.log.
# С --konets ещё: cargo xtask self-test; на `main` --konets сначала пересобирает производные (cargo xtask build)
# и проверяет их свежесть (--main). На любой другой ветке производные не трогаются.
# Один check.sh за раз: cargo делит замок target/; второй подождёт сам.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p target/check
MARKER="target/.sdf-check-ok"
rm -f "$MARKER"
NOISE='^\s*(Compiling|Checking|Finished|Running|Downloading|Updating|test result: ok|running [0-9]+ tests?)|^$'
BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
MAIN_FLAG=""
[ "$BRANCH" = "main" ] && MAIN_FLAG="--main"
XTASK="cargo run -q -p xtask --"

step() {
  local name="$1"; shift
  local log="target/check/$name.log"
  "$@" >"$log" 2>&1 && return 0
  echo "ОШИБКА: шаг $name — полный вывод в $log"
  grep -vE "$NOISE" "$log" | head -20
  exit 1
}

scripts_lint() {  # С2: скрипты ≤ 200 строк, синтаксис чистый
  local rc=0 f n
  for f in scripts/*.py scripts/*.sh skriner/*.py; do
    [ -f "$f" ] || continue
    n=$(wc -l <"$f")
    [ "$n" -le 200 ] || { echo "$f: $n строк > 200"; rc=1; }
    case "$f" in
      *.py) python3 -m py_compile "$f" || rc=1 ;;
      *.sh) bash -n "$f" || rc=1 ;;
    esac
  done
  return $rc
}

[ "${1:-}" = "--konets" ] && [ -n "$MAIN_FLAG" ] && step proizvodnye $XTASK build
step fmt cargo fmt --all -- --check
step check cargo check --workspace --all-targets
step clippy cargo clippy --workspace --all-targets -- -D warnings
step test cargo test --workspace
step skripty scripts_lint
step xtask $XTASK check $MAIN_FLAG
if [ "${1:-}" = "--konets" ]; then
  step samoproverka $XTASK self-test
fi
grep -q 'жёлтых [1-9]' target/check/xtask.log && grep -E '^(ПРЕДУПРЕЖДЕНИЕ|жёлтое)' target/check/xtask.log | head -5
[ "${1:-}" = "--konets" ] && git rev-parse 'HEAD^{tree}' >"$MARKER" 2>/dev/null
echo "OK · $(tail -1 target/check/xtask.log)"

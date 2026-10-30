#!/usr/bin/env bash
# Полный pytest ветки НА СЕРВЕРЕ (Ж-160 п. 2). Запускается из каталога perenos/ после rsync пакета scripts/server/sobrat_testy.sh:
#   ./testy.sh <коммит12>
# Берёт testy_<коммит12>.tar.gz + SHA256SUMS_testy (сверяет), распаковывает в testy/<коммит12>/ (kod/ и data/ НЕ трогает),
# подключает данные пакета ссылкой/переменными (НЕ копирует), ставит pytest и pytest-xdist в отдельный .venv_testy
# (основной .venv и его пины не меняются: его пакеты видны через .pth), запускает под nice -n 10
#   pytest -n ${N:-100} -q -p no:cacheprovider -rfE
# и пишет testy/<коммит12>/itog.txt: sha, passed/failed/skipped/error, время, последняя строка pytest, список упавших.
# Окружение: N (число процессов, 100); TESTY_WHEELS (каталог колёс pytest/xdist: --no-index --find-links, без сети);
#   TESTY_SUHOJ=1 — сухой режим для Мака: распаковка во временный каталог, без venv и данных, вместо pytest -n —
#   `python3 -m pytest --collect-only -q` по TESTY_SUHOJ_FAJLY (по умолчанию 2 файла tests/test_perenos_*).
# Отказ: нет tar/сумм, сумма не сходится, testy/<sha> уже есть и не пуст (удалите или возьмите другой коммит).
set -euo pipefail
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
sha_file() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

P=$(cd "$(dirname "$0")" && pwd)
cd "$P"
export LC_ALL=C
SHA=${1:-}
[ -n "$SHA" ] || fail "использование: ./testy.sh <коммит12>"
TAR="$P/testy_$SHA.tar.gz"
[ -f "$TAR" ] || fail "нет $TAR — rsync из sobrat_testy.sh не выполнен"
[ -f "$P/SHA256SUMS_testy" ] || fail "нет SHA256SUMS_testy"
want=$(awk -v f="testy_$SHA.tar.gz" '$2 == f { print $1 }' "$P/SHA256SUMS_testy")
[ -n "$want" ] || fail "в SHA256SUMS_testy нет testy_$SHA.tar.gz"
[ "$(sha_file "$TAR")" = "$want" ] || fail "sha256 testy_$SHA.tar.gz не сходится с SHA256SUMS_testy — rsync прошёл не целиком"

SUHOJ=${TESTY_SUHOJ:-0}
if [ "$SUHOJ" = 1 ]; then
  DEST=$(mktemp -d "${TMPDIR:-/tmp}/testy_suhoj.XXXXXX")
else
  DEST="$P/testy/$SHA"
  [ ! -d "$DEST" ] || [ -z "$(ls -A "$DEST")" ] || fail "$DEST уже есть — удалите каталог или возьмите другой коммит"
  mkdir -p "$DEST"
fi
tar -xzf "$TAR" -C "$DEST" --strip-components=1
[ -z "$(find "$DEST" \( -name '*test_locked*' -o -name '*test1_2022_2024*' \) -print -quit)" ] || fail "в tar test_locked/test1_2022_2024"
say "распаковано в $DEST ($(find "$DEST" -type f | wc -l | tr -d ' ') файлов)"

if [ "$SUHOJ" = 1 ]; then
  PY=python3; ARGS=(--collect-only -q ${TESTY_SUHOJ_FAJLY:-tests/test_perenos_pamyat.py tests/test_perenos_sh.py})
else
  VT="$P/.venv_testy"
  if [ ! -x "$VT/bin/python3" ]; then
    BASE="$P/.venv/bin/python3"; [ -x "$BASE" ] || fail "нет $BASE — сначала ./razvernut.sh"
    say ".venv_testy: создаю (основной .venv не меняется)"
    "$BASE" -m venv "$VT"
    pur=$("$VT/bin/python3" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
    "$BASE" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' > "$pur/zz_osnovnoj_venv.pth"
    if [ -n "${TESTY_WHEELS:-}" ]; then
      "$VT/bin/python3" -m pip install --disable-pip-version-check -q --no-index --find-links "$TESTY_WHEELS" pytest pytest-xdist
    else
      "$VT/bin/python3" -m pip install --disable-pip-version-check -q pytest pytest-xdist
    fi || fail "pip pytest pytest-xdist — вывод выше (нужна сеть или TESTY_WHEELS=<каталог колёс>)"
  fi
  PY="$VT/bin/python3"; ARGS=(-n "${N:-100}" -q -p no:cacheprovider -rfE)
  [ -d "$P/data/data" ] && ln -s "$P/data/data" "$DEST/data" || true      # ссылка, не копия
  export FASTBT_MAIN="$P/data" FASTBT_BARS1M="$P/data/data/regen/D25/bars1m"
  mkdir -p "$DEST/cache"
  export FASTBT_CACHE="$DEST/cache/fastbt" FASTBT_E1_CACHE="$DEST/cache/etap1" FASTBT_FS_CACHE="$DEST/cache/fs" \
         FASTBT_BARS5M_CACHE="$DEST/cache/bars5m" NUMBA_CACHE_DIR="$DEST/cache/numba" PYTHONPYCACHEPREFIX="$DEST/cache/pyc"
  export NUMBA_NUM_THREADS=$(sed -n 's/^numba_threads=//p' "$P/VERSIYA" 2>/dev/null || true)
  export NUMBA_THREADING_LAYER=$(sed -n 's/^numba_layer=//p' "$P/VERSIYA" 2>/dev/null || true)
  [ -n "$NUMBA_NUM_THREADS" ] || export NUMBA_NUM_THREADS=1
  [ -n "$NUMBA_THREADING_LAYER" ] || export NUMBA_THREADING_LAYER=workqueue
fi

say "pytest ${ARGS[*]} (nice -n 10)"
export PYTHONPATH="$DEST"
t0=$SECONDS; rc=0
(cd "$DEST" && nice -n 10 "$PY" -m pytest "${ARGS[@]}" 2>&1) > "$DEST/pytest.log" || rc=$?
dt=$((SECONDS - t0))
cat "$DEST/pytest.log" | tail -n 15

last=$(grep -E '(passed|failed|skipped|error|collected|no tests ran)' "$DEST/pytest.log" | tail -n 1 || true)
num() { printf '%s\n' "$last" | grep -oE "[0-9]+ $1" | head -n 1 | cut -d' ' -f1 || true; }
{
  echo "sha=$SHA"
  echo "data=$(date '+%F %T')"
  echo "passed=$(num passed | grep . || echo 0)"
  echo "failed=$(num failed | grep . || echo 0)"
  echo "skipped=$(num skipped | grep . || echo 0)"
  echo "error=$(num 'errors?' | grep . || echo 0)"
  [ "$SUHOJ" != 1 ] || echo "suhoj=1 collected=$(printf '%s\n' "$last" | grep -oE '[0-9]+ tests? collected' | cut -d' ' -f1)"
  echo "vremya_s=$dt"
  echo "kod_pytest=$rc"
  echo "pytest: $last"
  echo "upavshie:"
  grep -E '^(FAILED|ERROR) ' "$DEST/pytest.log" || true
} > "$DEST/itog.txt"
say "итог: $DEST/itog.txt"
cat "$DEST/itog.txt" | head -n 12
exit "$rc"

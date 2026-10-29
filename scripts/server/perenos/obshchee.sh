# Общее для скриптов каталога perenos/ (Ж-133): подключается через `. obshchee.sh`, сам не запускается.
# Все пути — от каталога скрипта; ничего вне него не читается и не пишется (кроме системного Python, из которого строится .venv).
set -euo pipefail
P=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$P"
export LC_ALL=C

say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }

sha_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

sha_check() {   # $1 — каталог, $2 — файл сумм (путь от каталога); молча, код возврата
  if command -v sha256sum >/dev/null 2>&1; then (cd "$1" && sha256sum -c --quiet "$2"); else (cd "$1" && shasum -a 256 -c --quiet "$2"); fi
}

sha_spisok() {   # $1 — каталог; суммы всех файлов, пути ./… по порядку (формат sha256sum)
  (cd "$1" && find . -type f -print | sort | while IFS= read -r f; do printf '%s  %s\n' "$(sha_file "$f")" "$f"; done)
}

kv() {   # значение из VERSIYA (не «KOD»: на macOS имена без учёта регистра, рядом каталог kod/); stderr — чтобы было видно
  [ -f "$P/VERSIYA" ] || { echo "FAIL: нет файла VERSIYA — каталог perenos/ не собран или скопирован не целиком" >&2; exit 1; }
  sed -n "s/^$1=//p" "$P/VERSIYA"
}

KOD_SHA=$(kv tar_sha256)
KOD_SHORT=$(kv commit_short)
KOD_TAR="$P/kod_$KOD_SHORT.tar.gz"
KOD_DIR="$P/kod/$KOD_SHA"
KOD_SUMS="$P/kod/$KOD_SHA.SHA256SUMS"
ETALONY="$P/etalony_$KOD_SHORT.npz"
MANIFEST="$P/manifest_dannyh.tsv"
PY="$P/.venv/bin/python3"

yadra() { echo "${PERENOS_YADRA:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}"; }

processov() {   # $1 — период (необязателен). На цепочку: (ядра − 4) / PERENOS_CEPOCHEK (умолчание 2: train4 и v2 идут вместе, Ж-131 п. 4),
  # не меньше 1; wf — одна цепочка на сервер: PERENOS_PROCS_WF (умолчание 80: RSS 1,6–2,3 ГБ, потолок 2,8 на процесс, К-wf8; 124 не влезают
  # в 256 ГБ без общих массивов), но не больше ядра − 4. PERENOS_PROCS — явное число (локальная проверка), главнее всех.
  local n=${PERENOS_PROCS:-$(( ($(yadra) - 4) / ${PERENOS_CEPOCHEK:-2} ))} lim
  if [ "${1:-}" = wf ] && [ -z "${PERENOS_PROCS:-}" ]; then
    n=${PERENOS_PROCS_WF:-80}; lim=$(( $(yadra) - 4 )); [ "$n" -le "$lim" ] 2>/dev/null || n=$lim
  fi
  [ "$n" -ge 1 ] 2>/dev/null || n=1
  echo "$n"
}

otmetka_wf() {   # период wf читает данные ≥ 2022: только при отметке снятого замка (копия docs/resheniya/wf_zamok_snyat.json; ставит владелец)
  [ -s "$P/wf_zamok_snyat.json" ] || fail "wf: нет отметки $P/wf_zamok_snyat.json (её ставит владелец, не агент) — счёт не запускаю"
}

otmetka_paketa() {   # отметка wf в коде (kod/<sha>/docs/resheniya/): печатает SHA файла и SHA предрегистрации v2 из неё; нет отметки — ничего (wf откажет)
  local k="$KOD_DIR/docs/resheniya/wf_zamok_snyat.json" pre
  [ -f "$k" ] || { say "   отметки wf в пакете нет (период wf на сервере откажет)"; return 0; }
  pre=$(sed -n 's/.*"sha256": *"\([0-9a-f]\{64\}\)".*/\1/p' "$k" | head -n 1)
  [ -n "$pre" ] || fail "отметка wf без sha256 предрегистрации v2 (64 hex) — пакет не годен"
  [ -z "${PERENOS_PREREG_SHA:-}" ] || [ "$pre" = "$PERENOS_PREREG_SHA" ] || fail "SHA предрегистрации в отметке $pre ≠ ожидаемому PERENOS_PREREG_SHA"
  [ ! -f "$P/wf_zamok_snyat.json" ] || [ "$(sha_file "$P/wf_zamok_snyat.json")" = "$(sha_file "$k")" ] || fail "wf_zamok_snyat.json в корне пакета ≠ отметке в коде"
  say "   отметка wf: sha256 файла $(sha_file "$k"), предрегистрация v2 $pre"
}

chuzhie_processy() {   # $1 — свой период; сумма процессов других ЖИВЫХ цепочек (logs/sobol_<период>.pid + .procs)
  local s=0 f
  for f in "$P"/logs/sobol_*.pid; do
    [ -f "$f" ] || continue
    case "$f" in */sobol_"$1".pid) continue ;; esac
    kill -0 "$(cat "$f")" 2>/dev/null || continue
    s=$(( s + $(cat "${f%.pid}.procs" 2>/dev/null || echo 0) ))
  done
  echo "$s"
}

proverit_summu() {   # $1 — период, $2 — процессов этой цепочки: сумма по живым цепочкам ≤ ядра − 4, иначе FAIL
  local lim=$(( $(yadra) - 4 )) dr
  [ "$lim" -ge 1 ] || lim=1
  dr=$(chuzhie_processy "$1")
  [ $(( dr + $2 )) -le "$lim" ] || fail "процессов уже занято другой цепочкой: $dr + $2 > ядра − 4 = $lim — запуск отменён (PERENOS_PROCS меньше или дождитесь другой цепочки)"
}

rezhim_npy() {   # PERENOS_REZHIM=1 (по умолчанию: numpy как есть) | 2 (без AVX512-группы и SVML — выбор владельца после ./proverka_yakor.sh --rezhim2)
  unset NPY_DISABLE_CPU_FEATURES NPY_DISABLE_SVML || true
  case "${PERENOS_REZHIM:-1}" in
    1) ;;
    2) export NPY_DISABLE_CPU_FEATURES=${PERENOS_REZHIM2_FEATURES:-"AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"}
       export NPY_DISABLE_SVML=1 ;;
    *) fail "PERENOS_REZHIM: 1 или 2" ;;
  esac
}

okruzhenie() {   # данные — только perenos/data, кэши — perenos/cache; переменные, ведущие наружу, сняты
  unset FASTBT_DATA_ROOT FASTBT_VPRED FASTBT_Z2_VPRED FASTBT_Z2_R2FINAL FASTBT_NO_CACHE PYTHONHOME || true
  export PYTHONPATH="$KOD_DIR" FASTBT_MAIN="$P/data" FASTBT_BARS1M="$P/data/data/regen/D25/bars1m"
  export FASTBT_CACHE="$P/cache/fastbt" FASTBT_E1_CACHE="$P/cache/etap1" FASTBT_FS_CACHE="$P/cache/fs"
  export FASTBT_BARS5M_CACHE="$P/cache/bars5m" NUMBA_CACHE_DIR="$P/cache/numba/$KOD_SHA" PYTHONPYCACHEPREFIX="$P/cache/pyc"
  [ "${PERENOS_SYSTEM_SITE:-0}" = 1 ] || export PYTHONNOUSERSITE=1
  # потоки и слой numba — те же, что при сборке эталона на Маке (VERSIYA: numba_threads, numba_layer); печатаются
  NUMBA_NUM_THREADS=$(kv numba_threads); NUMBA_THREADING_LAYER=$(kv numba_layer)
  [ -n "$NUMBA_NUM_THREADS" ] && [ -n "$NUMBA_THREADING_LAYER" ] || fail "в VERSIYA нет numba_threads/numba_layer — пересоберите perenos/"
  export NUMBA_NUM_THREADS NUMBA_THREADING_LAYER
  rezhim_npy
  echo "   numba: NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS NUMBA_THREADING_LAYER=$NUMBA_THREADING_LAYER; режим numpy ${PERENOS_REZHIM:-1}" \
       "$([ "${PERENOS_REZHIM:-1}" = 2 ] && echo "(NPY_DISABLE_CPU_FEATURES=$NPY_DISABLE_CPU_FEATURES NPY_DISABLE_SVML=1)")"
  mkdir -p "$P/cache" "$P/logs" "$P/ledger" "$P/tables" "$P/proverka"
}

kod_cel() {   # tar тот же, распакованный код не правился (Ж-131 п. 6)
  [ -f "$KOD_TAR" ] || fail "нет $KOD_TAR"
  [ "$(sha_file "$KOD_TAR")" = "$KOD_SHA" ] || fail "sha256 $KOD_TAR ≠ VERSIYA (tar_sha256) — код подменён или скопирован не целиком"
  [ -d "$KOD_DIR" ] && [ -f "$KOD_SUMS" ] || fail "код не распакован — сначала ./razvernut.sh"
  sha_check "$KOD_DIR" "$KOD_SUMS" >/dev/null 2>&1 || fail "код в kod/$KOD_SHA изменён после распаковки — правка кода на сервере запрещена (Ж-131 п. 6); заново: ./razvernut.sh после удаления kod/"
  [ -z "$(find "$KOD_DIR" -perm -u+w -type f -print -quit)" ] || fail "в kod/$KOD_SHA есть файлы с правом записи — каталог кода только для чтения"
}

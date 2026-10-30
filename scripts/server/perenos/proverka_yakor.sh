#!/usr/bin/env bash
# Проверка эквивалентности (Ж-131 п. 3, Ж-133 п. 4): ./proverka_yakor.sh из каталога perenos/ после ./razvernut.sh.
# Два уровня против эталона Мака etalony_<коммит>.npz (якорь A_fix 6 Sk на train4 + 200 точек):
#   дискретное (хэши данных/куба, kod, vhody, ключи, статусы, сделки, индексы входа/выхода) — строго;
#   непрерывное (Шарп, доходности, ряды) — с допуском DOPUSK (по умолчанию 1e-12 и для якоря, и для точек, Ж-137 п. 3; меняет только владелец).
# Печатает OK / FAIL с первым расхождением и величиной, и отдельно max |Δ| по непрерывным — даже при OK.
#   ./proverka_yakor.sh            — режим numpy PERENOS_REZHIM (по умолчанию 1: как есть); отметка proverka/OK (режим 2: OK_r2)
#   ./proverka_yakor.sh --rezhim2  — оба режима подряд (1, затем 2 без AVX512-группы и SVML); печатает оба итога, владелец выбирает
# Код возврата: 0 — OK; 1 — FAIL дискретного (стоп); 3 — FAIL только непрерывного (max |Δ| — дирижёру; счёт не запускать).
. "$(dirname "$0")/obshchee.sh"

prover() {   # $1 — режим numpy; печатает итог режима, возвращает его код
  local r=$1 otm="$P/proverka/OK" log rc
  [ "$r" = 2 ] && otm="$P/proverka/OK_r2"
  export PERENOS_REZHIM=$r
  okruzhenie
  rm -f "$otm"
  export FASTBT_NO_CACHE=1                       # как при сборке эталона на Маке: всё считается заново
  log="$P/logs/proverka_r${r}_$(date +%Y%m%d_%H%M%S).log"
  n=$(processov)
  say "режим $r: пересчёт эталона: якорь A_fix × 6 Sk + 200 точек, плюс набор v6 (якорь v6 × 6 Sk + 200 точек, если он в эталоне), процессов $n (по Sk — не больше 6); лог $log"
  set +e
  (cd "$KOD_DIR" && "$PY" "$KOD_DIR/scripts/server/proverka_ekviv.py" --etalony "$ETALONY" --otmetka "$otm" --rezhim "$r" \
    --kod-sha "$KOD_SHA" --dannye-sha "$(sha_file "$MANIFEST")" --procs "$n") > "$log" 2>&1
  rc=$?
  set -e
  grep -E '^(proverka|OK|FAIL)' "$log" || { echo "FAIL [режим $r]: проверка упала (код $rc) — хвост лога:"; tail -n 5 "$log"; rc=1; }
  [ $rc -eq 0 ] && [ ! -f "$otm" ] && rc=1
  return $rc
}

okruzhenie_proverka() {
  [ -x "$PY" ] || fail "нет .venv — сначала ./razvernut.sh"
  say "1/2 код, данные, версии"
  kod_cel
  "$PY" manifest_proverka.py --manifest "$MANIFEST" --koren data --strogo \
    ${PERENOS_DO:+--do "$PERENOS_DO" --otmetka "${PERENOS_OTMETKA:-wf_zamok_snyat.json}"} >/dev/null || fail "данные не совпадают с манифестом — ./razvernut.sh"
  "$PY" versii.py requirements_server.txt >/dev/null || fail "версии пакетов ≠ пинам — ./razvernut.sh"
}

okruzhenie_proverka
say "2/2 сравнение с эталоном Мака (DOPUSK=${DOPUSK:-1e-12})"
if [ "${1:-}" = --rezhim2 ]; then
  rc1=0; prover 1 || rc1=$?
  rc2=0; prover 2 || rc2=$?
  say "ИТОГ режим 1 (numpy как есть): код $rc1; режим 2 (без AVX512-группы и SVML): код $rc2   [0 — OK, 1 — FAIL дискретного, 3 — FAIL непрерывного]"
  echo "Выбор режима — за владельцем; выбранный фиксируется переменной PERENOS_REZHIM=<1|2> при ./zapusk_sobol.sh (нужна отметка proverka/OK или OK_r2 того же режима)."
  [ $rc1 -eq 0 ] || [ $rc2 -eq 0 ] || exit $(( rc1 > rc2 ? rc1 : rc2 ))
  exit 0
fi
rc=0; prover "${PERENOS_REZHIM:-1}" || rc=$?
exit $rc

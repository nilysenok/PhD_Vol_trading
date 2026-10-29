#!/usr/bin/env bash
# Проверка эквивалентности (Ж-131 п. 3, Ж-133 п. 4): ./proverka_yakor.sh из каталога perenos/ после ./razvernut.sh.
# Якорь A_fix 6 Sk на train4 (max |Δ| ≤ 1e-12) и 200 точек Соболя побайтно против эталона Мака etalony_<коммит>.npz.
# Последняя строка — OK или FAIL с первым расхождением. При FAIL отметки proverka/OK нет и ./zapusk_sobol.sh не стартует;
# дальше не идти, вывод — дирижёру. Допуск не расширяется (Ж-131, толкование 3).
. "$(dirname "$0")/obshchee.sh"
okruzhenie
export FASTBT_NO_CACHE=1                       # как при сборке эталона на Маке: всё считается заново
rm -f "$P/proverka/OK"
[ -x "$PY" ] || fail "нет .venv — сначала ./razvernut.sh"

say "1/3 код и данные"
kod_cel
"$PY" manifest_proverka.py --manifest "$MANIFEST" --koren data --strogo >/dev/null || fail "данные не совпадают с манифестом — ./razvernut.sh"
"$PY" versii.py requirements_server.txt >/dev/null || fail "версии пакетов ≠ пинам — ./razvernut.sh"

n=$(processov)
log="$P/logs/proverka_$(date +%Y%m%d_%H%M%S).log"
say "2/3 пересчёт эталона: якорь A_fix × 6 Sk + 200 точек, процессов $n (по Sk — не больше 6); лог $log"
set +e
(cd "$P/cache" && "$PY" "$KOD_DIR/scripts/server/proverka_ekviv.py" --etalony "$ETALONY" --otmetka "$P/proverka/OK" \
  --kod-sha "$KOD_SHA" --dannye-sha "$(sha_file "$MANIFEST")" --procs "$n") > "$log" 2>&1
rc=$?
set -e
grep -E '^(proverka|OK|FAIL)' "$log" | grep -v '^FAIL' || true

say "3/3 вердикт"
if [ $rc -ne 0 ] || [ ! -f "$P/proverka/OK" ]; then
  first=$(grep -m1 '^FAIL' "$log" || true)
  [ -n "$first" ] || first="FAIL: проверка упала (код $rc) — хвост лога:
$(tail -n 5 "$log")"
  echo "$first"
  exit 1
fi
grep '^OK' "$log" | tail -1

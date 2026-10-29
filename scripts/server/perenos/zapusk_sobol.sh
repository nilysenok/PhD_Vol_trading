#!/usr/bin/env bash
# Запуск Соболя на сервере (Ж-133 п. 5): ./zapusk_sobol.sh train4 | ./zapusk_sobol.sh v2 — в фоне (nohup), лог в logs/,
# ledger в ledger/<период>/<Sk>/ (одна стратегия — один каталог), чекпоинт каждые 5 000 точек, строка состояния по стратегиям.
#   ./zapusk_sobol.sh status <период>   — строки состояния и жив ли процесс
# Отказ без отметки proverka/OK от ЭТОГО кода, данных и эталона (./proverka_yakor.sh), при изменённом коде, при живом прогоне.
# Процессов — ядра − 4 (Ж-131 п. 4); PERENOS_PROCS — явно. fastbt.progon_server есть в коде — он (--kub/--period/--seed/--out/
# --procs/--prodolzhit, кубы kuby/<PERENOS_KUBY, по умолчанию все *.yaml>, PERENOS_SEED); нет — замена fastbt.progon_sobol через
# scripts/server/sobol_zapusk.py (только train4, не больше 10 процессов). Локальная проверка: PERENOS_STRATEGII=S1 PERENOS_TOCHEK=50.
. "$(dirname "$0")/obshchee.sh"
okruzhenie

if [ "${1:-}" = __ryad ]; then              # внутренний режим: сама цепочка под nohup
  per=$2; shift 2
  if [ -f "$KOD_DIR/fastbt/progon_server.py" ]; then
    for k in kuby/${PERENOS_KUBY:-*.yaml}; do
      say "progon_server: $k"
      (cd "$KOD_DIR" &&"$PY" -m fastbt.progon_server --kub "$P/$k" --period "$per" --seed "${PERENOS_SEED:-20261029}" \
        --out "$P/ledger/$per/$(basename "$k" .yaml)" --procs "$(processov)" --prodolzhit)
    done
  else
    sl=""; [ "$(kv kod_slit)" = 1 ] && sl="--kod-slit"
    (cd "$KOD_DIR" &&"$PY" "$KOD_DIR/scripts/server/sobol_zapusk.py" --root "$P/ledger/$per" --procs "$(processov)" $sl \
      ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${PERENOS_TOCHEK:+--tochek "$PERENOS_TOCHEK"})
  fi
  say "цепочка $per завершена"
  exit 0
fi

rezhim=${1:-}
[ "$rezhim" = status ] && per=${2:-} || per=$rezhim
case "$per" in train4|v2) ;; *) fail "период: train4 или v2 (./zapusk_sobol.sh train4 | v2 | status <период>)" ;; esac
[ -x "$PY" ] || fail "нет .venv — сначала ./razvernut.sh"
pidf="$P/logs/sobol_$per.pid"
zhiv() { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; }
server=0; [ -f "$KOD_DIR/fastbt/progon_server.py" ] && server=1

if [ "$rezhim" = status ]; then
  zhiv && say "прогон $per идёт: pid $(cat "$pidf")" || say "прогон $per не идёт"
  ls -t "$P"/logs/sobol_"$per"_*.log 2>/dev/null | head -1 | while read -r l; do say "лог $l"; tail -n 3 "$l"; done
  if [ $server = 0 ] && [ -d "$P/ledger/$per" ]; then
    (cd "$KOD_DIR" &&"$PY" "$KOD_DIR/scripts/server/sobol_zapusk.py" --root "$P/ledger/$per" --sostoyanie \
      ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${PERENOS_TOCHEK:+--tochek "$PERENOS_TOCHEK"}) 2>/dev/null
  fi
  exit 0
fi

say "1/3 отметка proverka, код"
[ -f "$P/proverka/OK" ] || fail "нет отметки proverka/OK — сначала ./proverka_yakor.sh (строка OK); счёт не запускаю"
ozh="kod=$KOD_SHA dannye=$(sha_file "$MANIFEST") etalony=$(sha_file "$ETALONY")"
[ "$(tr '\n' ' ' < "$P/proverka/OK" | sed 's/ $//')" = "$ozh" ] || fail "отметка proverka/OK от другого кода/данных/эталона — повторите ./proverka_yakor.sh"
kod_cel
[ $server = 1 ] || [ "$per" = train4 ] || fail "v2: в коде нет fastbt.progon_server (PR не слит), замена progon_sobol считает только train4"
zhiv && fail "прогон $per уже идёт: pid $(cat "$pidf") (./zapusk_sobol.sh status $per)"

n=$(processov)
log="$P/logs/sobol_${per}_$(date +%Y%m%d_%H%M%S).log"
say "2/3 запуск $per: $([ $server = 1 ] && echo fastbt.progon_server || echo 'fastbt.progon_sobol (замена)'), процессов $n, ledger/$per/, лог $log"
mkdir -p "$P/ledger/$per"
nohup bash "$P/zapusk_sobol.sh" __ryad "$per" > "$log" 2>&1 < /dev/null &
echo $! > "$pidf"
sleep 3
zhiv || { tail -n 20 "$log"; fail "прогон $per упал на старте — лог выше"; }
say "3/3 идёт: pid $(cat "$pidf"); состояние — ./zapusk_sobol.sh status $per; остановить — kill $(cat "$pidf")"
echo "OK zapusk $per: pid $(cat "$pidf"), лог $log"

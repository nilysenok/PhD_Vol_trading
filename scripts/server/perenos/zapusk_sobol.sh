#!/usr/bin/env bash
# Запуск Соболя на сервере (Ж-133 п. 5): ./zapusk_sobol.sh train4 | ./zapusk_sobol.sh v2 — в фоне (nohup), лог в logs/,
# ledger в ledger/<период>/<Sk>/ (одна стратегия — один каталог), чекпоинт каждые 5 000 точек, строка состояния по стратегиям.
#   ./zapusk_sobol.sh status <период>   — строки состояния и жив ли процесс
# Отказ без отметки proverka/OK от ЭТОГО кода, данных, эталона и режима numpy (./proverka_yakor.sh), при изменённом коде, при живом
# прогоне. Режим numpy — переменная PERENOS_REZHIM=1|2 (выбор владельца после ./proverka_yakor.sh --rezhim2); NUMBA_* — как на Маке.
# Процессов — ядра − 4 (Ж-131 п. 4); PERENOS_PROCS — явно. Если fastbt.progon_server умеет куб v5 (в --help есть --k3; ветка
# progon-server-v5): проходы PERENOS_PROHODY (умолчание «s1 s2 k3») подряд, каждый — `--kub v5 --period <период> --seed 1|2 | --k3
# [--n1 PERENOS_N1] --out ledger/<период>_<проход> --prodolzhit`; PERENOS_STRATEGII=S1,S2 — подмножество. Иначе — замена
# fastbt.progon_sobol через scripts/server/sobol_zapusk.py (только train4, куб progon, ≤ 10 процессов; локально PERENOS_TOCHEK=50).
# Период v2 считает только progon_server с кубом v5 (сам откажет, если драйвер не умеет — вывод в лог).
. "$(dirname "$0")/obshchee.sh"
okruzhenie
novyj_cli() {   # fastbt.progon_server с кубом v5 (ветка progon-server-v5): в --help есть --k3; иначе — прежний CLI/замена
  [ -f "$KOD_DIR/fastbt/progon_server.py" ] && (cd "$KOD_DIR" && "$PY" -m fastbt.progon_server --help 2>&1 | grep -q -- '--k3')
}

if [ "${1:-}" = __ryad ]; then              # внутренний режим: сама цепочка под nohup
  per=$2; shift 2
  if novyj_cli; then
    # CLI progon_server под куб v5 (Л10, #608): --kub v5 --period train4|v2 --seed 1|2 | --k3 [--n1]; проходы подряд, свой каталог ledger
    # на проход: ledger/<период>_<s1|s2|k3>/<Sk>/. PERENOS_PROHODY="s1 s2 k3" (умолчание), PERENOS_N1 — как --n1, PERENOS_STRATEGII.
    sl=""; [ "$(kv kod_slit)" = 1 ] && sl="--kod-slit"
    for pr in ${PERENOS_PROHODY:-s1 s2 k3}; do
      case "$pr" in s1) a="--seed 1" ;; s2) a="--seed 2" ;; k3) a="--k3" ;; *) fail "проход $pr: s1 | s2 | k3" ;; esac
      say "progon_server: v5 $per $pr, ledger/${per}_$pr/<Sk>/, процессов $(processov)"
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_server --kub v5 --period "$per" $a ${PERENOS_N1:+--n1 "$PERENOS_N1"} \
        ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} --out "$P/ledger/${per}_$pr" --procs "$(processov)" --prodolzhit $sl)
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
server=0; novyj_cli && server=1

if [ "$rezhim" = status ]; then
  zhiv && say "прогон $per идёт: pid $(cat "$pidf")" || say "прогон $per не идёт"
  ls -t "$P"/logs/sobol_"$per"_*.log 2>/dev/null | head -1 | while read -r l; do say "лог $l"; tail -n 3 "$l"; done
  if [ $server = 1 ]; then      # строка состояния по стратегии — из checkpoint.json (каждые 5 000 точек)
    for c in "$P"/ledger/"$per"_*/S*/checkpoint.json "$P"/ledger/"$per"/S*/checkpoint.json; do
      [ -f "$c" ] || continue
      echo "  $(basename "$(dirname "$(dirname "$c")")")/$(basename "$(dirname "$c")"): N=$(grep -o '"N": [0-9]*' "$c" | grep -o '[0-9]*$') чекпоинт i=$(grep -o '"posledniy_i": [0-9]*' "$c" | grep -o '[0-9]*$') ($(sed -E 's/.*"vremya": "([^"]*)".*/\1/' "$c"))"
    done
  fi
  if [ $server = 0 ] && [ -d "$P/ledger/$per" ]; then
    (cd "$KOD_DIR" &&"$PY" "$KOD_DIR/scripts/server/sobol_zapusk.py" --root "$P/ledger/$per" --sostoyanie \
      ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${PERENOS_TOCHEK:+--tochek "$PERENOS_TOCHEK"}) 2>/dev/null
  fi
  exit 0
fi

say "1/3 отметка proverka, код"
rz=${PERENOS_REZHIM:-1}; otm="$P/proverka/OK"; [ "$rz" = 2 ] && otm="$P/proverka/OK_r2"     # режим numpy фиксирует владелец
[ -f "$otm" ] || fail "нет отметки ${otm#$P/} (режим numpy $rz) — сначала ./proverka_yakor.sh (строка OK в этом режиме); счёт не запускаю"
ozh="kod=$KOD_SHA dannye=$(sha_file "$MANIFEST") etalony=$(sha_file "$ETALONY") rezhim=$rz"
[ "$(tr '\n' ' ' < "$otm" | sed 's/ $//')" = "$ozh" ] || fail "отметка ${otm#$P/} от другого кода/данных/эталона/режима — повторите ./proverka_yakor.sh"
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

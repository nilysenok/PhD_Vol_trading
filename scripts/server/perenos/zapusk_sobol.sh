#!/usr/bin/env bash
# Запуск Соболя на сервере (Ж-133 п. 5): ./zapusk_sobol.sh train4 | v2 | wf — в фоне (nohup), лог в logs/,
# wf (Ж-136 п. 5: один проход по всему периоду 2014-06…2026-02, s1 s2 k3): только при отметке wf_zamok_snyat.json рядом со скриптом
# (ставит владелец), процессов PERENOS_PROCS_WF (умолчание 80), одна цепочка на сервер;
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
    # spravka (Ж-137 п. 6, решение дирижёра 29.10 19:10): те же точки, что s1, но выход по close дня (--rol spravka --t-vyhod close);
    # ledger 2.2 с rol = spravka в ledger/<период>_spravka_close/<Sk>/, в N не входит. Идёт ПОСЛЕ полного прохода той же сетки.
    # PERENOS_KUB — куб (умолчание v5; v6 — см. ниже и раздел «v6» docs/otchety/2026-10-29_server_perenos.md).
    # v6 (Ж-137 п. 2; С-v6): PERENOS_KUB=v6 — сначала Моррис (ledger/<период>_morris/<Sk>/zamorozka.yaml), потом проходы с --zamorozka
    # (кроме k3: контроль на решётке v5, новые оси «выкл.»); N1 — из куба (200 000), PERENOS_N1 — явно; ledger 2.2 (setka v6, rol).
    # PERENOS_MORRIS_R — траекторий Морриса (умолчание ~5 000 точек на Sk; 1 — малая проверка). Моррис упал — полный проход не начинается.
    kub=${PERENOS_KUB:-v5}; zm=""; n1v="--n1 ${PERENOS_N1:-262144}"
    if [ "$kub" = v6 ]; then
      mo="$P/ledger/${per}_morris"; zm="--zamorozka $mo"; n1v=${PERENOS_N1:+--n1 $PERENOS_N1}
      say "progon_morris_v6: $per, ledger/${per}_morris/<Sk>/zamorozka.yaml, процессов $(processov "$per")"
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_morris_v6 --kub v6 --period "$per" --out "$mo" --procs "$(processov "$per")" --prodolzhit \
        ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${PERENOS_MORRIS_R:+--r "$PERENOS_MORRIS_R"}) || fail "Моррис v6 упал — заморозки нет, полный проход не начинаю"
    fi
    for pr in ${PERENOS_PROHODY:-s1 s2 k3 spravka}; do
      # N1 по умолчанию 262 144 на стратегию (решение дирижёра, volna_17 п. 1: 200 000 → степень двойки); k3 — полная решётка, без --n1
      od="${per}_$pr"
      case "$pr" in s1) a="--seed 1 $n1v $zm" ;; s2) a="--seed 2 $n1v $zm" ;; k3) a="--k3" ;;
        spravka) a="--seed 1 $n1v $zm --rol spravka --t-vyhod close"; od="${per}_spravka_close" ;;
        *) fail "проход $pr: s1 | s2 | k3 | spravka" ;; esac
      say "progon_server: $kub $per $pr ($a), ledger/$od/<Sk>/, процессов $(processov "$per")"
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_server --kub "$kub" --period "$per" $a \
        ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} --out "$P/ledger/$od" --procs "$(processov "$per")" --prodolzhit $sl)
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
case "$per" in train4|v2|wf) ;; *) fail "период: train4, v2 или wf (./zapusk_sobol.sh train4 | v2 | wf | status <период>)" ;; esac
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

if [ "$per" = wf ]; then     # Ж-136: единый walk-forward, один проход по всему периоду; только после снятия замка владельцем
  [ $server = 1 ] || fail "wf: в коде нет fastbt.progon_server с кубом v5 — замена progon_sobol считает только train4"
  otmetka_wf
fi
n=$(processov "$per")
proverit_summu "$per" "$n"          # train4 и v2 одновременно: сумма процессов ≤ ядра − 4 (по умолчанию каждая берёт половину)
log="$P/logs/sobol_${per}_$(date +%Y%m%d_%H%M%S).log"
say "2/3 запуск $per: $([ $server = 1 ] && echo fastbt.progon_server || echo 'fastbt.progon_sobol (замена)'), процессов $n, ledger/$per/, лог $log"
mkdir -p "$P/ledger/$per"
nohup bash "$P/zapusk_sobol.sh" __ryad "$per" > "$log" 2>&1 < /dev/null &
echo $! > "$pidf"
echo "$n" > "$P/logs/sobol_$per.procs"
sleep 3
zhiv || { tail -n 20 "$log"; fail "прогон $per упал на старте — лог выше"; }
say "3/3 идёт: pid $(cat "$pidf"); состояние — ./zapusk_sobol.sh status $per; остановить — kill $(cat "$pidf")"
echo "OK zapusk $per: pid $(cat "$pidf"), лог $log"

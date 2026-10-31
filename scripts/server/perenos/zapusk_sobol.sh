#!/usr/bin/env bash
# Запуск Соболя на сервере (Ж-133 п. 5): ./zapusk_sobol.sh train4 | v2 | wf — в фоне (nohup), лог в logs/,
# wf (Ж-136 п. 5: один проход по всему периоду 2014-06…2026-02, s1 s2 k3): только при отметке wf_zamok_snyat.json рядом со скриптом
# (ставит владелец), процессов PERENOS_PROCS_WF (умолчание 80), одна цепочка на сервер;
# ledger в ledger/<период>/<Sk>/ (одна стратегия — один каталог), чекпоинт каждые 5 000 точек, строка состояния по стратегиям.
#   ./zapusk_sobol.sh status <период>   — строки состояния и жив ли процесс
# Куб v7 (V7-8b): PERENOS_KUB=v7 ./zapusk_sobol.sh polnaya — период polnaya (= границы и замок wf, процессов PERENOS_PROCS_WF), Моррис
# fastbt.progon_morris_v7, проходы s1 s2 через fastbt.progon_v7 (ledger/polnaya_v7_<проход>/); v7 — только polnaya, polnaya — только v7.
# Пилот-1 v7 (Ж-156, Ж-160 п. 3): PERENOS_KUB=v7 PERENOS_PILOT=1 ./zapusk_sobol.sh polnaya — fastbt.progon_v7 --pilot --n1 PERENOS_N1 (1000)
# по семействам PERENOS_SEMEYSTVA (S1…S6), без Морриса; процессов PERENOS_PROCS_PILOT (64); ledger/polnaya_v7_pilot1/pilot_zamer.json.
# Отказ без отметки proverka/OK от ЭТОГО кода, данных, эталона и режима numpy (./proverka_yakor.sh), при изменённом коде, при живом
# прогоне. Режим numpy — переменная PERENOS_REZHIM=1|2 (выбор владельца после ./proverka_yakor.sh --rezhim2); NUMBA_* — как на Маке.
# Процессов — 120, не больше ядра − 4 (Ж-137 п. 2; память проверяет pamyat.sh); PERENOS_PROCS — явно. Если fastbt.progon_server умеет куб v5 (в --help есть --k3; ветка
# progon-server-v5): проходы PERENOS_PROHODY (умолчание «s1 s2 k3») подряд, каждый — `--kub v5 --period <период> --seed 1|2 | --k3
# [--n1 PERENOS_N1] --out ledger/<период>_<проход> --prodolzhit`; PERENOS_STRATEGII=S1,S2 — подмножество. Иначе — замена
# fastbt.progon_sobol через scripts/server/sobol_zapusk.py (только train4, куб progon, ≤ 10 процессов; локально PERENOS_TOCHEK=50).
# Период v2 считает только progon_server с кубом v5 (сам откажет, если драйвер не умеет — вывод в лог).
. "$(dirname "$0")/obshchee.sh"
. "$P/pamyat.sh"
okruzhenie
novyj_cli() {   # fastbt.progon_server с кубом v5 (ветка progon-server-v5): в --help есть --k3; иначе — прежний CLI/замена
  # куб v7 (V7-8b): драйверы fastbt.progon_v7 / progon_morris_v7 (лист V7-8a); их нет в коде — прежний путь не подменяет v7 (проверка ниже)
  if [ "${PERENOS_KUB:-v5}" = v7 ]; then [ -f "$KOD_DIR/fastbt/progon_v7.py" ] && [ -f "$KOD_DIR/fastbt/progon_morris_v7.py" ]; return; fi
  [ -f "$KOD_DIR/fastbt/progon_server.py" ] && (cd "$KOD_DIR" && "$PY" -m fastbt.progon_server --help 2>&1 | grep -q -- '--k3')
}

SEM_PILOT=S1,S2,S3,S4,S5,S6
procs_pilot() {   # пилот-1: PERENOS_PROCS_PILOT (умолчание 64: RSS ≈ 3,5 ГБ на процесс), но не больше ядра − 4 и не меньше 1
  local n=${PERENOS_PROCS_PILOT:-64} lim=$(( $(yadra) - 4 ))
  [ "$n" -le "$lim" ] 2>/dev/null || n=$lim; [ "$n" -ge 1 ] || n=1; echo "$n"
}
progrev() {   # $1 — период. Одна точка каждой Sk (v5: --n1 1, seed 1; v6/v7: Моррис --r 1) одним процессом в cache/progrev_<период>/ (не в ледгере)
  local pg="$P/cache/progrev_$1" t0 rc=0
  t0=$(date +%s); rm -rf "$pg"
  say "прогрев кэшей одним процессом: $1, ${PERENOS_KUB:-v5}, по точке на Sk, $pg"
  if [ "${PERENOS_KUB:-v5}" = v6 ] || [ "${PERENOS_KUB:-v5}" = v7 ]; then
    (cd "$KOD_DIR" && "$PY" -m fastbt.progon_morris_${PERENOS_KUB} --kub "$PERENOS_KUB" --period "$1" --out "$pg" --procs 1 --r 1 \
      ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"}) || rc=$?
  else
    (cd "$KOD_DIR" && "$PY" -m fastbt.progon_server --kub v5 --period "$1" --seed 1 --n1 1 --out "$pg" --procs 1 \
      ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${fast:-}) || rc=$?
  fi
  [ $rc = 0 ] || fail "прогрев упал (код $rc) — пул не стартую, вывод выше"
  say "прогрев кэшей: $(( $(date +%s) - t0 )) с"
}
dolya_yadra() {   # $1 — каталог прохода: доля точек, посчитанных ядром (sloy = fast), по Sk
  (cd "$KOD_DIR" && "$PY" "$KOD_DIR/scripts/server/svodka_ledger.py" --yadro "$1") || say "ядро: сводка не получена ($1)"
}

if [ "${1:-}" = __ryad ]; then              # внутренний режим: сама цепочка под nohup
  per=$2; shift 2
  pk=$per; [ "$per" = polnaya ] && pk=wf       # polnaya (куб v7, 2014-06…2026-02-28) — по процессам и замку как wf (Ж-136)
  [ "${PERENOS_PILOT:-}" != 1 ] || [ "${PERENOS_KUB:-v5}" = v7 ] || fail "PERENOS_PILOT=1 — только с PERENOS_KUB=v7"
  if [ "${PERENOS_KUB:-v5}" = v7 ] && ! novyj_cli; then fail "v7: в коде нет fastbt.progon_v7 / progon_morris_v7 (лист V7-8a не слит) — счёт не начинаю"; fi
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
    kub=${PERENOS_KUB:-v5}; zm=""; n1v="--n1 ${PERENOS_N1:-200000}"      # Ж-137 п. 1: 200 000 на стратегию (262 144 отменены)
    # Быстрый слой (Ж-083; ядро #698/#699/#703/#706): --fast по умолчанию; PERENOS_FAST=0 — всё эталоном. Доля точек ядра (sloy = fast) —
    # строкой «ядро» после каждого прохода и в svodka_ledger.py.
    fast=""; [ "${PERENOS_FAST:-1}" = 0 ] || fast="--fast"
    progrev "$per"      # до пула: один процесс, одна точка каждой Sk (bars5m, дисковый кэш, numba) — без гонок при fork 120 процессов
    if [ "${PERENOS_PILOT:-}" = 1 ]; then      # пилот-1 (Ж-156, Ж-160 п. 3): 1 000 точек на семейство, без Морриса, ledger/polnaya_v7_pilot1/
      od="$P/ledger/${per}_v7_pilot1"; np=$(procs_pilot)
      say "пилот-1: progon_v7 --pilot $per, n1 ${PERENOS_N1:-1000}, семейства ${PERENOS_SEMEYSTVA:-$SEM_PILOT}, процессов $np, $od"
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_v7 --kub v7 --period "$per" --pilot --n1 "${PERENOS_N1:-1000}" \
        --semeystva "${PERENOS_SEMEYSTVA:-$SEM_PILOT}" --procs "$np" --out "$od" $fast) || fail "пилот-1 упал — вывод выше"
      say "пилот-1 готов: замер $od/pilot_zamer.json; забрать: ./zabrat.sh --agregaty \$SRV:perenos <каталог на Маке>"
      say "цепочка $per завершена"; exit 0
    fi
    # v7 (V7-8b; Ж-155 п. 7, Ж-156): PERENOS_KUB=v7 — период polnaya, Моррис fastbt.progon_morris_v7, проходы fastbt.progon_v7 (s1 s2;
    # k3 v7 — ссылка на ledger v6, не пересчёт; spravka не нужна), N1 200 000 на стратегию (PERENOS_N1 — явно), каталоги ledger/polnaya_v7_<проход>.
    prohody="s1 s2 k3 spravka"; [ "$kub" = v7 ] && prohody="s1 s2"
    if [ "$kub" = v6 ]; then
      mo="$P/ledger/${per}_morris"; zm="--zamorozka $mo"; n1v=${PERENOS_N1:+--n1 $PERENOS_N1}
      say "progon_morris_v6: $per, ledger/${per}_morris/<Sk>/zamorozka.yaml, процессов $(processov "$per")"
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_morris_v6 --kub v6 --period "$per" --out "$mo" --procs "$(processov "$per")" --prodolzhit \
        ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${PERENOS_MORRIS_R:+--r "$PERENOS_MORRIS_R"}) || fail "Моррис v6 упал — заморозки нет, полный проход не начинаю"
    fi
    if [ "$kub" = v7 ]; then      # те же аргументы; period polnaya считается как wf по процессам ($pk)
      mo="$P/ledger/${per}_morris"; zm="--zamorozka $mo"
      say "progon_morris_v7: $per, ledger/${per}_morris/<Sk>/zamorozka.yaml, процессов $(processov "$pk")"
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_morris_v7 --kub v7 --period "$per" --out "$mo" --procs "$(processov "$pk")" --prodolzhit \
        ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} ${PERENOS_MORRIS_R:+--r "$PERENOS_MORRIS_R"}) || fail "Моррис v7 упал — заморозки нет, полный проход не начинаю"
    fi
    for pr in ${PERENOS_PROHODY:-$prohody}; do
      # N1 по умолчанию 200 000 на стратегию (Ж-137 п. 1); k3 — полная решётка, без --n1
      # каталог прохода: v5 — <период>_<проход>; v6 — <период>_v6_<проход> (ledger 2.2 ≠ 2.1: в один каталог v6 после v5 не писать — --prodolzhit откажет по vhody)
      pv=""; { [ "$kub" = v6 ] || [ "$kub" = v7 ]; } && pv="${kub}_"
      od="${per}_$pv$pr"
      case "$pr" in s1) a="--seed 1 $n1v $zm" ;; s2) a="--seed 2 $n1v $zm" ;; k3) a="--k3" ;;
        spravka) a="--seed 1 $n1v $zm --rol spravka --t-vyhod close"; od="${per}_${pv}spravka_close" ;;
        *) fail "проход $pr: s1 | s2 | k3 | spravka" ;; esac
      say "progon_server: $kub $per $pr ($a), ledger/$od/<Sk>/, процессов $(processov "$pk")"
      rc=0; vyv=$(mktemp "$P/logs/progon_vyvod.XXXXXX")
      if [ "$kub" = v7 ]; then      # v7 — свой драйвер (лист V7-8a), те же аргументы
        (cd "$KOD_DIR" && "$PY" -m fastbt.progon_v7 --kub "$kub" --period "$per" $a \
          ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} --out "$P/ledger/$od" --procs "$(processov "$pk")" --prodolzhit $sl $fast) 2>&1 | tee "$vyv" && rc=0 || rc=${PIPESTATUS[0]}
      else
      (cd "$KOD_DIR" && "$PY" -m fastbt.progon_server --kub "$kub" --period "$per" $a \
        ${PERENOS_STRATEGII:+--strategii "$PERENOS_STRATEGII"} --out "$P/ledger/$od" --procs "$(processov "$pk")" --prodolzhit $sl $fast) 2>&1 | tee "$vyv" && rc=0 || rc=${PIPESTATUS[0]}
      fi
      dolya_yadra "$P/ledger/$od"
      [ $rc = 0 ] || fail_progon "$pr" "$rc" "$vyv"
      rm -f "$vyv"
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
case "$per" in train4|v2|wf|polnaya) ;; *) fail "период: train4, v2, wf или polnaya (./zapusk_sobol.sh train4 | v2 | wf | polnaya | status <период>)" ;; esac
# куб v7 считает только период polnaya (2014-06…2026-02-28), и polnaya — только куб v7 (V7-8b)
if [ "$rezhim" != status ]; then
  [ "${PERENOS_KUB:-v5}" != v7 ] || [ "$per" = polnaya ] || fail "PERENOS_KUB=v7: период только polnaya (получен $per)"
  [ "$per" != polnaya ] || [ "${PERENOS_KUB:-v5}" = v7 ] || fail "период polnaya — только с PERENOS_KUB=v7"
fi
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

if [ "$per" = wf ] || [ "$per" = polnaya ]; then     # Ж-136: единый walk-forward, один проход по всему периоду; только после снятия замка владельцем
  [ $server = 1 ] || fail "$per: в коде нет fastbt.progon_server с кубом v5 (polnaya — progon_v7) — замена progon_sobol считает только train4"
  otmetka_wf
fi
pk=$per; [ "$per" = polnaya ] && pk=wf
n=$(processov "$pk")
if [ "${PERENOS_PILOT:-}" = 1 ]; then
  [ "${PERENOS_KUB:-v5}" = v7 ] || fail "PERENOS_PILOT=1 — только с PERENOS_KUB=v7"
  n=$(procs_pilot)
fi
proverit_summu "$per" "$n"          # train4 и v2 одновременно: сумма процессов ≤ ядра − 4 (по умолчанию каждая берёт половину)
log="$P/logs/sobol_${per}_$(date +%Y%m%d_%H%M%S).log"
say "2/3 запуск $per: $([ $server = 1 ] && echo fastbt.progon_server || echo 'fastbt.progon_sobol (замена)'), процессов $n, ledger/$per/, лог $log"
mkdir -p "$P/ledger/$per"
nohup bash "$P/zapusk_sobol.sh" __ryad "$per" > "$log" 2>&1 < /dev/null &
echo $! > "$pidf"
echo "$n" > "$P/logs/sobol_$per.procs"
pamyat_sledit "$per" "$(cat "$pidf")" "$n" >> "$P/logs/pamyat_$per.log" 2>&1 &
sleep 3
zhiv || { tail -n 20 "$log"; fail "прогон $per упал на старте — лог выше"; }
say "3/3 идёт: pid $(cat "$pidf"); состояние — ./zapusk_sobol.sh status $per; остановить — kill $(cat "$pidf")"
echo "OK zapusk $per: pid $(cat "$pidf"), лог $log"

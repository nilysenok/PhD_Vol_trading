# Проверка памяти цепочки (Ж-137 п. 2): подключается из zapusk_sobol.sh после obshchee.sh.
# Процессов по умолчанию 120 (obshchee.sh, processov). Наблюдатель замеряет память периодически (PERENOS_PAMYAT_INTERVAL секунд, умолчание 45)
# на всём проходе, а не один раз: рост рабочих виден по максимуму USS.
# Память процесса — USS (Private_Clean + Private_Dirty из /proc/<pid>/smaps_rollup), не RSS: RSS форка включает общие страницы
# родителя (120 × 2,4 ГБ RSS «занимали» 295 ГБ при 251 ГБ физических). Доступная память — MemAvailable из /proc/meminfo
# (верх — MemTotal). Если n × USS_max × 1,15 > MemAvailable + сумма USS цепочки — цепочка останавливается и продолжается
# с меньшим числом процессов (--prodolzhit, чекпоинт); причина — в stdout и в logs/pamyat_<период>.log (строку переносит в журнал владелец).
# Нет /proc (Мак) — «не измерено», процессов без изменений. Это замер: потолок, не гарантия. PERENOS_PROC — корень /proc (тесты).

pamyat_reshenie() {   # $1 — процессов n, $2 — USS одного процесса (максимум), КБ, $3 — вся доступная цепочке память, КБ -> новое n (>= 1, не больше n)
  awk -v n="$1" -v r="$2" -v t="$3" 'BEGIN { m = int(t / (r * 1.15)); if (m < 1) m = 1; print (m < n ? m : n) }'
}

pamyat_uss() {   # $1 — pid цепочки: "<макс USS КБ> <сумма USS КБ>" потомков (ps -A -o pid=,ppid=; USS — smaps_rollup)
  local p s mx=0 sm=0 f="${PERENOS_PROC:-/proc}"
  for p in $(ps -A -o pid=,ppid= | awk -v root="$1" '{ par[$1] = $2 }
      END { for (p in par) { q = p; while (q in par && q != root) q = par[q]; if (q == root && p != root) print p } }'); do
    [ -r "$f/$p/smaps_rollup" ] || continue
    s=$(awk '/^Private_(Clean|Dirty):/ { t += $2 } END { print t + 0 }' "$f/$p/smaps_rollup")
    sm=$((sm + s)); [ "$s" -gt "$mx" ] && mx=$s
  done
  echo "$mx $sm"
}

pamyat_svobodno() {   # "<MemAvailable КБ> <MemTotal КБ>" из /proc/meminfo; пусто, если файла нет (Мак)
  local m="${PERENOS_PROC:-/proc}/meminfo"
  [ -r "$m" ] || return 0
  awk '/^MemAvailable:/ { a = $2 } /^MemTotal:/ { t = $2 } END { if (a != "") print a, t + 0 }' "$m"
}

pamyat_ostanovit() {   # $1 — pid: потомки и сам процесс
  local k
  for k in $(ps -A -o pid=,ppid= | awk -v root="$1" '{ par[$1] = $2 } END { for (p in par) { q = p; while (q in par && q != root) q = par[q]; if (q == root) print p } }'); do
    kill "$k" 2>/dev/null || true
  done
}

pamyat_sledit() {   # $1 — период, $2 — pid цепочки, $3 — n; фоновый наблюдатель: замер каждые PERENOS_PAMYAT_INTERVAL с, пока цепочка жива
  local per=$1 pid=$2 n=$3 lg="$P/logs/pamyat_$1.log" gran="${PERENOS_PAMYAT_INTERVAL:-45}" mx sm av tm tot nn np top=0 lastlog=0
  while kill -0 "$pid" 2>/dev/null; do
    read -r mx sm <<< "$(pamyat_uss "$pid")"
    read -r av tm <<< "$(pamyat_svobodno)"
    if [ -z "$av" ]; then
      echo "$(date '+%F %T') $per: память не измерена (нет /proc/meminfo); процессов $n без изменений" >> "$lg"; return 0
    fi
    if [ "${mx:-0}" -gt 0 ]; then
      [ "$mx" -gt "$top" ] && top=$mx
      tot=$((av + sm)); [ -n "$tm" ] && [ "$tm" -gt 0 ] && [ "$tot" -gt "$tm" ] && tot=$tm
      nn=$(pamyat_reshenie "$n" "$top" "$tot")
      if [ "$nn" -lt "$n" ]; then
        echo "$(date '+%F %T') $per: СНИЖЕНИЕ процессов $n → $nn: USS макс $((top / 1024)) МБ, доступно $((tot / 1024)) МБ, нужно $n × USS × 1,15 = $((n * top * 115 / 100 / 1024)) МБ" | tee -a "$lg"
        pamyat_ostanovit "$pid"; sleep 2
        PERENOS_PROCS=$nn nohup bash "$P/zapusk_sobol.sh" __ryad "$per" >> "$P/logs/sobol_${per}_pamyat.log" 2>&1 < /dev/null &
        np=$!; echo "$np" > "$P/logs/sobol_$per.pid"; echo "$nn" > "$P/logs/sobol_$per.procs"
        echo "$(date '+%F %T') $per: продолжение с $nn процессами, pid $np, лог logs/sobol_${per}_pamyat.log" >> "$lg"
        pid=$np; n=$nn; top=0; lastlog=0
      elif [ "$top" -ge $((lastlog * 11 / 10 + 1)) ]; then
        echo "$(date '+%F %T') $per: память: $n × $((top / 1024)) МБ × 1,15 помещается в $((tot / 1024)) МБ; процессов $n" >> "$lg"; lastlog=$top
      fi
    fi
    sleep "$gran"
  done
  echo "$(date '+%F %T') $per: цепочка закончилась; USS макс $((top / 1024)) МБ" >> "$lg"
}

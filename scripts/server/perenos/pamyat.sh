# Проверка памяти при старте цепочки (Ж-137 п. 2): подключается из zapusk_sobol.sh после obshchee.sh.
# Процессов по умолчанию 120 (obshchee.sh, processov). После первого чекпоинта (или через PERENOS_PAMYAT_ZHDAT секунд, умолчание 600)
# измеряется максимум RSS среди процессов цепочки и свободная память (free -k, MemAvailable + то, что цепочка уже заняла).
# Если n × RSS × 1,15 не помещается — цепочка останавливается и продолжается с меньшим числом процессов (--prodolzhit, чекпоинт);
# причина — в stdout и в logs/pamyat_<период>.log (строку переносит в журнал владелец). Это замер на первом куске: потолок, не гарантия.

pamyat_reshenie() {   # $1 — процессов n, $2 — RSS одного процесса, КБ, $3 — вся доступная цепочке память, КБ -> новое n (>= 1, не больше n)
  awk -v n="$1" -v r="$2" -v t="$3" 'BEGIN { m = int(t / (r * 1.15)); if (m < 1) m = 1; print (m < n ? m : n) }'
}

pamyat_rss() {   # $1 — pid цепочки: "<макс RSS КБ> <сумма RSS КБ>" потомков (ps -A -o pid=,ppid=,rss=)
  ps -A -o pid=,ppid=,rss= | awk -v root="$1" '{ par[$1] = $2; rss[$1] = $3 }
    END { for (p in par) { q = p; while (q in par && q != root) q = par[q]
            if (q == root && p != root) { s += rss[p]; if (rss[p] > m) m = rss[p] } }
          print m + 0, s + 0 }'
}

pamyat_svobodno() {   # КБ MemAvailable из `free -k`; пусто, если free нет
  command -v free >/dev/null 2>&1 || return 0
  free -k | awk '/^Mem:/ { print $7 }'
}

pamyat_ostanovit() {   # $1 — pid: потомки и сам процесс
  local k
  for k in $(ps -A -o pid=,ppid= | awk -v root="$1" '{ par[$1] = $2 } END { for (p in par) { q = p; while (q in par && q != root) q = par[q]; if (q == root) print p } }'); do
    kill "$k" 2>/dev/null || true
  done
}

pamyat_sledit() {   # $1 — период, $2 — pid цепочки, $3 — n; фоновый наблюдатель
  local per=$1 pid=$2 n=$3 lg="$P/logs/pamyat_$1.log" t=0 mx sm av tot nn np
  while ! ls "$P"/ledger/"$per"*/S*/checkpoint.json >/dev/null 2>&1 && [ "$t" -lt "${PERENOS_PAMYAT_ZHDAT:-600}" ]; do
    kill -0 "$pid" 2>/dev/null || break; sleep 1; t=$((t + 1))
  done
  kill -0 "$pid" 2>/dev/null || { echo "$(date '+%F %T') $per: цепочка закончилась до замера памяти" >> "$lg"; return 0; }
  read -r mx sm <<< "$(pamyat_rss "$pid")"
  av=$(pamyat_svobodno)
  if [ -z "$av" ] || [ "$mx" -le 0 ]; then
    echo "$(date '+%F %T') $per: память не измерена (free: '${av:-нет}', RSS макс $mx КБ); процессов $n без изменений" >> "$lg"; return 0
  fi
  tot=$((av + sm)); nn=$(pamyat_reshenie "$n" "$mx" "$tot")
  if [ "$nn" -ge "$n" ]; then
    echo "$(date '+%F %T') $per: память: $n × $((mx / 1024)) МБ × 1,15 помещается в $((tot / 1024)) МБ; процессов $n" >> "$lg"; return 0
  fi
  echo "$(date '+%F %T') $per: СНИЖЕНИЕ процессов $n → $nn: RSS макс $((mx / 1024)) МБ, доступно $((tot / 1024)) МБ, нужно $n × RSS × 1,15 = $((n * mx * 115 / 100 / 1024)) МБ" | tee -a "$lg"
  pamyat_ostanovit "$pid"; sleep 2
  PERENOS_PROCS=$nn nohup bash "$P/zapusk_sobol.sh" __ryad "$per" >> "$P/logs/sobol_${per}_pamyat.log" 2>&1 < /dev/null &
  np=$!; echo "$np" > "$P/logs/sobol_$per.pid"; echo "$nn" > "$P/logs/sobol_$per.procs"
  echo "$(date '+%F %T') $per: продолжение с $nn процессами, pid $np, лог logs/sobol_${per}_pamyat.log" >> "$lg"
}

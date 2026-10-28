#!/usr/bin/env bash
# Фоновый запуск тяжёлого счёта на Маке под nice -n 10.
#   scripts/bg.sh <имя> <команда…>   запуск; pid -> data/run/<имя>.pid, лог -> data/run/<имя>.log
#   scripts/bg.sh stop <имя>         снять процесс по pid
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$ROOT/data/run"
mkdir -p "$RUN"

if [ "${1:-}" = "stop" ]; then
  name="${2:?usage: bg.sh stop <имя>}"
  pidf="$RUN/$name.pid"
  [ -f "$pidf" ] || { echo "нет pid-файла: $pidf" >&2; exit 1; }
  pid="$(cat "$pidf")"
  if kill -0 "$pid" 2>/dev/null; then kill "$pid" && echo "снят pid $pid ($name)"; else echo "pid $pid уже не работает"; fi
  rm -f "$pidf"
  exit 0
fi

name="${1:?usage: bg.sh <имя> <команда…>}"
shift
[ "$#" -ge 1 ] || { echo "usage: bg.sh <имя> <команда…>" >&2; exit 1; }
if [ -f "$RUN/$name.pid" ] && kill -0 "$(cat "$RUN/$name.pid")" 2>/dev/null; then
  echo "уже запущен: pid $(cat "$RUN/$name.pid")" >&2
  exit 1
fi
cd "$ROOT"
nohup nice -n 10 "$@" >"$RUN/$name.log" 2>&1 &
echo $! >"$RUN/$name.pid"
echo "запущен $name pid $(cat "$RUN/$name.pid"), лог data/run/$name.log"

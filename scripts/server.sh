#!/usr/bin/env zsh
# Сервер второго раунда гибрида (R2). Решение владельца 26.10, 14:30: весь R2 считается только на сервере; Мак — код и журналы.
# Подробно — docs/server_r2.md.
#
#   scripts/server.sh push [--dry-run] [--update-data]   код (git ls-files) и данные на сервер; test_locked исключён жёстко
#   scripts/server.sh setup                              окружение на сервере (env/setup_server.sh): Python 3.9.6, пакеты, Rust, moex_iss
#   scripts/server.sh run <имя> [команда…]               фон на сервере под nice: лог и pid в data/run_server/<имя>.{log,pid}
#                                                        без команды — готовые задачи: repro, checkenv (см. TASKS ниже)
#   scripts/server.sh status                             задачи, загрузка, диск, проверка test_locked
#   scripts/server.sh pull [--dry-run]                   data/regen/R2/** и логи на Мак; существующее не перезаписывается
#   scripts/server.sh stop [--kill] <pid>                снять задачу по pid (только pid из data/run_server/*.pid)
#
# Адрес, пользователь, ключ — только из окружения или файла ~/.config/phd_server.env (вне репозитория), ключи:
#   PHD_SERVER_HOST, PHD_SERVER_USER, PHD_SERVER_KEY (путь к ключу), PHD_SERVER_PORT (22), PHD_REMOTE_DIR (PhD_Vol_trading — от $HOME сервера)
# Окружение важнее файла. Секретов в скрипте нет. `push --dry-run` сети не трогает: список строится find / git ls-files.
# Нужен GNU rsync ≥ 3 на Маке (brew install rsync; openrsync из macOS не годится) и rsync на сервере.
emulate -R zsh
setopt err_exit no_unset pipe_fail extended_glob

SELF=${0:A}
CODE_ROOT=${PHD_CODE_ROOT:-${SELF:h:h}}                           # откуда код: эта копия (git ls-files)
MAC_ROOT='/Users/nilysenok/Desktop/Старое/PhD_Vol_trading'      # основная копия: там данные (data/ не в git)
DATA_ROOT=${PHD_LOCAL_ROOT:-$MAC_ROOT}
ENV_FILE=${PHD_ENV_FILE:-$HOME/.config/phd_server.env}
LOCK='test_locked'
RUNDIR='data/run_server'                                         # на сервере: логи, pid, коды выхода
CODE_DIRS=(scripts models env moex_iss fastbt tests docs)        # код — только отслеживаемые git файлы этих каталогов
DATA_DIRS=(                                                      # данные — всё внутри, кроме *test_locked*
  data/regen
  data/calendar
  moex_discovery/data/external/moex_iss/indices_10m             # l2_lib.RVI10
  moex_discovery/data/final_v6/candles_10m                      # l2_lib.CANDLES
)
DATA_DIRS+=(${=PHD_PUSH_EXTRA:-})                                # дополнительные каталоги данных через пробел
PULL_DIRS=(data/regen/R2)
typeset -A TASKS
TASKS=(
  repro    "python3 scripts/server_repro_check.py --workers ${PHD_REPRO_WORKERS:-4}"
  repro_A  "python3 scripts/analysis/r2_repro_A.py --workers ${PHD_REPRO_WORKERS:-4}"
  checkenv "python3 env/check_env.py"
)

die() { print -ru2 -- "server.sh: $*"; exit 1 }
say() { print -r -- "$*" }

usage() {
  sed -n '2,20p' $SELF | sed 's/^# \{0,1\}//'
  exit ${1:-0}
}

# ---------------------------------------------------------------- конфигурация (без секретов в скрипте)
load_cfg() {
  local k v line
  if [[ -f $ENV_FILE ]]; then
    local perm=$(stat -f %Lp $ENV_FILE 2>/dev/null || stat -c %a $ENV_FILE 2>/dev/null || print 600)
    [[ $perm == *00 ]] || print -ru2 -- "server.sh: внимание: $ENV_FILE доступен не только вам (права $perm) — chmod 600"
    while IFS= read -r line || [[ -n $line ]]; do
      [[ $line == [[:space:]]#(\#*|) ]] && continue
      [[ $line =~ '^[[:space:]]*(export[[:space:]]+)?(PHD_SERVER_HOST|PHD_SERVER_USER|PHD_SERVER_KEY|PHD_SERVER_PORT|PHD_REMOTE_DIR)=(.*)$' ]] || {
        print -ru2 -- "server.sh: $ENV_FILE: строка пропущена (разрешены только PHD_SERVER_HOST/USER/KEY/PORT, PHD_REMOTE_DIR)"; continue; }
      k=$match[2]; v=$match[3]
      v=${v%%[[:space:]]#}; v=${v#\"}; v=${v%\"}; v=${v#\'}; v=${v%\'}
      [[ -z ${(P)k:-} ]] && typeset -g $k=$v                    # окружение важнее файла
    done < $ENV_FILE
  fi
  PHD_SERVER_PORT=${PHD_SERVER_PORT:-22}
  PHD_REMOTE_DIR=${PHD_REMOTE_DIR:-PhD_Vol_trading}
}

need_cfg() {
  load_cfg
  [[ -n ${PHD_SERVER_HOST:-} && -n ${PHD_SERVER_USER:-} && -n ${PHD_SERVER_KEY:-} ]] ||
    die "нет PHD_SERVER_HOST / PHD_SERVER_USER / PHD_SERVER_KEY — заполните $ENV_FILE (docs/server_r2.md)"
  PHD_SERVER_KEY=${PHD_SERVER_KEY/#\~/$HOME}
  [[ -r $PHD_SERVER_KEY ]] || die "ключ не читается: $PHD_SERVER_KEY"
  [[ $PHD_SERVER_PORT == <1-65535> ]] || die "PHD_SERVER_PORT: число"
  [[ $PHD_REMOTE_DIR == [A-Za-z0-9_./-]## && $PHD_REMOTE_DIR != *..* ]] || die "PHD_REMOTE_DIR: только латиница, цифры, _ . / - (без ~ и ..)"
  [[ $PHD_REMOTE_DIR == *$LOCK* ]] && die "PHD_REMOTE_DIR не может содержать $LOCK"
  TARGET="$PHD_SERVER_USER@$PHD_SERVER_HOST"
  SSH_OPTS=(-i $PHD_SERVER_KEY -p $PHD_SERVER_PORT -o BatchMode=yes -o IdentitiesOnly=yes
            -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
  RSH="ssh ${(j: :)${(@qq)SSH_OPTS}}"
  RD_Q=${(qq)PHD_REMOTE_DIR}
}

need_rsync() {
  RSYNC=${PHD_RSYNC:-}
  if [[ -z $RSYNC ]]; then
    local c
    for c in /opt/homebrew/bin/rsync /usr/local/bin/rsync $(whence -p rsync 2>/dev/null); do
      [[ -x $c ]] && { RSYNC=$c; break }
    done
  fi
  [[ -n $RSYNC ]] || die "нет rsync (brew install rsync)"
  $RSYNC --version 2>/dev/null | head -1 | grep -q 'rsync  *version  *3' ||
    die "$RSYNC — не GNU rsync 3 (openrsync не поддерживает нужные --files-from/--ignore-existing); brew install rsync или PHD_RSYNC=<путь>"
}

# удалённый bash-скрипт через stdin (секреты не передаются); PHD_PRINT_REMOTE=1 — только напечатать скрипт, без сети
rsh() {
  local script="set -eu -o pipefail; cd $RD_Q 2>/dev/null || { echo 'нет каталога $PHD_REMOTE_DIR на сервере — сначала push' >&2; exit 3; }
$1"
  if [[ ${PHD_PRINT_REMOTE:-0} == 1 ]]; then print -r -- $script; return 0; fi
  print -r -- $script | ssh $SSH_OPTS $TARGET 'bash -s'
}

# ---------------------------------------------------------------- списки файлов (локально, без сети)
build_push_lists() {   # $1 — список кода, $2 — список данных (пути от корней CODE_ROOT / DATA_ROOT)
  local code_out=$1 data_out=$2 d f
  [[ -d $DATA_ROOT/data/regen ]] || die "нет $DATA_ROOT/data/regen (PHD_LOCAL_ROOT)"
  # отслеживаемые и новые неигнорируемые файлы (git ls-files --cached --others --exclude-standard)
  git -C $CODE_ROOT -c core.quotePath=false ls-files --cached --others --exclude-standard -- $CODE_DIRS |
    grep -v -E "$LOCK|(^|/)\.env($|/)|^\.claude/|\.DS_Store$|__pycache__|\.pyc$" | sort -u > $code_out || true
  : > $data_out
  for d in $DATA_DIRS; do
    [[ $d == *$LOCK* ]] && die "каталог данных с $LOCK в списке push: $d"
    [[ -e $DATA_ROOT/$d ]] || { print -ru2 -- "server.sh: внимание: нет $DATA_ROOT/$d — пропуск"; continue; }
    find $DATA_ROOT/$d -name "*$LOCK*" -prune -o -type f ! -name .DS_Store ! -name '*.pyc' ! -name '.env' ! -path '*/__pycache__/*' -print |
      while IFS= read -r f; do print -r -- ${f#${(b)DATA_ROOT}/}; done >> $data_out
  done
  # вторая страховка: ни одной строки с test_locked
  if grep -q $LOCK $code_out $data_out; then
    grep -n $LOCK $code_out $data_out | head >&2
    die "в списке push оказался $LOCK — передача отменена"
  fi
}

list_size() {   # $1 — корень, $2 — список -> байты
  zmodload -F zsh/stat b:zstat
  local root=$1 f s=0 n
  local -a st
  while IFS= read -r f; do
    zstat -A st +size -- $root/$f 2>/dev/null && (( s += st[1] ))
  done < $2
  print $s
}

human() { local LC_NUMERIC=C b=$1; if (( b > 1073741824 )); then printf '%.1f ГБ' $(( b / 1073741824.0 )); else printf '%.1f МБ' $(( b / 1048576.0 )); fi }

code_version() {   # -> файл CODE_VERSION.txt
  local out=$1
  {
    print -r -- "push: $(date '+%F %T %Z') с $(hostname -s)"
    print -r -- "код: $CODE_ROOT"
    print -r -- "ветка: $(git -C $CODE_ROOT rev-parse --abbrev-ref HEAD 2>/dev/null)  коммит: $(git -C $CODE_ROOT rev-parse --short HEAD 2>/dev/null)  незакоммиченных файлов: $(git -C $CODE_ROOT status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
    print -r -- "данные: $DATA_ROOT ($DATA_DIRS)"
  } > $out
}

# ---------------------------------------------------------------- push
cmd_push() {
  local dry=0 upd=0 a
  for a in "$@"; do
    case $a in
      --dry-run) dry=1 ;;
      --update-data) upd=1 ;;
      *) die "push: неизвестный ключ $a" ;;
    esac
  done
  local tmp=$(mktemp -d "${TMPDIR:-/tmp}/phd_push.XXXXXX")
  trap "rm -f $tmp/code.txt $tmp/data.txt $tmp/CODE_VERSION.txt; rmdir $tmp 2>/dev/null" EXIT
  build_push_lists $tmp/code.txt $tmp/data.txt
  local nc=$(wc -l < $tmp/code.txt | tr -d ' ') nd=$(wc -l < $tmp/data.txt | tr -d ' ')
  local sc=$(list_size $CODE_ROOT $tmp/code.txt) sd=$(list_size $DATA_ROOT $tmp/data.txt)
  if (( dry )); then
    awk '{print "code\t" $0}' $tmp/code.txt
    awk '{print "data\t" $0}' $tmp/data.txt
    print -ru2 -- "push --dry-run: код $nc файлов ($(human $sc)) из $CODE_ROOT; данные $nd файлов ($(human $sd)) из $DATA_ROOT"
    print -ru2 -- "  исключено: *$LOCK* (find -prune + grep), .env, __pycache__, .DS_Store; данные — $([[ $upd == 1 ]] && print 'только более новые (--update)' || print 'без перезаписи на сервере (--ignore-existing)')"
    print -ru2 -- "  строк с $LOCK в списке: $(cat $tmp/code.txt $tmp/data.txt | grep -c $LOCK || true)"
    return 0
  fi
  need_cfg; need_rsync
  code_version $tmp/CODE_VERSION.txt
  say "push: код $nc файлов ($(human $sc)), данные $nd файлов ($(human $sd)) -> $TARGET:$PHD_REMOTE_DIR"
  ssh $SSH_OPTS $TARGET "mkdir -p $RD_Q"
  local -a common=(-a --human-readable --info=stats1,progress2 --exclude="*$LOCK*" --exclude='.env' -e $RSH)
  $RSYNC $common --files-from=$tmp/code.txt $CODE_ROOT/ $TARGET:$PHD_REMOTE_DIR/
  $RSYNC -a -e $RSH $tmp/CODE_VERSION.txt $TARGET:$PHD_REMOTE_DIR/CODE_VERSION.txt
  $RSYNC $common $([[ $upd == 1 ]] && print -- --update || print -- --ignore-existing) --files-from=$tmp/data.txt $DATA_ROOT/ $TARGET:$PHD_REMOTE_DIR/
  check_remote_locked || die "ПОСЛЕ ПЕРЕДАЧИ на сервере найден $LOCK — удалите его на сервере вручную и сообщите владельцу"
  say "push готов; на сервере $LOCK нет"
}

check_remote_locked() {   # 0 — на сервере нет ничего с test_locked в имени (в копии проекта и по пути Мака)
  local found
  found=$(rsh "find . -name '*$LOCK*' -print 2>/dev/null | head -20
if [ -e '$MAC_ROOT' ]; then find -L '$MAC_ROOT' -maxdepth 3 -name '*$LOCK*' -print 2>/dev/null | head -5; fi" || true)
  if [[ -n $found ]]; then
    print -ru2 -- "$found"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- setup
cmd_setup() {
  need_cfg
  local pass='' v
  for v in PHD_PY_TOOL PHD_SKIP_RUST PHD_ENV_DIR; do
    [[ -n ${(P)v:-} ]] && pass+="export $v=${(qq)${(P)v}}; "
  done
  rsh "[ -f env/setup_server.sh ] || { echo 'нет env/setup_server.sh — сначала scripts/server.sh push'; exit 1; }
mkdir -p $RUNDIR
$pass bash env/setup_server.sh 2>&1 | tee $RUNDIR/setup_\$(date +%Y%m%d_%H%M%S).log"
}

# ---------------------------------------------------------------- run
cmd_run() {
  (( $# >= 1 )) || die "run <имя> [команда…]"
  local name=$1; shift
  [[ $name =~ '^[A-Za-z0-9][A-Za-z0-9_.-]*$' ]] || die "имя задачи: латиница, цифры, _ . -"
  local -a cmd
  if (( $# == 0 )); then
    [[ -n ${TASKS[$name]:-} ]] || die "нет готовой задачи «$name» (есть: ${(kj:, :)TASKS}); иначе: run <имя> <команда…>"
    cmd=(${(z)TASKS[$name]})
  else
    [[ $1 == -- ]] && shift
    cmd=("$@")
  fi
  (( ${#cmd} )) || die "пустая команда"
  [[ "$cmd" == *$LOCK* ]] && die "команда упоминает $LOCK — на сервере такого нет и быть не должно"
  need_cfg
  rsh "set -- ${(j: :)${(@qq)cmd}}
name=${(qq)name}; pidf=$RUNDIR/\$name.pid; log=$RUNDIR/\$name.log
mkdir -p $RUNDIR
if [ -f \"\$pidf\" ] && kill -0 \"\$(cat \"\$pidf\")\" 2>/dev/null; then echo \"уже запущен: \$name pid \$(cat \"\$pidf\")\"; exit 1; fi
[ -f env/activate_r2.sh ] || { echo 'нет env/activate_r2.sh — сначала push и setup'; exit 1; }
. env/activate_r2.sh
if find . -name '*$LOCK*' -print -quit | grep -q .; then echo 'на сервере есть $LOCK — счёт не запускаю (docs/server_r2.md)'; exit 2; fi
rm -f \"\$pidf\" \"$RUNDIR/\$name.exit\"
{ echo \"# \$(date '+%F %T %Z') \$(hostname) nice 10; \$(sed -n 3p CODE_VERSION.txt 2>/dev/null)\"; printf '# команда:'; printf ' %q' \"\$@\"; echo; } > \"\$log\"
nohup setsid bash -c 'echo \$\$ > \"\$0\"; nice -n 10 \"\$@\"; echo \$? > \"\${0%.pid}.exit\"' \"\$pidf\" \"\$@\" >> \"\$log\" 2>&1 < /dev/null &
for i in \$(seq 1 50); do [ -s \"\$pidf\" ] && break; sleep 0.1; done
echo \"запущен \$name pid \$(cat \"\$pidf\"), лог $PHD_REMOTE_DIR/\$log\""
}

# ---------------------------------------------------------------- status
cmd_status() {
  need_cfg
  rsh "echo \"== \$(hostname)  \$(date '+%F %T %Z')\"
uptime; echo \"ядер: \$(nproc)\"; free -g 2>/dev/null | sed -n 1,2p || true; df -h . | tail -1
[ -f CODE_VERSION.txt ] && sed -n 1,3p CODE_VERSION.txt
echo '== задачи ($RUNDIR)'
for f in $RUNDIR/*.pid; do
  [ -e \"\$f\" ] || { echo '  нет'; break; }
  n=\$(basename \"\$f\" .pid); p=\$(cat \"\$f\")
  if kill -0 \"\$p\" 2>/dev/null; then st=\"работает \$(ps -o etime= -p \"\$p\" | tr -d ' ')\"
  elif [ -f $RUNDIR/\$n.exit ]; then st=\"завершён, код \$(cat $RUNDIR/\$n.exit)\"
  else st='не работает (снят или упал без кода)'; fi
  printf '  %-28s pid %-8s %s\n' \"\$n\" \"\$p\" \"\$st\"
  tail -n 2 $RUNDIR/\$n.log 2>/dev/null | cut -c1-200 | sed 's/^/      /' || true
done
if find . -name '*$LOCK*' -print -quit | grep -q .; then echo 'ВНИМАНИЕ: на сервере есть $LOCK:'; find . -name '*$LOCK*' -print | head; else echo '$LOCK на сервере: нет'; fi"
}

# ---------------------------------------------------------------- pull
cmd_pull() {
  local dry=0 a
  for a in "$@"; do
    case $a in
      --dry-run) dry=1 ;;
      *) die "pull: неизвестный ключ $a" ;;
    esac
  done
  need_cfg
  local tmp=$(mktemp -d "${TMPDIR:-/tmp}/phd_pull.XXXXXX")
  trap "rm -f $tmp/*.txt; rmdir $tmp 2>/dev/null" EXIT
  # удалённый список: размер<TAB>путь (GNU find), без test_locked
  rsh "for d in ${(j: :)${(@qq)PULL_DIRS}}; do [ -d \"\$d\" ] && find \"\$d\" -name '*$LOCK*' -prune -o -type f -printf '%s\t%p\n'; done; true" > $tmp/r2_remote.txt
  rsh "[ -d $RUNDIR ] && find $RUNDIR -maxdepth 1 -type f \\( -name '*.log' -o -name '*.exit' \\) -printf '%P\n'; true" > $tmp/logs.txt
  local sz f lsz
  local -a st
  zmodload -F zsh/stat b:zstat
  : > $tmp/r2_new.txt; : > $tmp/r2_skip.txt; : > $tmp/r2_conf.txt
  while IFS=$'\t' read -r sz f; do
    [[ $f == *$LOCK* ]] && die "в списке pull оказался $LOCK: $f"
    [[ $f == data/regen/R2/* && $f != *..* ]] || die "неожиданный путь из списка сервера: $f"
    if [[ -e $DATA_ROOT/$f ]]; then
      zstat -A st +size -- $DATA_ROOT/$f
      if [[ $st[1] == $sz ]]; then print -r -- $f >> $tmp/r2_skip.txt; else print -r -- "$f (Мак $st[1] Б, сервер $sz Б)" >> $tmp/r2_conf.txt; fi
    else
      print -r -- $f >> $tmp/r2_new.txt
    fi
  done < $tmp/r2_remote.txt
  local stamp=$(date +%Y%m%d_%H%M%S)
  local nn=$(wc -l < $tmp/r2_new.txt | tr -d ' ') ns=$(wc -l < $tmp/r2_skip.txt | tr -d ' ') nk=$(wc -l < $tmp/r2_conf.txt | tr -d ' ') nl=$(wc -l < $tmp/logs.txt | tr -d ' ')
  if (( dry )); then
    awk '{print "new\t" $0}' $tmp/r2_new.txt
    awk -v d="data/run/server/$stamp/" '{print "log\t" d $0}' $tmp/logs.txt
    awk '{print "КОНФЛИКТ\t" $0}' $tmp/r2_conf.txt
    print -ru2 -- "pull --dry-run: новых $nn, уже есть (тот же размер) $ns, конфликтов (есть, другой размер — НЕ перезаписываются) $nk, логов $nl -> data/run/server/$stamp/"
    return 0
  fi
  need_rsync
  if (( nn )); then
    $RSYNC -a --ignore-existing --exclude="*$LOCK*" --info=stats1 -e $RSH --files-from=$tmp/r2_new.txt $TARGET:$PHD_REMOTE_DIR/ $DATA_ROOT/
  fi
  if (( nl )); then
    mkdir -p $DATA_ROOT/data/run/server/$stamp
    $RSYNC -a --ignore-existing -e $RSH --files-from=$tmp/logs.txt $TARGET:$PHD_REMOTE_DIR/$RUNDIR/ $DATA_ROOT/data/run/server/$stamp/
  fi
  if [[ -d $DATA_ROOT/data/run/server/$stamp ]] && find $DATA_ROOT/data/run/server/$stamp -name "*$LOCK*" | grep -q .; then
    die "после pull найден $LOCK в забранных логах — проверьте вручную"
  fi
  (( nk )) && { print -ru2 -- "server.sh: конфликты (не перезаписаны):"; cat $tmp/r2_conf.txt >&2 }
  say "pull готов: новых $nn, пропущено $ns, конфликтов $nk, логи -> data/run/server/$stamp/"
}

# ---------------------------------------------------------------- stop
cmd_stop() {
  local sig=TERM
  [[ ${1:-} == --kill ]] && { sig=KILL; shift }
  local pid=${1:-}
  [[ $pid == <2-> ]] || die "stop [--kill] <pid>"
  need_cfg
  rsh "p=$pid
found=\$(grep -lx \"\$p\" $RUNDIR/*.pid 2>/dev/null | head -1 || true)
[ -n \"\$found\" ] || { echo \"pid \$p нет в $RUNDIR/*.pid — не трогаю\"; exit 1; }
if ! kill -0 \"\$p\" 2>/dev/null; then echo \"pid \$p уже не работает (\$found)\"; rm -f \"\$found\"; exit 0; fi
kill -$sig -- \"-\$p\" 2>/dev/null || kill -$sig \"\$p\"
for i in \$(seq 1 30); do kill -0 \"\$p\" 2>/dev/null || break; sleep 0.5; done
if kill -0 \"\$p\" 2>/dev/null; then echo \"pid \$p не завершился за 15 с; принудительно: scripts/server.sh stop --kill \$p\"; exit 1; fi
rm -f \"\$found\"; echo \"снят pid \$p (\$(basename \"\$found\" .pid), сигнал $sig группе процессов)\""
}

# ---------------------------------------------------------------- разбор
(( $# )) || usage 1
sub=$1; shift
case $sub in
  push) cmd_push "$@" ;;
  setup) cmd_setup "$@" ;;
  run) cmd_run "$@" ;;
  status) cmd_status "$@" ;;
  pull) cmd_pull "$@" ;;
  stop) cmd_stop "$@" ;;
  -h|--help|help) usage 0 ;;
  *) die "неизвестная подкоманда: $sub (push|setup|run|status|pull|stop)" ;;
esac

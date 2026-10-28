#!/usr/bin/env bash
# Рабочие копии агентов (урок terminal 10.10: 102 копии, 404 ГБ): scripts/kopii.sh spisok|chistka|disk|sozdat
#   spisok — все копии, их ветки, занятое место и копии сверх постоянного набора (их быть не должно)
#   chistka — cargo clean во всех копиях, кроме основной (кэш crates общий в ~/.cargo, его не трогает)
#   disk — свободное место; ниже порога (SDF_DISK_MIN, по умолчанию 50 ГБ) — красное
#   sozdat — завести постоянный набор копий через git worktree (один раз; агентам worktree add запрещён)
set -uo pipefail
cd "$(dirname "$0")/.."
KOPII="${SDF_KOPII:-$HOME/Developer/.sdf-kopii}"
NABOR="kod-1 kod-2 prov-1 pisar check"
case "${1:-spisok}" in
  spisok)
    git worktree list
    for d in "$KOPII"/*/; do
      [ -d "$d" ] || continue
      n=$(basename "$d")
      sz=$(du -sh "$d" 2>/dev/null | cut -f1)
      extra=""
      case " $NABOR " in *" $n "*) ;; *) extra=" ← СВЕРХ НАБОРА";; esac
      echo "$n · $(git -C "$d" rev-parse --abbrev-ref HEAD 2>/dev/null) · $sz$extra"
    done ;;
  chistka)
    for d in "$KOPII"/*/; do
      [ -d "$d" ] || continue
      if pgrep -f "$d" >/dev/null 2>&1; then echo "пропуск $d: в ней идёт процесс"; continue; fi
      (cd "$d" && cargo clean >/dev/null 2>&1 && echo "очищено $d")
    done ;;
  disk)
    MIN="${SDF_DISK_MIN:-50}"
    free=$(df -g / 2>/dev/null | awk 'NR==2{print $4}')
    [ -n "$free" ] || free=$(( $(df -k / | awk 'NR==2{print $4}') / 1024 / 1024 ))
    if [ "$free" -lt "$MIN" ]; then echo "КРАСНОЕ: свободно $free ГБ < $MIN ГБ — scripts/kopii.sh chistka"; exit 1; fi
    echo "свободно $free ГБ" ;;
  sozdat)
    mkdir -p "$KOPII"
    for n in $NABOR; do
      [ -d "$KOPII/$n" ] && { echo "есть $n"; continue; }
      git worktree add -q --detach "$KOPII/$n" origin/main && echo "создана $n"
    done ;;
  *) echo "scripts/kopii.sh spisok|chistka|disk|sozdat"; exit 2 ;;
esac

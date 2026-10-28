#!/usr/bin/env bash
# Д16.7 / Д16.9: цепочка расчётов (h=22, затем h=5), 4 воркера. HARX_DIR — копия harx.py агента B до слияния.
set -u
cd "$(dirname "$0")/../.."
export HARX_DIR=${HARX_DIR:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-acdd56e718522aa6c/scripts/analysis}
LG=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D16_79/logs
mkdir -p "$LG"
for h in ${HS:-22 5}; do
  python3 -W ignore scripts/analysis/d16_79.py --h $h --task pwlin --steps 1-9 > "$LG/pwlin_h$h.log" 2>&1
  python3 -W ignore scripts/analysis/d16_79.py --h $h --task bf --steps 1-9 > "$LG/bf_h$h.log" 2>&1
  python3 -W ignore scripts/analysis/d16_79.py --h $h --task pw --steps 1-9 > "$LG/pw_h$h.log" 2>&1
done
date > "$LG/chain.done"

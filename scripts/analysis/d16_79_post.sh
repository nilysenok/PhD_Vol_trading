#!/usr/bin/env bash
# Д16.7 / Д16.9: выходы кандидатов, гибрид (c1-c4) и сводка. Аргументы: h [основа]. База — та же точка V9 на HAR-J.
# Без основы: D12_h22 / F2m_h5 / D12_h1_B_b1, имена D16_7_h{h}, D16_9_h{h}; с основой (D12_h5, L2v3_h5): D16_7_<основа>, D16_9_<основа>.
set -u
cd "$(dirname "$0")/../.."
h=$1; B=${2:-}
case $h in 22) base=D12_h22;; 5) base=F2m_h5;; 1) base=D12_h1_B_b1;; esac
if [ -n "$B" ]; then base=$B; n7=D16_7_$B; n9=D16_9_$B; export D16_BASE=$B; else n7=D16_7_h$h; n9=D16_9_h$h; fi
export HARX_DIR=${HARX_DIR:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-acdd56e718522aa6c/scripts/analysis}
M=scripts/analysis/d16_79.py
python3 -W ignore $M --h $h --task final --final bf stop $n7
python3 -W ignore $M --h $h --task final --final bf 0 ${n7}_K0
python3 -W ignore $M --h $h --task final --final pw 0 $n9
scripts/analysis/d16_79_gibrid.sh $base:$h ${n7}_K0:$h $n7:$h $n9:$h
python3 -W ignore $M --h $h --task summ > "/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D16_79/logs/summ${B:+_$B}_h$h.log" 2>&1

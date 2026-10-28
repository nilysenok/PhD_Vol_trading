#!/usr/bin/env bash
# Д16.7 / Д16.9, вторая очередь (после указания 01:19): основы D12_h5 и L2v3_h5 (h=5), затем гибрид и сводки по всем выходам.
set -u
cd "$(dirname "$0")/../.."
export HARX_DIR=${HARX_DIR:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/.claude/worktrees/agent-acdd56e718522aa6c/scripts/analysis}
LG=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D16_79/logs
M=scripts/analysis/d16_79.py
for B in D12_h5 L2v3_h5; do
  D16_BASE=$B python3 -W ignore $M --h 5 --task bf --steps 1-9 > "$LG/bf_${B}.log" 2>&1
  D16_BASE=$B python3 -W ignore $M --h 5 --task pw --steps 1-9 > "$LG/pw_${B}.log" 2>&1
done
scripts/analysis/d16_79_post.sh 5 D12_h5
scripts/analysis/d16_79_post.sh 5 L2v3_h5
scripts/analysis/d16_79_post.sh 5
scripts/analysis/d16_79_post.sh 22
scripts/analysis/d16_79_post.sh 1
date > "$LG/all.done"

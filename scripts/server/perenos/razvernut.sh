#!/usr/bin/env bash
# Развёртывание на сервере (Ж-133 п. 3): ./razvernut.sh из каталога perenos/. Без вопросов, стоп на первой ошибке.
#   1) SHA256SUMS каталога; 2) Python 3.9; 3) .venv и пакеты с пинами Мака (только колёса); 4) код из tar в kod/<sha256>,
#   только чтение; 5) данные по манифесту (sha256, max_data ≤ 2021-12-31, лишних файлов нет). Конец — строка OK и версии.
# PERENOS_PYTHON — путь к python3.9, если его нет в PATH. PERENOS_SYSTEM_SITE=1 — только локальная проверка на Маке
# (venv видит пакеты Мака; вместе с PIP_NO_INDEX=1 pip не ходит в сеть).
. "$(dirname "$0")/obshchee.sh"

say "1/5 сверка SHA256SUMS каталога $P"
[ -f SHA256SUMS ] || fail "нет SHA256SUMS"
sha_check "$P" SHA256SUMS >/dev/null || fail "SHA256SUMS не сходится — каталог скопирован не целиком или изменён; повторите rsync"

say "2/5 Python 3.9"
PYB=${PERENOS_PYTHON:-$(command -v python3.9 || true)}
[ -n "$PYB" ] && [ -x "$PYB" ] || fail "нет python3.9 (Мак — 3.9.6). Поставьте: apt install python3.9 python3.9-venv (deadsnakes) или uv python install 3.9.6; путь — PERENOS_PYTHON=…"
v=$("$PYB" --version 2>&1)
case "$v" in "Python 3.9."*) say "   $PYB: $v" ;; *) fail "$PYB: $v, нужен 3.9.x" ;; esac

say "3/5 .venv и пакеты из requirements_server.txt (только колёса, без зависимостей сверх пинов)"
if [ ! -x "$PY" ]; then
  if [ "${PERENOS_SYSTEM_SITE:-0}" = 1 ]; then "$PYB" -m venv --system-site-packages .venv; else "$PYB" -m venv .venv; fi
fi
[ "${PERENOS_SYSTEM_SITE:-0}" = 1 ] || export PYTHONNOUSERSITE=1
"$PY" -m pip install --disable-pip-version-check -q --only-binary=:all: --no-deps -r requirements_server.txt \
  || fail "pip: нет колеса под эту платформу или нет сети — вывод выше; версии не менять, прислать вывод дирижёру"
"$PY" versii.py requirements_server.txt | tee logs_versii.txt
grep -q '^OK versii' logs_versii.txt || fail "версии пакетов ≠ пинам Мака"

say "4/5 код: kod_$KOD_SHORT.tar.gz -> kod/$KOD_SHA (только чтение)"
[ "$(sha_file "$KOD_TAR")" = "$KOD_SHA" ] || fail "sha256 tar ≠ KOD"
if [ ! -d "$KOD_DIR" ]; then
  tmp="$P/kod/.raspakovka_$$"
  mkdir -p "$tmp"
  tar -xzf "$KOD_TAR" -C "$tmp" --strip-components=1
  [ -z "$(find "$tmp" \( -path '*/data/*' -o -name '*test_locked*' -o -name '*test1_2022_2024*' \) -print -quit)" ] || fail "в tar кода есть data/ или test_locked"
  nakl=$(kv nakladka)                   # файлы манифеста, которые fastbt читает от корня кода (KOD: nakladka)
  if [ -n "$nakl" ]; then
    awk -F'\t' 'NR > 1 && NF { print (NF == 7 ? $6 : $1) }' "$MANIFEST" | { grep -E "$nakl" || true; } \
      | while IFS= read -r p; do mkdir -p "$tmp/$(dirname "$p")"; cp "data/$p" "$tmp/$p"; done
  fi
  sha_spisok "$tmp" > "$KOD_SUMS"
  mv "$tmp" "$KOD_DIR"
  chmod -R a-w "$KOD_DIR"
fi
kod_cel
say "   файлов кода $(wc -l < "$KOD_SUMS" | tr -d ' '), каталог только для чтения"

say "5/5 данные: data/ по manifest_dannyh.tsv"
"$PY" manifest_proverka.py --manifest "$MANIFEST" --koren data --strogo || fail "данные не совпадают с манифестом"
mkdir -p ledger tables logs proverka cache
mv logs_versii.txt logs/versii.txt
echo "OK razvernut: $(sed -n 's/^OK versii: //p' logs/versii.txt | cut -d';' -f1); kod $KOD_SHORT (tar sha256 ${KOD_SHA:0:16}…)"

#!/usr/bin/env bash
# Сборка кода ветки для полного pytest НА СЕРВЕРЕ (Ж-160 п. 2). Запускается на Маке, ничего не считает:
#   scripts/server/sobrat_testy.sh <ветка|коммит>
# -> testy_out/ (TESTY_OUT): testy_<коммит12>.tar.gz (git archive: fastbt/, tests/, scripts/, conftest.py, pytest.ini,
#    pyproject.toml, setup.cfg — что есть в коммите; корень в tar — testy/), SHA256SUMS_testy (имя НЕ SHA256SUMS: файл
#    едет в тот же perenos/, а SHA256SUMS пакета переноса trogat нельзя), iskluchenno.txt (что вырезано фильтром 2022+).
# Фильтр 2022+ — тот же список, что у пакета: scripts/server/tar_iskluchit_2022plus.txt (из коммита); плюс test_locked,
# test1_2022_2024, __pycache__, numba-кэш. Данные (data/) в tar не идут: на сервере подключаются из perenos/data.
# Окружение: TESTY_OUT (по умолчанию testy_out/ в корне), TESTY_REPO (корень репозитория), TESTY_MAX_MB (50).
# Отказ: каталог назначения не пуст; нет списка исключений; tar больше TESTY_MAX_MB; в tar попал файл из исключений.
set -euo pipefail
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
sha_file() { shasum -a 256 "$1" | cut -d' ' -f1; }

REPO=${TESTY_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
[ $# -eq 1 ] || fail "использование: scripts/server/sobrat_testy.sh <ветка|коммит>"
FULL=$(git -C "$REPO" rev-parse --verify "$1^{commit}") || fail "нет коммита $1"
SHORT=${FULL:0:12}
OUT=${TESTY_OUT:-$REPO/testy_out}
[ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT")" ] || fail "$OUT не пуст — укажите новый TESTY_OUT"
mkdir -p "$OUT"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/testy_mak_$SHORT.XXXXXX")
say "коммит $FULL; каталог $OUT"

git -C "$REPO" ls-tree -r --name-only "$FULL" -- fastbt tests scripts conftest.py pytest.ini pyproject.toml setup.cfg \
  | grep -v -E '(^|/)__pycache__/|\.py[co]$|\.nb[ic]$|test_locked|test1_2022_2024' > "$TMP/vse.txt" || true
git -C "$REPO" show "$FULL:scripts/server/tar_iskluchit_2022plus.txt" > "$TMP/iskl.txt" 2>/dev/null \
  || fail "нет scripts/server/tar_iskluchit_2022plus.txt в коммите $SHORT"
grep -v -E '^(#|$)' "$TMP/iskl.txt" > "$TMP/iskl_puti.txt" || true
grep -v -x -F -f "$TMP/iskl_puti.txt" "$TMP/vse.txt" > "$TMP/belyj.txt" || true          # FILTR2022
grep -x -F -f "$TMP/iskl_puti.txt" "$TMP/vse.txt" > "$OUT/iskluchenno.txt" || true
[ -s "$TMP/belyj.txt" ] || fail "в коммите $SHORT нет ни tests/, ни fastbt/"
spisok=()
while IFS= read -r f; do spisok+=("$f"); done < "$TMP/belyj.txt"

TAR="$OUT/testy_$SHORT.tar.gz"
git -C "$REPO" archive --format=tar --prefix=testy/ "$FULL" -- "${spisok[@]}" | gzip -n -9 > "$TAR"
MAXB=$(( ${TESTY_MAX_MB:-50} * 1048576 ))
[ "$(wc -c < "$TAR")" -lt "$MAXB" ] || fail "tar $(du -h "$TAR" | cut -f1) ≥ ${TESTY_MAX_MB:-50} МБ"
tar -tzf "$TAR" | sed 's|^testy/||' | grep -x -F -f "$TMP/iskl_puti.txt" > "$TMP/plohie.txt" || true
[ ! -s "$TMP/plohie.txt" ] || fail "в tar попали пути из tar_iskluchit_2022plus.txt: $(head -n 3 "$TMP/plohie.txt" | tr '\n' ' ')"
tar -tzf "$TAR" | grep -E 'test_locked|test1_2022_2024' >/dev/null && fail "в tar test_locked/test1_2022_2024" || true
tar -tzf "$TAR" | grep -q '^testy/tests/' && tar -tzf "$TAR" | grep -q '^testy/fastbt/' || fail "в tar нет tests/ или fastbt/"
git -C "$REPO" show "$FULL:scripts/server/perenos/testy.sh" > "$OUT/testy.sh" 2>/dev/null \
  || fail "нет scripts/server/perenos/testy.sh в коммите $SHORT"
chmod +x "$OUT/testy.sh"
(cd "$OUT" && shasum -a 256 "$(basename "$TAR")" > SHA256SUMS_testy)
say "   $(tar -tzf "$TAR" | grep -vc '/$') файлов, $(du -h "$TAR" | cut -f1), sha256 $(sha_file "$TAR")"
say "   вырезано фильтром 2022+: $(wc -l < "$OUT/iskluchenno.txt" | tr -d ' ') (список — $OUT/iskluchenno.txt)"
say "   нетекстовые фикстуры tests/ (едут как есть, проверьте даты): $(grep -E '^tests/.*\.(npy|npz|csv|parquet|pkl|json)$' "$TMP/belyj.txt" | tr '\n' ' ')"
rm -rf "$TMP"

echo "OK testy: $OUT; коммит $SHORT"
echo "   rsync -av $OUT/testy_$SHORT.tar.gz $OUT/SHA256SUMS_testy $OUT/testy.sh \"\$SRV\":perenos/"
echo "   ssh \"\$SRV\" 'cd perenos && ./testy.sh $SHORT'"
echo "   итог потом (Мак): scripts/server/perenos/zabrat.sh --testy \"\$SRV\":perenos <каталог на Маке>   # testy/$SHORT/itog.txt"

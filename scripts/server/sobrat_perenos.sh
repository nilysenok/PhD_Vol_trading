#!/usr/bin/env bash
# Сборка каталога переноса на Маке (Ж-131 п. 5, Ж-133 п. 1), одна команда:
#   scripts/server/sobrat_perenos.sh <коммит>
# -> perenos/: kod_<коммит12>.tar.gz (git archive коммита: fastbt/, scripts/server/, env/requirements_server.txt, справочные
#    csv ≤ 2021 из docs/issledovaniya/, которые читает fastbt; без data/),
#    VERSIYA (коммит, sha256 tar), requirements_server.txt, manifest_dannyh.tsv, data/ (файлы манифеста, sha256 сверены),
#    kuby/ (yaml кубов из коммита), etalony_<коммит12>.npz (якорь A_fix + 200 точек, считается здесь из кода tar и data/),
#    razvernut.sh, proverka_yakor.sh, zapusk_sobol.sh, zabrat.sh, obshchee.sh, manifest_proverka.py, versii.py,
#    spisok_perenosa.txt (для rsync --files-from), SHA256SUMS (всё, кроме data/ — её сверяет манифест).
# Окружение: PERENOS_OUT (perenos/ в корне копии), PERENOS_MANIFEST (docs/server/manifest_dannyh_2021.tsv),
#   PERENOS_DANNYE (основная копия — откуда данные), PERENOS_ETALONY_PROCS (6), PERENOS_ETALONY_TOCHEK (200).
# Отказ: каталог назначения не пуст; манифест с test_locked / test1_2022_2024 / max_data > 2021-12-31 / sha256 не сходится.
set -euo pipefail
say() { printf '%s  %s\n' "$(date '+%F %T')" "$*"; }
fail() { printf 'FAIL: %s\n' "$*"; exit 1; }
sha_file() { shasum -a 256 "$1" | cut -d' ' -f1; }

REPO=$(cd "$(dirname "$0")/../.." && pwd)
[ $# -eq 1 ] || fail "использование: scripts/server/sobrat_perenos.sh <коммит>"
FULL=$(git -C "$REPO" rev-parse --verify "$1^{commit}") || fail "нет коммита $1"
SHORT=${FULL:0:12}
OUT=${PERENOS_OUT:-$REPO/perenos}
MANIFEST=${PERENOS_MANIFEST:-$REPO/docs/server/manifest_dannyh_2021.tsv}
DANNYE=${PERENOS_DANNYE:-/Users/nilysenok/Desktop/Старое/PhD_Vol_trading}
[ -f "$MANIFEST" ] || fail "нет манифеста $MANIFEST"
[ ! -e "$OUT" ] || [ -z "$(ls -A "$OUT")" ] || fail "$OUT не пуст — укажите новый PERENOS_OUT"
mkdir -p "$OUT"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/perenos_mak_$SHORT.XXXXXX")
say "коммит $FULL; каталог $OUT; рабочий каталог Мака $TMP"

say "1/6 tar кода (git archive $SHORT)"
TAR="$OUT/kod_$SHORT.tar.gz"
# БЕЛЫЙ список (в git есть файлы с данными 2022+): fastbt/**.py|yaml без tests/ и designs/; из designs/ — v5/, zh126/,
# progon/*.yaml (кубы замены progon_sobol) и sweep/*.yaml (базовая точка A_fix ekviv_20, сетки S{k}_setka_e1b — их читает
# оценщик); scripts/server/*.py|sh; requirements.
# Таблицы (csv/parquet) в tar не идут: справочные ≤ 2021 едут усечёнными копиями манифеста (накладка ниже).
git -C "$REPO" ls-tree -r --name-only "$FULL" -- fastbt scripts/server env/requirements_server.txt | grep -E \
  '^(fastbt/.*\.(py|yaml)|scripts/server/[^/]*\.(py|sh)|scripts/server/perenos/[^/]*\.sh|env/requirements_server\.txt)$' \
  | grep -v -E '^fastbt/(tests|designs)/' > "$TMP/belyj.txt" || true
git -C "$REPO" ls-tree -r --name-only "$FULL" -- fastbt/designs | grep -E \
  '^fastbt/designs/((v5|zh126)/.*\.(py|yaml)|(progon|sweep)/[^/]*\.yaml)$' >> "$TMP/belyj.txt" || true
git -C "$REPO" show "$FULL:scripts/server/tar_iskluchit_2022plus.txt" > "$TMP/iskl.txt" || fail "нет scripts/server/tar_iskluchit_2022plus.txt в коммите"
plohie=$(grep -v '^#' "$TMP/iskl.txt" | grep -Fx -f - "$TMP/belyj.txt" || true)
[ -z "$plohie" ] || fail "в белый список попали пути из tar_iskluchit_2022plus.txt: $(echo "$plohie" | head -3 | tr '\n' ' ')"
spisok=()
while IFS= read -r f; do spisok+=("$f"); done < "$TMP/belyj.txt"
git -C "$REPO" archive --format=tar --prefix=kod/ "$FULL" -- "${spisok[@]}" | gzip -n -9 > "$TAR"
! tar -tzf "$TAR" | grep -E '^kod/data/|test_locked|test1_2022_2024|\.(csv|parquet|npz|pkl)$' >/dev/null \
  || fail "в tar попали data/, test_locked или таблицы"
! tar -tzf "$TAR" | grep -E '__pycache__|\.py[co]$|\.nb[ic]$' >/dev/null \
  || fail "в tar попали __pycache__ или numba-кэш (привязан к машине)"
TAR_SHA=$(sha_file "$TAR")
tar -xzf "$TAR" -C "$TMP"
K="$TMP/kod"
for f in scripts/server/perenos/razvernut.sh scripts/server/manifest_proverka.py env/requirements_server.txt; do
  [ -f "$K/$f" ] || fail "в коммите $SHORT нет $f — соберите с коммита, где есть scripts/server/"
done
say "   $(tar -tzf "$TAR" | grep -vc '/$') файлов, sha256 $TAR_SHA"

say "2/6 данные по манифесту -> $OUT/data (sha256 источника и копии)"
cp "$MANIFEST" "$OUT/manifest_dannyh.tsv"
python3 "$K/scripts/server/manifest_proverka.py" --manifest "$OUT/manifest_dannyh.tsv" --koren "$OUT/data" \
  --istochnik "$DANNYE" ${PERENOS_DANNYE_DOP:+--istochnik "$PERENOS_DANNYE_DOP"} --strogo --spisok "$TMP/spisok_dannyh.txt" \
  ${PERENOS_DO:+--do "$PERENOS_DO" --otmetka "${PERENOS_OTMETKA:-$REPO/docs/resheniya/wf_zamok_snyat.json}"} || fail "манифест отвергнут — перенос не собран"

# fastbt читает часть таблиц от корня КОДА, не от FASTBT_MAIN (etap1_data: ROOT/data/calendar/cbr.csv; ставка
# fastbt/rates/; dividendy, msp, fazy: docs/issledovaniya/): эти файлы манифеста (усечённые копии) кладутся в распакованный
# код — здесь и в razvernut.sh тем же правилом, до суммы и chmod; сверены манифестом
NAKLADKA='^(data/calendar|fastbt/rates|docs/issledovaniya)/'
{ grep -E "$NAKLADKA" "$TMP/spisok_dannyh.txt" || true; } | while IFS= read -r p; do mkdir -p "$K/$(dirname "$p")"; cp "$OUT/data/$p" "$K/$p"; done

say "3/6 скрипты, requirements, кубы"
cp "$K"/scripts/server/perenos/*.sh "$K/scripts/server/manifest_proverka.py" "$K/scripts/server/versii.py" "$OUT/"
cp "$K/env/requirements_server.txt" "$OUT/"
# все пины = версиям этого Мака (иначе эталон не сравним); numba и llvmlite обязаны быть в пинах
python3 "$K/scripts/server/versii.py" "$OUT/requirements_server.txt" || fail "пины requirements_server.txt ≠ версиям пакетов на Маке"
grep -q '^numba==' "$OUT/requirements_server.txt" && grep -q '^llvmlite==' "$OUT/requirements_server.txt" || fail "в пинах нет numba/llvmlite"
chmod +x "$OUT"/*.sh
for d in progon v5 v6; do
  ls "$K/fastbt/designs/$d/"*.yaml >/dev/null 2>&1 || continue
  mkdir -p "$OUT/kuby/$d"
  cp "$K/fastbt/designs/$d/"*.yaml "$OUT/kuby/$d/"
done
slit=0
git -C "$REPO" merge-base --is-ancestor "$FULL" origin/kanon-v22-pereschet 2>/dev/null && slit=1
# потоки и слой numba: одни и те же при эталоне на Маке и во всех скриптах сервера (obshchee.sh читает их из VERSIYA)
NT=${PERENOS_NUMBA_THREADS:-1}; NL=${PERENOS_NUMBA_LAYER:-workqueue}
printf 'commit=%s\ncommit_short=%s\ntar=kod_%s.tar.gz\ntar_sha256=%s\nkod_slit=%s\nnakladka=%s\nnumba_threads=%s\nnumba_layer=%s\nsobran=%s\n' \
  "$FULL" "$SHORT" "$SHORT" "$TAR_SHA" "$slit" "$NAKLADKA" "$NT" "$NL" "$(date '+%F %T %Z')" > "$OUT/VERSIYA"
say "   numba: NUMBA_NUM_THREADS=$NT NUMBA_THREADING_LAYER=$NL (на эталоне и на сервере)"

say "4/6 эталоны Мака: якорь A_fix × 6 Sk на train4 + ${PERENOS_ETALONY_TOCHEK:-200} точек (код tar, данные $OUT/data)"
mkdir -p "$TMP/cache"
(cd "$K" && env NUMBA_NUM_THREADS="$NT" NUMBA_THREADING_LAYER="$NL" PYTHONPATH="$K" FASTBT_MAIN="$OUT/data" FASTBT_BARS1M="$OUT/data/data/regen/D25/bars1m" \
  FASTBT_NO_CACHE=1 FASTBT_CACHE="$TMP/cache/fastbt" FASTBT_E1_CACHE="$TMP/cache/etap1" FASTBT_FS_CACHE="$TMP/cache/fs" \
  FASTBT_BARS5M_CACHE="$TMP/cache/bars5m" NUMBA_CACHE_DIR="$TMP/cache/numba" PYTHONPYCACHEPREFIX="$TMP/cache/pyc" \
  python3 "$K/scripts/server/etalony_mak.py" --out "$OUT/etalony_$SHORT.npz" \
  --tochek "${PERENOS_ETALONY_TOCHEK:-200}" --procs "${PERENOS_ETALONY_PROCS:-6}") > "$TMP/etalony.log" 2>&1 \
  || { tail -n 15 "$TMP/etalony.log"; fail "эталоны не собраны (лог $TMP/etalony.log)"; }
grep '^etalony' "$TMP/etalony.log"
[ -f "$OUT/etalony_$SHORT.npz" ] || fail "нет $OUT/etalony_$SHORT.npz"
find "$OUT/data" -name '.*.lock' -size 0 -delete          # flock daily_1850 при счёте эталонов, не данные

say "5/6 SHA256SUMS и список rsync"
(cd "$OUT" && find . -path ./data -prune -o -type f ! -name SHA256SUMS -print | sort | while IFS= read -r f; do
  printf '%s  %s\n' "$(sha_file "$f")" "$f"; done) > "$TMP/SHA256SUMS"
mv "$TMP/SHA256SUMS" "$OUT/SHA256SUMS"
{ (cd "$OUT" && find . -path ./data -prune -o -type f -print | sed 's|^\./||' | sort); sed 's|^|data/|' "$TMP/spisok_dannyh.txt"; } > "$TMP/spisok"
mv "$TMP/spisok" "$OUT/spisok_perenosa.txt"
python3 "$K/scripts/server/manifest_proverka.py" --manifest "$OUT/manifest_dannyh.tsv" --koren "$OUT/data" --strogo \
  ${PERENOS_DO:+--do "$PERENOS_DO" --otmetka "${PERENOS_OTMETKA:-$REPO/docs/resheniya/wf_zamok_snyat.json}"} >/dev/null \
  || fail "после эталонов в data/ появились файлы вне манифеста"

say "6/6 готово"
echo "   перенос на сервер (одна команда; user@host — ваш):"
echo "   rsync -av --files-from=$OUT/spisok_perenosa.txt $OUT/ user@host:perenos/"
echo "   на сервере: cd perenos && ./razvernut.sh && ./proverka_yakor.sh"
nd=$(wc -l < "$TMP/spisok_dannyh.txt" | tr -d ' ')
echo "OK perenos: $OUT; kod $SHORT (tar sha256 $TAR_SHA); данных $nd файлов, $(du -sh "$OUT/data" | cut -f1); kod_slit=$slit"

#!/usr/bin/env bash
# Обновление КОДА на сервере без повторной передачи данных (1,1 ГБ). Пакет perenos/ уже развёрнут (./razvernut.sh).
#
# Владельцу — 4 команды:
#   Мак:     PERENOS_PREDYDUSHCHIJ=<прежний perenos/> scripts/server/sobrat_perenos.sh <коммит>   (PERENOS_OUT — новый каталог)
#   Мак:     rsync -av --files-from=<новый>/spisok_obnovleniya.txt <новый>/ user@host:perenos/
#   Сервер:  cd perenos && ./obnovit_kod.sh
#   Сервер:  ./proverka_yakor.sh      (новый код -> новая отметка proverka/OK; без неё zapusk_sobol.sh откажет)
# Откат: старый kod/<sha> не удаляется; вернуть прежние VERSIYA, kod_<sha>.tar.gz, etalony_<sha>.npz, SHA256SUMS и повторить
# ./obnovit_kod.sh + ./proverka_yakor.sh.
#
# Что делает: 1) SHA256SUMS каталога; 2) отказ при живом прогоне (logs/sobol_*.pid); 3) .venv не пересоздаётся: пины
# requirements_server.txt обязаны совпасть с установленными версиями, иначе отказ -> ./razvernut.sh; 4) данные: sha манифеста тот же,
# что в отметке proverka/OK -> не трогаются; иначе сверка data/ по новому манифесту (не совпало — отказ); 5) новый код -> kod/<sha>
# (только чтение), старые kod/* не трогаются. Данные из spisok_obnovleniya.txt едут только если манифест изменился.
. "$(dirname "$0")/obshchee.sh"

say "1/5 сверка SHA256SUMS каталога $P"
[ -f SHA256SUMS ] || fail "нет SHA256SUMS"
sha_check "$P" SHA256SUMS >/dev/null || fail "SHA256SUMS не сходится — rsync прошёл не целиком; повторите rsync с spisok_obnovleniya.txt"
[ "$(sha_file "$KOD_TAR")" = "$KOD_SHA" ] || fail "sha256 tar ≠ VERSIYA"

say "2/5 живой прогон"
for f in "$P"/logs/sobol_*.pid; do
  [ -f "$f" ] || continue
  ! kill -0 "$(cat "$f")" 2>/dev/null || fail "прогон идёт: pid $(cat "$f") ($(basename "$f" .pid)) — дождитесь конца или kill, потом обновляйте"
done

say "3/5 .venv и requirements_server.txt"
[ -x "$PY" ] || fail "нет .venv — это первое развёртывание: ./razvernut.sh"
"$PY" versii.py requirements_server.txt 2>/dev/null | grep -q '^OK versii' \
  || fail "requirements изменились (версии в .venv ≠ пинам нового пакета) — обновление кода не годится; запустите ./razvernut.sh"

say "4/5 данные"
msha=$(sha_file "$MANIFEST"); staryj=""
for o in "$P/proverka/OK" "$P/proverka/OK_r2"; do
  [ -f "$o" ] && staryj="$staryj $(tr ' ' '\n' < "$o" | sed -n 's/^dannye=//p')"
done
case " $staryj " in
  *" $msha "*) say "   манифест тот же (sha ${msha:0:16}…) — data/ не трогаю" ;;
  *) say "   манифест изменился или отметки нет — сверка data/ по новому манифесту"
     "$PY" manifest_proverka.py --manifest "$MANIFEST" --koren data --strogo \
       ${PERENOS_DO:+--do "$PERENOS_DO" --otmetka "${PERENOS_OTMETKA:-wf_zamok_snyat.json}"} >/dev/null \
       || fail "данные не совпадают с новым манифестом — передайте данные (сборщик с PERENOS_PREDYDUSHCHIJ) или ./razvernut.sh" ;;
esac

say "5/5 код: kod_$KOD_SHORT.tar.gz -> kod/$KOD_SHA (только чтение; прежние kod/* остаются)"
raspakovat_kod
kod_cel
otmetka_paketa
echo "OK obnovit_kod: kod $KOD_SHORT (tar sha256 ${KOD_SHA:0:16}…). Дальше обязательно: ./proverka_yakor.sh (новая отметка proverka/OK)"

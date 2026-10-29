# Отметка снятого замка в пакете переноса (Ж-136; решение дирижёра, класс B 29.10). Подключается `. otmetka_paketa.sh`, сам не запускается.
# Отметку в репозитории ставит владелец; здесь она только копируется: в tar кода по тому же пути docs/resheniya/wf_zamok_snyat.json
# (попадает в kod/<sha>.SHA256SUMS на сервере), в корень пакета (его читает zapusk_sobol.sh wf) и в SHA256SUMS пакета.
# Без отметки пакет собирается как раньше (wf на сервере откажет).

otmetka_v_tar() {   # $1 — несжатый tar (префикс kod/), $2 — файл отметки, может не существовать; пусто/нет файла — tar не меняется
  [ -s "${2:-}" ] || return 0
  local d; d=$(mktemp -d "${TMPDIR:-/tmp}/otmetka.XXXXXX")
  mkdir -p "$d/kod/docs/resheniya"
  cp "$2" "$d/kod/docs/resheniya/wf_zamok_snyat.json"
  tar -rf "$1" -C "$d" kod/docs/resheniya/wf_zamok_snyat.json
  rm -rf "$d"
}

summy_paketa() {   # $1 — каталог пакета: суммы всех файлов кроме data/ и самого SHA256SUMS (формат sha256sum, пути ./…)
  (cd "$1" && find . -path ./data -prune -o -type f ! -name SHA256SUMS -print | sort | while IFS= read -r f; do
    printf '%s  %s\n' "$(sha_file "$f")" "$f"; done)
}

# GATES — T4, доработки скрипта открытия после #313 (дирижёр 27.10, плюс #317 Н5), arhitektor, 27.10

Ошибка процесса: файл гейтов написан после правок, а не до (polnota.md п. 1). Требования взяты из сообщений дирижёра дословно, гейты задним числом не подгонялись: каждый CHECK перезапущен в 19:20.

Только `tests/test_otkrytie_gl4.py`. `pytest tests` целиком и `tests/test_a_weights_future.py` не запускались (запрет с 19:15).

## G1 — все тесты скрипта открытия проходят
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q
EXPECT: 0 failed
EVIDENCE: 63 passed in 4.37s

## G2 — «да»: время заголовка записи журнала строго позже zamorozheno_v и равно data; строка содержит sha_spec; подделка через старую запись отвергается
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k bez_da
EXPECT: passed (в том числе test_bez_da_staraya_zapis_poddelka, test_bez_da_staroe_da_bez_sha_spec)
EVIDENCE: 14 passed, 49 deselected

## G3 — daty обязательны; Ледуа–Вольф по совпавшим датам; равная длина со сдвинутыми датами — отказ; перестановка порядка — то же Δ
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k daty
EXPECT: 3 passed
EVIDENCE: 3 passed, 60 deselected

## G4 — проверка сдвига в две части: 2018–2021 до замка (провал не тратит открытие, функция видит только vhody_dev), 2022+ после замка; каждая метка входа проверена
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k "proverka_sdviga or proverki"
EXPECT: passed
EVIDENCE: 5 passed, 58 deselected

## G5 — замок от корня основной копии репо, FASTBT_MAIN на путь не влияет; чтение под data/ до замка — только хэш (open, io.open, Path.read_bytes/read_text из O.sha), парсеры — только под замком
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k zamok
EXPECT: passed
EVIDENCE: 4 passed, 59 deselected

## G6 — Н5: ряд 2022+ — отдельный файл; ряд ≤ 2021 после открытия побайтно тот же; запись ряда 2022+ в каталог ряда разработки — отказ
CHECK: python3 -m pytest tests/test_otkrytie_gl4.py -q -k ryad
EXPECT: passed
EVIDENCE: 3 passed, 60 deselected

## G7 — мутации ловятся (копия в scratchpad/mut, fastbt — ссылка; тест пути замка в копии исключён, копия не git)
CHECK: для мутанта: python3 -m pytest <копия>/tests/test_otkrytie_gl4.py -q -k "not test_zamok_ne_zavisit"; test $? -eq 1
EXPECT: м5–м7 — pytest вернул 1
EVIDENCE: база в копии — 62 passed (+1 тест пути замка: в копии без git он ожидаемо падает); м5 «да» без проверки sha_spec — 1 failed (test_bez_da_staroe_da_bez_sha_spec); м6 Ледуа–Вольф по позиции — 2 failed (daty_sdvinutye, daty_vyravnivanie); м7 без проверки сдвига до замка — 4 failed (zamok_do_chteniya, proverka_sdviga_2018_2021_*, ne_vse_metki). 3 из 3, у всех pytest=1

## G8 — мутант м8 «замок от FASTBT_MAIN» ловится
CHECK: python3 -m pytest <копия>/tests/test_otkrytie_gl4.py -q -k test_zamok_ne_zavisit; test $? -eq 1
EXPECT: pytest вернул 1
EVIDENCE: не запускался — см. ABANDON

## G9 — изменены только файлы владения
CHECK: git diff --name-only origin/kanon-v22-pereschet...HEAD
EXPECT: scripts/otkrytie_gl4.py, tests/test_otkrytie_gl4.py, docs/final_gl4/zagotovki/spec_otkrytiya.shablon.json, docs/kanon/v24/gates/E4_otkrytie_2.md
EVIDENCE: те же 4 пути (проверено перед коммитом, 19:21)

## ABANDON (только с причиной; гейт не удалять и не переписывать)
ABANDON: G8 копия в scratchpad — не git-репозиторий, поэтому `koren_repo` в ней отказывает. git вне своей рабочей копии запрещён, мутировать рабочую копию нельзя. Покрытие без мутанта: test_zamok_ne_zavisit_ot_fastbt_main проверяет, что путь замка = <корень git>/data/regen/test_locked/…, что FASTBT_MAIN подменён и что каталога FASTBT_MAIN в пути замка нет. Закрыть может проверяющий: мутант в отдельной git-копии.

## Реестр
закрыто 8 из 9; ABANDON: 1 (G8); pending: 0

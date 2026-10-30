"""Правила усечения данных ≤ 2021-12-31 для сервера (Ж-131 п. 2): какие файлы, по каким колонкам дат (usech_2021.py)."""
KAL = ("date", "known_from")
# (шаблон от корня, вид, колонки дат, очищаемые колонки, читатель)
PRAVILA = [
    ("moex_discovery/data/raw/candles_10m/*.parquet", "pq", ("timestamp",), (), "fastbt/sigma_real.py rv22, bars_agg (ТФ tf_os_v5), daily_1850"),
    ("moex_discovery/data/final_v6/candles_10m/external/*.parquet", "pq", ("begin",), (), "rezhim_e1.imoex_daily, e1_sektor, vneshnie_1850 (IMOEX, секторные 10м)"),
    # оценщик v6 (fastbt/istochniki_v6.py, vneshnie_1850.py; трассировка открытий 1 точки Морриса S1, train4 и v2, 29.10)
    ("data/iss_ext/bars_10m/BR*.parquet", "pq", ("begin",), (), "vneshnie_1850.contract bars: Brent BR* (istochniki_v6.Fajly.brent, faktor 'brent')"),
    ("data/iss_ext/br_contracts.json", "json", ("lsttrade",), (), "vneshnie_1850.contract_table (календарь контрактов Brent; записи с датой > 2021 отброшены)"),
    ("moex_discovery/data/external/moex_iss/currency_10m/USD000UTSTOM.parquet", "pq", ("begin",), (), "vneshnie_1850 PATHS['usdrub'] (фактор USDRUB, indeks('usdrub'))"),
    ("moex_volatility/results/final/strategies/data/vpred_aligned.parquet", "pq", ("date",), (), "fastbt/data.py VPRED (load_ticker -> istochniki_v6.Fajly.zakrytie)"),
    ("data/corporate_actions.csv", "csv", ("date",), (), "fastbt/data.py corp_actions"),
    ("data/events.csv", "csv", ("data_obyavleniya", "data_sobytiya"), (), "fastbt/sobytiya_pit.py (календарь событий PIT: объявление и событие)"),
    ("data/calendar/torgovye_dni_moex.csv","csv", ("date",), (), "торговый календарь MOEX"),
    ("fastbt/rates/key_rate_cbr_vladelec.csv", "csv", ("data_vstupleniya",), (), "ставка ЦБ владельца (дата вступления; объявление раньше входа)"),
    *[(f"data/calendar/{n}.csv", "csv", KAL, (), f"календарь {n} (PIT: date и known_from; retrieved_at — дата выгрузки, не дата события: не очищается)")
      for n in ("cbr", "expirations", "fomc", "imoex_rebal", "imoex_rebal_official", "cpi_us")],
    ("moex_discovery/data/external/macro/raw/key_rate_cbr.parquet", "pq", ("date",), (), "fastbt/msp.py rate_cbr (parquet ЦБ; до 2022 берётся ряд владельца)"),
    # справочные таблицы исследователя: fastbt читает их от корня КОДА (накладка sobrat_perenos.sh / razvernut.sh);
    # accessed — дата скачивания (2026), не дата данных: очищается; строки без даты реестра (нет выплат) отбрасываются
    ("docs/issledovaniya/2026-10-27_dividendy_2017_2021.csv", "csv", ("record_date", "announce_date", "last_cum_date"),
     ("accessed",), "fastbt/dividendy.py TABLE, sobytiya_e1 (реестр и объявление PIT)"),
    ("docs/issledovaniya/2026-10-28_dividendy_2015_2016.csv", "csv", ("record_date",), ("accessed",), "fastbt/dividendy.py TABLE_2015"),
    ("docs/issledovaniya/2026-10-28_kluchevaya_stavka_2013_2017.csv", "csv", ("data_vstupleniya", "data_resheniya"),
     ("accessed",), "fastbt/msp.py RATE_RESEARCH"),
    ("docs/issledovaniya/2026-10-28_sobytiya_2016_2021.csv", "csv", ("data_obyavleniya", "data_sobytiya"),
     ("accessed",), "fastbt события E (PIT: объявление и событие)"),
    # справочник без колонки дат (тикер → сектор; даты только в accessed и as-of в source_url, ≤ 2021-12-30): «без дат»
    ("docs/issledovaniya/2026-10-27_sektory.csv", "csv", (), ("accessed",), "fastbt/e1_sektor.py, fazy.py (сектор тикера)"),
    ("docs/issledovaniya/2026-10-28_sostav_imoex.csv", "csv", ("date_eff", "known_from"), (), "fastbt/vselennaya.py SOSTAV"),
    ("docs/issledovaniya/div_sverka_svod.csv", "csv", ("record_date", "announce_pit"), (),
     "fastbt/sobytiya_e1.py SVOD_CSV, istochniki_v6.div_otfiltrovan (свод Д8: без него ось дивидендов v6 закрыта)"),
    # календарь основных сессий MOEX 2025–2026 (fastbt/kalendar_moex.py): нужен ТОЛЬКО пакету wf (--do 2026-02-28); при --do 2021
    # копия пуста по датам и отвергается (в манифест ≤ 2021 не входит); читается лениво (daily_1850.aggregate, calendar_extend)
    ("docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv", "csv", ("date",), ("accessed",), "fastbt/kalendar_moex.py CALENDAR_MOEX (wf: дни ≥ 2025)"),
    ("docs/issledovaniya/2026-10-28_sostav_sektorov.csv", "csv", ("дата_ребалансировки",), (), "состав секторных индексов"),
    # составы 2022–2026 нужны ТОЛЬКО пакету wf (--do 2026-02-28); при --do 2021 копия пуста и отвергается
    ("docs/issledovaniya/2026-10-30_sostav_imoex_2022_2026.csv", "csv", ("date_eff", "known_from"), (), "fastbt/vselennaya.py SOSTAV (wf: 2022–2026)"),
    ("docs/issledovaniya/2026-10-30_sostav_sektorov_2022_2026.csv", "csv", ("дата_ребалансировки",), (), "состав секторных индексов (wf: 2022–2026)"),
    # CNY: хвост 06.2024–02.2026 фактора usdrub (vneshnie_1850.PATHS['cnyrub']); при --do 2021 копия пуста и отвергается
    ("moex_discovery/data/external/moex_iss/currency_10m/CNYRUB_TOM.parquet", "pq", ("begin",), (), "vneshnie_1850 PATHS['cnyrub'] (фактор CNY)"),
]

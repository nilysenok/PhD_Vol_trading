"""Правила усечения данных ≤ 2021-12-31 для сервера (Ж-131 п. 2): какие файлы, по каким колонкам дат (usech_2021.py)."""
KAL = ("date", "known_from")
# (шаблон от корня, вид, колонки дат, очищаемые колонки, читатель)
PRAVILA = [
    ("moex_discovery/data/raw/candles_10m/*.parquet", "pq", ("timestamp",), (), "fastbt/sigma_real.py rv22, bars_agg (ТФ tf_os_v5), daily_1850"),
    ("moex_discovery/data/final_v6/candles_10m/external/*.parquet", "pq", ("begin",), (), "rezhim_e1.imoex_daily, e1_sektor, vneshnie_1850 (IMOEX, секторные 10м)"),
    ("data/corporate_actions.csv", "csv", ("date",), (), "fastbt/data.py corp_actions"),
    ("data/calendar/torgovye_dni_moex.csv", "csv", ("date",), (), "торговый календарь MOEX"),
    ("fastbt/rates/key_rate_cbr_vladelec.csv", "csv", ("data_vstupleniya",), (), "ставка ЦБ владельца (дата вступления; объявление раньше входа)"),
    *[(f"data/calendar/{n}.csv", "csv", KAL, ("retrieved_at",), f"календарь {n} (PIT: date и known_from)")
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
    ("docs/issledovaniya/2026-10-28_sostav_sektorov.csv", "csv", ("дата_ребалансировки",), (), "состав секторных индексов"),
]

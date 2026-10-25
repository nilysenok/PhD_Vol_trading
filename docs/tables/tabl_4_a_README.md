# Таблицы главы 4, подход A (fastbt, маска [0])

Каждая таблица ниже воспроизводится командой из корня репозитория:

- `tabl_4_msp_a.csv` — МСП-A по окнам: `python3 -m fastbt.chapter4_a --workers 10 --boot-reps 1000`
- `tabl_4_msp_a_po_godam.csv` — МСП-A по тестовым годам 2020–2026: `python3 -m fastbt.chapter4_a --workers 10 --boot-reps 1000`
- `tabl_4_9_h3.csv` — H3, бенчмарки MCFTR/RGBITR: `python3 -m fastbt.chapter4_a --workers 10 --boot-reps 1000`
- `tabl_4_otsev_a.csv` — отсев фазы 1: `python3 -m fastbt.chapter4_a --workers 10 --boot-reps 1000`
- `tabl_4_msp_a_po_godam_mask1.csv`, `tabl_4_msp_a_mask1.csv`, `tabl_4_a_utechka.csv` — до/после утечки: сначала `python3 -m fastbt.chapter4_a --workers 10 --boot-reps 1000`, затем `python3 -m fastbt.chapter4_a --workers 10 --boot-reps 1000 --leak-mask-old`
- `tabl_4_a_razlozhenie.csv`, `tabl_4_a_vklad_strategiy.csv` — разложение МСП-A по годам и вклад S1–S6 (маски [0] и [1]): после двух команд выше `python3 -m fastbt.decomp`

Конфигурация: `{"mask_old": false, "fix_c_s12_val": true, "fix_b_median": true, "fix_d_units": true, "fix_c_fill": true, "fix_rate": true, "fix_sharpe": true, "comm": 0.0005, "msp_comm": 0.0005, "years_a": [2020, 2021, 2022, 2023, 2024, 2025, 2026], "years_bcd": [2022, 2023, 2024, 2025, 2026], "corp_actions": true, "val_start": 2020, "approaches": "A"}`

Время счёта позиций: 6 с.

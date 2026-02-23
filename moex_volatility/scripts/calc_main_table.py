"""
Main Table: pure strategy results (no MMF) + separate MMF supplement.
Reads from capital_efficiency_details.csv + exposure_scaling_analysis.csv.
Outputs: MAIN_TABLE.md, MAIN_TABLE.csv
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
BPY = 252

# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
def calc_metrics(r):
    """Annualized metrics from daily return array."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return dict(Return_pct=np.nan, Vol_pct=np.nan, Sharpe=np.nan,
                    MDD_pct=np.nan, Calmar=np.nan)
    ann_ret = np.mean(r) * BPY
    ann_vol = np.std(r, ddof=1) * np.sqrt(BPY)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-12 else 0
    cum = np.cumprod(1 + r)
    rmax = np.maximum.accumulate(cum)
    mdd = np.min((cum - rmax) / rmax)
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-12 else 0
    return dict(Return_pct=round(ann_ret * 100, 2),
                Vol_pct=round(ann_vol * 100, 2),
                Sharpe=round(sharpe, 2),
                MDD_pct=round(mdd * 100, 2),
                Calmar=round(calmar, 2))


def alpha_beta(strat_r, idx_r):
    mask = np.isfinite(strat_r) & np.isfinite(idx_r)
    s, i = strat_r[mask], idx_r[mask]
    beta = np.cov(s, i)[0, 1] / np.var(i, ddof=1) if np.var(i) > 1e-12 else 0
    alpha = (np.mean(s) - beta * np.mean(i)) * BPY * 100
    return round(beta, 4), round(alpha, 2)


# ──────────────────────────────────────────────────────────
# LOAD
# ──────────────────────────────────────────────────────────
print("Loading data...")
det = pd.read_csv('results/capital_efficiency_details.csv',
                   index_col=0, parse_dates=True)

ALL_METAS = ['META-A', 'META-B', 'META-C', 'META-D',
             'META-BEST', 'META-MEAN(BCD)']
SCALE_METAS = ['META-A', 'META-B', 'META-D', 'META-BEST']

# Extract data
data = {}
for p in SCALE_METAS:
    strat_r = det[f'{p}_strat'].values
    exp = det[f'{p}_exposure'].values
    mmf_col = det[f'{p}_mmf'].values
    free_cap_1x = np.maximum(0, 1 - exp)
    mask_free = free_cap_1x > 0.01
    mmf_rate = np.where(mask_free, mmf_col / free_cap_1x, 0)
    data[p] = dict(strat_r=strat_r, exp=exp, mmf_rate=mmf_rate)

imoex_r = det['IMOEX'].values
n_days = len(det)
print(f"Loaded {n_days} trading days")


# ──────────────────────────────────────────────────────────
# TASK 1: MAIN TABLE (no MMF)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TASK 1: MAIN TABLE (pure strategy, no MMF)")
print("=" * 80)

COMMENTS = {
    'META-A': 'Базовый (без прогнозов)',
    'META-B': 'Адаптивные стопы',
    'META-C': 'Режимный фильтр',
    'META-D': 'Vol-gate',
    'META-BEST': 'Лучший подход по стратегии',
    'META-MEAN(BCD)': 'Средний эффект прогнозов',
}

rows_main = []

# 1. IMOEX B&H
m = calc_metrics(imoex_r)
rows_main.append(dict(
    Portfolio='IMOEX B&H', k='1x', Avg_Exposure=100.0,
    Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
    Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
    Beta=1.0, Alpha_pct='—', Comment='Benchmark (ценовой индекс)'))

# 2-7. Base meta-portfolios ×1 (no MMF)
for p in ALL_METAS:
    strat_col = f'{p}_strat'
    exp_col = f'{p}_exposure'
    strat_r = det[strat_col].values
    exp = det[exp_col].values
    m = calc_metrics(strat_r)
    beta, alpha = alpha_beta(strat_r, imoex_r)
    avg_exp = round(np.mean(exp) * 100, 1)
    rows_main.append(dict(
        Portfolio=p, k='1x', Avg_Exposure=avg_exp,
        Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
        Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
        Beta=beta, Alpha_pct=alpha, Comment=COMMENTS[p]))

# 8-12. Scaled (no MMF): META-BEST ×{2,3,5}, META-D ×{2,3}
SCALE_LIST = [
    ('META-BEST', 2), ('META-BEST', 3), ('META-BEST', 5),
    ('META-D', 2), ('META-D', 3),
]

for p, k in SCALE_LIST:
    d = data[p]
    scaled_r = k * d['strat_r']
    eff_exp = k * d['exp']
    m = calc_metrics(scaled_r)
    avg_exp = round(np.mean(eff_exp) * 100, 1)
    max_exp = round(np.max(eff_exp) * 100, 1)
    pct_over = round(np.mean(eff_exp > 1.0) * 100, 1)
    note = f'Экспозиция {avg_exp}% (макс {max_exp}%)'
    if pct_over > 0:
        note += f', >100% в {pct_over}% дней'
    rows_main.append(dict(
        Portfolio=f'{p} × {k}', k=f'{k}x', Avg_Exposure=avg_exp,
        Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
        Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
        Beta='—', Alpha_pct='—', Comment=note))

main_df = pd.DataFrame(rows_main)
COLS_MAIN = ['Portfolio', 'k', 'Avg_Exposure', 'Return_pct', 'Vol_pct',
             'Sharpe', 'MDD_pct', 'Calmar', 'Beta', 'Alpha_pct', 'Comment']
main_df = main_df[COLS_MAIN]

print("\nMain Table:")
print(main_df.to_string(index=False))

# Sanity check: Sharpe should be ~constant across multipliers
print("\n--- Sharpe constancy check ---")
for p_name in ['META-BEST', 'META-D']:
    base_sharpe = None
    for _, row in main_df.iterrows():
        if row['Portfolio'].startswith(p_name):
            s = row['Sharpe']
            if base_sharpe is None:
                base_sharpe = s
            print(f"  {row['Portfolio']:20s}  Sharpe={s}  (delta from 1x: {s - base_sharpe:+.2f})")


# ──────────────────────────────────────────────────────────
# TASK 2: MMF SUPPLEMENT
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TASK 2: MMF SUPPLEMENT (effect of ФДР)")
print("=" * 80)

rows_mmf = []

# All base metas ×1 + MMF
for p in ALL_METAS:
    strat_col = f'{p}_strat'
    exp_col = f'{p}_exposure'
    strat_r = det[strat_col].values
    exp_v = det[exp_col].values

    # Strategy-only metrics
    m_strat = calc_metrics(strat_r)

    # With MMF
    if p in data:
        d = data[p]
        free_cap = np.maximum(0, 1 - d['exp'])
        mmf_r = free_cap * d['mmf_rate']
        total_r = strat_r + mmf_r
    else:
        # META-C, META-MEAN(BCD): compute MMf from any available mmf_rate
        mmf_rate = data['META-A']['mmf_rate']  # same CBR rate for all
        free_cap = np.maximum(0, 1 - exp_v)
        mmf_r = free_cap * mmf_rate
        total_r = strat_r + mmf_r

    m_total = calc_metrics(total_r)
    avg_free = round(np.mean(free_cap) * 100, 1)
    mmf_ann = round(np.mean(mmf_r) * BPY * 100, 2)

    rows_mmf.append(dict(
        Portfolio=p, k='1x', Free_Capital_pct=avg_free,
        MMF_Return_pct=mmf_ann,
        Total_Return_pct=m_total['Return_pct'],
        Total_Vol_pct=m_total['Vol_pct'],
        Total_Sharpe=m_total['Sharpe'],
        Total_MDD_pct=m_total['MDD_pct'],
        Total_Calmar=m_total['Calmar'],
        Strat_Sharpe=m_strat['Sharpe'],
        Delta_Sharpe=round(m_total['Sharpe'] - m_strat['Sharpe'], 2)))

# Scaled + MMF
for p, k in SCALE_LIST:
    d = data[p]
    scaled_r = k * d['strat_r']
    eff_exp = k * d['exp']
    free_cap = np.maximum(0, 1 - eff_exp)
    mmf_r = free_cap * d['mmf_rate']
    total_r = scaled_r + mmf_r

    m_strat = calc_metrics(scaled_r)
    m_total = calc_metrics(total_r)
    avg_free = round(np.mean(free_cap) * 100, 1)
    mmf_ann = round(np.mean(mmf_r) * BPY * 100, 2)

    rows_mmf.append(dict(
        Portfolio=f'{p} × {k}', k=f'{k}x', Free_Capital_pct=avg_free,
        MMF_Return_pct=mmf_ann,
        Total_Return_pct=m_total['Return_pct'],
        Total_Vol_pct=m_total['Vol_pct'],
        Total_Sharpe=m_total['Sharpe'],
        Total_MDD_pct=m_total['MDD_pct'],
        Total_Calmar=m_total['Calmar'],
        Strat_Sharpe=m_strat['Sharpe'],
        Delta_Sharpe=round(m_total['Sharpe'] - m_strat['Sharpe'], 2)))

mmf_df = pd.DataFrame(rows_mmf)
COLS_MMF = ['Portfolio', 'k', 'Free_Capital_pct', 'MMF_Return_pct',
            'Total_Return_pct', 'Total_Vol_pct', 'Total_Sharpe',
            'Total_MDD_pct', 'Total_Calmar', 'Strat_Sharpe', 'Delta_Sharpe']
mmf_df = mmf_df[COLS_MMF]

print("\nMMF Supplement:")
print(mmf_df.to_string(index=False))


# ──────────────────────────────────────────────────────────
# TASK 3: SCALING THRESHOLDS (META-BEST, without MMF)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TASK 3: SCALING THRESHOLDS (pure strategy, no MMF)")
print("=" * 80)

multipliers = np.arange(1, 15.5, 0.5)
scale_rows = []

for p in ['META-BEST', 'META-D', 'META-B']:
    d = data[p]
    for k in multipliers:
        scaled_r = k * d['strat_r']
        eff_exp = k * d['exp']
        m = calc_metrics(scaled_r)
        scale_rows.append(dict(
            Portfolio=p, Multiplier=k,
            Avg_Exposure=round(np.mean(eff_exp) * 100, 1),
            Max_Exposure=round(np.max(eff_exp) * 100, 1),
            Pct_Over100=round(np.mean(eff_exp > 1.0) * 100, 1),
            Return_pct=m['Return_pct'],
            Vol_pct=m['Vol_pct'],
            Sharpe=m['Sharpe'],
            MDD_pct=m['MDD_pct'],
            Calmar=m['Calmar']))

scale_df = pd.DataFrame(scale_rows)

# Key display: META-BEST thresholds
KEY_K = [1, 2, 3, 4, 5]
for p_name in ['META-BEST', 'META-D']:
    print(f"\n--- {p_name} scaling (no MMF) ---")
    pscan = scale_df[scale_df['Portfolio'] == p_name]
    for k in KEY_K:
        row = pscan[pscan['Multiplier'] == k].iloc[0]
        borrowed = 'Нет'
        if row['Pct_Over100'] > 0 and row['Pct_Over100'] <= 1:
            borrowed = f'Нет ({row["Pct_Over100"]}% дней >100%)'
        elif row['Pct_Over100'] > 1:
            borrowed = f'Редко ({row["Pct_Over100"]}% дней)'
        print(f"  {k}x: exp={row['Avg_Exposure']}%, max={row['Max_Exposure']}%, "
              f"ret={row['Return_pct']}%, vol={row['Vol_pct']}%, "
              f"sharpe={row['Sharpe']}, mdd={row['MDD_pct']}%, borrowed={borrowed}")


# ──────────────────────────────────────────────────────────
# TASK 4: SAVE
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SAVING...")
print("=" * 80)

out = 'results'

# ──── CSV ────
csv_rows = []
# Main table
for _, row in main_df.iterrows():
    csv_rows.append({
        'Section': 'Main',
        'Portfolio': row['Portfolio'], 'k': row['k'],
        'Avg_Exposure_pct': row['Avg_Exposure'],
        'Return_pct': row['Return_pct'], 'Vol_pct': row['Vol_pct'],
        'Sharpe': row['Sharpe'], 'MDD_pct': row['MDD_pct'],
        'Calmar': row['Calmar'], 'Beta': row['Beta'],
        'Alpha_pct': row['Alpha_pct'], 'Comment': row['Comment']})
# MMF supplement
for _, row in mmf_df.iterrows():
    csv_rows.append({
        'Section': 'MMF',
        'Portfolio': row['Portfolio'], 'k': row['k'],
        'Free_Capital_pct': row['Free_Capital_pct'],
        'MMF_Return_pct': row['MMF_Return_pct'],
        'Total_Return_pct': row['Total_Return_pct'],
        'Total_Vol_pct': row['Total_Vol_pct'],
        'Total_Sharpe': row['Total_Sharpe'],
        'Total_MDD_pct': row['Total_MDD_pct'],
        'Total_Calmar': row['Total_Calmar'],
        'Strat_Sharpe': row['Strat_Sharpe'],
        'Delta_Sharpe': row['Delta_Sharpe']})

csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv(f'{out}/MAIN_TABLE.csv', index=False)
print(f"Saved: {out}/MAIN_TABLE.csv")


# ──── MARKDOWN ────
period_start = det.index[0].date()
period_end = det.index[-1].date()

# Recover avg CBR from mmf_rate
avg_mmf_rate = np.mean(data['META-BEST']['mmf_rate']) * 365 * 100
avg_cbr = avg_mmf_rate + 1.5

with open(f'{out}/MAIN_TABLE.md', 'w') as f:
    f.write("# Основные результаты\n\n")
    f.write(f"**Период**: {period_start} — {period_end} ({n_days} торговых дней)\n")
    f.write("**Комиссия**: 0.40% на сторону\n")
    f.write(f"**Ключевая ставка ЦБ**: ср. ~{avg_cbr:.0f}% (диапазон 7.5%–21.0%)\n\n")

    # ──── Section 1: Main table ────
    f.write("## 1. Чистые результаты стратегий (без ФДР)\n\n")
    f.write("| Портфель | k | Ср. экспозиция | Return% | Vol% | Sharpe "
            "| MDD% | Calmar | Beta | Alpha% | Комментарий |\n")
    f.write("|" + "|".join(["---"] * 11) + "|\n")
    for _, row in main_df.iterrows():
        beta_str = str(row['Beta'])
        alpha_str = str(row['Alpha_pct'])
        f.write(f"| {row['Portfolio']} | {row['k']} | {row['Avg_Exposure']}% "
                f"| {row['Return_pct']} | {row['Vol_pct']} | {row['Sharpe']} "
                f"| {row['MDD_pct']} | {row['Calmar']} | {beta_str} | {alpha_str} "
                f"| {row['Comment']} |\n")

    f.write("\n**Примечания:**\n")
    f.write("- Все числа — **без учёта размещения свободного капитала в ФДР**\n")
    f.write("- Sharpe при масштабировании (×k) инвариантен: k × return / k × vol = const\n")
    f.write("- MDD растёт приблизительно пропорционально множителю\n")
    f.write("- Alpha и Beta — OLS на дневных доходностях IMOEX, только для ×1\n")
    f.write("- Все стратегии market-neutral (beta ≈ 0, корреляция с IMOEX < 0.06)\n\n")

    # ──── Section 2: MMF supplement ────
    f.write("## 2. Эффект размещения свободного капитала в ФДР\n\n")
    f.write("При размещении свободного капитала (1 − экспозиция) в фонде денежного рынка "
            f"(доходность ≈ ключевая ставка ЦБ − 1.5%, ср. ~{avg_mmf_rate:.1f}% годовых):\n\n")
    f.write("| Портфель | k | Своб. капитал | ФДР доход% | Итого Return% | Итого Vol% "
            "| Итого Sharpe | Итого MDD% | Итого Calmar | Δ Sharpe |\n")
    f.write("|" + "|".join(["---"] * 10) + "|\n")
    for _, row in mmf_df.iterrows():
        f.write(f"| {row['Portfolio']} | {row['k']} | {row['Free_Capital_pct']}% "
                f"| +{row['MMF_Return_pct']} | {row['Total_Return_pct']} "
                f"| {row['Total_Vol_pct']} | {row['Total_Sharpe']} "
                f"| {row['Total_MDD_pct']} | {row['Total_Calmar']} "
                f"| +{row['Delta_Sharpe']} |\n")

    f.write("\n**Примечания:**\n")
    f.write("- ФДР добавляет безрисковую доходность на незадействованный капитал\n")
    f.write("- При низкой экспозиции (META-B: 4.1%) бо́льшая часть капитала в ФДР "
            "→ Sharpe растёт значительно\n")
    f.write("- При масштабировании ×k свободный капитал уменьшается → "
            "вклад ФДР падает\n")
    f.write("- Vol и MDD с ФДР практически не меняются "
            "(ФДР добавляет стабильный положительный return)\n\n")

    # ──── Section 3: Scaling thresholds ────
    f.write("## 3. Масштабирование экспозиции (META-BEST, без ФДР)\n\n")
    f.write("| Множитель | Ср. экспозиция | Макс. экспозиция | Заёмные средства? "
            "| Return% | Vol% | Sharpe | MDD% |\n")
    f.write("|" + "|".join(["---"] * 8) + "|\n")

    best_scan = scale_df[scale_df['Portfolio'] == 'META-BEST']
    for k in [1, 2, 3, 4, 5]:
        row = best_scan[best_scan['Multiplier'] == k].iloc[0]
        if row['Pct_Over100'] == 0:
            borrowed = 'Нет'
        elif row['Pct_Over100'] <= 1:
            borrowed = f'Нет ({row["Pct_Over100"]}% дней >100%)'
        else:
            borrowed = f'Редко ({row["Pct_Over100"]}% дней)'
        f.write(f"| {k}× | {row['Avg_Exposure']}% | {row['Max_Exposure']}% "
                f"| {borrowed} | {row['Return_pct']} | {row['Vol_pct']} "
                f"| {row['Sharpe']} | {row['MDD_pct']} |\n")

    f.write("\n**Выводы:**\n\n")

    # Find thresholds
    first_over100 = best_scan[best_scan['Max_Exposure'] > 100]
    threshold_k = first_over100.iloc[0]['Multiplier'] if len(first_over100) > 0 else '>15'
    threshold_pct = first_over100.iloc[0]['Pct_Over100'] if len(first_over100) > 0 else 0

    f.write(f"1. **Реальный leverage** (экспозиция >100%) для META-BEST начинается "
            f"при множителе **~{threshold_k}×** и затрагивает <{max(threshold_pct, 1)}% "
            f"торговых дней.\n\n")
    f.write("2. Sharpe **инвариантен** к масштабированию "
            "(k × μ / k × σ = μ / σ = const). "
            "Различия в таблице — эффект compounding и дискретности.\n\n")
    f.write("3. Рекомендуемый диапазон для институционального инвестора: **3–5×** "
            "(экспозиция 37–62%, заёмные средства не требуются при ~3×, "
            "минимально при 5×).\n\n")
    f.write("4. **Терминология**: «масштабирование экспозиции» (exposure scaling / "
            "position sizing), **не** «leverage». Leverage подразумевает заёмные "
            "средства и экспозицию >100%.\n\n")

    # ──── Section 4: Same for META-D ────
    f.write("## 4. Масштабирование экспозиции (META-D, без ФДР)\n\n")
    f.write("| Множитель | Ср. экспозиция | Макс. экспозиция | Заёмные средства? "
            "| Return% | Vol% | Sharpe | MDD% |\n")
    f.write("|" + "|".join(["---"] * 8) + "|\n")

    d_scan = scale_df[scale_df['Portfolio'] == 'META-D']
    for k in [1, 2, 3, 4, 5]:
        row = d_scan[d_scan['Multiplier'] == k].iloc[0]
        if row['Pct_Over100'] == 0:
            borrowed = 'Нет'
        elif row['Pct_Over100'] <= 1:
            borrowed = f'Нет ({row["Pct_Over100"]}% дней >100%)'
        else:
            borrowed = f'Редко ({row["Pct_Over100"]}% дней)'
        f.write(f"| {k}× | {row['Avg_Exposure']}% | {row['Max_Exposure']}% "
                f"| {borrowed} | {row['Return_pct']} | {row['Vol_pct']} "
                f"| {row['Sharpe']} | {row['MDD_pct']} |\n")

    # ──── Section 5: Quick summary ────
    f.write("\n## 5. Сводка: что выбрать?\n\n")
    f.write("| Цель | Портфель | Настройка | Return% | Sharpe | MDD% |\n")
    f.write("|" + "|".join(["---"] * 6) + "|\n")

    # Min risk
    b1 = main_df[main_df['Portfolio'] == 'META-B'].iloc[0]
    b1_mmf = mmf_df[mmf_df['Portfolio'] == 'META-B'].iloc[0]
    f.write(f"| Минимальный риск | META-B | ×1 + ФДР | {b1_mmf['Total_Return_pct']} "
            f"| {b1_mmf['Total_Sharpe']} | {b1_mmf['Total_MDD_pct']} |\n")

    # Max Sharpe
    best1_mmf = mmf_df[mmf_df['Portfolio'] == 'META-BEST'].iloc[0]
    f.write(f"| Макс. Sharpe | META-BEST | ×1 + ФДР | {best1_mmf['Total_Return_pct']} "
            f"| {best1_mmf['Total_Sharpe']} | {best1_mmf['Total_MDD_pct']} |\n")

    # Balanced
    best3_mmf = mmf_df[(mmf_df['Portfolio'] == 'META-BEST × 3')].iloc[0]
    f.write(f"| Баланс доходность/риск | META-BEST | ×3 + ФДР | {best3_mmf['Total_Return_pct']} "
            f"| {best3_mmf['Total_Sharpe']} | {best3_mmf['Total_MDD_pct']} |\n")

    # Max return (no real leverage)
    best5_mmf = mmf_df[(mmf_df['Portfolio'] == 'META-BEST × 5')].iloc[0]
    f.write(f"| Макс. доходность | META-BEST | ×5 + ФДР | {best5_mmf['Total_Return_pct']} "
            f"| {best5_mmf['Total_Sharpe']} | {best5_mmf['Total_MDD_pct']} |\n")

print(f"Saved: {out}/MAIN_TABLE.md")
print("\nDone!")

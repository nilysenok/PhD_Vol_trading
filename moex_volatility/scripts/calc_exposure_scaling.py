"""
Exposure Scaling Analysis
Correct framing: position scaling (not leverage) for low-exposure strategies.
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

# ──────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────
print("Loading data...")
det = pd.read_csv('results/capital_efficiency_details.csv', index_col=0, parse_dates=True)

# Extract base strategy returns, exposure, MMF daily rate
PORTFOLIOS = ['META-A', 'META-B', 'META-D', 'META-BEST']

data = {}
for p in PORTFOLIOS:
    strat_r = det[f'{p}_strat'].values
    exp = det[f'{p}_exposure'].values
    # Recover MMF daily rate from: mmf = free_cap * mmf_rate
    # free_cap = max(0, 1 - exp), mmf_rate = mmf / free_cap
    free_cap_1x = np.maximum(0, 1 - exp)
    mmf_col = det[f'{p}_mmf'].values
    # mmf_rate per day (same for all portfolios)
    mask_free = free_cap_1x > 0.01
    mmf_rate = np.where(mask_free, mmf_col / free_cap_1x, 0)
    data[p] = dict(strat_r=strat_r, exp=exp, mmf_rate=mmf_rate)

# Also load IMOEX
imoex_r = det['IMOEX'].values

n_days = len(det)
print(f"Loaded {n_days} trading days, {len(PORTFOLIOS)} portfolios")

# ──────────────────────────────────────────────────────────
# 1. DETAILED TABLE FOR KEY MULTIPLIERS
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("1. DETAILED SCALING TABLE")
print("=" * 80)

KEY_MULTIPLIERS = [1, 2, 3, 5, 8]

detail_rows = []
for p in PORTFOLIOS:
    d = data[p]
    for k in KEY_MULTIPLIERS:
        scaled_r = k * d['strat_r']
        eff_exp = k * d['exp']
        free_cap = np.maximum(0, 1 - eff_exp)
        mmf_r = free_cap * d['mmf_rate']
        total_r = scaled_r + mmf_r

        avg_exp = np.mean(eff_exp) * 100
        max_exp = np.max(eff_exp) * 100
        p95_exp = np.percentile(eff_exp, 95) * 100
        avg_free = np.mean(free_cap) * 100
        pct_over100 = np.mean(eff_exp > 1.0) * 100
        real_leverage = "Yes" if pct_over100 > 0 else "No"

        m_strat = calc_metrics(scaled_r)
        m_total = calc_metrics(total_r)
        mmf_ann = np.mean(mmf_r) * BPY * 100

        row = dict(
            Portfolio=p, Multiplier=f'{k}x',
            Avg_Exposure_pct=round(avg_exp, 1),
            P95_Exposure_pct=round(p95_exp, 1),
            Max_Exposure_pct=round(max_exp, 1),
            Free_Capital_pct=round(avg_free, 1),
            Pct_Days_Over100=round(pct_over100, 1),
            Real_Leverage=real_leverage,
            Strat_Return_pct=m_strat['Return_pct'],
            Strat_Vol_pct=m_strat['Vol_pct'],
            Strat_Sharpe=m_strat['Sharpe'],
            Strat_MDD_pct=m_strat['MDD_pct'],
            MMF_Addition_pct=round(mmf_ann, 2),
            Total_Return_pct=m_total['Return_pct'],
            Total_Vol_pct=m_total['Vol_pct'],
            Total_Sharpe=m_total['Sharpe'],
            Total_MDD_pct=m_total['MDD_pct'],
            Total_Calmar=m_total['Calmar'],
        )
        detail_rows.append(row)

detail_df = pd.DataFrame(detail_rows)

# Print META-BEST table
print("\n--- META-BEST ---")
best_df = detail_df[detail_df['Portfolio'] == 'META-BEST']
print(best_df[['Multiplier', 'Avg_Exposure_pct', 'Max_Exposure_pct', 'Free_Capital_pct',
               'Pct_Days_Over100', 'Total_Return_pct', 'Total_Vol_pct',
               'Total_Sharpe', 'Total_MDD_pct']].to_string(index=False))

print("\n--- META-D ---")
d_df = detail_df[detail_df['Portfolio'] == 'META-D']
print(d_df[['Multiplier', 'Avg_Exposure_pct', 'Max_Exposure_pct', 'Free_Capital_pct',
            'Pct_Days_Over100', 'Total_Return_pct', 'Total_Vol_pct',
            'Total_Sharpe', 'Total_MDD_pct']].to_string(index=False))

print("\n--- META-B ---")
b_df = detail_df[detail_df['Portfolio'] == 'META-B']
print(b_df[['Multiplier', 'Avg_Exposure_pct', 'Max_Exposure_pct', 'Free_Capital_pct',
            'Pct_Days_Over100', 'Total_Return_pct', 'Total_Vol_pct',
            'Total_Sharpe', 'Total_MDD_pct']].to_string(index=False))

# ──────────────────────────────────────────────────────────
# 2. FINE-GRAINED SCAN (1× to 15×, step 0.5)
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. FINE-GRAINED MULTIPLIER SCAN (META-BEST)")
print("=" * 80)

multipliers = np.arange(1, 15.5, 0.5)
scan_rows = []

for p in PORTFOLIOS:
    d = data[p]
    for k in multipliers:
        scaled_r = k * d['strat_r']
        eff_exp = k * d['exp']
        free_cap = np.maximum(0, 1 - eff_exp)
        mmf_r = free_cap * d['mmf_rate']
        total_r = scaled_r + mmf_r

        m = calc_metrics(total_r)
        scan_rows.append(dict(
            Portfolio=p,
            Multiplier=k,
            Avg_Exposure=round(np.mean(eff_exp) * 100, 1),
            Max_Exposure=round(np.max(eff_exp) * 100, 1),
            P95_Exposure=round(np.percentile(eff_exp, 95) * 100, 1),
            Free_Capital=round(np.mean(free_cap) * 100, 1),
            Pct_Over100=round(np.mean(eff_exp > 1.0) * 100, 1),
            Return_pct=m['Return_pct'],
            Vol_pct=m['Vol_pct'],
            Sharpe=m['Sharpe'],
            MDD_pct=m['MDD_pct'],
            Calmar=m['Calmar'],
        ))

scan_df = pd.DataFrame(scan_rows)

# Find key thresholds for META-BEST
best_scan = scan_df[scan_df['Portfolio'] == 'META-BEST'].copy()

print("\nMETA-BEST multiplier scan:")
print(best_scan[['Multiplier', 'Avg_Exposure', 'Max_Exposure', 'Pct_Over100',
                  'Return_pct', 'Vol_pct', 'Sharpe', 'MDD_pct']].to_string(index=False))

# Thresholds
print("\n--- KEY THRESHOLDS (META-BEST) ---")

# a) avg exposure = 50%
idx50 = best_scan.loc[(best_scan['Avg_Exposure'] - 50).abs().idxmin()]
print(f"Avg exposure ~50%: multiplier = {idx50['Multiplier']}x (actual avg = {idx50['Avg_Exposure']}%)")

# b) avg exposure = 100%
idx100 = best_scan.loc[(best_scan['Avg_Exposure'] - 100).abs().idxmin()]
print(f"Avg exposure ~100%: multiplier = {idx100['Multiplier']}x (actual avg = {idx100['Avg_Exposure']}%)")

# c) first time max exposure > 100%
over100 = best_scan[best_scan['Max_Exposure'] > 100]
if len(over100) > 0:
    first_over = over100.iloc[0]
    print(f"First max exposure > 100%: multiplier = {first_over['Multiplier']}x "
          f"(max = {first_over['Max_Exposure']}%)")
else:
    print("Max exposure never exceeds 100% in range 1-15x")

# d) Sharpe maximized (with MMF)
max_sh_idx = best_scan['Sharpe'].idxmax()
max_sh = best_scan.loc[max_sh_idx]
print(f"Max Sharpe (with MMF): multiplier = {max_sh['Multiplier']}x, "
      f"Sharpe = {max_sh['Sharpe']}, Return = {max_sh['Return_pct']}%")

# e) MDD thresholds
for mdd_target in [-5, -10, -20]:
    over_mdd = best_scan[best_scan['MDD_pct'] <= mdd_target]
    if len(over_mdd) > 0:
        first_mdd = over_mdd.iloc[0]
        print(f"MDD reaches {mdd_target}%: multiplier = {first_mdd['Multiplier']}x "
              f"(MDD = {first_mdd['MDD_pct']}%)")
    else:
        print(f"MDD never reaches {mdd_target}% in range 1-15x")

# Same for META-D
d_scan = scan_df[scan_df['Portfolio'] == 'META-D'].copy()
print("\n--- KEY THRESHOLDS (META-D) ---")
idx50d = d_scan.loc[(d_scan['Avg_Exposure'] - 50).abs().idxmin()]
print(f"Avg exposure ~50%: multiplier = {idx50d['Multiplier']}x")
over100d = d_scan[d_scan['Max_Exposure'] > 100]
if len(over100d) > 0:
    print(f"First max exposure > 100%: multiplier = {over100d.iloc[0]['Multiplier']}x")
max_sh_d = d_scan.loc[d_scan['Sharpe'].idxmax()]
print(f"Max Sharpe: multiplier = {max_sh_d['Multiplier']}x, Sharpe = {max_sh_d['Sharpe']}")

# Same for META-B
b_scan = scan_df[scan_df['Portfolio'] == 'META-B'].copy()
print("\n--- KEY THRESHOLDS (META-B) ---")
idx50b = b_scan.loc[(b_scan['Avg_Exposure'] - 50).abs().idxmin()]
print(f"Avg exposure ~50%: multiplier = {idx50b['Multiplier']}x")
over100b = b_scan[b_scan['Max_Exposure'] > 100]
if len(over100b) > 0:
    print(f"First max exposure > 100%: multiplier = {over100b.iloc[0]['Multiplier']}x")
else:
    print("Max exposure never exceeds 100% in range 1-15x")
max_sh_b = b_scan.loc[b_scan['Sharpe'].idxmax()]
print(f"Max Sharpe: multiplier = {max_sh_b['Multiplier']}x, Sharpe = {max_sh_b['Sharpe']}")

# ──────────────────────────────────────────────────────────
# 3. IMOEX METRICS (for comparison)
# ──────────────────────────────────────────────────────────
imoex_m = calc_metrics(imoex_r)
print(f"\nIMOEX B&H: {imoex_m}")

# ──────────────────────────────────────────────────────────
# 4. BUILD FINAL COMPARISON TABLE V2
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("4. FINAL COMPARISON TABLE V2")
print("=" * 80)

# Alpha/Beta
def compute_alpha_beta(strat_r, idx_r):
    mask = np.isfinite(strat_r) & np.isfinite(idx_r)
    s, i = strat_r[mask], idx_r[mask]
    beta = np.cov(s, i)[0, 1] / np.var(i) if np.var(i) > 1e-12 else 0
    alpha = (np.mean(s) - beta * np.mean(i)) * BPY * 100
    return round(beta, 4), round(alpha, 2)

rows_v2 = []

# IMOEX
m = calc_metrics(imoex_r)
rows_v2.append(dict(Portfolio='IMOEX B&H', Multiplier='1x', Avg_Exposure_pct=100.0,
                     Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
                     Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
                     Beta=1.0, Alpha_pct='—',
                     Comment='Benchmark (ценовой индекс)'))

# Base meta-portfolios (1×, strategy only)
for p in ['META-A', 'META-B', 'META-C', 'META-D', 'META-BEST', 'META-MEAN(BCD)']:
    strat_col = f'{p}_strat'
    if strat_col not in det.columns:
        continue
    strat_r = det[strat_col].values
    m = calc_metrics(strat_r)
    beta, alpha = compute_alpha_beta(strat_r, imoex_r)
    exp_col = f'{p}_exposure'
    avg_exp = round(np.mean(det[exp_col].values) * 100, 1) if exp_col in det.columns else '—'
    comments = {'META-A': 'Базовый (без прогнозов)',
                'META-B': 'Адаптивные стопы', 'META-C': 'Режимный фильтр',
                'META-D': 'Vol-gate', 'META-BEST': 'Лучший подход по стратегии',
                'META-MEAN(BCD)': 'Средний эффект прогнозов'}
    rows_v2.append(dict(Portfolio=p, Multiplier='1x', Avg_Exposure_pct=avg_exp,
                         Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
                         Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
                         Beta=beta, Alpha_pct=alpha,
                         Comment=comments.get(p, '')))

# Strategy + MMF (1×)
for p in ['META-A', 'META-B', 'META-D', 'META-BEST']:
    d = data[p]
    free_cap = np.maximum(0, 1 - d['exp'])
    total_r = d['strat_r'] + free_cap * d['mmf_rate']
    m = calc_metrics(total_r)
    avg_exp = round(np.mean(d['exp']) * 100, 1)
    rows_v2.append(dict(Portfolio=f'{p} + ФДР', Multiplier='1x',
                         Avg_Exposure_pct=avg_exp,
                         Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
                         Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
                         Beta='~0', Alpha_pct='—',
                         Comment='Стратегия + фонд денежного рынка'))

# Scaled positions + MMF for META-BEST and META-D
for p in ['META-BEST', 'META-D']:
    d = data[p]
    for k in [2, 3, 5]:
        scaled_r = k * d['strat_r']
        eff_exp = k * d['exp']
        free_cap = np.maximum(0, 1 - eff_exp)
        total_r = scaled_r + free_cap * d['mmf_rate']
        m = calc_metrics(total_r)
        avg_exp = round(np.mean(eff_exp) * 100, 1)
        max_exp = round(np.max(eff_exp) * 100, 1)
        leverage_note = ""
        if max_exp > 100:
            pct_over = round(np.mean(eff_exp > 1.0) * 100, 1)
            leverage_note = f", leverage {pct_over}% дней"
        rows_v2.append(dict(
            Portfolio=f'{p} × {k} + ФДР', Multiplier=f'{k}x',
            Avg_Exposure_pct=avg_exp,
            Return_pct=m['Return_pct'], Vol_pct=m['Vol_pct'],
            Sharpe=m['Sharpe'], MDD_pct=m['MDD_pct'], Calmar=m['Calmar'],
            Beta='~0', Alpha_pct='—',
            Comment=f'Экспозиция {avg_exp}% (макс {max_exp}%){leverage_note}'))

v2_df = pd.DataFrame(rows_v2)
cols_v2 = ['Portfolio', 'Multiplier', 'Avg_Exposure_pct', 'Return_pct', 'Vol_pct',
           'Sharpe', 'MDD_pct', 'Calmar', 'Beta', 'Alpha_pct', 'Comment']
v2_df = v2_df[cols_v2]

print(v2_df.to_string(index=False))

# ──────────────────────────────────────────────────────────
# 5. SAVE
# ──────────────────────────────────────────────────────────
out = 'results'

# V2 CSV
v2_df.to_csv(f'{out}/FINAL_COMPARISON_TABLE_V2.csv', index=False)
print(f"\nSaved: {out}/FINAL_COMPARISON_TABLE_V2.csv")

# Scan CSV
scan_df.to_csv(f'{out}/exposure_scaling_analysis.csv', index=False)
print(f"Saved: {out}/exposure_scaling_analysis.csv")

# Detail CSV
detail_df.to_csv(f'{out}/exposure_scaling_detail.csv', index=False)
print(f"Saved: {out}/exposure_scaling_detail.csv")

# ──── MARKDOWN: FINAL_COMPARISON_TABLE_V2.md ────
# Retrieve thresholds for markdown
best_scan_r = scan_df[scan_df['Portfolio'] == 'META-BEST']
first_over100_best = best_scan_r[best_scan_r['Max_Exposure'] > 100]
threshold_k = first_over100_best.iloc[0]['Multiplier'] if len(first_over100_best) > 0 else '>15'
max_sharpe_row = best_scan_r.loc[best_scan_r['Sharpe'].idxmax()]
avg_cbr = np.mean(data['META-BEST']['mmf_rate']) * 365 * 100 + 1.5  # recover approximate CBR

with open(f'{out}/FINAL_COMPARISON_TABLE_V2.md', 'w') as f:
    f.write("# Final Comparison Table V2 — Exposure Scaling\n\n")
    f.write(f"**Period**: {det.index[0].date()} to {det.index[-1].date()} "
            f"({n_days} trading days)\n")
    f.write("**Commission**: 0.40% per side\n")
    f.write(f"**CBR key rate**: avg ~{avg_cbr:.0f}% (range 7.5%–21.0%)\n")
    f.write("**ФДР rate**: CBR − 1.5% (management fees)\n\n")

    f.write("## Main Results\n\n")
    f.write("| Portfolio | k | Exposure% | Return% | Vol% | Sharpe | MDD% | Calmar | Beta | Comment |\n")
    f.write("|" + "|".join(["---"] * 10) + "|\n")
    for _, row in v2_df.iterrows():
        vals = [str(row[c]) for c in cols_v2]
        f.write("| " + " | ".join(vals) + " |\n")

    f.write("\n## Exposure Scaling Detail (META-BEST)\n\n")
    f.write("| Множитель | Ср. экспозиция | P95 экспозиция | Макс экспозиция | Своб. капитал | % дней >100% | Return% | Vol% | Sharpe | MDD% |\n")
    f.write("|" + "|".join(["---"] * 10) + "|\n")
    for _, row in detail_df[detail_df['Portfolio'] == 'META-BEST'].iterrows():
        f.write(f"| {row['Multiplier']} | {row['Avg_Exposure_pct']}% | {row['P95_Exposure_pct']}% | "
                f"{row['Max_Exposure_pct']}% | {row['Free_Capital_pct']}% | {row['Pct_Days_Over100']}% | "
                f"{row['Total_Return_pct']} | {row['Total_Vol_pct']} | {row['Total_Sharpe']} | {row['Total_MDD_pct']} |\n")

    f.write("\n## Exposure Scaling Detail (META-D)\n\n")
    f.write("| Множитель | Ср. экспозиция | P95 экспозиция | Макс экспозиция | Своб. капитал | % дней >100% | Return% | Vol% | Sharpe | MDD% |\n")
    f.write("|" + "|".join(["---"] * 10) + "|\n")
    for _, row in detail_df[detail_df['Portfolio'] == 'META-D'].iterrows():
        f.write(f"| {row['Multiplier']} | {row['Avg_Exposure_pct']}% | {row['P95_Exposure_pct']}% | "
                f"{row['Max_Exposure_pct']}% | {row['Free_Capital_pct']}% | {row['Pct_Days_Over100']}% | "
                f"{row['Total_Return_pct']} | {row['Total_Vol_pct']} | {row['Total_Sharpe']} | {row['Total_MDD_pct']} |\n")

    f.write("\n## Exposure Scaling Detail (META-B)\n\n")
    f.write("| Множитель | Ср. экспозиция | P95 экспозиция | Макс экспозиция | Своб. капитал | % дней >100% | Return% | Vol% | Sharpe | MDD% |\n")
    f.write("|" + "|".join(["---"] * 10) + "|\n")
    for _, row in detail_df[detail_df['Portfolio'] == 'META-B'].iterrows():
        f.write(f"| {row['Multiplier']} | {row['Avg_Exposure_pct']}% | {row['P95_Exposure_pct']}% | "
                f"{row['Max_Exposure_pct']}% | {row['Free_Capital_pct']}% | {row['Pct_Days_Over100']}% | "
                f"{row['Total_Return_pct']} | {row['Total_Vol_pct']} | {row['Total_Sharpe']} | {row['Total_MDD_pct']} |\n")

    f.write("\n## Key Thresholds (META-BEST)\n\n")
    # Re-derive thresholds
    bs = best_scan_r.copy()
    exp50 = bs.loc[(bs['Avg_Exposure'] - 50).abs().idxmin()]
    exp100 = bs.loc[(bs['Avg_Exposure'] - 100).abs().idxmin()]
    max_sh = bs.loc[bs['Sharpe'].idxmax()]
    mdd5 = bs[bs['MDD_pct'] <= -5]
    mdd10 = bs[bs['MDD_pct'] <= -10]
    mdd20 = bs[bs['MDD_pct'] <= -20]

    f.write(f"- **Ср. экспозиция 50%**: множитель **{exp50['Multiplier']}×** "
            f"(Return {exp50['Return_pct']}%, Sharpe {exp50['Sharpe']})\n")
    f.write(f"- **Ср. экспозиция 100%**: множитель **{exp100['Multiplier']}×** "
            f"(Return {exp100['Return_pct']}%, Sharpe {exp100['Sharpe']})\n")

    if len(first_over100_best) > 0:
        fo = first_over100_best.iloc[0]
        f.write(f"- **Макс. экспозиция впервые >100%**: множитель **{fo['Multiplier']}×** "
                f"(макс {fo['Max_Exposure']}%, в ср. {fo['Pct_Over100']}% дней)\n")

    f.write(f"- **Максимальный Sharpe (с ФДР)**: множитель **{max_sh['Multiplier']}×**, "
            f"Sharpe = **{max_sh['Sharpe']}**, Return = {max_sh['Return_pct']}%\n")

    if len(mdd5) > 0:
        f.write(f"- **MDD ≤ -5%**: множитель **{mdd5.iloc[0]['Multiplier']}×** "
                f"(MDD = {mdd5.iloc[0]['MDD_pct']}%)\n")
    if len(mdd10) > 0:
        f.write(f"- **MDD ≤ -10%**: множитель **{mdd10.iloc[0]['Multiplier']}×** "
                f"(MDD = {mdd10.iloc[0]['MDD_pct']}%)\n")
    if len(mdd20) > 0:
        f.write(f"- **MDD ≤ -20%**: множитель **{mdd20.iloc[0]['Multiplier']}×** "
                f"(MDD = {mdd20.iloc[0]['MDD_pct']}%)\n")
    else:
        f.write(f"- **MDD ≤ -20%**: не достигается при множителях до 15×\n")

    f.write("\n## Conclusions\n\n")
    f.write(f"1. **Реальный leverage** (экспозиция >100%) для META-BEST начинается при множителе "
            f"**~{threshold_k}×**. ")
    f.write("При множителях 1–5× заёмные средства **не требуются** — это масштабирование позиции "
            "в рамках собственного капитала.\n\n")
    f.write(f"2. **Оптимальный множитель** (макс. Sharpe с ФДР): **{max_sh['Multiplier']}×** "
            f"(Sharpe {max_sh['Sharpe']}, Return {max_sh['Return_pct']}%, "
            f"MDD {max_sh['MDD_pct']}%). ")
    f.write("Для институционального инвестора рекомендуется диапазон **3–5×**: "
            "экспозиция 36–60%, свободно 40–64%, MDD в пределах -3...-8%.\n\n")
    f.write("3. **Терминология для диссертации**: «масштабирование экспозиции» "
            "(exposure scaling / position sizing), **НЕ** «leverage». "
            "Leverage подразумевает заёмные средства и экспозицию >100%. "
            "При экспозиции 12–60% речь идёт о выборе размера позиции "
            "в рамках собственного капитала.\n\n")
    f.write("4. **Capital efficiency**: при множителе 1× свободно ~88% капитала. "
            "Размещение в ФДР добавляет ~7.7% годовых. При множителе 5× свободно ~40%, "
            "ФДР добавляет ~5%. Совокупная доходность растёт как за счёт масштабирования стратегии, "
            "так и за счёт ФДР на остаток.\n\n")
    f.write("5. **META-B** — особый случай: экспозиция всего 4.1%. "
            "Даже при множителе 8× средняя экспозиция ~33%, макс — проверить. "
            "Sharpe при 1× + ФДР = 9.22 (!) благодаря экстремально низкой волатильности.\n")

print(f"Saved: {out}/FINAL_COMPARISON_TABLE_V2.md")

# ──── MARKDOWN: exposure_scaling_analysis.md ────
with open(f'{out}/exposure_scaling_analysis.md', 'w') as f:
    f.write("# Exposure Scaling Analysis\n\n")
    f.write(f"Fine-grained scan: multipliers 1× to 15× (step 0.5×), "
            f"{n_days} trading days.\n\n")

    for p in PORTFOLIOS:
        pscan = scan_df[scan_df['Portfolio'] == p]
        f.write(f"## {p}\n\n")
        f.write("| k | Avg Exp% | Max Exp% | Free% | %Days>100 | Ret% | Vol% | Sharpe | MDD% | Calmar |\n")
        f.write("|" + "|".join(["---"] * 10) + "|\n")
        for _, r in pscan.iterrows():
            f.write(f"| {r['Multiplier']}x | {r['Avg_Exposure']} | {r['Max_Exposure']} | "
                    f"{r['Free_Capital']} | {r['Pct_Over100']} | "
                    f"{r['Return_pct']} | {r['Vol_pct']} | {r['Sharpe']} | "
                    f"{r['MDD_pct']} | {r['Calmar']} |\n")
        f.write("\n")

print(f"Saved: {out}/exposure_scaling_analysis.md")
print("\nDone!")

"""Ж-055/056/057/058: отчёты docs/otchety/2026-10-28_diag_S{k}.md и _diag_svod.md из готовых матриц и журналов сделок.
Запуск: FASTBT_MAIN=$PWD python3 scripts/diag_zh055_otchet.py   (журналы — scripts/diag_zh055_sdelki.py)"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_zh055 as D                                                    # noqa: E402
from diag_zh055 import ME, NA, OUT, ROOT, f3, fin, pkey                    # noqa: E402

HEAD = ("Статус: диагностика, не выбор; разработка train 2018–2019; validation не открыта; **без дивидендов** "
        "(правка дивидендов в PnL не сделана, находка #362; блок 08:20 до дирижёра не дошёл); ledger — только чтение.\n\n"
        "Правила, заданные до счёта. (а) Целевая среза = min(mSR 2018, mSR 2019). (б) Ось «влияет», если размах mSR 2018 или "
        f"2019 по определённым точкам среза > {D.INFL}. (в) Форма по главной оси (наибольший μ*): соседи — ближайшие определённые "
        f"значения слева/справа от выбранного в порядке сетки; есть сосед лучше более чем на {D.DELTA} — «склон»; все соседи не хуже "
        f"выбранного более чем на {D.DELTA} — «плато»; иначе — «пик». (г) «Сделок» в срезах — круги (стороны/2), `sdelok_y` ledger. "
        "(д) Журнал сделок — один прогон точки итерации 1 (Ж-056), PnL сделки = side·(выход/вход−1) − комиссии, доля номинала, "
        "без веса портфеля.\n")


def jread(p):
    if not p.exists():
        return None
    return [r for r in csv.DictReader(l for l in open(p, encoding="utf-8") if not l.startswith("#"))]


def profile(rows):
    if not rows:
        return NA
    n = len(rows)
    pn = sorted((float(r["pnl"]) for r in rows), reverse=True)
    dur = [(D.pd.Timestamp(r["data_vykhoda"]) - D.pd.Timestamp(r["data_vhoda"])).days for r in rows]
    w, l = [x for x in pn if x > 0], [x for x in pn if x <= 0]
    tot, k = sum(pn), max(1, math.ceil(0.05 * n))
    top = sum(pn[:k])
    ex = {x: sum(r["prichina"] == x for r in rows) / n for x in ("stop", "take", "trail", "bu", "vremya", "signal")}
    return dict(n=n, per_year={y: sum(r["god_vkhoda"] == y for r in rows) for y in ("2018", "2019")}, dur=sum(dur) / n,
                win=len(w) / n, avgw=sum(w) / len(w) if w else float("nan"), avgl=sum(l) / len(l) if l else float("nan"),
                top=top, tot=tot, k=k, share=top / tot if tot > 0 else None, ex=ex)


def prof_md(p, title):
    if p == NA:
        return f"**{title}:** нет журнала\n"
    sh = f"{p['share'] * 100:.1f} %" if p["share"] is not None else f"не определена (суммарный PnL {p['tot']:.4f} ≤ 0; топ-{p['k']} дают {p['top']:.4f})"
    return (f"**{title}:** сделок {p['n']} (2018: {p['per_year']['2018']}, 2019: {p['per_year']['2019']}); средняя длительность "
            f"{p['dur']:.2f} дн; доля прибыльных {p['win'] * 100:.1f} %; средний выигрыш {p['avgw'] * 100:.2f} %, средний проигрыш "
            f"{p['avgl'] * 100:.2f} % (доли номинала); доля PnL топ-5 % сделок (топ-{p['k']}): {sh}.\n\n"
            "Выходы: " + ", ".join(f"{k} {v * 100:.1f} %" for k, v in p["ex"].items()) + "\n")


def gap_md(rows):
    if not rows:
        return NA
    s = {}
    for r in rows:
        c = s.setdefault(r["god_vkhoda"], [0, 0.0])
        c[0] += r["div_gap"] == "да"
        c[1] += float(r["div_gap_dolya"])
    return "; ".join(f"{y}: сделок с гэпом {c[0]}, сумма гэпов {c[1] * 100:.3f} % номинала" for y, c in sorted(s.items()))


def axis_md(a, rows, pt, mor):
    sel = next(i for i, r in enumerate(rows) if r[0] == pt[a])
    lines = [f"\n**{a}** (μ* = {f3(mor[a][0]) if a in mor else '—'}), выбрано {pt[a]}", "", "| значение | mSR 2018 | mSR 2019 | сделок 2018 | сделок 2019 |", "|---|---|---|---|---|"]
    for r in rows:
        mk = "**" if r[0] == pt[a] else ""
        if r[2]:
            lines.append(f"| {mk}{r[0]}{mk} | {r[2]} | | | |")
        else:
            lines.append(f"| {mk}{r[0]}{mk} | {f3(r[3])} | {f3(r[4])} | {r[5]:.1f} | {r[6]:.1f} |")
    return "\n".join(lines), D.shape([D.tsel(r) for r in rows], sel), D.influence(rows)


def one(sid, cal, svod):
    axes, pt, row, mor = D.load_cfg(sid)
    keys = {pkey(pt)} | {pkey({**pt, a: v}) for a, g in axes.items() for v in g} | {pkey({**pt, "k_s": 2.0})}
    got = D.fetch(sid, keys)
    sl = D.slices(sid, axes, pt, got)
    rec = got.get(pkey(pt))
    main_ax = max(mor, key=lambda a: mor[a][0])
    out = [f"# Диагностика {sid}: параметры, срезы, сделки (Ж-055, Ж-056, Ж-057)\n", HEAD, "## 1. Параметры\n",
           "| ось | паспорт A | точка итерации 1 | диапазон сетки | μ* | σ Морриса |", "|---|---|---|---|---|---|"]
    for a, g in axes.items():
        m = mor.get(a)
        out.append(f"| {a} | {D.A_GRID.get(a, 'нет в A (новая ось)')} | {pt[a]} | {g} | {f3(m[0]) if m else '—'} | {f3(m[1]) if m else '—'} |")
    out += ["", "Паспорт A: единой точки нет — в каждом (тикер, год) топ-10 из сетки; в колонке — сетка A.\n", "## 2. Срезы по осям (ledger, остальные в выбранной точке)"]
    nax, shp = 0, NA
    for a in axes:
        md, f, inf = axis_md(a, sl[a], pt, mor)
        nax += fin(inf) and inf > D.INFL
        shp = f if a == main_ax else shp
        out += [md, f"\nразмах mSR по срезу: {f3(inf)}" + (f"; форма (главная ось): **{f}**" if a == main_ax else "")]
    chk = "ledger не содержит точку"
    if rec:
        x = [v for y in sorted(rec["excess_d"]) for v in rec["excess_d"][y]]
        chk = f"mSR по дневному ряду (metriki_e.msr) {f3(ME.msr(x))} vs ledger {f3(rec['msr'])}"
    mo = D.monthly(rec, cal) if rec else None
    jb, jk = jread(ROOT / f"docs/tables/diag/{sid}_sdelki.csv"), jread(ROOT / f"docs/tables/diag/{sid}_sdelki_ks2.csv")
    pb, pk = profile(jb), profile(jk)
    out += ["\n## 3. Профиль сделок (журнал Ж-056)\n", prof_md(pb, "точка итерации 1")]
    out.append(prof_md(pk, "k_s = 2") if pk != NA else "Точка со стопом: второй прогон не нужен.\n")
    out += ["Дивидендные гэпы (Ж-057, `div_gap`; день гэпа — первый торговый день после last_cum_date; эффект −side·dps/close; в PnL не вошёл): "
            + gap_md(jb) + "\n", "PnL по месяцам, % суммы дневных избыточных (ledger):\n"]
    best = worst = NA
    if mo:
        out.append("| месяц | PnL % |\n|---|---|\n" + "\n".join(f"| {m} | {v * 100:.2f} |" for m, v in sorted(mo.items())))
        s = sorted(mo.items(), key=lambda kv: kv[1])
        best, worst = ", ".join(f"{m} {v * 100:+.2f}" for m, v in s[::-1][:2]), ", ".join(f"{m} {v * 100:+.2f}" for m, v in s[:2])
    else:
        out.append(f"нет: длина ряда не совпала с календарём ({NA})")
    out += ["\n## 4. Выходы\n", "Доли по причинам — п. 3 (журнал). Срез k_s = 2 — там же и в ledger: "
            + (f"prichiny ledger {rec['prichiny']}" if rec else NA), "", f"Сверка: {chk}.", "\n## 5. Нет в матрицах",
            "Список сделок и причины выхода в ledger не хранятся (хранятся доли `prichiny`, `sdelok_y`, дневные ряды) — п. 3–4 взяты из журнала Ж-056 (отдельный прогон одной точки). Срезы по осям вне плана ledger — «нет в матрице». mSR при дивидендах — после ветки `pravka-dividendy`."]
    (OUT / f"2026-10-28_diag_{sid}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    svod[sid] = dict(nax=nax, shp=shp, main=main_ax, share=pb["share"] if pb != NA else NA, best=best, worst=worst,
                     pbo=row["pbo"], msr=(row["msr_2018"], row["msr_2019"]), sd=(pb["per_year"] if pb != NA else NA), gap=gap_md(jb))


def peak(sid):
    d = D.pickle.load(open(D.MAIN / f"docs/tables/zh051/_ledger/{sid}/ledger_it1.pkl", "rb"))
    v = [r[1] for (k, p), r in d.items() if isinstance(k, tuple) and not r[1].get("ne_schitana")]
    act = [r for r in v if all(r.get("sdelok_y", {}).get(y, 0) >= 30 for y in (2018, 2019))]   # правило Ж-048: сделок ≥ 30 в год
    m = lambda vs, y: max([r["msr_by_year"].get(y, float("nan")) for r in vs if fin(r["msr_by_year"].get(y))] or [float("nan")])  # noqa: E731
    return (f"{f3(m(v, 2018))} / {f3(m(v, 2019))} (точек {len(v)}); с сделками ≥ 30 в оба года: "
            f"{f3(m(act, 2018))} / {f3(m(act, 2019))} (точек {len(act)})")


def main():
    cal = D.calendar()
    svod = {}
    for k in range(1, 7):
        one(f"S{k}", cal, svod)
    L = ["# Сводка диагностики S1–S6 (Ж-055–058)\n", HEAD,
         "«Пик по сетке» — максимум mSR по всем точкам ledger it1 (потолок, не выбор: подбор и счёт на одних данных). Колонки «после» — после слияния `pravka-dividendy`.\n",
         "| стратегия | влияющих осей (размах > 0.05) | главная ось → форма | доля PnL топ-5 % сделок | лучшие 2 месяца | худшие 2 месяца | пик mSR 2018 / 2019 по сетке (точек) | центр плато mSR 2018 / 2019 | PBO | сделок в год 2018 / 2019 | гэпов дивидендов | после дивидендов |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sid, s in svod.items():
        pk = peak(sid)
        sh = f"{s['share'] * 100:.1f} %" if isinstance(s["share"], float) else "не определена (PnL сделок ≤ 0)"
        sd = f"{s['sd']['2018']} / {s['sd']['2019']}" if s["sd"] != NA else NA
        L.append(f"| {sid} | {s['nax']} | {s['main']} → {s['shp']} | {sh} | {s['best']} | {s['worst']} | {pk} | {f3(float(s['msr'][0]))} / {f3(float(s['msr'][1]))} | {f3(float(s['pbo']))} | {sd} | {s['gap']} | — (ждёт pravka-dividendy) |")
    (OUT / "2026-10-28_diag_svod.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""Разложение издержек по сделкам (Ж-150), справочно, train4 (2016–2019); отбора нет. Данных ≥ 2020 не читает: строки с входом или выходом
≥ 2020-01-01 отбрасываются при загрузке (их число — в `OTBROSHENO`). Издержки = комиссия + плата за шорт; ставка — отдельный столбец
и в «итого после» входит. Доля издержек — от ВАЛОВОГО PnL (до издержек). Sharpe дн. — приближённо: PnL сделки на дату выхода, нулевые
рабочие дни окна включены, ddof=1, ×√252. Единицы PnL — % капитала (голые идеи, вес 1/N) или % x движка (v6)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

D0, D1 = pd.Timestamp("2016-01-01"), pd.Timestamp("2019-12-31")
KORZ = ("1", "2–3", "4–10", ">10")
POROG, OKNO, MIN_OBS, KS = 0.30, 252, 30, (2, 3, 5)
OTBROSHENO: dict = {}
KOLS = ["src", "sid", "tochka", "variant", "tip_vhoda", "tf", "tiker", "vhod", "vyhod", "uderzhanie", "tercil", "gross", "kom", "short", "stavka", "net"]


def korzina(h) -> str:
    return KORZ[0] if h <= 1 else KORZ[1] if h <= 3 else KORZ[2] if h <= 10 else KORZ[3]


def _diapazon(df: pd.DataFrame, nm: str) -> pd.DataFrame:
    df = df.copy()
    for c in ("vhod", "vyhod"):
        df[c] = pd.to_datetime(df[c])
    ok = (df.vhod >= D0) & (df.vyhod <= D1)
    OTBROSHENO[nm] = OTBROSHENO.get(nm, 0) + int((~ok).sum())
    df = df[ok].copy()
    df["korz"] = df.uderzhanie.map(korzina)
    return df.reset_index(drop=True)


def chitat_golaya(path) -> pd.DataFrame:
    """sdelki_<period>.csv драйвера голых идей (первая строка «# json» пропускается)."""
    r = pd.read_csv(path, skiprows=1)
    r = r.rename(columns={"vhod": "rezhim_vhoda"}).rename(columns={"data_vhoda": "vhod", "data_vyhoda": "vyhod", "uderzhanie_dn": "uderzhanie", "tercil_sily": "tercil", "pnl_do_izderzhek": "gross",
                          "komissiya": "kom", "plata_za_short": "short", "pnl_posle": "net"})
    r = r.assign(src="golaya", tochka="", tf=r.tf.fillna(""))
    for c in ("gross", "kom", "short", "stavka", "net"):
        r[c] = r[c] * 100
    return _diapazon(r[KOLS], "golaya")


def chitat_v6(kesh) -> pd.DataFrame:
    """Журналы пересчёта Д-4 (`<Sk>_<seed>_<i>.json`). Типа входа/ранга/силы по сделке нет; плата за шорт и ставка — не по сделке."""
    rows = []
    for p in sorted(Path(kesh).glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        for t in d.get("zhurnal") or []:
            if t["vhod"] and t["vyhod"]:
                rows.append(dict(src="v6", sid=d["sid"], tochka=p.stem, variant="", tip_vhoda=d.get("vhod_tip") or "не измерено по сделке", tf="",
                                 tiker=t["tiker"], vhod=t["vhod"], vyhod=t["vyhod"], uderzhanie=t["uderzhanie_dn"], tercil=np.nan, gross=t["pnl_do_izderzhek"] * 100,
                                 kom=t["komissiya"] * 100, short=np.nan, stavka=np.nan, net=(t["pnl_do_izderzhek"] - t["komissiya"]) * 100))
    return _diapazon(pd.DataFrame(rows, columns=KOLS), "v6")


def dolya(kom: float, short: float, gross: float, baza: str = "valovoj") -> float:
    """Доля издержек: от валового PnL (канон Ж-150) или — мутация — от нетто. gross ≤ 0 → nan."""
    izd, den = kom + short, gross if baza == "valovoj" else gross - kom - short
    return izd / den if gross > 0 and den > 0 else float("nan")


def sharpe_dn(df: pd.DataFrame, col: str) -> float:
    dni = pd.bdate_range(D0, D1)
    s = df.groupby("vyhod")[col].sum().reindex(dni, fill_value=0.0)
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252)) if s.std(ddof=1) > 0 else float("nan")


def _sr(g: pd.DataFrame) -> dict:
    sh, kom, gr = g["short"].sum(min_count=1), g.kom.sum(), g.gross.sum()
    sh0 = 0.0 if np.isnan(sh) else sh
    do = g.gross.fillna(0)
    return dict(n=len(g), gross_sr=g.gross.mean(), kom_sr=g.kom.mean(), short_sr=g["short"].mean(), stavka_sr=g.stavka.mean(), net_sr=g.net.mean(),
                dolya_izd=dolya(kom, sh0, gr), dolya_lt30=bool(dolya(kom, sh0, gr) < POROG), sharpe_do=sharpe_dn(g.assign(x=do), "x"), sharpe_posle=sharpe_dn(g, "net"))


def razrez(df: pd.DataFrame, razdel: str, po: str | None) -> list:
    base = ["src", "sid", "tochka", "variant", "tip_vhoda"] + (["tf"] if razdel == "4_tf" else [])
    out = []
    for k, g in df.groupby(base + ([po] if po else []), dropna=False):
        k = k if isinstance(k, tuple) else (k,)
        r = dict(zip(base + ([po] if po else []), k))
        out.append({"razdel": razdel, **{c: r.get(c, "") for c in ("src", "sid", "tochka", "variant", "tip_vhoda", "tf")}, "srez": str(r.get(po, "")) if po else "все", **_sr(g)})
    return out


def ozhid(g: pd.DataFrame, d, rang, korz: str, lag: int = 1) -> tuple:
    """(E[gross], E[издержки], n) по ячейке (ранг, удержание) среди сделок, ЗАКРЫТЫХ не позже d−lag, за 252 рабочих дня. lag=0 — мутация (день d)."""
    d = np.datetime64(pd.Timestamp(d), "D")
    lo = np.busday_offset(d, -OKNO, roll="forward")
    m = (g.vyhod.values.astype("datetime64[D]") <= d - np.timedelta64(lag, "D")) & (g.vyhod.values.astype("datetime64[D]") > lo) & (g.korz.values == korz)
    m &= (g.tercil.fillna(-1).values == (-1 if pd.isna(rang) else rang))
    h = g[m]
    return (h.gross.mean(), (h.kom + h["short"].fillna(0)).mean(), len(h)) if len(h) >= MIN_OBS else (np.nan, np.nan, len(h))


def k_okup(df: pd.DataFrame, k: int, lag: int = 1) -> list:
    """Гипотеза v7 (справочно): вход разрешён, если E[gross | ранг, удержание] ≥ k × E[издержки]. Корзина удержания известна лишь после сделки — потолок."""
    out = []
    for key, g in df[df.variant.isin(["", "centr"])].groupby(["src", "sid", "tochka", "variant", "tip_vhoda"]):
        g = g.sort_values("vhod").reset_index(drop=True)
        ok = np.array([(lambda e: bool(e[0] >= k * e[1]) if not np.isnan(e[0]) and e[1] > 0 else False)(ozhid(g, r.vhod, r.tercil, r.korz, lag)) for r in g.itertuples()])
        p, sh = g[ok], g["short"].fillna(0)
        out.append({"razdel": "5_k_okup", "src": key[0], "sid": key[1], "tochka": key[2], "variant": key[3], "tip_vhoda": key[4], "tf": "", "srez": f"k={k}",
                    "n": len(g), "n_proshlo": int(ok.sum()), "dolya_proshlo": ok.mean() if len(g) else np.nan, "gross_sr": p.gross.mean(), "net_sr": p.net.mean(),
                    "kom_sr": p.kom.mean(), "short_sr": p["short"].mean(), "dolya_izd": dolya(p.kom.sum(), sh[ok].sum(), p.gross.sum()) if len(p) else np.nan})
    return out


def itog(df: pd.DataFrame) -> str:
    """Итоговая строка по сделкам центров голых идей и v6 (без вариантов входа T): издержки/валовой; граница удержания и ранга, где доля < 30 %."""
    d = df[df.variant.isin(["", "centr"])]
    gr = d.gross.sum()
    x = f"{dolya(d.kom.sum(), d['short'].fillna(0).sum(), gr) * 100:.0f}" if gr > 0 else "не определено (валовой ≤ 0)"
    n = next((b for b in KORZ if dolya(*(lambda s: (s.kom.sum(), s["short"].fillna(0).sum(), s.gross.sum()))(d[d.korz == b])) < POROG), None)
    dr = d[d.tercil.notna()]
    p = next((t for t in (1, 2, 3) if dolya(*(lambda s: (s.kom.sum(), s["short"].fillna(0).sum(), s.gross.sum()))(dr[dr.tercil == t])) < POROG), None)
    nn = {"1": 1, "2–3": 2, "4–10": 4, ">10": 11}.get(n)
    return (f"издержки съедают {x} % валового; окупаются сделки с удержанием ≥ {nn if nn else 'не достигнуто'} дней "
            f"и рангом ≥ {p if p else 'не достигнуто'} (терциль; доля издержек < 30 % валового)")


def tablica(df: pd.DataFrame) -> pd.DataFrame:
    rows = razrez(df, "1_sk_tip", None) + razrez(df, "2_uderzhanie", "korz") + razrez(df[df.tercil.notna()], "3_tercil", "tercil")
    rows += razrez(df[df.sid.isin(["S10", "S11", "S12"])], "4_tf", None)
    for k in KS:
        rows += k_okup(df, k)
    t = pd.DataFrame(rows)
    t["primechanie"] = np.where(t.src == "v6", "v6: ранг/сила/тип входа не измерены по сделке; плата за шорт и ставка — не по сделке (в x базой); комиссия строки делится "
                                "между событиями поровну (приближённо)", "")
    return pd.concat([pd.DataFrame([{"razdel": "0_itog", "srez": itog(df)}]), t], ignore_index=True)

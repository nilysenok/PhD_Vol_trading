"""Л3.3 (Ж-146 п. 3, Ж-093, Ж-149 п. 1): драйвер A′-расширенной поверх дневных рядов центров S1–S12, train4 (2016–2019).
Запуск: python3 scripts/analysis/a_shtrih_v7.py <вход.npz> [папка] -> docs/tables/a_shtrih_v7/a_shtrih_v7.csv. Прогон на данных —
отдельным заданием; здесь только код и синтетика.

Вход npz: X[S,N,T] excess пар, poz[S,N,T] ЗНАКОВЫЕ позиции, sektora[N], imoex[T], mix[T], ks[T], daty[T] (datetime64[D]),
znak="znakovyj", istochnik (не "ledger": в ledger 2.1/2.2 дневных рядов и знака позиции нет). |позиция| и данные ≥ 2020 — отказ.
Строки: основная (паритет LW + сектор + таргет + выключатель + хедж), контроли «равный риск» и «без хеджа», аблация вперёд
1 → 4 (1 равный вес; 2 + паритет; 3 + нейтрализация: хедж и сектор; 4 + таргет и выключатель = основная). Метрики — `metriki_e`
(Д7) и диагностика Ж-148 `metriki_diag` (в пороги не входит). «Доля рынка» = R² ряда к IMOEX (Ж-093 п. 1в; до = аблация 1,
после = основная)."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import a_shtrih_v7_kniga as K, metriki_diag as md, metriki_e as me  # noqa: E402

GRANICA = np.datetime64("2020-01-01")
VYKL = dict(hedzh=False, sektor_netto=False, tsel_vol=False, vykl=False)
KONF = {
    "osnovnaya": dict(vesa="paritet"),
    "kontrol_ravnyj_risk": dict(vesa="ravnyj_risk"),
    "kontrol_bez_hedzha": dict(vesa="paritet", hedzh=False),
    "ablaciya_1": dict(vesa="vykl", **VYKL),
    "ablaciya_2": dict(vesa="paritet", **VYKL),
    "ablaciya_3": dict(vesa="paritet", **{**VYKL, "hedzh": True, "sektor_netto": True}),
    "ablaciya_4": dict(vesa="paritet"),
}
KOLONKI = ("sharpe", "msr", "sortino", "mdd", "kalmar", "sharpe_ned", "msr_ned", "sortino_ned", "n_nedel", "sharpe_norm",
           "ekspoziciya", "cena_prostoev", "prichina_nan", "dolya_rynka", "kom_tik_god", "kom_hedzh_god", "n_dnej", "mix_nan_dnej")
POLYA_SHAPKI = ("status", "okno", "komissiya", "hash_vhodov", "hash_koda", "stroki", "metriki")
KLYUCHI = ("X", "poz", "sektora", "imoex", "mix", "ks", "daty")


def _proverit_daty(daty) -> None:
    daty = np.asarray(daty).astype("datetime64[D]")
    if daty.size == 0 or (daty >= GRANICA).any():
        raise ValueError(f"данные ≥ 2020-01-01 или пустая лента: максимум {daty.max() if daty.size else None}; train4 — до 2019")


def _proverit_znak(znak, istochnik, poz) -> None:
    if znak != "znakovyj":
        raise ValueError(f"вход не помечен znak='znakovyj' (получено {znak!r}): |позиция| как вход — отказ")
    if "ledger" in str(istochnik).lower():
        raise ValueError("источник ledger: в ledger 2.1/2.2 нет знака позиции и дневных рядов — вход из рядов центров")
    if poz is None or not (np.nanmin(np.asarray(poz, float)) < 0):
        raise ValueError("в poz нет ни одного шорта: похоже на |позиция|, нужны знаковые позиции")


def proverit_vhod(d: dict) -> None:
    """Дата раньше всего остального (G3), затем знак (G2), затем формы (проверяет движок книги)."""
    _proverit_daty(d["daty"])
    _proverit_znak(d.get("znak"), d.get("istochnik", ""), d.get("poz"))
    X = np.asarray(d["X"], float)
    if X.ndim != 3 or np.asarray(d["daty"]).shape != (X.shape[2],):
        raise ValueError("X должна быть [S,N,T], daty[T]")


def zagruzit(src) -> dict:
    """src — путь к npz или отображение. Сначала daty/znak/istochnik: при отказе массивы данных не читаются."""
    if isinstance(src, (str, Path)):
        src = np.load(src, allow_pickle=False)
    daty = src["daty"]
    _proverit_daty(daty)
    d = {"daty": np.asarray(daty).astype("datetime64[D]"), "znak": str(src["znak"]), "istochnik": str(src["istochnik"])}
    _proverit_znak(d["znak"], d["istochnik"], None if "poz" not in src else src["poz"])
    d.update({k: np.asarray(src[k]) for k in KLYUCHI if k != "daty"})
    proverit_vhod(d)
    return d


def schitat(d: dict) -> dict:
    proverit_vhod(d)
    a = (d["X"], d["poz"], d["sektora"], d["imoex"], d["mix"], d["ks"])
    return {nazv: K.kniga_ryada(*a, **kw) for nazv, kw in KONF.items()}


def dolya_rynka(r, imoex) -> float:
    """Доля дисперсии ряда от рынка: R² = corr(r, IMOEX)² по дням, где оба известны; знак не влияет, нулевая дисперсия -> nan."""
    r, im = np.asarray(r, float), np.asarray(imoex, float)
    ok = np.isfinite(r) & np.isfinite(im)
    if ok.sum() < 3 or r[ok].std() < 1e-12 or im[ok].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(r[ok], im[ok])[0, 1] ** 2)


def stroka(res, d: dict) -> dict:
    ok = np.isfinite(res.ryad)
    x, daty = res.ryad[ok], np.asarray(d["daty"])[ok]
    poz = (np.abs(res.poz_itog).sum(0) * res.m * res.f + np.abs(res.h))[ok]
    ned = me.metriki_ned(x, daty)
    return dict(sharpe=me.sharpe(x), msr=me.msr(x), sortino=me.sortino_e(x), mdd=me.mdd(x), kalmar=me.kalmar(x),
                sharpe_ned=ned["sharpe_ned"], msr_ned=ned["msr_ned"], sortino_ned=ned["sortino_ned"], n_nedel=ned["n_nedel"],
                sharpe_norm=md.sharpe_norm(x, poz), ekspoziciya=md.ekspoziciya(poz), cena_prostoev=md.cena_prostoev(x, poz),
                prichina_nan=md.prichina_norm(poz) or "", dolya_rynka=dolya_rynka(res.ryad, d["imoex"]),
                kom_tik_god=float(res.kom_tik.mean() * 252), kom_hedzh_god=float(res.kom_hedzh.mean() * 252),
                n_dnej=int(ok.sum()), mix_nan_dnej=int(res.mix_nan.sum()))


def tablica(d: dict, rez: dict) -> pd.DataFrame:
    return pd.DataFrame({nazv: stroka(r, d) for nazv, r in rez.items()}).T[list(KOLONKI)]


def _hash(parts) -> str:
    h = hashlib.sha256()
    for nazv, a in parts:
        a = np.ascontiguousarray(a)
        h.update(f"{nazv}|{a.dtype}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def shapka(d: dict) -> dict:
    kod = [(p.name, np.frombuffer(p.read_bytes(), np.uint8)) for p in
           (ROOT / "fastbt" / "a_shtrih_v7_kniga.py", ROOT / "fastbt" / "a_shtrih_v7.py", ROOT / "fastbt" / "kniga_msp.py",
            Path(__file__))]
    daty = np.asarray(d["daty"]).astype("datetime64[D]")
    return dict(status="разработка train4 2016–2019, до аудита, потолок (на ряде центров, не на движке)",
                okno=f"{daty.min()} .. {daty.max()}, {daty.size} дней",
                komissiya="0,04 % на каждое изменение позиции (Ж-149): хедж — полностью, тикеры — сверх оплаченного центрами",
                hash_vhodov=_hash([(k, d[k]) for k in KLYUCHI]), hash_koda=_hash(kod),
                stroki=", ".join(KONF), metriki="metriki_e (Д7) + metriki_diag (Ж-148, диагностика)")


def zapisat(tab: pd.DataFrame, sh: dict, papka) -> Path:
    papka = Path(papka)
    papka.mkdir(parents=True, exist_ok=True)
    p = papka / "a_shtrih_v7.csv"
    p.write_text("".join(f"# {k}: {sh[k]}\n" for k in POLYA_SHAPKI) + tab.to_csv(index_label="stroka"), encoding="utf-8")
    return p


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    d = zagruzit(argv[0])
    out = zapisat(tablica(d, schitat(d)), shapka(d), argv[1] if len(argv) > 1 else ROOT / "docs" / "tables" / "a_shtrih_v7")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Б5 (Ж-130, Ж-131 п. 1): свод возврата сервера (`zabrat.sh` → data/server_rezultaty/<дата>/) на Маке. Счёта Соболя нет — только
пост-обработка ledger: калибратор v5 (`kalibr_v5`), контроли к1/к2 (`kontroli_v5`), к3 (проход k3), второй seed (`kontroli_v5.seed2`).
Проход = каталог с manifest.json (`<корень>/<sid>/`), ключ (sid, period, k3, sobol_klyuch); раскладка каталогов любая.
ledger 2.0 (ряды) — полный калибратор; ledger 2.1 (суммы сегментов по кварталам) — только фаза 1 по окну [2016; 31.12(t−1)]
(`Chitatel.okno`): плато и голос требуют дневных рядов, их в 2.1 нет — ↺ класс B, откат: пересчёт на 2.0.
Таблицы: docs/tables/server_v5/*.csv; шапка из строк `#` — статус, код (sha файлов), хэш входов (proverka.md п. 4)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from fastbt import canon_v5 as cv5, kalibr_v5 as K, kontroli_v5 as kv, ledger_v2 as lm
from fastbt.kalibr_v5 import PERIODY

STATUS = "разработка train4 / В2, после аудита"
OUT = Path("docs/tables/server_v5")
FAJLY = ("svod_server_v5.py", "kalibr_v5.py", "kontroli_v5.py", "ledger_v2.py")
TOP = 10


def kod_sha() -> str:
    d = Path(__file__).parent
    return hashlib.sha256(b"".join((d / f).read_bytes() for f in FAJLY)).hexdigest()[:12]


def vhod_sha(root: Path, sid: str) -> str:
    h, d = hashlib.sha256(), Path(root) / sid
    for f in ("manifest.json", lm.SUMS, "daty.npy"):
        h.update((d / f).read_bytes() if (d / f).exists() else b"-")
    return h.hexdigest()[:16]


def prohody(katalog: Path) -> list:
    """Все ledger под каталогом возврата: [{root, sid, kl, v, vhod}], порядок стабильный."""
    out = []
    for mp in sorted(Path(katalog).rglob("manifest.json")):
        mf = json.loads(mp.read_text(encoding="utf-8"))
        if mf.get("v") not in (lm.V, lm.V21):
            continue
        root, sid = mp.parent.parent, mp.parent.name
        kl = (sid, mf.get("period"), bool(mf.get("k3")), mf.get("sobol_klyuch"))
        out.append(dict(root=root, sid=sid, kl=kl, v=mf["v"], vhod=vhod_sha(root, sid), mf=mf))
    return out


def osi_kuba(sid: str) -> dict:
    return {n: [lm.tok(v) for v in vals] for n, vals in cv5.osi(cv5.load_kub(sid))}


def imya(kl: tuple) -> str:
    return f"{kl[0]}|{kl[1]}|{'k3' if kl[2] else 'v5'}|{kl[3]}"


def nachalo(period: str) -> str:
    return PERIODY[period][0]


def proc(period: str):
    return lambda p, g: kv.kalibr(p, g, nachalo(period))


def tablica_kalibr(pr: dict, pul, period: str) -> list:
    rows = []
    for t, r in K.kalibrovat(pul, period).items():
        for k, x in r["tikery"].items():
            for rang, (kl, pl) in enumerate(zip(x["chleny"], x["plato"]) if x else (), 1):
                rows.append([imya(pr["kl"]), t, *(r["okno"] or ("", "")), k, rang, kl, f"{pl:.6f}", r["n_proby"]])
    return rows


def tablica_kontroli(pr: dict, pul, period: str, n_shift: int) -> list:
    v = kv.kalibr(pul, kv.GODY, nachalo(period))
    k2 = kv.k2(pul, v, n_shift)
    return [[imya(pr["kl"]), f"{k2['sharpe']:.6f}", f"{kv.k1(pul):.6f}", f"{k2['p']:.6f}", n_shift, len(pul.tochki)]]


def okna_v21(pr: dict) -> list:
    """Фаза 1 на 2.1: ворота v4 (sh > 0, mdd > −30 %, экспозиция > 5 %) и top-10 по Шарпу окна, год t = 2017–2019."""
    ch, rows, out = lm.Chitatel(pr["root"], pr["sid"]), None, []
    rows = ch.rows()
    kom = float(rows[0]["klyuch"]["komissiya"]) if rows and ch.manifest.get("poz_konec") else 0.0
    for t in (2017, 2018, 2019):
        sh, mdd, ex, _ = (np.asarray(a, float) for a in ch.okno(t, rows, "port", kom))
        ok = (sh > 0) & (mdd > K.MDD_MIN) & (ex > K.EXP_MIN)
        idx = [i for i in np.argsort(-sh, kind="stable") if ok[i]][:TOP]
        out += [[imya(pr["kl"]), t, rang, json.dumps(rows[i]["klyuch"], sort_keys=True, ensure_ascii=False),
                 f"{sh[i]:.6f}", f"{mdd[i]:.6f}", f"{ex[i]:.4f}"] for rang, i in enumerate(idx, 1)]
    return out


def sravnenie(pary: list) -> list:
    """Расхождение Шарпа ансамбля по годам: seed 1 против seed 2 (одна стратегия, период, не k3)."""
    out = []
    for (a, pa), (b, pb) in pary:
        for t, d in kv.seed2(proc(a["kl"][1]), pa, pb).items():
            out.append([imya(a["kl"]), imya(b["kl"]), t, f"{d['sharpe1']:.6f}", f"{d['sharpe2']:.6f}", f"{d['rashozhdenie']:.6f}"])
    return out


def zapis(path: Path, shapka: list, kolonki: list, rows: list, vhody: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        for s in (f"# статус: {STATUS}", f"# код: {kod_sha()}", f"# входы: {' '.join(sorted(set(vhody)))}", *shapka):
            f.write(s + "\n")
        w = csv.writer(f, delimiter=";")
        w.writerow(kolonki)
        w.writerows(rows)


def svod(katalog: Path, out: Path = OUT, osi: dict | None = None, n_shift: int = 199) -> list:
    """Таблицы по всем проходам каталога возврата; osi[sid] — оси (по умолчанию из kub_<sid>.yaml). Возвращает пути."""
    prs, kal, kon, okn, pul_ = prohody(katalog), [], [], [], {}
    for pr in prs:
        sid, period = pr["sid"], pr["kl"][1]
        if pr["v"] == lm.V21:
            okn += okna_v21(pr)
            continue
        pul = K.pul_iz_ledger(pr["root"], sid, (osi or {}).get(sid) or osi_kuba(sid))
        pul_[pr["kl"]] = (pr, pul)
        kal += tablica_kalibr(pr, pul, period)
        kon += tablica_kontroli(pr, pul, period, n_shift)
    pary = [(pul_[k], pul_[(k[0], k[1], False, "seed2")]) for k in pul_ if not k[2] and k[3] == "seed" and (k[0], k[1], False, "seed2") in pul_]
    vh = [p["vhod"] for p in prs]
    seed = sravnenie(pary)
    spec = [("kalibr_top10.csv", ["prohod", "god", "okno_s", "okno_po", "tiker", "rang", "tochka", "plato", "n_proby"], kal),
            ("kontroli_k1_k2_k3.csv", ["prohod", "sharpe_ansambl_k2", "k1_sharpe", "k2_p", "n_shift", "tochek"], kon),
            ("okna_ledger21.csv", ["prohod", "god", "rang", "tochka", "sharpe", "mdd", "expo_pct"], okn),
            ("seed2_ustojchivost.csv", ["prohod1", "prohod2", "god", "sharpe1", "sharpe2", "rashozhdenie"], seed)]
    paths = []
    for name, kol, rows in spec:
        zapis(Path(out) / name, [], kol, rows, vh)
        paths.append(Path(out) / name)
    return paths


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m fastbt.svod_server_v5 <каталог возврата>")
    for p in svod(Path(sys.argv[1])):
        print(p)

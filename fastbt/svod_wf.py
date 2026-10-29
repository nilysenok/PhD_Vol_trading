"""Свод walk-forward (предрегистрация v2, §4–§7) по каталогу возврата сервера: ledger 2.1 (v5) и 2.2 (v6), period = wf, проходы s1 / s2 / k3 / spravka.
Расчёт — `svod_wf_raschet`; каталог, N и шапка — здесь; `svod_server_v5` переиспользован (хэш входов, оси куба). Таблицы — docs/tables/wf/*.csv.
Проход определяется manifest: k3 → k3; sobol_klyuch seed2 → s2; spravka → spravka; иначе s1. spravka и роль kontrol в N не входят (Ж-132 п. 4).
Основная сетка — v6 (У-11), v5 — справочно; N = Obedinenie(v5 ∪ v6, без spravka). Оси: osi[sid] или куб `kub_<sid>.yaml` (v6 без куба — передать osi)."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from fastbt import ledger_v2 as lm, svod_server_v5 as SV, svod_wf_raschet as R

STATUS = "после аудита / тест-wf"
OUT = Path("docs/tables/wf")
PREREG = Path(__file__).resolve().parents[1] / "docs/preregistraciya_v2.md"
FAJLY = ("svod_wf.py", "svod_wf_raschet.py", "svod_server_v5.py", "kalibr_v5_21.py", "pbo_wf.py", "ledger_v2.py", "kontroli_v5.py")


def rol_prohoda(mf: dict) -> str:
    if mf.get("sobol_klyuch") == "spravka" or mf.get("spravka"):
        return "spravka"
    return "k3" if mf.get("k3") else ("s2" if mf.get("sobol_klyuch") == "seed2" else "s1")


def sha_koda() -> str:
    d = Path(__file__).parent
    return hashlib.sha256(b"".join((d / f).read_bytes() for f in FAJLY)).hexdigest()[:12]


def sha_prereg() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()[:12] if PREREG.exists() else "-"


def prohody(katalog: Path) -> dict:
    """{sid: {(setka, prohod): {chitatel, vhod}}}: только ledger с суммами и period = wf."""
    out = {}
    for mp in sorted(Path(katalog).rglob("manifest.json")):
        mf = json.loads(mp.read_text(encoding="utf-8"))
        if mf.get("v") not in lm.SUMMY_V or mf.get("period") != "wf":
            continue
        root, sid = mp.parent.parent, mp.parent.name
        setka = "v6" if mf["v"] == lm.V22 else "v5"
        out.setdefault(sid, {})[setka, rol_prohoda(mf)] = dict(ch=lm.Chitatel(root, sid), vhod=SV.vhod_sha(root, sid))
    return out


def zapis(path: Path, kolonki: list, rows: list, vhody: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shapka = (f"# статус: {STATUS}", f"# код: {sha_koda()}", f"# входы: {' '.join(sorted(set(vhody)))}", f"# предрегистрация v2 sha256: {sha_prereg()}",
              "# Шарп ансамбля — нижняя оценка из сумм кварталов (см. svod_wf_raschet); результатов сервера здесь нет, если вход синтетический")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.writelines(s + "\n" for s in shapka)
        w = csv.writer(f, delimiter=";")
        w.writerow(kolonki)
        w.writerows(rows)


def f6(x) -> str:
    return f"{x:.6f}"


def po_sid(sid: str, ps: dict, osi: dict) -> dict:
    """Строки таблиц одной стратегии."""
    t = dict(okna=[], itog=[], kontroli=[], seed2=[], dsr_pbo=[])
    pul, res = {}, {}
    for (setka, pr), x in sorted(ps.items()):
        if pr == "spravka":
            continue
        pul[setka, pr] = p = R.pul_wf(x["ch"], osi)
        res[setka, pr] = w = R.wf(p)
        t["okna"] += [[sid, setka, pr, g, len(v["sel"]), v["n_proby"], v["okno"][1], f6(w["god"].get(g, float("nan")))] for g, v in w["vybor"].items()]
        t["itog"].append([sid, setka, pr, f6(w["sharpe"]), f6(w["mediana"]), int(w["rezultat"]), int(w["dni"])])
    for (setka, pr), p in pul.items():
        if pr != "s1":
            continue
        k2 = R.k2(p, res[setka, pr]["vybor"])
        t["kontroli"] += [[sid, setka, "k1", f6(R.k1(p)["sharpe"]), "", ""], [sid, setka, "k2", f6(k2["sharpe"]), f6(k2["p"]), len(k2["perm"])],
                          [sid, setka, "odna_tochka", f6(R.odna_tochka(p)["sharpe"]), "", ""]]
        if (setka, "s2") in res:
            t["seed2"] += [[sid, setka, g, f6(a), f6(b), f6(abs(a - b))] for g, (a, b) in R.seed2(res[setka, "s1"], res[setka, "s2"]).items()]
    osn = "v6" if ("v6", "s1") in res else "v5"
    chs = [x["ch"] for (s, pr), x in sorted(ps.items()) if pr != "spravka"]
    ob = lm.Obedinenie(*chs)
    n_osn = lm.Obedinenie(*[x["ch"] for (s, pr), x in ps.items() if s == osn and pr != "spravka"]).n_k()
    if (osn, "s1") in res:
        r, nb = res[osn, "s1"], pul[osn, "s1"].nbars
        d, b = R.dsr_wf(ob, nb, r["sharpe"], r["dni"]), R.pbo_ob(ob, chs, osi, sid)
        t["dsr_pbo"].append([sid, d["N"], n_osn, f6(d["V"]), f6(d["sr_zvezda"]), f6(d["dsr"]), f6(d["dsr_n1"]), f6(b["pbo"]), b["n"], b["n_propusk"]])
    return t


def svod(katalog: Path, out: Path = OUT, osi: dict | None = None) -> list:
    """Таблицы по каталогу возврата; osi[sid] — оси (по умолчанию куб sid). Возвращает пути."""
    ps, acc = prohody(katalog), dict(okna=[], itog=[], kontroli=[], seed2=[], dsr_pbo=[])
    for sid, x in ps.items():
        for k, v in po_sid(sid, x, (osi or {}).get(sid) or SV.osi_kuba(sid)).items():
            acc[k] += v
    vh = [y["vhod"] for x in ps.values() for y in x.values()]
    spec = [("wf_okna.csv", ["sid", "setka", "prohod", "god", "tikerov_s_chlenami", "n_proby", "okno_po", "sharpe_nizhn"], "okna"),
            ("wf_itog.csv", ["sid", "setka", "prohod", "sharpe_oos", "mediana_2017_2025", "rezultat_p7", "dni_oos"], "itog"),
            ("wf_kontroli.csv", ["sid", "setka", "kontrol", "sharpe", "p", "n_sdvigov"], "kontroli"),
            ("wf_seed2.csv", ["sid", "setka", "god", "sharpe_s1", "sharpe_s2", "rashozhdenie"], "seed2"),
            ("wf_dsr_pbo.csv", ["sid", "N", "N_osnovnaya", "V_dnevnoj", "sr_zvezda", "dsr", "dsr_N1_spravochno", "pbo", "razbienij", "propusk"], "dsr_pbo")]
    for name, kol, key in spec:
        zapis(Path(out) / name, kol, acc[key], vh)
    return [Path(out) / s[0] for s in spec]


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m fastbt.svod_wf <каталог возврата>")
    for p in svod(Path(sys.argv[1])):
        print(p)

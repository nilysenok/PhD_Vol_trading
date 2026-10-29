"""Ж-131 п. 5: сводка по ledger серверного Соболя (fastbt.progon_server) — одна таблица по каталогам <Sk>/ любых проходов:
train4 (ledger 2.1) s1/s2, v2 (ledger 2.0) s1/s2, k3. Аргументы — каталоги: <out> (с <Sk>/), сам <Sk>/ или папка с несколькими <out>.
Сверка: шарды против SHA256SUMS (Chitatel) и полнота шардов (файл из SHA256SUMS есть, лишних нет), один kod и один vhody
на каталог, чекпоинт = manifest по kod/vhody/kub_sha/seed/периоду. «несовм.» = точки [0; чекпоинт], которых нет в ledger (canon_v5).
Шарп справочно, без выбора точек: train4 — окно [2016; 2019] (ledger 2.1, stats.window, comm = 0); v2 (ledger 2.0) — ряд портфеля
за весь период v2 [okno_s; okno_po] manifest, √252, ddof = 1 (как stats.window), комиссия уже в ряду.
Запуск: python3 scripts/server/svodka_ledger.py <каталог>.. [--top N]; код выхода 1 — расхождение, отказ читателя или пустой вход."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastbt import ledger_v2 as lm  # noqa: E402

KOLONKI = ["prohod", "strategiya", "tochek", "ok", "oshibok", "nesovm", "unikalnyh", "chekpoint_i", "sverka", "kod", "vhody",
           "obyom_mb", "vremya_ch", "s_na_tochku_p50", "yadro_pct", "top_sharpe_okna"]
SVERKA_CHP = ("kod", "vhody", "kub_sha", "sobol_seed", "period")
SHAPKA = ("Шарп справочно, без выбора: train4 (ledger 2.1) — окно [2016; 2019], stats.window, comm = 0; "
          "v2 (ledger 2.0) — ряд портфеля за весь период v2 [okno_s; okno_po] manifest, √252, ddof = 1")


def obyom_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6


def metka(mf: dict) -> str:
    if mf.get("rol") == lm.SPRAVKA:      # справочная строка (Ж-137 п. 6): отдельной строкой, вне N
        return f"spravka {mf.get('t_vyhod')} {mf.get('period', '?')}"
    return "k3" if mf.get("k3") else f"{mf.get('period', '?')} s{2 if mf.get('sobol_klyuch') == 'seed2' else 1}"


def sverka_chekpointa(d: Path, mf: dict) -> list:
    """Поля чекпоинта, не совпавшие с manifest.json (пусто — совпали или чекпоинта нет)."""
    p = d / "checkpoint.json"
    if not p.exists():
        return []
    chp = json.loads(p.read_text(encoding="utf-8"))
    return [k for k in SVERKA_CHP if k in mf and chp.get(k) != mf[k]]


def polnota_shardov(d: Path) -> list:
    """Файлы из SHA256SUMS, которых нет, и .npy шардов, которых нет в SHA256SUMS (хэши сверяет Chitatel)."""
    sm = lm._sums(d)
    est = {f.relative_to(d).as_posix() for sub in ("port", "tik") for f in d.glob(f"p*/{sub}/*.npy")}   # zaem/diag — не в SHA256SUMS
    return [f"нет файла {k}" for k in sorted(sm) if not (d / k).exists()] + [f"нет в SHA256SUMS {k}" for k in sorted(est - set(sm))]


def sharpe_ryada(m: np.ndarray) -> np.ndarray:
    """m [T, n] → Шарп √252 с ddof = 1 по столбцам; нулевой разброс — 0 (как stats.window)."""
    sd = m.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd < 1e-12, 0.0, m.mean(axis=0) / sd * np.sqrt(252.0))


def top_okna(ch, n: int) -> str:
    """«sobol_i:Шарп» лучших n строк ok — справочно (см. SHAPKA); пусто или n ≤ 0 — «—»."""
    rows = ch.rows()
    if not rows or n <= 0:
        return "—"
    sh = np.asarray(ch.okno(2020, rows)[0] if ch.v in lm.SUMMY_V else sharpe_ryada(ch.matrica(rows)), float)
    por = sorted(range(len(rows)), key=lambda j: (np.isnan(sh[j]), -np.nan_to_num(sh[j])))[:n]
    return " ".join(f"{rows[j].get('sobol_i')}:{sh[j]:.3f}" for j in por)


def nesovmestnyh(vse: list, posl) -> object:
    """Точки [0; posl], которых нет в ledger (выброшены canon_v5); без чекпоинта — «—»."""
    return "—" if posl is None else posl + 1 - len({r["sobol_i"] for r in vse if r.get("sobol_i", posl + 1) <= posl})


def dolya_yadra(vse: list) -> object:
    """Доля строк (все статусы), посчитанных ядром быстрого слоя (sloy = fast), %; пусто — «—». Строка без sloy — не ядро."""
    return f"{100 * sum(r.get('sloy') == 'fast' for r in vse) / len(vse):.1f}" if vse else "—"


def yadro_stroki(paths) -> list:
    """['<Sk>: 87.5 % (n из N) — <проход>'] по каталогам; ледже не читается — «нет данных»."""
    out = []
    for r, s in (x for p in paths for x in nayti(p)):
        try:
            vse = lm.Chitatel(r, s).rows(None)
        except ValueError as e:
            out.append(f"{s}: нет данных ({str(e)[:60]})")
            continue
        n = sum(x.get("sloy") == "fast" for x in vse)
        out.append(f"{s}: {dolya_yadra(vse)} % ({n} из {len(vse)}) — {r.name}")
    return out


def stroka(root: Path, sid: str, top: int) -> dict:
    d = root / sid
    mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    chp = d / "checkpoint.json"
    posl = json.loads(chp.read_text(encoding="utf-8"))["posledniy_i"] if chp.exists() else None
    out = dict(prohod=metka(mf), strategiya=sid, obyom_mb=f"{obyom_mb(d):.1f}", chekpoint_i="—" if posl is None else posl)
    raz = sverka_chekpointa(d, mf)
    if (pol := polnota_shardov(d)) or raz:
        return {**out, "sverka": f"ОТКАЗ: {'; '.join(pol[:3])}"[:120] if pol else f"чекпоинт ≠ manifest: {raz}"}
    try:
        ch = lm.Chitatel(root, sid)
    except ValueError as e:                     # отказ читателя — это и есть результат сверки, строка его показывает
        return {**out, "sverka": f"ОТКАЗ: {e}"[:120]}
    vse = ch.rows(None)
    t = np.array([r.get("t_sek", np.nan) for r in vse], float)
    out.update(tochek=len(vse), ok=len(ch.rows()), oshibok=sum(r.get("status") == "oshibka" for r in vse),
               nesovm=nesovmestnyh(vse, posl), unikalnyh=len({r["klyuch_sha"] for r in vse}), kod=str(mf["kod"])[:12],
               vhody=str(mf["vhody"])[:12], sverka="ok", vremya_ch=f"{np.nansum(t) / 3600:.3f}",
               s_na_tochku_p50=f"{np.nanmedian(t):.3f}" if t.size else "—", yadro_pct=dolya_yadra(vse), top_sharpe_okna=top_okna(ch, top))
    return out


def nayti(path) -> list:
    """[(root, sid)]: сам <Sk>/ (есть manifest.json), каталог <out> с <Sk>/ или папка с несколькими <out>."""
    p = Path(path)
    if (p / "manifest.json").exists():
        return [(p.parent, p.name)]
    sub = sorted(q for q in p.iterdir() if q.is_dir()) if p.is_dir() else []
    if any((q / "manifest.json").exists() for q in sub):
        return [(p, q.name) for q in sub if (q / "manifest.json").exists()]
    return [x for q in sub for x in nayti(q)]


def svodka_vse(paths, top: int = 5) -> list:
    pary = [x for p in paths for x in nayti(p)]
    rows = [stroka(r, s, top) for r, s in pary]
    chislo = [r for r in rows if isinstance(r.get("tochek"), int)]
    rows.append(dict(prohod="—", strategiya="ИТОГ", tochek=sum(r["tochek"] for r in chislo), ok=sum(r["ok"] for r in chislo),
                     oshibok=sum(r["oshibok"] for r in chislo),
                     sverka="ok" if pary and all(r.get("sverka") == "ok" for r in rows) else "есть расхождения",
                     obyom_mb=f"{sum(float(r['obyom_mb']) for r in rows):.1f}",
                     vremya_ch=f"{sum(float(r['vremya_ch']) for r in chislo):.3f}"))
    return rows


def svodka(root, top: int = 5) -> list:
    return svodka_vse([root], top)


def tablica(rows: list) -> str:
    lines = [SHAPKA, "", "| " + " | ".join(KOLONKI) + " |", "|" + "---|" * len(KOLONKI)]
    lines += ["| " + " | ".join(str(r.get(k, "—")) for k in KOLONKI) + " |" for r in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Сводка ledger серверного Соболя (справочно, без выбора)")
    ap.add_argument("katalog", type=Path, nargs="+")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--yadro", action="store_true", help="только доля точек ядра по Sk (по строке на Sk), без таблицы")
    a = ap.parse_args(argv)
    if a.yadro:
        print("\n".join("ядро " + x for x in yadro_stroki(a.katalog)))
        return 0
    rows = svodka_vse(a.katalog, a.top)
    print(tablica(rows))
    return 0 if rows[-1]["sverka"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

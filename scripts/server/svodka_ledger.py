"""Ж-131 п. 5: сводка по ledger серверного Соболя (fastbt.progon_server) — одна таблица по каталогам <Sk>/.
Сверка хэшей: шарды против SHA256SUMS (Chitatel), один kod и один vhody на каталог, чекпоинт = manifest по kod/vhody/kub_sha.
Top-N по Шарпу окна 2016–2019 (ledger 2.1, stats.window, comm = 0) — справочно, без выбора. Метрик 2020+ нет по построению.
Запуск: python3 scripts/server/svodka_ledger.py <каталог> [--top N]."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fastbt import ledger_v2 as lm  # noqa: E402

KOLONKI = ["strategiya", "tochek", "ok", "oshibok", "unikalnyh", "chekpoint_i", "sverka", "kod", "vhody",
           "obyom_mb", "vremya_ch", "s_na_tochku_p50", "top_sharpe_okna"]
SVERKA_CHP = ("kod", "vhody", "kub_sha", "sobol_seed", "period")


def obyom_mb(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6


def sverka_chekpointa(d: Path, mf: dict) -> list:
    """Поля чекпоинта, не совпавшие с manifest.json (пусто — совпали или чекпоинта нет)."""
    p = d / "checkpoint.json"
    if not p.exists():
        return []
    chp = json.loads(p.read_text(encoding="utf-8"))
    return [k for k in SVERKA_CHP if k in mf and chp.get(k) != mf[k]]


def top_okna(ch, n: int) -> str:
    """«sobol_i:Шарп» лучших n строк ok по Шарпу окна [2016; 2019] — справочно; ledger 2.0 — «—»."""
    rows = ch.rows()
    if ch.v != lm.V21 or not rows or n <= 0:
        return "—"
    sh = np.asarray(ch.okno(2020, rows)[0], float)
    por = sorted(range(len(rows)), key=lambda j: (np.isnan(sh[j]), -np.nan_to_num(sh[j])))[:n]
    return " ".join(f"{rows[j].get('sobol_i')}:{sh[j]:.3f}" for j in por)


def stroka(root: Path, sid: str, top: int) -> dict:
    d = root / sid
    mf = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    chp = d / "checkpoint.json"
    out = dict(strategiya=sid, obyom_mb=f"{obyom_mb(d):.1f}",
               chekpoint_i=json.loads(chp.read_text(encoding="utf-8"))["posledniy_i"] if chp.exists() else "—")
    try:
        ch = lm.Chitatel(root, sid)
    except ValueError as e:                     # отказ читателя — это и есть результат сверки, строка его показывает
        return {**out, "sverka": f"ОТКАЗ: {e}"[:120]}
    vse = ch.rows(None)
    t = np.array([r.get("t_sek", np.nan) for r in vse], float)
    raz = sverka_chekpointa(d, mf)
    out.update(tochek=len(vse), ok=len(ch.rows()), oshibok=sum(r.get("status") == "oshibka" for r in vse),
               unikalnyh=len({r["klyuch_sha"] for r in vse}), kod=str(mf["kod"])[:12], vhody=str(mf["vhody"])[:12],
               sverka="ok" if not raz else f"чекпоинт ≠ manifest: {raz}",
               vremya_ch=f"{np.nansum(t) / 3600:.3f}", s_na_tochku_p50=f"{np.nanmedian(t):.3f}" if t.size else "—",
               top_sharpe_okna=top_okna(ch, top))
    return out


def svodka(root, top: int = 5) -> list:
    root = Path(root)
    sids = sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists())
    rows = [stroka(root, s, top) for s in sids]
    chislo = [r for r in rows if isinstance(r.get("tochek"), int)]
    rows.append(dict(strategiya="ИТОГ", tochek=sum(r["tochek"] for r in chislo), ok=sum(r["ok"] for r in chislo),
                     oshibok=sum(r["oshibok"] for r in chislo),
                     sverka="ok" if all(r.get("sverka") == "ok" for r in rows) else "есть расхождения",
                     obyom_mb=f"{sum(float(r['obyom_mb']) for r in rows):.1f}",
                     vremya_ch=f"{sum(float(r['vremya_ch']) for r in chislo):.3f}"))
    return rows


def tablica(rows: list) -> str:
    lines = ["| " + " | ".join(KOLONKI) + " |", "|" + "---|" * len(KOLONKI)]
    lines += ["| " + " | ".join(str(r.get(k, "—")) for k in KOLONKI) + " |" for r in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Сводка ledger серверного Соболя (справочно, без выбора)")
    ap.add_argument("katalog", type=Path)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    rows = svodka(a.katalog, a.top)
    print(tablica(rows))
    return 0 if rows[-1]["sverka"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())

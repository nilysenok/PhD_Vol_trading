"""П-0: профиль точки куба v5 (медленный путь), cProfile на N точках Соболя стратегии Sk, train4.

Малая проверка, не счёт (Ж-135): один процесс, метрик не печатает, пишет только в --out (scratchpad), не в docs/tables.
Запуск: python3 scripts/zamer_profil_v5.py --sid S1 --n 20 --out <каталог вне репозитория>
Самопроверка: python3 scripts/zamer_profil_v5.py (без аргументов)
"""
import argparse
import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PERIOD = "train4"


def top(st: pstats.Stats, k: int = 10, key: str = "tottime") -> list:
    """[(функция 'файл:строка(имя)', вызовов, tottime, cumtime)] по убыванию key."""
    idx = 2 if key == "tottime" else 3
    rows = [(f"{Path(f).name}:{ln}({nm})", v[1], v[2], v[3]) for (f, ln, nm), v in st.stats.items()]
    return sorted(rows, key=lambda r: -r[idx])[:k]


def fmt(rows: list, total: float) -> str:
    out = ["функция | вызовов | собств. с | % | накопл. с"]
    out += [f"{r[0]} | {r[1]} | {r[2]:.3f} | {100 * r[2] / total:.1f} | {r[3]:.3f}" for r in rows]
    return "\n".join(out)


def profil(sid: str, n: int) -> tuple:
    """(Stats, секунд на точку, ok). Сборка среды и первая точка (кэши, numba) — вне профиля."""
    from fastbt import canon_v5
    from fastbt import progon_sobol as ps
    rz = ps.Rezhim("v5", PERIOD, "seed", False, canon_v5.load_kub(sid)["fdr"])
    sr = ps.fabrika_real(sid, rz)
    pts = ps.tochki(ps.load_kub(sid, rz), 0, n + 1, rz)
    ps.schitat_tochku(sr, sid, pts[0][0], pts[0][1], "profil_p0", False, False, rz=rz)
    pr, k = cProfile.Profile(), 0
    t0 = time.perf_counter()
    pr.enable()
    for i, pt in pts[1:]:
        row, _, _ = ps.schitat_tochku(sr, sid, i, pt, "profil_p0", False, False, rz=rz)
        k += row["status"] == "ok"
    pr.disable()
    return pstats.Stats(pr, stream=io.StringIO()), (time.perf_counter() - t0) / max(1, len(pts) - 1), k


def selftest() -> None:
    def a():
        return sum(i * i for i in range(200000))

    def b():
        return [a() for _ in range(3)]

    pr = cProfile.Profile()
    pr.runcall(b)
    st = pstats.Stats(pr, stream=io.StringIO())
    cum = top(st, 3, "cumtime")
    assert cum[0][0].endswith("(b)") or cum[0][0].endswith("(<module>)") or "(b)" in cum[0][0] or "(a)" in cum[0][0], cum
    tt = top(st, 2, "tottime")
    assert tt[0][2] >= tt[1][2]
    assert "функция" in fmt(cum, 1.0)
    print("selftest ok")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.sid is None and a.out is None:
        selftest()
        return 0
    txt = []
    for sid in a.sid or ["S1"]:
        st, sec, k = profil(sid, a.n)
        tot = sum(v[2] for v in st.stats.values())
        txt.append(f"## {sid}: {k} ok из {a.n}, {sec:.2f} с/точку (с профайлером), сумма собственного времени {tot:.2f} с")
        txt.append("### по собственному времени\n" + fmt(top(st, 10, "tottime"), tot))
        txt.append("### по накопленному\n" + fmt(top(st, 10, "cumtime"), tot))
    s = "\n".join(txt)
    print(s)
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "profil_v5.md").write_text(s + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

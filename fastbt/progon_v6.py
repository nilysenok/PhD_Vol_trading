"""Серверный Соболь по кубу v6 (Ж-137 п. 2, 6; лист С-v6): то, что добавляет `--kub v6` к драйверу progon_sobol и обёртке progon_server.
Данных не читает. Порядок работ: Моррис → zamorozka.yaml → полный проход (s1, s2) → k3 → spravka.
  s1 / s2 — точки Соболя по кубу v6, замороженные оси Морриса вынуты из куба (на «выкл.»: размерности нет, в каноне осей нет), rol = sobol;
  k3 — контроль: полная решётка раздела k3 куба v5 (оси v4), ВСЕ новые оси «выкл.» и расширения v5 не взяты, заморозка не нужна; точки те же,
       что у k3 v5 (тот же klyuch_n), строки в ledger 2.2 с setka = v6, rol = kontrol (вне N);
  spravka — точки s1 (seed 1) с t_vyhod = close, rol = spravka (вне N).
Все три пишут ledger 2.2 (setka = v6, rol, klyuch_n); якоря куба v5 (yakorya_progon) лежат в точке, как у v5 (v5 ⊂ v6 по klyuch_n)."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import yaml

from fastbt import canon_v5 as c5, canon_v6
from fastbt.progon_morris_v6_polnyj import tochki_polnye

IST = "fastbt.istochniki_v6"           # лист Ист-v6: zagruzit(period, tikery) -> IstochnikiV6


def sha_fajla(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def zamorozka(katalog, sids, kub_put) -> tuple:
    """((sid, sha zamorozka.yaml, (замороженные оси)), …) из <каталог Морриса>/<Sk>/zamorozka.yaml. Отказ: нет файла; заморозка
    считана на другом кубе (vhody.kub_sha ≠ sha куба); замороженная ось не новая (ось v5 заморозке не подлежит)."""
    out = []
    for s in sids:
        p = Path(katalog) / s / "zamorozka.yaml"
        if not p.exists():
            raise SystemExit(f"{p}: нет — сначала Моррис (python3 -m fastbt.progon_morris_v6), полный проход без заморозки не считается")
        d = yaml.safe_load(p.read_text(encoding="utf-8"))
        if d.get("vhody", {}).get("kub_sha") != sha_fajla(kub_put(s)):
            raise SystemExit(f"{p}: заморозка посчитана на другом кубе v6 (kub_sha не совпал) — отказ")
        osi = tuple(sorted(d["osi"]))
        if set(osi) - set(canon_v6.novye(canon_v6.load_kub(s, kub_put(s).parent))):
            raise SystemExit(f"{p}: замороженная ось не новая в кубе v6 — отказ")
        out.append((s, sha_fajla(p), osi))
    return tuple(out)


def zam_sid(rz, sid: str) -> tuple:
    """(sha zamorozka.yaml, frozenset осей) стратегии; не задана — ошибка (молчаливый счёт без заморозки запрещён)."""
    for s, sha, osi in rz.zam:
        if s == sid:
            return sha, frozenset(osi)
    raise ValueError(f"{sid}: заморозка Морриса не задана — полный проход v6 без неё не считается")


def tochki(kub: dict, n0: int, n1: int, rz, tochki_v5) -> list:
    """[(i, якоря v5 ∪ канон v6)], i ∈ [n0, n1). k3 — решётка k3 куба v5 (tochki_v5 драйвера на кубе-основе, новые оси «выкл.»)."""
    if rz.k3:
        return tochki_v5(kub["v5"], n0, n1, rz._replace(setka="v5", rol=None))
    zam = zam_sid(rz, kub["sid"])[1]
    ya = kub["v5"].get("yakorya_progon", {})
    return [(i, {**ya, **c}) for i, c in tochki_polnye(kub["sid"], zam, n0, n1, rz.seed, kub)[0]]


def ocenshchik(sid: str, inp, base_p, komissiya: float, rz):
    """EvaluatorV6 с рядами v6 (istochniki_v6.zagruzit); модуля нет (Ист-v6 не слит) — отказ, не оценщик без рядов."""
    try:
        mod = importlib.import_module(IST)
    except ModuleNotFoundError as e:
        if e.name != IST:
            raise
        raise SystemExit(f"{IST} (лист Ист-v6) не слит — ряды v6 считать нечем — отказ") from e
    from fastbt.eval_v6 import EvaluatorV6
    return EvaluatorV6(sid, inp, base_p, komissiya, fdr=rz.fdr, ist=mod.zagruzit(rz.period, list(inp.sk.tickers)))


def vhody(digest: str, evr) -> str:
    """Д5: хэш входов = входы v5 + хэш рядов v6 (`IstochnikiV6.sha`, если загрузчик его даёт)."""
    sha = getattr(evr.ist, "sha", None)
    return digest if not sha else hashlib.sha256(f"{digest}|{sha}".encode()).hexdigest()


def obem_k3(kub: dict) -> int:
    return c5.obem(c5.k3_kub(kub["v5"]))

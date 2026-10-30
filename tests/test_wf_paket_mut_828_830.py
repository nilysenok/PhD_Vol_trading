"""Тестировщик #828–#830: мутации, не пойманные тестами авторов. Синтетика, сети и данных нет; /proc — каталог PERENOS_PROC.
#828: «порог угрозы 5 % → 50 %» (ugroza при 40 % свободной — срез по одному замеру); «родитель progon_server в максимуме» на цепочке
105 процессов (родитель 3,8 ГБ, рабочие 1,9 ГБ, 251 ГБ) — срез зря.
#829: «контроль утечки = сам ряд» (ok2 никогда не истина); «ok2 спасает любой ряд» (убрано условие sigma_).
#830: window_idx без min (IndexError-индекс len); i1 всегда последний день (сдвиг на train4); zona_perioda без возврата зоны при ошибке."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SH = ROOT / "scripts/server/perenos"
SRC = (SH / "pamyat.sh").read_text()
GB = 1024 * 1024
NORMA, PIK = 1992294, 3984588            # КБ: 1,9 ГБ и 3,8 ГБ


def _bash(tmp: Path, code: str, pamyat: str | None = None, **env) -> subprocess.CompletedProcess:
    (tmp / "obshchee.sh").write_text((SH / "obshchee.sh").read_text())
    (tmp / "pamyat.sh").write_text(pamyat if pamyat is not None else SRC)
    (tmp / "VERSIYA").write_text("commit_short=abc\ntar_sha256=ff\nnumba_threads=1\nnumba_layer=workqueue\n")
    return subprocess.run(["bash", "-c", f". ./obshchee.sh; . ./pamyat.sh; {code}"], cwd=tmp, env={**os.environ, **env},
                          capture_output=True, text=True, timeout=60)


def _zamery(tmp: Path, seriya: list[int], av_gb: int, pamyat: str | None = None, zhit: int = 5) -> str:
    (tmp / "logs").mkdir()
    (tmp / "zapusk_sobol.sh").write_text("#!/bin/sh\necho restart\n")
    cnt = tmp / "cnt"
    cnt.write_text("0")
    arr = " ".join(map(str, seriya))
    body = (f'i=$(( $(cat {cnt}) + 1 )); echo $i > {cnt}; a=({arr}); u=${{a[$(( i - 1 < {len(seriya)} ? i - 1 : {len(seriya) - 1} ))]}}; '
            f'echo "$u $(( u * 107 ))"')
    sleeper = subprocess.Popen(["sleep", str(zhit)])
    threading.Thread(target=sleeper.wait, daemon=True).start()
    try:
        code = (f'pamyat_uss(){{ {body}; }}; pamyat_svobodno(){{ echo "{av_gb * GB} {251 * GB}"; }}; pamyat_ostanovit(){{ :; }}; '
                f'pamyat_sledit train4 {sleeper.pid} 107')
        assert _bash(tmp, code, pamyat=pamyat, PERENOS_PAMYAT_INTERVAL="1").returncode == 0
    finally:
        sleeper.kill()
    return (tmp / "logs/pamyat_train4.log").read_text()


def test_porog_ugrozy_5_procentov_a_ne_50(tmp_path):
    log = _zamery(tmp_path, [NORMA, PIK, NORMA], av_gb=100)       # 40 % свободно: не угроза, один пик — среза нет
    assert "СНИЖЕНИЕ" not in log and "ждём второго" in log
    mut = SRC.replace('"$((av * 20))"', '"$((av * 2))"').replace("[ $((av * 20)) -lt", "[ $((av * 2)) -lt")
    assert mut != SRC
    d = tmp_path / "m"
    d.mkdir()
    assert "СНИЖЕНИЕ" in _zamery(d, [NORMA, PIK, NORMA], av_gb=100, pamyat=mut)      # 50 % → срез по пику: тест выше падает


def test_srez_srazu_nizhe_5_no_ne_vyshe(tmp_path):
    assert "СНИЖЕНИЕ" in _zamery(tmp_path, [PIK], av_gb=12, zhit=1)            # 4,8 % — угроза, один замер
    d = tmp_path / "b"
    d.mkdir()
    assert "СНИЖЕНИЕ" not in _zamery(d, [PIK], av_gb=13, zhit=1)               # 5,2 % — не угроза


def test_tsepochka_105_roditel_3_8_rabochie_1_9_bez_sreza(tmp_path):
    proc, bind = tmp_path / "proc", tmp_path / "bin"
    bind.mkdir()
    pids = list(range(11, 11 + 105))                                           # 105 процессов: родитель 11 + 104 рабочих
    for pid in pids:
        (proc / str(pid)).mkdir(parents=True)
        uss = PIK if pid == 11 else NORMA
        (proc / str(pid) / "smaps_rollup").write_text(f"Private_Clean:   {uss} kB\nPrivate_Dirty:   0 kB\n")
        (proc / str(pid) / "cmdline").write_bytes(b"python\0-m\0fastbt.progon_server")
    (bind / "ps").write_text("#!/bin/sh\nprintf '10 1\\n11 10\\n" + "".join(f"{p} 11\\n" for p in pids[1:]) + "'\n")
    (bind / "ps").chmod(0o755 | stat.S_IEXEC)
    env = dict(PATH=f"{bind}:{os.environ['PATH']}", PERENOS_PROC=str(proc))
    code = ('read -r mx sm <<< "$(pamyat_uss 10)"; tot=$((40 * 1048576 + sm)); [ $tot -gt $((251 * 1048576)) ] && tot=$((251 * 1048576)); '
            'pamyat_reshenie 105 $mx $tot')
    assert int(_bash(tmp_path, code, **env).stdout.split()[-1]) == 105           # родитель вне максимума: 251 ГБ / (1,9 × 1,15) ≥ 105
    mut = SRC.replace("grep -q 'progon_server' && continue", "grep -q 'NEVER' && continue")
    assert mut != SRC
    assert int(_bash(tmp_path, code, pamyat=mut, **env).stdout.split()[-1]) < 105   # родитель в максимуме: срез зря, тест выше падает


# ---- #829 ----
def _modul(zamena: tuple | None = None):
    src = (ROOT / "scripts/proverka_sdviga_wf.py").read_text()
    if zamena:
        assert zamena[0] in src
        src = src.replace(*zamena)
    m = types.ModuleType("psw_mut")
    m.__file__ = str(ROOT / "scripts/proverka_sdviga_wf.py")
    sys.path.insert(0, str(ROOT))
    try:
        exec(compile(src, m.__file__, "exec"), m.__dict__)
    finally:
        sys.path.remove(str(ROOT))
    return m


def _ryady(n=900, seed=1):
    rng = np.random.default_rng(seed)
    y = pd.DataFrame(rng.normal(size=(n, 6)), index=pd.bdate_range("2018-01-01", periods=n), columns=list("ABCDEF"))
    s = 0.15 * y + 0.6 * y.shift(1) + 0.4 * y.shift(2) + 0.2 * y.shift(3) + 0.05 * rng.normal(size=y.shape)
    return s, y


def test_kontrol_utechki_ne_sam_ryad():
    s, y = _ryady()
    assert _modul().vtoraya_sigma(s, y)["ok2"]
    m = _modul(("chastnaya(s.shift(-1).diff(), lrv)", "chastnaya(s.diff(), lrv)"))
    assert not m.vtoraya_sigma(s, y)["ok2"]                        # контроль = сам ряд: |f| < ½|f| ложно — мутация видна


def test_ok2_spasaet_tolko_sigma():
    t = pd.DataFrame({"ryad": ["sigma_h5", "atr14", "sigma_x"], "ok": [False, False, True], "ok2": [True, True, False]})
    r = _modul().itog(t)
    assert list(r["d1_itog"]) == [True, False, True]
    m = _modul(('t["ryad"].str.startswith("sigma_") & ok2', "ok2"))
    assert list(m.itog(t)["d1_itog"]) != [True, False, True]


# ---- #830 ----
def test_window_idx_bez_min_i_bez_sdviga():
    from fastbt import etap1
    d = pd.bdate_range("2019-12-20", "2020-01-20").to_numpy("datetime64[ns]")
    for end in ("2020-01-01", "2020-01-03", "2020-01-04"):          # внутри ряда: первый день >= end, min не вмешивается
        i1 = etap1.window_idx(d, pd.Timestamp("2019-12-20"), 2019, pd.Timestamp(end))[2]
        assert i1 == int(np.searchsorted(d, np.datetime64(end))) < len(d) - 1
    assert etap1.window_idx(d, pd.Timestamp("2019-12-20"), 2019, pd.Timestamp("2021-01-01"))[2] == len(d) - 1   # за концом — последний, не len
    assert etap1.window_idx(d, pd.Timestamp("2019-12-20"), 2019, d[-1])[2] == len(d) - 1                        # ровно на последнем дне


def test_zona_perioda_vozvrashchaet_zonu_pri_isklyuchenii():
    sys.path.insert(0, str(ROOT / "scripts/server"))
    try:
        import ekviv
    finally:
        sys.path.remove(str(ROOT / "scripts/server"))
    from fastbt import zona
    zona.set_zone("dev")
    with pytest.raises(RuntimeError):
        with ekviv.zona_perioda("wf"):
            assert zona.name() == "wf"
            raise RuntimeError("x")
    assert zona.name() == "dev"
    with ekviv.zona_perioda("train4"):
        assert zona.name() == "dev"                                 # не-wf зону не трогает
    with ekviv.zona_perioda("wf"):
        pass
    assert zona.name() == "dev"

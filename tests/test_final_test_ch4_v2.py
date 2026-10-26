"""Замок и SHA-сверка второго открытия главы 4 (scripts/analysis/final_test_ch4_v2.py) — на поддельных временных файлах.

Реальные test_locked/ и годы ≥ 2022 не читаются: все пути (spec, r2spec, locked, lock, main, root) подменены tmp_path.
    python3 -m pytest -q tests/test_final_test_ch4_v2.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
import final_test_ch4_v2 as F  # noqa: E402


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _put(p: Path, data: bytes) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """Поддельное окружение: замороженная спецификация с верными SHA поддельных файлов (поддельная папка — не git:
    проверка «код не раньше ce57e41» подменена, она проверяется отдельно на настоящем репозитории)."""
    monkeypatch.setattr(F, 'require_base', lambda root, commit: None)
    root, main, locked = tmp_path / 'root', tmp_path / 'main', tmp_path / 'main' / 'data' / 'regen' / 'test_locked'
    real = yaml.safe_load((ROOT / 'models' / 'designs' / 'ch4_v2' / 'ch4_v2_sborka.yaml').read_text())
    spec = json.loads(json.dumps(real))
    fv = _put(root / spec['final_v2']['put'], 'FINAL v2 (поддельный)'.encode())
    spec['final_v2']['sha256'] = _sha(fv)
    sob = {}
    for lab, v in spec['vhody']['sigma_test'].items():
        f = _put(locked / v['fayl'], f'поддельный σ {lab}'.encode())
        v['sha256'] = sob[v['fayl']] = _sha(f)
    for k in ('features', 'bary', 'stavka'):
        f = _put(main / spec['vhody'][k]['fayl'], f'поддельный {k}'.encode())
        spec['vhody'][k]['sha256'] = _sha(f)
    for p in list(spec['kod']):
        spec['kod'][p] = _sha(_put(root / p, f'код {p}'.encode()))
    _put(root / 'fastbt' / '_poddelnyy.py', 'поддельный fastbt'.encode())
    spec['fastbt_sha_vse'] = F.fastbt_agg(root)
    spec['dizayny'] = {k: dict(v) for k, v in real['selftest_zaglushki'].items() if k in F.PODHODY}
    spec['status'] = 'zamorozheno'
    spec['da_vladeltsa'] = 'да (поддельное, тест)'
    sp = _put(root / 'models' / 'designs' / 'ch4_v2' / 'ch4_v2_sborka.yaml',
              yaml.safe_dump(spec, allow_unicode=True).encode())
    r2 = _put(root / 'models' / 'designs' / 'r2' / 'r2_A_sborka.yaml',
              yaml.safe_dump({'sobrano': sob}, allow_unicode=True).encode())
    P = F.Puti(root=root, main=main, spec=sp, r2spec=r2, locked=locked, lock=locked / 'ch4_v2' / 'OPENED_CH4_V2.json',
               dev=tmp_path / 'dev', out_tables_test=tmp_path / 'tab', out_data_test=tmp_path / 'dat',
               out_selftest=tmp_path / 'self')
    return P, spec


def _rewrite(P, spec):
    P.spec.write_text(yaml.safe_dump(spec, allow_unicode=True))


def test_lock_path_constant():
    p = F.Puti().lock
    assert str(p).endswith('data/regen/test_locked/ch4_v2/OPENED_CH4_V2.json')


def test_real_spec_refuses_before_data(monkeypatch):
    """Настоящая спецификация не заморожена: отказ на шаге 1, до замка и до чтения любых данных."""
    touched = []
    monkeypatch.setattr(F, 'check_lock_free', lambda P: touched.append('lock'))
    monkeypatch.setattr(F, 'verify_sha', lambda s, P: touched.append('sha'))
    monkeypatch.setattr(F, 'sources_test', lambda *a: touched.append('data'))
    with pytest.raises(F.Stop) as ei:
        F.main(['--scope', 'test'])
    assert 'не заморожена' in ei.value.msg and touched == []


@pytest.mark.parametrize('mut, frag', [
    (lambda s: s.update(status='ждёт «да» владельца'), 'zamorozheno'),
    (lambda s: s.update(da_vladeltsa=''), '«да»'),
    (lambda s: s['dizayny'].pop('D'), 'подхода D'),
    (lambda s: s['dizayny'].update(C2={'semya': 'C2'}), 'лишние'),
    (lambda s: s['final_v2'].update(sha256=''), 'FINAL'),
    (lambda s: s['vhody']['bary'].update(sha256=''), 'bary'),
    (lambda s: s.update(fastbt_sha_vse=''), 'fastbt_sha_vse'),
])
def test_not_frozen(fake, mut, frag):
    P, spec = fake
    mut(spec)
    _rewrite(P, spec)
    with pytest.raises(F.Stop) as ei:
        F.precheck_test(P)
    assert frag in ei.value.msg
    assert not P.lock.exists()


def test_sha_ok_then_lock_once(fake):
    P, spec = fake
    s, shas = F.precheck_test(P)
    assert len(shas) == 1 + 4 + 3 + len(spec['kod']) + 1
    F.acquire_lock(P, dict(proba=1))
    assert json.loads(P.lock.read_text())['status'] == 'начато'
    with pytest.raises(F.Stop) as ei:                 # второй запуск: замок есть
        F.precheck_test(P)
    assert 'второе открытие' in ei.value.msg
    with pytest.raises(F.Stop):                       # гонка: O_EXCL
        F.acquire_lock(P, dict(proba=2))
    F.finish_lock(P, dict(vyhody={}))
    d = json.loads(P.lock.read_text())
    assert d['status'] == 'завершено' and d['proba'] == 1


def test_require_base_real_repo():
    """Настоящий репозиторий: код не раньше ce57e41 (исправления FA #238, #239); несуществующий коммит — отказ."""
    F.require_base(ROOT, 'ce57e41')
    with pytest.raises(F.Stop):
        F.require_base(ROOT, '0' * 40)


@pytest.mark.parametrize('what', ['sigma', 'sobrano', 'final', 'features', 'kod', 'fastbt'])
def test_sha_mismatch_refuses_without_lock(fake, what):
    P, spec = fake
    lab, v = next(iter(spec['vhody']['sigma_test'].items()))
    if what == 'sigma':
        (P.locked / v['fayl']).write_bytes('подменён'.encode())
    elif what == 'sobrano':
        P.r2spec.write_text(yaml.safe_dump({'sobrano': {v['fayl']: '0' * 64}}))
    elif what == 'final':
        (P.root / spec['final_v2']['put']).write_text('FINAL v2 изменён после заморозки')
    elif what == 'features':
        (P.main / spec['vhody']['features']['fayl']).write_bytes(b'x')
    elif what == 'fastbt':
        (P.root / 'fastbt' / 'novyy.py').write_text('новый файл fastbt после заморозки')
    else:
        (P.root / next(iter(spec['kod']))).write_text('код изменён')
    with pytest.raises(F.Stop) as ei:
        F.precheck_test(P)
    assert 'SHA' in ei.value.msg
    assert not P.lock.exists()


def test_run_test_writes_lock_before_data(fake, monkeypatch):
    """run_test: замок пишется до чтения рядов теста; второй run_test отказывает."""
    P, spec = fake
    seen = {}

    class Sentinel(Exception):
        pass

    def fake_sources(spec_, P_, out):
        seen['lock_at_read'] = P_.lock.exists()
        raise Sentinel
    monkeypatch.setattr(F, 'sources_test', fake_sources)
    import fastbt.s9 as s9
    monkeypatch.setattr(s9, 'set_scope', lambda *a, **k: None)
    with pytest.raises(Sentinel):
        F.run_test(P, workers=1)
    assert seen['lock_at_read'] is True
    with pytest.raises(F.Stop) as ei:
        F.run_test(P, workers=1)
    assert 'второе открытие' in ei.value.msg


def test_boot_reps_fixed(fake, monkeypatch):
    P, spec = fake
    with pytest.raises(F.Stop) as ei:
        F.run_test(P, workers=1, reps_override=1000)
    assert 'бутстрап' in ei.value.msg and not P.lock.exists()

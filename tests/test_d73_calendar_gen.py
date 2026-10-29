"""Генератор календаря экспираций даёт побайтно data/calendar/expirations.csv (без сети)."""
import importlib.util, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REPO_CSV = os.path.join(ROOT, 'data', 'calendar', 'expirations.csv')
MARK = '2026-10-25'  # retrieved_at правил-строк в репозитории


def load():
    sys.argv = ['x']
    spec = importlib.util.spec_from_file_location('d73', os.path.join(ROOT, 'scripts', 'analysis', 'd73_build_calendar.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gen(tmp, **kw):
    m = load()
    m.write('expirations', m.expirations(MARK, **kw), out=str(tmp))
    return open(os.path.join(tmp, 'expirations.csv'), 'rb').read()


def test_bajt_v_bajt(tmp_path):
    assert gen(tmp_path) == open(REPO_CSV, 'rb').read()


def test_mutaciya_isklyuchenie_ne_primeneno_lovitsya(tmp_path):
    out = gen(tmp_path, exceptions={})
    assert out != open(REPO_CSV, 'rb').read()
    assert b'2022-03-17' in out and b'2022-03-29' not in out

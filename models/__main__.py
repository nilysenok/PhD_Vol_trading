"""python3 -m models run --design <yaml> [--smoke] [--confirm | --finalists A,B] [--scope val|test] [--data-root DIR] [--workers N]
python3 -m models tune --h <h> --model <xgb|lgbm> [--trials 30] [--confirm] [--design <yaml>]
python3 -m models tables [--design <yaml>] [--scope val|test] [--confirm]
python3 -m models mkdesign --selected <B_selected.csv> --name <имя> [--train B|default] [--max-bin N|none] [--features P] [--target 1=..,5=..,22=..]
Вывод — одна таблица; логи — в data/run/ (в консоль — только хвост при ошибке)."""
import argparse, logging, os, sys, time, traceback
from pathlib import Path
from . import config as C


def setup_log(cmd):
    C.RUN.mkdir(parents=True, exist_ok=True)
    p = C.RUN / f'{cmd}_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(filename=p, level=logging.INFO, format='%(asctime)s %(message)s', force=True)
    return p


def tail(p, n=15):
    lines = Path(p).read_text().splitlines()
    return '\n'.join(lines[-n:])


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python3 -m models')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run'); r.add_argument('--design', required=True); r.add_argument('--smoke', action='store_true')
    t = sub.add_parser('tune'); t.add_argument('--h', type=int, required=True, choices=C.HS); t.add_argument('--model', required=True)
    t.add_argument('--trials', type=int, default=30); t.add_argument('--design', default=None); t.add_argument('--smoke', action='store_true')
    tb = sub.add_parser('tables'); tb.add_argument('--design', default=str(C.ROOT / 'models/designs/B_baseline.yaml'))
    g = sub.add_parser('gate'); g.add_argument('--design', required=True); g.add_argument('--ref-dir', required=True)
    g.add_argument('--smoke', action='store_true'); g.add_argument('--boost-name', default='B', help='B | P1 — имена hyb_<имя>_*.parquet; прогнозы бустинга B_<m>_h<h>_*')
    for p in (r, t, tb, g):
        p.add_argument('--confirm', action='store_true', help='шаги 7-9 (подтверждающий набор val 2022-2024)')
        p.add_argument('--finalists', default='', help='имена финалистов; включает шаги 7-9')
        p.add_argument('--data-root', default=None, help='подмена входов дизайна: каталог prepared/ или .parquet матрицы (детектор утечек)')
        p.add_argument('--workers', type=int, default=os.cpu_count())
    for p in (r, tb):
        p.add_argument('--scope', default='val', choices=['val', 'test'])
    mk = sub.add_parser('mkdesign'); mk.add_argument('--selected', required=True); mk.add_argument('--name', required=True)
    mk.add_argument('--train', default='B', choices=['B', 'default']); mk.add_argument('--max-bin', default=None)
    mk.add_argument('--features', default=None, help='матрица признаков (каталог prepared/ или .parquet)')
    mk.add_argument('--target', default=None, help='колонки цели: 1=rv_target_h1,5=rv_target_mean_h5,22=rv_target_mean_h22')
    a = ap.parse_args(argv)
    if a.cmd == 'mkdesign':
        from .design import from_selected
        print(from_selected(a.selected, a.name, a.train, a.max_bin, a.features, a.target)); return 0
    confirm = a.confirm or bool(a.finalists)
    if getattr(a, 'scope', 'val') == 'test':
        from .summary import require_final
        require_final()
    lp = setup_log(a.cmd)
    t0 = time.time()
    try:
        import pandas as pd
        pd.set_option('display.width', 200); pd.set_option('display.max_colwidth', 60)
        if a.cmd == 'run':
            from . import pipeline, summary
            d = C.load_design(a.design)
            man = pipeline.run(d, a.data_root, a.workers, confirm, a.smoke)
            tab = summary.table(man, a.scope, confirm and not a.smoke)
            tab.to_csv(C.RUN / f'summary_{d["name"]}_{a.scope}.csv', index=False)
            logging.info('stage timer\n%s', pd.DataFrame(man['timer']).to_string())
        elif a.cmd == 'tune':
            from . import tune
            tab = tune.run(a.h, a.model.split(','), a.trials, a.design, a.data_root, a.workers, confirm, a.smoke)
        elif a.cmd == 'gate':
            import json
            from . import gate, pipeline
            d = C.load_design(a.design)
            man = json.loads(pipeline.manifest_path(d['name'], confirm, a.smoke, a.data_root).read_text())
            rows = ([] if d.get('boost_external') else gate.boost(man, a.ref_dir)) + gate.hybrid(man, a.ref_dir, a.boost_name)
            tab = pd.DataFrame(rows)
        else:
            from . import tables
            tab = tables.run(C.load_design(a.design), a.scope, confirm, a.data_root)
        print(tab.to_string(index=False))
        print(f'время {time.time() - t0:.1f} с; лог {lp}')
        return 0
    except SystemExit:
        raise
    except Exception:
        logging.error(traceback.format_exc())
        print(f'ОШИБКА (хвост {lp}):\n{tail(lp)}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

"""Д7.3: календарные признаки, известные на дату t (point-in-time).

Для даты t и горизонта h (календарные дни), по каждому типу события:
  n_<тип>_h<h>   - число событий в окне [t+1, t+h] среди известных на t (known_from <= t);
  dte_<тип>      - дней до ближайшего известного события строго после t (NaN, если такого нет).
Внеплановые (vneplan == 1) в признаки не входят. Типы = имена CSV в data/calendar/.

    python3 d73_features.py            # пример по реальным CSV
    python3 d73_features.py --test     # тест на синтетике
"""
import os
import sys

import numpy as np
import pandas as pd

CAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'calendar')
TYPES = ['cbr', 'fomc', 'cpi_ru', 'cpi_us', 'dividends_expected', 'dividends_actual', 'expirations',
         'imoex_rebal']


def load_events(cal_dir=CAL, types=TYPES):
    """{тип: DataFrame[date, known_from]} (datetime64), без внеплановых. Пустой/отсутствующий файл -> пустой."""
    out = {}
    for ty in types:
        p = os.path.join(cal_dir, ty + '.csv')
        df = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame(columns=['date', 'known_from', 'vneplan'])
        if 'vneplan' in df:
            df = df[df['vneplan'].fillna(0).astype(int) == 0]
        out[ty] = pd.DataFrame({'date': pd.to_datetime(df['date']), 'known_from': pd.to_datetime(df['known_from'])})
    return out


def features(t, h, events):
    """Признаки одного дня t (str/Timestamp) для горизонта h (дней). events - результат load_events."""
    t = pd.Timestamp(t).normalize()
    res = {}
    for ty, df in events.items():
        known = df[df['known_from'] <= t]  # только то, что известно на t
        d = known['date']
        res['n_%s_h%d' % (ty, h)] = int(((d > t) & (d <= t + pd.Timedelta(days=h))).sum())
        fut = d[d > t]
        res['dte_%s' % ty] = float((fut.min() - t).days) if len(fut) else np.nan
    return res


def features_frame(dates, h, events):
    """DataFrame признаков для набора дат (индекс = даты)."""
    dates = pd.DatetimeIndex(dates)
    return pd.DataFrame([features(t, h, events) for t in dates], index=dates)


def _test():
    t = pd.Timestamp('2020-06-10')
    ev = {'x': pd.DataFrame({'date': pd.to_datetime(['2020-06-12', '2020-06-20']),
                             'known_from': pd.to_datetime(['2020-06-01', '2020-06-11'])})}
    f = features(t, 5, ev)
    # событие 06-20 известно только с 06-11 -> на t=06-10 не влияет (ни в окне, ни в dte)
    assert f['n_x_h5'] == 1 and f['dte_x'] == 2.0, f
    f = features(t, 30, ev)
    assert f['n_x_h30'] == 1 and f['dte_x'] == 2.0, f  # даже при широком окне 06-20 не видно
    f = features(t + pd.Timedelta(days=1), 30, ev)  # на следующий день оно уже известно
    assert f['n_x_h30'] == 2 and f['dte_x'] == 1.0, f
    # событие, известное с t+1 и единственное: на t - 0 и NaN
    ev2 = {'y': pd.DataFrame({'date': pd.to_datetime(['2020-06-15']), 'known_from': pd.to_datetime(['2020-06-11'])})}
    f = features(t, 10, ev2)
    assert f['n_y_h10'] == 0 and np.isnan(f['dte_y']), f
    assert features(t + pd.Timedelta(days=1), 10, ev2)['n_y_h10'] == 1
    # событие в день t не считается (окно с t+1); прошедшее не считается
    ev3 = {'z': pd.DataFrame({'date': pd.to_datetime(['2020-06-10', '2020-06-01']),
                              'known_from': pd.to_datetime(['2020-01-01', '2020-01-01'])})}
    f = features(t, 10, ev3)
    assert f['n_z_h10'] == 0 and np.isnan(f['dte_z']), f
    # пустой тип
    f = features(t, 10, {'e': pd.DataFrame({'date': pd.to_datetime([]), 'known_from': pd.to_datetime([])})})
    assert f['n_e_h10'] == 0 and np.isnan(f['dte_e'])
    # реальные CSV: внеплановые исключены (ЦБ 28.02.2022)
    real = load_events()
    assert not (real['cbr']['date'] == '2022-02-28').any()
    print('OK: тесты пройдены')


if __name__ == '__main__':
    if '--test' in sys.argv:
        _test()
    else:
        ev = load_events()
        pd.set_option('display.width', 250, 'display.max_columns', 30)
        print(features_frame(['2022-02-25', '2023-03-15', '2024-12-12', '2026-06-10'], 14, ev).T.to_string())

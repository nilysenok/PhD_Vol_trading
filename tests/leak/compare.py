"""Побайтное сравнение выходов эталонного и «зашумлённого» миров, QLIKE и гипотезы причины расхождения.
Выход модели — длинная таблица: stage, kind (val|test), ty (год теста шага), date, ticker, rv_actual, rv_pred."""
from dataclasses import dataclass
import numpy as np, pandas as pd
from .noise import pos_t, row_idx

KEYS = ['stage', 'kind', 'date', 'ticker']


@dataclass
class Verdict:
    stage: str
    check: str      # 'future' | 'permute'
    t: str
    ok: bool
    detail: str

    def line(self):
        return f"{'PASS' if self.ok else 'FAIL'} [{self.check}] стадия={self.stage} t={self.t}: {self.detail}"


def eligible(df, t, h, cal, info_end):
    """Строки, выход которых по правилам walk-forward не может зависеть от данных после t: date <= t и данные шага
    (до info_end(ty) включительно) вместе с горизонтом цели h заканчиваются не позже t."""
    p = pos_t(cal, t)
    ok_ty = {ty: pos_t(cal, info_end(int(ty))) + h <= p for ty in df['ty'].unique()}
    return ((pd.to_datetime(df['date']) <= pd.Timestamp(t)) & df['ty'].map(ok_ty)).values


def hypothesis(bad_dates, t, h, cal):
    p = pos_t(cal, t)
    near = np.mean(row_idx(cal, bad_dates) > p - h) if len(bad_dates) else 0
    if near > 0.8:
        return (f'расхождения в последних {h} днях до t: метка/окно/лаг заглядывает вперёд на <= h '
                f'(цель s использует [s+1,s+h]; лаг < h или перекрытие меток)')
    return ('расхождения по всей истории до t: обучение, валидация, ранняя остановка или параметры подбираются '
            'на данных после t (маска окна включает будущее) либо признак построен из будущих значений')


def compare_future(ref, new, t, h, cal, info_end):
    """Список Verdict по стадиям (stage, kind): побайтно rv_pred на eligible-строках."""
    out = []
    for (st, kd), g0 in ref.groupby(['stage', 'kind'], sort=False):
        g0 = g0[eligible(g0, t, h, cal, info_end)].sort_values(['date', 'ticker'])
        g1 = new[(new.stage == st) & (new.kind == kd)]
        g1 = g1[eligible(g1, t, h, cal, info_end)].sort_values(['date', 'ticker'])
        name = f'{st}/{kd}'
        if len(g0) == 0:
            out.append(Verdict(name, 'future', str(t), False, 'нет сравнимых строк (проверка вхолостую)'))
            continue
        k0 = g0[['date', 'ticker']].astype(str).agg('|'.join, axis=1).values
        k1 = g1[['date', 'ticker']].astype(str).agg('|'.join, axis=1).values
        if not np.array_equal(k0, k1):
            out.append(Verdict(name, 'future', str(t), False, f'набор строк изменился ({len(k0)} -> {len(k1)})'))
            continue
        a, b = g0['rv_pred'].values.astype('<f8'), g1['rv_pred'].values.astype('<f8')
        if a.tobytes() == b.tobytes():
            out.append(Verdict(name, 'future', str(t), True, f'{len(a)} прогнозов совпали побайтно'))
            continue
        bad = a.view('<u8') != b.view('<u8')
        i = int(np.flatnonzero(bad)[0])
        d = f"{g0['date'].iloc[i]:%Y-%m-%d} {g0['ticker'].iloc[i]}"
        out.append(Verdict(name, 'future', str(t), False,
                           f'разошлось {int(bad.sum())} из {len(a)}; первый выход {d}: {a[i]!r} -> {b[i]!r}; '
                           f'гипотеза: {hypothesis(g0["date"].values[bad], t, h, cal)}'))
    return out


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def compare_permute(res, perm_frame, h, dev_last_val_year=2021, tol=0.0):
    """Валидационный QLIKE на переставленной цели (разработочный набор: val-годы <= 2021) против константы —
    средней по тикеру RV обучающего окна шага (годы <= ty-2). Модель обязана быть не лучше константы."""
    out, col = [], f'rv_target_h{h}'
    for st, g in res[res.kind == 'val'].groupby('stage', sort=False):
        g = g[g['ty'] - 1 <= dev_last_val_year]
        qm, qc, n = [], [], 0
        for ty, gg in g.groupby('ty'):
            tr = perm_frame[(perm_frame['year'] <= ty - 2)].groupby('ticker')[col].mean()
            c = gg['ticker'].map(tr).values
            ok = ~np.isnan(c) & (gg['rv_actual'].values > 1e-12)  # фильтр как в gibrid_io.merged
            qm.append(qlike(gg['rv_actual'].values[ok], gg['rv_pred'].values[ok]))
            qc.append(qlike(gg['rv_actual'].values[ok], c[ok]))
            n += int(ok.sum())
        if not qm:
            out.append(Verdict(f'{st}/val', 'permute', '-', False, 'нет валидационных строк разработочного набора'))
            continue
        m, c = float(np.mean(qm)), float(np.mean(qc))
        out.append(Verdict(f'{st}/val', 'permute', '-', m >= c - tol,
                           f'QLIKE модели {m:.4f} vs константа {c:.4f} (шагов {len(qm)}, строк {n})'
                           + ('' if m >= c - tol else '; модель лучше константы на случайной цели: утечка цели или переобучение на val')))
    return out

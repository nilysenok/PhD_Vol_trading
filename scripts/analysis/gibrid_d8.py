"""Д8 - структура гибрида (объявлено владельцем 25.10). Все параметры - по валидационному году шага; смешивание - S.comb (--blend rv|log).
V10_MinQ (Д8.2): веса - минимум val-QLIKE на симплексе (w >= 0, sum w = 1, SLSQP), гребень к равным lam*||w - 1/3||^2, lam in {0; 0,1; 0,3}
  выбирается 2-кратной перекрёстной проверкой по половинам val-года (первая/вторая половина дат), затем веса - по всему году.
V7_HalfLife (Д8.3): V7 с экспоненциальным средним потерь, полураспад {10, 20, 60, 120} дней, лаг h; полураспад - по val-QLIKE шага.
V11_Gap (Д8.5): w_HAR = sigmoid((|ln F_ML - ln F_HAR| - theta)/tau), F_ML - смесь XGB и LGB поровну; theta - децили рассогласования
  val-года и {1; 2}, tau in {0,02; 0,05; 0,1; 0,2; 0,5; 1}; сетка по val-QLIKE шага; XGB и LGB делят (1 - w_HAR) поровну.
Med3 (Д8.6): медиана трёх (HAR-J, XGB, LGB) по наблюдению. HAR-CJ - gibrid_harcj.py.
Gate_L2 (Д8.7): w_HAR = sigmoid(b0 + z'b), z - стандартизованные (по val-году) признаки состояния: ln RV_d, ln(RV_w/RV_m), рассогласование,
  ln VIX(t-1), ln RVI(t-1); b - минимум val-QLIKE + ALPHA*||b||^2 (b0 без штрафа, |b0| <= 4),
  L-BFGS-B, старт - лучший постоянный вес (сетка 0,02..0,98)."""
import numpy as np
from scipy.optimize import minimize
import gibrid_schemes as S

HL = (10, 20, 60, 120)
V10_LAMS = (0.0, 0.1, 0.3)
TAUS = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
ALPHA = 0.01  # сильная L2: на средний QLIKE, ~ C = 1/(2*ALPHA*n) ~ 0,0125 в терминах sklearn при n ~ 4000 (val-год)
STATE = ('rv_d', 'rv_w', 'rv_m', 'vix1', 'rvi1')


def P3(d):
    return np.column_stack([d['har'].values, d['xgb'].values, d['lgb'].values])


def _lg(v):
    return np.log(np.clip(np.asarray(v, dtype=float), 1e-10, None))


def _row_w(w):
    return np.column_stack([w, (1 - w) / 2, (1 - w) / 2])


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def gap(d):
    ml = S.comb(np.column_stack([d['xgb'].values, d['lgb'].values]), np.array([0.5, 0.5]))
    return np.abs(_lg(ml) - _lg(d['har'].values))


def simplex(P, y, lam):
    f = lambda w: S.qlike(y, S.comb(P, w)) + lam * float(np.sum((w - 1 / 3) ** 2))
    r = minimize(f, np.full(3, 1 / 3), method='SLSQP', bounds=[(0, 1)] * 3,
                 constraints=({'type': 'eq', 'fun': lambda w: w.sum() - 1},), options=dict(ftol=1e-10, maxiter=300))
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def v10(d):
    P, y, dt = P3(d), d['rv_actual'].values, d['date'].values
    u = np.sort(np.unique(dt))
    first = dt <= u[len(u) // 2 - 1]
    cv = {lam: sum(S.qlike(y[~m], S.comb(P[~m], simplex(P[m], y[m], lam))) for m in (first, ~first)) for lam in V10_LAMS}
    lam = min(cv, key=cv.get)
    return simplex(P, y, lam), lam


def v11_apply(d, th, tau):
    W = _row_w(_sig((gap(d) - th) / tau))
    return S.comb(P3(d), W), W


def v11_fit(d):
    y, g = d['rv_actual'].values, gap(d)
    ths = list(np.quantile(g, np.arange(0.1, 0.95, 0.1))) + [1.0, 2.0]  # 1 и 2 - чтобы вес HAR мог уйти почти в 0
    grid = [(float(th), tau) for th in ths for tau in TAUS]
    return min(grid, key=lambda p: S.qlike(y, v11_apply(d, *p)[0]))


def med3(d):
    P = P3(d)
    med = np.median(P, axis=1)
    W = (P == med[:, None]).astype(float)
    return np.clip(med, 1e-10, None), W / W.sum(axis=1, keepdims=True)


def _state(d):
    X = np.column_stack([_lg(d['rv_d']), _lg(d['rv_w']) - _lg(d['rv_m']), gap(d), _lg(d['vix1']), _lg(d['rvi1'])])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def gate_fit(d):
    X = _state(d)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-12
    Z, P, y = (X - mu) / sd, P3(d), d['rv_actual'].values
    f = lambda b: S.qlike(y, S.comb(P, _row_w(_sig(b[0] + Z @ b[1:])))) + ALPHA * float(np.sum(b[1:] ** 2))
    w0 = min(np.linspace(0.02, 0.98, 49), key=lambda w: S.qlike(y, S.comb(P, np.array([w, (1 - w) / 2, (1 - w) / 2]))))
    b0 = np.zeros(1 + Z.shape[1]); b0[0] = np.log(w0 / (1 - w0))  # старт - лучший постоянный вес; |b0| <= 4 - без насыщения сигмоиды
    r = minimize(f, b0, method='L-BFGS-B', bounds=[(-4, 4)] + [(None, None)] * Z.shape[1])
    return dict(b=r.x, mu=mu, sd=sd)


def gate_apply(d, g):
    W = _row_w(_sig(g['b'][0] + ((_state(d) - g['mu']) / g['sd']) @ g['b'][1:]))
    return S.comb(P3(d), W), W

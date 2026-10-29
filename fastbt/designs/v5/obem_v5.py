"""Объём куба v5 по стратегиям (архитектор, 29.10). Числа уровней осей — из таблицы docs/otchety/2026-10-29_setka_v5.md."""
import math

SIG = {
    "S1": dict(ma_tip=7, okno=19, porog=23, z_vyhod=4),
    "S2": dict(ma_tip=7, okno=19, shirina=13, podtv=5),
    "S3": dict(okno=17, podtv=6),
    "S4": dict(period=15, mnozh=15),
    "S5": dict(tip=5, uroven=3, period=3, atr_filtr=5, fs=4),
    "S6": dict(okno=15, porog=12, dir_bias=2, fs=5),
}
TREND = {"S3", "S4"}
FILTR_K = dict(adx=7, atr_lo=6, obyom=6, rsi=6, mom=7, dow=4, hurst=4, szhatie=5, naklon_max=4)
FILTR_T = dict(adx=5, atr_lo=6, obyom=6, rsi=2, mom=7, dow=4, naklon_min=4)      # rsi у тренда: в Р действует только «> 0»
FILTR_T_NOM = dict(FILTR_T, rsi=6)
VHOD_K = dict(k_e=6, T=4, reentry=2)
VHOD_T = dict(k_e=4, T=4, reentry=2)
VYHOD_K = dict(stop=31, tejk=36, trail=15, bezub=12, stop_plyus=4, tau=11, chast=7, pir=3, smena=2, razmer_tip=2, m_sily=3, rang_tip=3)
VYHOD_T = dict(VYHOD_K, trail=20)
EXEC_V4 = dict(consec=2, min_hold=2, cooldown=3, min_sily=9)          # фаза 3, полный перебор над top-20
MSP = dict(uroven_razmera=17, L=5, sigma_cel=4, okno_sigma=2)         # слой МСП, вне куба отбора


def prod(d):
    return math.prod(d.values())


def kub_37000():
    """Куб одной линии Р (daily_upper_bound.py@6a7a4636:349-406) без v4 и v2: сигнал × фильтры × РМ."""
    rm = 25 * 33 * 9 * 5          # стопы × тейки × mh × be = 37 125 — вероятный источник имени «37 000»
    fk, ft = 7 * 6 * 6 * 6 * 7 * 4, 5 * 6 * 6 * 6 * 7 * 4
    sig = dict(S1=6 * 19 * 23, S2=6 * 19 * 13, S3=17, S4=15 * 15, S5=4 * 3, S6=15 * 12)
    for sid, s in sig.items():
        f = ft if sid in TREND else fk
        print(f"37000 {sid}: сигнал {s} × фильтры {f} × РМ {rm} = {s * f * rm:.3e}")


def main():
    kub_37000()
    print("sid | сигнал | фильтры | вход | выход | куб отбора (Соболь) | ×фаза3 (108) | log10 | осей | 2 крупнейшие")
    tot = 0
    for sid, s in SIG.items():
        t = sid in TREND
        F, E, X = (FILTR_T, VHOD_T, VYHOD_T) if t else (FILTR_K, VHOD_K, VYHOD_K)
        osi = {**s, **F, **E, **X}
        kub = prod(osi)
        tot += kub
        top2 = sorted(osi.values(), reverse=True)[:2]
        print(f"{sid} | {prod(s)} | {prod(F)} | {prod(E)} | {prod(X)} | {kub:.3e} | {kub * prod(EXEC_V4):.3e} | "
              f"{math.log10(kub):.2f} | {len(osi)} | {top2}")
    print(f"сумма кубов отбора: {tot:.3e}; фильтры тренда номинально {prod(FILTR_T_NOM)}; фаза 3 = {prod(EXEC_V4)}; МСП = {prod(MSP)}")
    # Соболь: N1 = наименьшая степень двойки >= max(20000 (Ж-111 п. 19), 20*k1*k2)
    k1, k2 = 36, 31
    n1 = 1 << math.ceil(math.log2(max(20000, 20 * k1 * k2)))
    print(f"N1 = {n1}; на уровень крупнейшей оси >= {n1 // k1}; на клетку двух крупнейших >= {n1 // (k1 * k2)}")
    n = 6 * n1
    for name, s in (("0,02 (владелец)", 0.02), ("быстрый холодный 0,027", 0.027), ("медленный 0,38", 0.38)):
        cpu = n * s
        print(f"{name}: {n} точек, ЦП {cpu / 3600:.2f} ч; стена 10 проц. p50 (×7) {cpu / 7 / 3600:.2f} ч, p90 (×4) {cpu / 4 / 3600:.2f} ч")
    pol = tot * 0.02
    print(f"полный перебор при 0,02 с: {pol / 3600 / 24 / 365:.3e} лет ЦП; доля Соболя {n / tot:.2e}")
    # фаза 2: 102 единицы × 3 года WF × (top-50 центров × ~10 осей с эффектом × 2 соседа), дедуп не учтён
    f2 = 102 * 3 * 50 * 10 * 2
    print(f"фаза 2 (верх, без дедупа): {f2} оценок точек; фаза 3: {102 * 3 * 20 * prod(EXEC_V4)} прогонов одного тикера")


main()

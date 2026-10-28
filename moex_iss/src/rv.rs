//! RV, BV, J, RSV± по дням. Определение — как у хранимых признаков:
//! moex_volatility/src/data/intraday_features.py
//!  - :105-107 r = ln(close_t / close_{t-1}) по ВСЕМУ ряду баров (сдвиг сквозной, первый бар дня
//!    даёт ночную доходность), :111 первый бар ряда отбрасывается;
//!  - :122-125 лаг |r| для BV берётся внутри дня (у первой доходности дня лага нет);
//!  - :139-146 RV = Σr², BV = (π/2)·Σ|r_i||r_{i-1}|, :149-150 RSV± = Σr² при r>0 / r<0;
//!  - :173 J = max(RV − BV, 0); :189-195 годовое — только *_annualized (sqrt(·)·sqrt(252)),
//!    поля rv/bv/jump — дневной масштаб (дисперсия), его и считаем.
//! Файл баров там читается целиком, без фильтра сессии (:78-93, :95-203).
use crate::model::Bar;
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DayRv {
    pub rv: f64,
    pub bv: f64,
    pub jump: f64,
    pub rsv_pos: f64,
    pub rsv_neg: f64,
    pub n: usize,
}

/// Бары должны быть отсортированы по времени без дублей.
pub fn daily(bars: &[Bar]) -> BTreeMap<i64, DayRv> {
    // значение: (накопитель, |r| предыдущей доходности дня, есть ли она)
    let mut out: BTreeMap<i64, (DayRv, f64, bool)> = BTreeMap::new();
    for w in bars.windows(2) {
        let (p, b) = (&w[0], &w[1]);
        if p.close <= 0.0 || b.close <= 0.0 {
            continue;
        }
        let r = (b.close / p.close).ln();
        let e = out.entry(b.day()).or_insert((DayRv::default(), 0.0, false));
        if e.2 {
            e.0.bv += r.abs() * e.1;
        }
        e.0.rv += r * r;
        if r > 0.0 {
            e.0.rsv_pos += r * r;
        } else if r < 0.0 {
            e.0.rsv_neg += r * r;
        }
        e.0.n += 1;
        e.1 = r.abs();
        e.2 = true;
    }
    out.into_iter()
        .map(|(k, (mut d, _, _))| {
            d.bv *= std::f64::consts::FRAC_PI_2;
            d.jump = (d.rv - d.bv).max(0.0);
            (k, d)
        })
        .collect()
}

/// По тикерам параллельно (rayon).
pub fn daily_many(sets: &[(String, Vec<Bar>)]) -> Vec<(String, BTreeMap<i64, DayRv>)> {
    sets.par_iter().map(|(t, b)| (t.clone(), daily(b))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day_bars(closes: &[f64]) -> Vec<Bar> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Bar {
                ts: 36_000_000 + i as i64 * 600_000,
                open: c,
                high: c,
                low: c,
                close: c,
                volume: 1,
                value: 1.0,
            })
            .collect()
    }

    /// 52 бара: цена чередуется x -> x*e^a -> x ... Известный ответ.
    #[test]
    fn synthetic_day_52_bars() {
        let a = 0.001_f64;
        let mut closes = vec![100.0];
        for i in 0..51 {
            let last = closes[i];
            closes.push(if i % 2 == 0 { last * a.exp() } else { last * (-a).exp() });
        }
        let d = daily(&day_bars(&closes));
        let x = d.values().next().copied().unwrap_or_default();
        assert_eq!(x.n, 51);
        let rv = 51.0 * a * a;
        let bv = std::f64::consts::FRAC_PI_2 * 50.0 * a * a;
        assert!((x.rv - rv).abs() < 1e-12, "rv {}", x.rv);
        assert!((x.bv - bv).abs() < 1e-12, "bv {}", x.bv);
        assert!((x.jump - (rv - bv).max(0.0)).abs() < 1e-12);
        // r>0 на 26 шагах, r<0 на 25
        assert!((x.rsv_pos - 26.0 * a * a).abs() < 1e-12);
        assert!((x.rsv_neg - 25.0 * a * a).abs() < 1e-12);
        assert!((x.rsv_pos + x.rsv_neg - x.rv).abs() < 1e-12);
    }

    #[test]
    fn jump_floor_zero() {
        // равные по модулю доходности: BV = π/2·RV·(n-1)/n > RV -> J = 0
        let d = daily(&day_bars(&[100.0, 101.0, 100.0, 101.0, 100.0]));
        assert_eq!(d.values().next().map(|x| x.jump), Some(0.0));
    }
}

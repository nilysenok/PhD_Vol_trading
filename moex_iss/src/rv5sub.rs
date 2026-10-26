//! Д25.6: дневная RV по 5-минутным доходностям с подвыборкой (5 сдвигов сетки по 1 минуте, среднее) из 1-минутных баров ISS
//! (interval=1). Окно сессии — то же, что у 10-минутной RV: begin ∈ [09:50; 18:50) (rv1850::W0950), без карантина.
//! Цена в узле сетки τ — close последнего бара с begin < τ (бар закрыт к τ, previous-tick). Узлы сдвига k: τ = 09:50 + k + 5j мин.
//! Ряд цен дня: close последнего бара прошлого дня (ночная доходность, как сквозной сдвиг rv::daily) → узлы сетки → close
//! последнего бара дня. RV_k = Σ ln²(p_i / p_{i−1}); RV = среднее RV_k по k = 0..4. Используется только как обучающая цель.
//! Статус: модуль готов и проверен тестами; выгрузка 1-минутных баров не выполнена (оценка объёма — d25_6.py), к CLI не подключён.
#![allow(dead_code)]
use crate::model::{minute_of_day, Bar};
use crate::rv1850::{cut, Win};
use std::collections::BTreeMap;

pub const STEP: i64 = 5;
pub const NSUB: i64 = 5;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DaySub {
    pub rv: f64,
    /// число 1-минутных баров дня в окне
    pub n_bars: usize,
    /// среднее число доходностей на сдвиг
    pub n_ret: f64,
}

fn rv_offset(day: &[Bar], prev_close: Option<f64>, w: Win, k: i64) -> (f64, usize) {
    let mut px: Vec<f64> = Vec::new();
    if let Some(p) = prev_close {
        px.push(p);
    }
    let mut i = 0usize;
    let mut last: Option<f64> = None;
    let mut tau = w.from_min + k;
    while tau < w.to_min {
        while i < day.len() && minute_of_day(day[i].ts) < tau {
            last = Some(day[i].close);
            i += 1;
        }
        if let Some(p) = last {
            px.push(p); // повтор цены — нулевая доходность (учитывается в n_ret)
        }
        tau += STEP;
    }
    if let Some(b) = day.last() {
        px.push(b.close);
    }
    let mut rv = 0.0;
    let mut n = 0usize;
    for wp in px.windows(2) {
        if wp[0] > 0.0 && wp[1] > 0.0 {
            let r = (wp[1] / wp[0]).ln();
            rv += r * r;
            n += 1;
        }
    }
    (rv, n)
}

/// bars — 1-минутные бары одного тикера, отсортированы по времени без дублей (весь ряд; окно режется здесь).
pub fn daily_sub(bars: &[Bar], w: Win) -> BTreeMap<i64, DaySub> {
    let b = cut(bars, w, None);
    let mut by_day: BTreeMap<i64, Vec<Bar>> = BTreeMap::new();
    for x in b {
        by_day.entry(x.day()).or_default().push(x);
    }
    let mut out = BTreeMap::new();
    let mut prev: Option<f64> = None;
    for (d, v) in by_day {
        let mut s = 0.0;
        let mut nr = 0usize;
        for k in 0..NSUB {
            let (rv, n) = rv_offset(&v, prev, w, k);
            s += rv;
            nr += n;
        }
        out.insert(d, DaySub { rv: s / NSUB as f64, n_bars: v.len(), n_ret: nr as f64 / NSUB as f64 });
        prev = v.last().map(|x| x.close);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rv1850::W0950;

    fn bar(day: i64, min: i64, close: f64) -> Bar {
        Bar { ts: day * crate::model::MS_DAY + min * 60_000, open: close, high: close, low: close, close, volume: 1, value: close }
    }

    #[test]
    fn constant_price_zero() {
        let v: Vec<Bar> = (600..1130).map(|m| bar(19000, m, 100.0)).collect();
        let r = daily_sub(&v, W0950);
        assert_eq!(r[&19000].rv, 0.0);
    }

    #[test]
    fn single_jump_counted_once_per_offset() {
        // скачок цены в 12:00 с 100 до 110: на каждом сдвиге ровно одна ненулевая доходность ln(1,1)
        let v: Vec<Bar> = (600..1130).map(|m| bar(19000, m, if m < 720 { 100.0 } else { 110.0 })).collect();
        let r = daily_sub(&v, W0950);
        let j = (1.1f64).ln().powi(2);
        assert!((r[&19000].rv - j).abs() < 1e-12);
    }

    #[test]
    fn overnight_return_included_and_window_cut() {
        let mut v: Vec<Bar> = (600..1130).map(|m| bar(19000, m, 100.0)).collect();
        v.push(bar(19000, 1200, 500.0)); // вечерний бар 20:00 — вне окна, не влияет
        v.extend((600..1130).map(|m| bar(19001, m, 105.0)));
        let r = daily_sub(&v, W0950);
        assert_eq!(r[&19000].rv, 0.0);
        let on = (1.05f64).ln().powi(2);
        assert!((r[&19001].rv - on).abs() < 1e-12);
    }
}

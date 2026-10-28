//! Д23.2 / Д23.4: корреляционные и ликвидностные признаки дня t из 10-мин баров (окно begin ∈ [09:50; 18:50), без карантина).
//! Отдельный бинарник (общие модули крейта подключаются через #[path], main.rs не меняется):
//!   cargo run --release --bin d23_corr -- <каталог баров> <выход.parquet> [ГГГГ-ММ-ДД — срез для детектора утечек]
//! Сетка дня — 54 слота 09:50..18:40; цена слота — close бара, при пропуске — последняя известная цена дня
//! (доходность 0); доходность слота k ≥ 1 — ln(p_k / p_{k-1}); слоты до первого бара дня не участвуют.
//! Признаки (всё — из баров дня t и раньше, известны к 18:50 дня t):
//!   corr_d / corr_w / corr_m — средняя по парам 17 тикеров реализованная корреляция Σ r_i r_j / √(Σ r_i² Σ r_j²),
//!     суммы — за 1 / 5 / 22 последних торговых дня (по объединённому календарю) включая t; пара — если у обоих есть данные;
//!   disp_lnrv — СКО ln RV (Σ r² дня) по тикерам дня t;
//!   amihud — |ln(C_t / C_{t-1})| / оборот дня (млрд руб.), C — close последнего бара окна; lasthour — доля RV дня в слотах [17:50; 18:50).
#[path = "../model.rs"]
#[allow(dead_code)]
mod model;
#[path = "../rv.rs"]
#[allow(dead_code)]
mod rv;
#[path = "../store.rs"]
#[allow(dead_code)]
mod store;

use anyhow::Result;
use arrow::array::{ArrayRef, Date32Array, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use model::{minute_of_day, Bar, TICKERS};
use parquet::arrow::ArrowWriter;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

const FROM: i64 = 9 * 60 + 50;
const TO: i64 = 18 * 60 + 50;
const NS: usize = ((TO - FROM) / 10) as usize; // 54
const LAST_HOUR: usize = ((17 * 60 + 50 - FROM) / 10) as usize; // слот 17:50

/// День тикера: доходности слотов (None — нет цены), Σ оборота, close последнего бара.
#[derive(Clone)]
struct Day {
    r: [f64; NS],
    ok: [bool; NS],
    value: f64,
    close: f64,
}

fn days_of(bars: &[Bar]) -> BTreeMap<i64, Day> {
    let mut by: BTreeMap<i64, Vec<&Bar>> = BTreeMap::new();
    for b in bars {
        let m = minute_of_day(b.ts);
        if (FROM..TO).contains(&m) {
            by.entry(b.day()).or_default().push(b);
        }
    }
    let mut out = BTreeMap::new();
    for (d, bs) in by {
        let mut px = [f64::NAN; NS];
        let mut value = 0.0;
        let mut close = f64::NAN;
        for b in &bs {
            let k = ((minute_of_day(b.ts) - FROM) / 10) as usize;
            if k < NS && b.close > 0.0 {
                px[k] = b.close;
                close = b.close;
            }
            if b.value.is_finite() {
                value += b.value;
            }
        }
        let (mut r, mut ok) = ([0.0; NS], [false; NS]);
        let mut last = f64::NAN;
        for k in 0..NS {
            if px[k].is_nan() && !last.is_nan() {
                px[k] = last; // пропуск бара — цена не изменилась
            }
            if k > 0 && px[k - 1] > 0.0 && px[k] > 0.0 {
                r[k] = (px[k] / px[k - 1]).ln();
                ok[k] = true;
            }
            if !px[k].is_nan() {
                last = px[k];
            }
        }
        out.insert(d, Day { r, ok, value, close });
    }
    out
}

/// Суммы дня по паре: (Σ r_i r_j, Σ r_i², Σ r_j²) по общим слотам.
fn pair_sums(a: &Day, b: &Day) -> (f64, f64, f64) {
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for k in 0..NS {
        if a.ok[k] && b.ok[k] {
            c += a.r[k] * b.r[k];
            va += a.r[k] * a.r[k];
            vb += b.r[k] * b.r[k];
        }
    }
    (c, va, vb)
}

fn mean(v: &[f64]) -> Option<f64> {
    let v: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = Path::new(args.get(1).map(String::as_str).unwrap_or("data/iss_check/bars_10m"));
    let out = Path::new(args.get(2).map(String::as_str).unwrap_or("d23_corr.parquet"));
    let tk: Vec<&str> = TICKERS.to_vec();
    // детектор «будущее не влияет»: третий аргумент ГГГГ-ММ-ДД — бары позже этого дня отбрасываются
    let cut = args.get(3).and_then(|s| model::parse_day(s)).unwrap_or(i64::MAX);
    let mut data: Vec<BTreeMap<i64, Day>> = Vec::new();
    for t in &tk {
        let bars: Vec<Bar> = store::read_bars(&dir.join(format!("{t}.parquet")))?.into_iter().filter(|b| b.day() <= cut).collect();
        data.push(days_of(&bars));
    }
    let cal: Vec<i64> = {
        let mut s: std::collections::BTreeSet<i64> = Default::default();
        data.iter().for_each(|m| s.extend(m.keys()));
        s.into_iter().collect()
    };
    let n = tk.len();
    // суммы пар по дням календаря
    let mut ps: Vec<Vec<Option<(f64, f64, f64)>>> = vec![Vec::new(); cal.len()];
    for (di, d) in cal.iter().enumerate() {
        for i in 0..n {
            for j in (i + 1)..n {
                ps[di].push(match (data[i].get(d), data[j].get(d)) {
                    (Some(a), Some(b)) => Some(pair_sums(a, b)),
                    _ => None,
                });
            }
        }
    }
    let corr_win = |di: usize, w: usize| -> Option<f64> {
        let lo = di + 1 - w.min(di + 1);
        let np = ps[di].len();
        let mut cs = Vec::with_capacity(np);
        for p in 0..np {
            let (mut c, mut va, mut vb, mut any) = (0.0, 0.0, 0.0, false);
            for q in &ps[lo..=di] {
                if let Some((x, y, z)) = q[p] {
                    c += x;
                    va += y;
                    vb += z;
                    any = true;
                }
            }
            if any && va > 0.0 && vb > 0.0 {
                cs.push(c / (va * vb).sqrt());
            }
        }
        mean(&cs)
    };
    let (mut d_col, mut t_col) = (Vec::new(), Vec::new());
    let mut cols: Vec<Vec<Option<f64>>> = vec![Vec::new(); 6];
    for (di, d) in cal.iter().enumerate() {
        let (cd, cw, cm) = (corr_win(di, 1), corr_win(di, 5), corr_win(di, 22));
        let lnrv: Vec<f64> = (0..n)
            .filter_map(|i| data[i].get(d).map(|x| x.r.iter().zip(x.ok.iter()).filter(|(_, o)| **o).map(|(r, _)| r * r).sum::<f64>()))
            .filter(|v| *v > 0.0)
            .map(f64::ln)
            .collect();
        let disp = if lnrv.len() > 2 {
            let m = lnrv.iter().sum::<f64>() / lnrv.len() as f64;
            Some((lnrv.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (lnrv.len() - 1) as f64).sqrt())
        } else {
            None
        };
        for i in 0..n {
            let Some(x) = data[i].get(d) else { continue };
            let rv: f64 = (0..NS).filter(|k| x.ok[*k]).map(|k| x.r[k] * x.r[k]).sum();
            let rv_lh: f64 = (LAST_HOUR..NS).filter(|k| x.ok[*k]).map(|k| x.r[k] * x.r[k]).sum();
            let prev = data[i].range(..*d).next_back().map(|(_, p)| p.close);
            let ami = match prev {
                Some(c0) if c0 > 0.0 && x.close > 0.0 && x.value > 0.0 => Some((x.close / c0).ln().abs() / (x.value / 1e9)),
                _ => None,
            };
            d_col.push(*d as i32);
            t_col.push(tk[i].to_string());
            for (c, v) in cols.iter_mut().zip([cd, cw, cm, disp, ami, if rv > 0.0 { Some(rv_lh / rv) } else { None }]) {
                c.push(v);
            }
        }
    }
    let names = ["corr_d", "corr_w", "corr_m", "disp_lnrv", "amihud", "lasthour"];
    let mut fields = vec![Field::new("date", DataType::Date32, false), Field::new("ticker", DataType::Utf8, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Date32Array::from(d_col)), Arc::new(StringArray::from(t_col))];
    for (nm, c) in names.iter().zip(cols) {
        fields.push(Field::new(*nm, DataType::Float64, true));
        arrays.push(Arc::new(Float64Array::from(c)));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut w = ArrowWriter::try_new(std::fs::File::create(out)?, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    eprintln!("d23_corr: {} строк, {} дней, {} тикеров -> {}", batch.num_rows(), cal.len(), n, out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bar(min: i64, c: f64) -> Bar {
        Bar { ts: min * 60_000, open: c, high: c, low: c, close: c, volume: 1, value: 1e9 }
    }
    #[test]
    fn grid_carries_price_and_cuts_window() {
        let v = [bar(9 * 60 + 40, 1.0), bar(9 * 60 + 50, 100.0), bar(10 * 60 + 10, 110.0), bar(18 * 60 + 50, 1.0)];
        let d = days_of(&v);
        let x = d.values().next().unwrap();
        assert!((x.r[1]).abs() < 1e-15 && x.ok[1]); // 10:00 пропущен -> доходность 0
        assert!((x.r[2] - (1.1f64).ln()).abs() < 1e-12);
        assert_eq!(x.close, 110.0); // бар 18:50 вне окна
    }
    #[test]
    fn perfect_correlation_is_one() {
        let a = days_of(&[bar(600, 100.0), bar(610, 101.0), bar(620, 99.0)]);
        let b = days_of(&[bar(600, 50.0), bar(610, 50.5), bar(620, 49.5)]);
        let (c, va, vb) = pair_sums(a.values().next().unwrap(), b.values().next().unwrap());
        assert!((c / (va * vb).sqrt() - 1.0).abs() < 1e-12);
    }
}

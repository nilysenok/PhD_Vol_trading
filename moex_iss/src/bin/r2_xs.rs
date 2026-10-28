//! R2, агент F (26.10): кросс-секция (G5) и ликвидность/форма дня (G7, G3) из 10-мин баров с начала истории ISS (ISS_start).
//! Отдельный бинарник (общие модули крейта — через #[path], main.rs и d23_corr.rs не меняются). Определения — как у d23_corr (Д23.2/23.4):
//!   окно begin ∈ [09:50; 18:50), без карантина, 54 слота; цена слота — close бара, пропуск — последняя известная цена дня (доходность 0);
//!   доходность слота k ≥ 1 — ln(p_k / p_{k-1}); слоты до первого бара дня не участвуют.
//! Отличия от d23_corr: несколько каталогов баров (история iss_hist + основная выгрузка iss_check), режим EQNL (`{T}_EQNL.parquet`),
//! начало ряда тикера из CSV (правило отрезка Д24.10), дополнительно оборот дня и число доходностей.
//!   cargo run --release --bin r2_xs -- <выход.parquet> <срез ГГГГ-ММ-ДД | -> <начала.csv: ticker,from> <каталог1> [<каталог2> ...]
//! Склейка: бары тикера из всех каталогов, при совпадении begin берётся бар более позднего каталога (основная выгрузка важнее истории).
//! Срез (детектор «будущее не влияет»): бары дней позже среза отбрасываются.
//! Признаки строки (date = t, ticker) — только бары дня t и раньше, известны к 18:50 дня t (информационное множество d):
//!   corr_d / corr_w / corr_m — средняя по парам 17 тикеров реализованная корреляция Σ r_i r_j / √(Σ r_i² Σ r_j²), суммы за 1 / 5 / 22
//!     последних дня объединённого календаря включая t; пара — если у обоих есть данные;
//!   disp_lnrv — СКО ln Σr² дня t по тикерам (≥ 3 тикера); n_xs — число тикеров с данными в день t;
//!   amihud — |ln(C_t / C_{t-1})| / оборот дня (млрд руб.), C — close последнего бара окна;
//!   lasthour — доля Σr² дня в слотах [17:50; 18:50); value — оборот окна (руб.); n_ret — число доходностей окна.
#[path = "../model.rs"]
#[allow(dead_code)]
mod model;
#[path = "../rv.rs"]
#[allow(dead_code)]
mod rv;
#[path = "../store.rs"]
#[allow(dead_code)]
mod store;

use anyhow::{bail, Context, Result};
use arrow::array::{ArrayRef, Date32Array, Float64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use model::{minute_of_day, Bar, TICKERS};
use parquet::arrow::ArrowWriter;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const FROM: i64 = 9 * 60 + 50;
const TO: i64 = 18 * 60 + 50;
const NS: usize = ((TO - FROM) / 10) as usize; // 54
const LAST_HOUR: usize = ((17 * 60 + 50 - FROM) / 10) as usize; // слот 17:50

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
                px[k] = last;
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
    if v.is_empty() {
        None
    } else {
        Some(v.iter().sum::<f64>() / v.len() as f64)
    }
}

/// Склейка баров тикера по каталогам (поздний каталог важнее), начиная с дня `from`, не позже среза `cut`.
fn merged_bars(dirs: &[PathBuf], t: &str, from: i64, cut: i64) -> Result<Vec<Bar>> {
    let mut by: BTreeMap<i64, Bar> = BTreeMap::new();
    for d in dirs {
        for name in [format!("{t}_EQNL.parquet"), format!("{t}.parquet")] {
            let p = d.join(&name);
            if p.exists() {
                for b in store::read_bars(&p)? {
                    by.insert(b.ts, b);
                }
            }
        }
    }
    Ok(by.into_values().filter(|b| b.day() >= from && b.day() <= cut).collect())
}

fn read_starts(p: &Path) -> Result<HashMap<String, i64>> {
    let s = std::fs::read_to_string(p).with_context(|| format!("нет {}", p.display()))?;
    let mut m = HashMap::new();
    for l in s.lines().skip(1) {
        let f: Vec<&str> = l.split(',').collect();
        if f.len() >= 2 {
            if let Some(d) = model::parse_day(f[1].trim()) {
                m.insert(f[0].trim().to_string(), d);
            }
        }
    }
    Ok(m)
}

struct Out {
    date: Vec<i32>,
    ticker: Vec<String>,
    cols: Vec<Vec<Option<f64>>>,
}

const NAMES: [&str; 10] = ["corr_d", "corr_w", "corr_m", "disp_lnrv", "n_xs", "amihud", "lasthour", "value", "n_ret", "rv_win"];

fn compute(data: &[BTreeMap<i64, Day>], tk: &[&str]) -> Out {
    let cal: Vec<i64> = {
        let mut s: BTreeSet<i64> = Default::default();
        data.iter().for_each(|m| s.extend(m.keys()));
        s.into_iter().collect()
    };
    let n = tk.len();
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
    let mut o = Out { date: Vec::new(), ticker: Vec::new(), cols: vec![Vec::new(); NAMES.len()] };
    for (di, d) in cal.iter().enumerate() {
        let (cd, cw, cm) = (corr_win(di, 1), corr_win(di, 5), corr_win(di, 22));
        let rvs: Vec<f64> = (0..n)
            .filter_map(|i| data[i].get(d).map(|x| (0..NS).filter(|k| x.ok[*k]).map(|k| x.r[k] * x.r[k]).sum::<f64>()))
            .collect();
        let lnrv: Vec<f64> = rvs.iter().copied().filter(|v| *v > 0.0).map(f64::ln).collect();
        let disp = if lnrv.len() > 2 {
            let m = lnrv.iter().sum::<f64>() / lnrv.len() as f64;
            Some((lnrv.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (lnrv.len() - 1) as f64).sqrt())
        } else {
            None
        };
        let nxs = Some(rvs.len() as f64);
        for i in 0..n {
            let Some(x) = data[i].get(d) else { continue };
            let rv: f64 = (0..NS).filter(|k| x.ok[*k]).map(|k| x.r[k] * x.r[k]).sum();
            let rv_lh: f64 = (LAST_HOUR..NS).filter(|k| x.ok[*k]).map(|k| x.r[k] * x.r[k]).sum();
            let nret = (0..NS).filter(|k| x.ok[*k]).count() as f64;
            let prev = data[i].range(..*d).next_back().map(|(_, p)| p.close);
            let ami = match prev {
                Some(c0) if c0 > 0.0 && x.close > 0.0 && x.value > 0.0 => Some((x.close / c0).ln().abs() / (x.value / 1e9)),
                _ => None,
            };
            o.date.push(*d as i32);
            o.ticker.push(tk[i].to_string());
            let vals = [cd, cw, cm, disp, nxs, ami, if rv > 0.0 { Some(rv_lh / rv) } else { None }, Some(x.value), Some(nret), Some(rv)];
            for (c, v) in o.cols.iter_mut().zip(vals) {
                c.push(v);
            }
        }
    }
    o
}

fn write(o: Out, out: &Path) -> Result<usize> {
    let mut fields = vec![Field::new("date", DataType::Date32, false), Field::new("ticker", DataType::Utf8, false)];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(Date32Array::from(o.date)), Arc::new(StringArray::from(o.ticker))];
    for (nm, c) in NAMES.iter().zip(o.cols) {
        fields.push(Field::new(*nm, DataType::Float64, true));
        arrays.push(Arc::new(Float64Array::from(c)));
    }
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let mut w = ArrowWriter::try_new(std::fs::File::create(out)?, schema, None)?;
    w.write(&batch)?;
    w.close()?;
    Ok(batch.num_rows())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        bail!("r2_xs <выход.parquet> <срез|-> <начала.csv> <каталог1> [<каталог2> ...]");
    }
    let out = Path::new(&args[1]);
    let cut = if args[2] == "-" { i64::MAX } else { model::parse_day(&args[2]).context("срез")? };
    let starts = read_starts(Path::new(&args[3]))?;
    let dirs: Vec<PathBuf> = args[4..].iter().map(PathBuf::from).collect();
    let tk: Vec<&str> = TICKERS.to_vec();
    let mut data = Vec::new();
    for t in &tk {
        let from = starts.get(*t).copied().unwrap_or(i64::MIN);
        data.push(days_of(&merged_bars(&dirs, t, from, cut)?));
    }
    let o = compute(&data, &tk);
    let ndays = o.date.iter().collect::<BTreeSet<_>>().len();
    let n = write(o, out)?;
    eprintln!("r2_xs: {n} строк, {ndays} дней, {} тикеров -> {}", tk.len(), out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bar(day: i64, min: i64, c: f64) -> Bar {
        Bar { ts: day * model::MS_DAY + min * 60_000, open: c, high: c, low: c, close: c, volume: 1, value: 1e9 }
    }

    #[test]
    fn window_and_carry() {
        let v = [bar(0, 9 * 60 + 40, 1.0), bar(0, 9 * 60 + 50, 100.0), bar(0, 10 * 60 + 10, 110.0), bar(0, 18 * 60 + 50, 1.0)];
        let d = days_of(&v);
        let x = d.values().next().unwrap();
        assert!(x.r[1].abs() < 1e-15 && x.ok[1]);
        assert!((x.r[2] - (1.1f64).ln()).abs() < 1e-12);
        assert_eq!(x.close, 110.0);
        assert!((x.value - 2e9).abs() < 1e-6); // бары 09:50 и 10:10 в окне
    }

    #[test]
    fn day_t_features_ignore_later_days() {
        // два тикера, три дня; признаки дня 1 не зависят от дня 2
        let mk = |k: f64| -> Vec<Bar> {
            (0..3).flat_map(|d| (0..10).map(move |i| bar(d, 600 + 10 * i, 100.0 * (1.0 + 0.01 * k * ((i * (d + 1)) % 3) as f64)))).collect()
        };
        let full = vec![days_of(&mk(1.0)), days_of(&mk(2.0))];
        let cut: Vec<BTreeMap<i64, Day>> = full.iter().map(|m| m.range(..=1).map(|(k, v)| (*k, v.clone())).collect()).collect();
        let tk = ["A", "B"];
        let (a, b) = (compute(&full, &tk), compute(&cut, &tk));
        for c in 0..NAMES.len() {
            for r in 0..b.date.len() {
                assert_eq!(a.cols[c][r].map(f64::to_bits), b.cols[c][r].map(f64::to_bits), "{} строка {r}", NAMES[c]);
            }
        }
    }

    #[test]
    fn perfect_correlation_is_one() {
        let a = days_of(&[bar(0, 600, 100.0), bar(0, 610, 101.0), bar(0, 620, 99.0)]);
        let b = days_of(&[bar(0, 600, 50.0), bar(0, 610, 50.5), bar(0, 620, 49.5)]);
        let (c, va, vb) = pair_sums(a.values().next().unwrap(), b.values().next().unwrap());
        assert!((c / (va * vb).sqrt() - 1.0).abs() < 1e-12);
    }
}

//! Д17.2: реализованный размах (RRV) из 10-минутных баров ISS со срезом 18:50.
//! Окно — бары с begin в [09:50; 18:50) МСК (rv1850::W0950), бары карантина выброшены
//! (метки — функция пайплайна detect_quarantine_flags, файл quarantine_28.parquet от d172_quarantine28.py).
//! По каждому 10-минутному бару i дня:
//!   Parkinson:     RRV_PK = Σ (ln H_i/L_i)² / (4 ln 2)
//!   Garman–Klass:  RRV_GK = Σ [0,5·(ln H_i/L_i)² − (2 ln 2 − 1)·(ln C_i/O_i)²]
//! Бары с неположительной ценой O/H/L/C или H < L пропускаются (в n_bars не входят).
use crate::model::{day_of, Bar};
use crate::rv1850::{cut, W0950};
use crate::store;
use anyhow::Result;
use arrow::array::{ArrayRef, Date32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

pub const T28: [&str; 28] = [
    "SBER", "VTBR", "MOEX", "LKOH", "TATN", "NVTK", "ALRS", "MTLR", "HYDR", "IRAO", "OGKB", "RTKM", "MTSS", "MGNT",
    "AFLT", "PHOR", "LSRG", "TGKA", "MSNG", "TRMK", "UTAR", "RGSS", "MSRS", "GCHE", "SELG", "MRKC", "PIKK", "MRKP",
];
/// DATE_FROM пайплайна (с него считается карантин), 2014-06-09.
pub const FROM: &str = "2014-06-09";

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DayRange {
    pub pk: f64,
    pub gk: f64,
    pub n: usize,
}

/// Дневные RRV по барам (уже отсортированы, в окне, без карантина).
pub fn daily(bars: &[Bar]) -> BTreeMap<i64, DayRange> {
    let c_pk = 1.0 / (4.0 * std::f64::consts::LN_2);
    let c_gk = 2.0 * std::f64::consts::LN_2 - 1.0;
    let mut out: BTreeMap<i64, DayRange> = BTreeMap::new();
    for b in bars {
        if !(b.open > 0.0 && b.high > 0.0 && b.low > 0.0 && b.close > 0.0 && b.high >= b.low) {
            continue;
        }
        let hl = (b.high / b.low).ln();
        let co = (b.close / b.open).ln();
        let e = out.entry(b.day()).or_default();
        e.pk += hl * hl * c_pk;
        e.gk += 0.5 * hl * hl - c_gk * co * co;
        e.n += 1;
    }
    out
}

fn load_quarantine(path: &Path) -> Result<HashMap<String, HashSet<i64>>> {
    let mut m: HashMap<String, HashSet<i64>> = HashMap::new();
    for b in store::read_batches(path)? {
        let (t, ts) = (store::col_str(&b, "ticker")?, store::col_ts(&b, "ts")?);
        for (t, ts) in t.into_iter().zip(ts) {
            m.entry(t).or_default().insert(ts);
        }
    }
    Ok(m)
}

type Rows = Vec<(String, i64, DayRange)>;

fn job(out: &Path, t: &str, q: &HashSet<i64>, from: i64) -> Result<Rows> {
    let full = store::read_bars(&out.join("bars_10m_full").join(format!("{t}.parquet")))?;
    let full: Vec<Bar> = full.into_iter().filter(|b| day_of(b.ts) >= from).collect();
    Ok(daily(&cut(&full, W0950, Some(q))).into_iter().map(|(d, r)| (t.to_string(), d, r)).collect())
}

pub fn write(path: &Path, rows: &[(String, i64, DayRange)]) -> Result<()> {
    let fields = vec![
        Field::new("date", DataType::Date32, false),
        Field::new("ticker", DataType::Utf8, false),
        Field::new("rrv_pk", DataType::Float64, false),
        Field::new("rrv_gk", DataType::Float64, false),
        Field::new("n_bars", DataType::Int64, false),
    ];
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Date32Array::from_iter_values(rows.iter().map(|r| r.1 as i32))),
        Arc::new(StringArray::from_iter_values(rows.iter().map(|r| r.0.as_str()))),
        Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.2.pk))),
        Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.2.gk))),
        Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.2.n as i64))),
    ];
    store::write(path, Schema::new(fields), cols)
}

pub fn run(out: &Path, quarantine: &Path) -> Result<()> {
    let q = load_quarantine(quarantine)?;
    let empty = HashSet::new();
    let from = crate::model::parse_day(FROM).unwrap_or(0);
    let jobs: Vec<Result<Rows>> = T28.par_iter().map(|t| job(out, t, q.get(*t).unwrap_or(&empty), from)).collect();
    let mut rows = Vec::new();
    for j in jobs {
        rows.extend(j?);
    }
    write(&out.join("rrv_1850.parquet"), &rows)?;
    let neg = rows.iter().filter(|r| r.2.gk <= 0.0).count();
    eprintln!("rrv_1850: {} строк, тикеров {}, дней с RRV_GK <= 0: {neg}", rows.len(), T28.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bar(day: i64, h: i64, m: i64, o: f64, hi: f64, lo: f64, c: f64) -> Bar {
        Bar { ts: day * crate::model::MS_DAY + (h * 60 + m) * 60_000, open: o, high: hi, low: lo, close: c, volume: 1, value: 1.0 }
    }

    /// Синтетический день: H/L = e^0.01 на каждом из 54 баров окна, C/O = e^0.005 (рост), ответ известен в закрытой форме.
    #[test]
    fn synthetic_day_known_answer() {
        let d = 20_000;
        let (a, b) = (0.01f64, 0.005f64);
        let mut v = Vec::new();
        // вне окна: 06:50 и 18:50, 19:00 — должны отсечься
        v.push(bar(d, 6, 50, 100.0, 200.0, 50.0, 150.0));
        let mut p = 100.0;
        for k in 0..54 {
            let (h, m) = ((9 * 60 + 50 + 10 * k) / 60, (9 * 60 + 50 + 10 * k) % 60);
            let lo = p * (-0.0025f64).exp();
            v.push(bar(d, h, m, p, lo * a.exp(), lo, p * b.exp()));
            p *= b.exp();
        }
        v.push(bar(d, 18, 50, 100.0, 200.0, 50.0, 150.0));
        v.push(bar(d, 19, 0, 100.0, 200.0, 50.0, 150.0));
        let r = daily(&cut(&v, W0950, None));
        let x = r[&d];
        assert_eq!(x.n, 54);
        let pk = 54.0 * a * a / (4.0 * 2f64.ln());
        let gk = 54.0 * (0.5 * a * a - (2.0 * 2f64.ln() - 1.0) * b * b);
        assert!((x.pk - pk).abs() < 1e-15, "pk {} vs {pk}", x.pk);
        assert!((x.gk - gk).abs() < 1e-15, "gk {} vs {gk}", x.gk);
        // карантин: убираем бар 10:00 — сумма уменьшается ровно на один член
        let q: HashSet<i64> = [bar(d, 10, 0, 0.0, 0.0, 0.0, 0.0).ts].into_iter().collect();
        let y = daily(&cut(&v, W0950, Some(&q)))[&d];
        assert_eq!(y.n, 53);
        assert!((y.pk - pk * 53.0 / 54.0).abs() < 1e-15);
        assert!((y.gk - gk * 53.0 / 54.0).abs() < 1e-15);
    }

    #[test]
    fn flat_and_broken_bars() {
        let d = 20_001;
        // плоский бар: вклад 0, но учитывается; битый (H < L) и нулевая цена — пропускаются
        let v = vec![bar(d, 10, 0, 5.0, 5.0, 5.0, 5.0), bar(d, 10, 10, 5.0, 4.0, 6.0, 5.0), bar(d, 10, 20, 0.0, 5.0, 4.0, 5.0)];
        let x = daily(&v)[&d];
        assert_eq!((x.n, x.pk, x.gk), (1, 0.0, 0.0));
    }
}

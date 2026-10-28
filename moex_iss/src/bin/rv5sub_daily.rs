//! Д25.6: дневная RV по 5-минутным доходностям с подвыборкой (5 сдвигов по 1 мин, среднее) из 1-минутных баров ISS.
//! Отдельный бинарник (модули крейта — через #[path], main.rs не меняется):
//!   cargo run --release --bin rv5sub_daily -- <каталог bars1m> <выход.csv> [ГГГГ-ММ-ДД — срез для детектора утечек]
//! Вход: {T}_{Y}.csv (begin,open,high,low,close,volume,value; begin «ГГГГ-ММ-ДД ЧЧ:ММ:СС», МСК) — выгрузка d25_6.py download.
//! Выход: ticker,date,rv,n_bars,n_ret — по торговым дням тикера; окно W0950 = [09:50; 18:50), без карантина.
//! Срез (третий аргумент): бары с begin >= среза заменяются детерминированным шумом цены (×(1 ± 5%)) — детектор
//! «будущее не влияет»: RV дней < среза обязана совпасть побайтно с расчётом без шума.
#[path = "../model.rs"]
#[allow(dead_code)]
mod model;
#[path = "../rv5sub.rs"]
mod rv5sub;

use anyhow::{Context, Result};
use model::{day_str, parse_ts, Bar};
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};

fn load(files: &[PathBuf], cut: Option<i64>) -> Result<Vec<Bar>> {
    let mut v = Vec::new();
    for f in files {
        let s = std::fs::read_to_string(f).with_context(|| format!("{f:?}"))?;
        for l in s.lines().skip(1) {
            let p: Vec<&str> = l.split(',').collect();
            if p.len() != 7 {
                continue;
            }
            let ts = match parse_ts(p[0]) {
                Some(t) => t,
                None => continue,
            };
            let g = |i: usize| p[i].parse::<f64>().unwrap_or(f64::NAN);
            let mut b = Bar { ts, open: g(1), high: g(2), low: g(3), close: g(4), volume: g(5) as i64, value: g(6) };
            if let Some(c) = cut {
                if ts >= c {
                    // шум: знак и размер — от минуты (детерминированно), цена ×(1 ± 5%)
                    let k = ((ts / 60_000) % 7) as f64 - 3.0;
                    b.close *= 1.0 + 0.05 * k / 3.0;
                }
            }
            if b.close.is_finite() && b.close > 0.0 {
                v.push(b);
            }
        }
    }
    v.sort_by_key(|b| b.ts);
    v.dedup_by_key(|b| b.ts);
    Ok(v)
}

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = Path::new(&a[1]);
    let out = Path::new(&a[2]);
    let cut = a.get(3).and_then(|d| parse_ts(&format!("{d} 00:00:00")));
    let mut by_t: std::collections::BTreeMap<String, Vec<PathBuf>> = Default::default();
    for e in std::fs::read_dir(dir)? {
        let p = e?.path();
        if p.extension().map_or(false, |x| x == "csv") {
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            if let Some((t, _)) = stem.split_once('_') {
                if p.with_extension("done").exists() {
                    by_t.entry(t.to_string()).or_default().push(p.clone());
                }
            }
        }
    }
    let res: Vec<(String, Vec<String>)> = by_t
        .par_iter()
        .map(|(t, fs)| {
            let mut fs = fs.clone();
            fs.sort();
            let bars = load(&fs, cut).unwrap_or_default();
            let d = rv5sub::daily_sub(&bars, rv5sub::W0950);
            let lines = d
                .iter()
                .map(|(day, x)| format!("{t},{},{:e},{},{:.2}", day_str(*day), x.rv, x.n_bars, x.n_ret))
                .collect();
            (t.clone(), lines)
        })
        .collect();
    let mut f = std::fs::File::create(out)?;
    writeln!(f, "ticker,date,rv,n_bars,n_ret")?;
    let mut n = 0;
    for (_, ls) in &res {
        for l in ls {
            writeln!(f, "{l}")?;
            n += 1;
        }
    }
    eprintln!("тикеров {}, тикер-дней {n} -> {out:?}", res.len());
    Ok(())
}

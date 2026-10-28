//! Загрузка 10-мин баров: страницы start=N по 500, прогресс на диске, докачка после обрыва.
//! Сырьё: raw/{T}.csv (дописывается постранично), прогресс: progress/{T}.json.
use crate::fetch::Http;
use crate::model::Bar;
use crate::parse::parse_candles;
use crate::{session, store};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

pub const PARALLEL: usize = 4;
pub const PAUSE: Duration = Duration::from_millis(100);

#[derive(Serialize, Deserialize, Default)]
struct Progress {
    next_start: u64,
    done: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
fn url(ticker: &str, from: &str, till: &str, start: u64) -> String {
    candles_url(&tqbr_base(ticker), from, till, start)
}

/// Путь инструмента ISS без хвоста (Д24): /iss/engines/{engine}/markets/{market}[/boards/{board}]/securities/{secid}.
pub fn tqbr_base(ticker: &str) -> String {
    format!("/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}")
}

pub fn candles_url(base: &str, from: &str, till: &str, start: u64) -> String {
    format!("{base}/candles.json?interval=10&from={from}&till={till}&start={start}")
}

/// Задание выгрузки (Д24): имя файла, путь инструмента, период.
#[derive(Clone, Debug)]
pub struct Spec {
    pub name: String,
    pub base: String,
    pub from: String,
    pub till: String,
}

fn line(b: &Bar) -> String {
    format!("{},{},{},{},{},{},{}\n", b.ts, b.open, b.high, b.low, b.close, b.volume, b.value)
}

/// Читает сырой csv; битые (оборванные) строки пропускаются, дубли убираются.
pub fn load_raw(path: &Path) -> Result<Vec<Bar>> {
    let mut v = Vec::new();
    if let Ok(s) = std::fs::read_to_string(path) {
        for l in s.lines() {
            let p: Vec<&str> = l.split(',').collect();
            if p.len() != 7 {
                continue;
            }
            let f = |i: usize| p[i].parse::<f64>().ok();
            if let (Some(ts), Some(o), Some(h), Some(lo), Some(c), Some(vol), Some(val)) =
                (p[0].parse::<i64>().ok(), f(1), f(2), f(3), f(4), p[5].parse::<i64>().ok(), f(6))
            {
                v.push(Bar { ts, open: o, high: h, low: lo, close: c, volume: vol, value: val });
            }
        }
    }
    v.sort_by_key(|b| b.ts);
    v.dedup_by_key(|b| b.ts);
    Ok(v)
}

async fn one(http: Http, out: PathBuf, ticker: String, from: String, till: String) -> Result<usize> {
    let sp = Spec { base: tqbr_base(&ticker), name: ticker, from, till };
    one_spec(http, out, sp).await
}

async fn one_spec(http: Http, out: PathBuf, sp: Spec) -> Result<usize> {
    let (ticker, from, till) = (sp.name.clone(), sp.from.clone(), sp.till.clone());
    let raw = out.join("raw").join(format!("{ticker}.csv"));
    let prog = out.join("progress").join(format!("{ticker}.json"));
    std::fs::create_dir_all(raw.parent().context("raw")?)?;
    std::fs::create_dir_all(prog.parent().context("progress")?)?;
    let mut p: Progress = std::fs::read(&prog).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if p.next_start > 0 && !p.done {
        eprintln!("{ticker}: докачка с start={}", p.next_start);
    }
    while !p.done {
        let page = parse_candles(&http.get_text(&candles_url(&sp.base, &from, &till, p.next_start)).await?)
            .with_context(|| format!("{ticker} start={}", p.next_start))?;
        if page.is_empty() {
            p.done = true;
        } else {
            let text: String = page.iter().map(line).collect();
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&raw)?;
            f.write_all(text.as_bytes())?;
            f.sync_data()?;
            p.next_start += page.len() as u64;
        }
        std::fs::write(&prog, serde_json::to_vec(&p)?)?;
        tokio::time::sleep(PAUSE).await;
    }
    let all = load_raw(&raw)?;
    store::write_bars(&out.join("bars_10m_full").join(format!("{ticker}.parquet")), &all)?;
    let main = session::filter_main(&all);
    store::write_bars(&out.join("bars_10m").join(format!("{ticker}.parquet")), &main)?;
    eprintln!("{ticker}: готово, баров {} (основная сессия {})", all.len(), main.len());
    Ok(all.len())
}

/// Грузит произвольные инструменты (Д24), не более PARALLEL одновременно, с той же докачкой.
pub async fn run_specs(specs: &[Spec], out: &Path) -> Result<usize> {
    let http = Http::new()?;
    let sem = Arc::new(Semaphore::new(PARALLEL));
    let mut tasks = Vec::new();
    for sp in specs {
        let (h, o, s2, s) = (http.clone(), out.to_path_buf(), sp.clone(), sem.clone());
        tasks.push(tokio::spawn(async move {
            let _permit = s.acquire_owned().await?;
            one_spec(h, o, s2).await
        }));
    }
    let (mut total, mut errs) = (0, 0);
    for (sp, task) in specs.iter().zip(tasks) {
        match task.await? {
            Ok(n) => total += n,
            Err(e) => {
                errs += 1;
                eprintln!("{}: ОШИБКА {e:#} (прогресс сохранён, перезапуск продолжит)", sp.name)
            }
        }
    }
    eprintln!("всего баров: {total}, ошибок: {errs}");
    anyhow::ensure!(errs == 0, "ошибок выгрузки: {errs}");
    Ok(total)
}

/// Грузит тикеры, не более PARALLEL одновременно.
pub async fn run(tickers: &[String], out: &Path, from: &str, till: &str) -> Result<()> {
    let http = Http::new()?;
    let sem = Arc::new(Semaphore::new(PARALLEL));
    let mut tasks = Vec::new();
    for t in tickers {
        let (h, o, t2, f, tl, s) = (http.clone(), out.to_path_buf(), t.clone(), from.to_string(), till.to_string(), sem.clone());
        tasks.push(tokio::spawn(async move {
            let _permit = s.acquire_owned().await?;
            one(h, o, t2, f, tl).await
        }));
    }
    let mut total = 0;
    for (t, task) in tickers.iter().zip(tasks) {
        match task.await? {
            Ok(n) => total += n,
            Err(e) => eprintln!("{t}: ОШИБКА {e:#} (прогресс сохранён, перезапуск продолжит)"),
        }
    }
    eprintln!("всего баров: {total}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_roundtrip_skips_partial_line() -> Result<()> {
        let p = std::env::temp_dir().join("moex_iss_raw_test.csv");
        let b = Bar { ts: 1000, open: 1.1, high: 2.2, low: 0.5, close: 1.3, volume: 5, value: 6.5 };
        std::fs::write(&p, format!("{}{}{}1000,1.", line(&b), line(&b), "garbage\n"))?;
        assert_eq!(load_raw(&p)?, vec![b]);
        Ok(())
    }

    #[test]
    fn url_shape() {
        let u = url("SBER", "2014-06-01", "2026-02-27", 500);
        assert!(u.contains("/boards/TQBR/securities/SBER/candles.json?interval=10&from=2014-06-01&till=2026-02-27&start=500"));
    }
}

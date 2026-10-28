//! Д24.8 / Д24.10 (26.10): внешние ряды ISS и история акций до 2014-06.
//!  - USD000UTSTOM: валютный рынок, engine currency, market selt, board CETS;
//!  - ближний фьючерс Brent (BR): engine futures, market forts. Контракты — по месяцам поставки, SECID `BR{буква}{цифра}` или
//!    `BR{буква}{цифра}_{год}` (старые серии ISS переименовывает с суффиксом года); принимается тот, у кого SHORTNAME = `BR-{м}.{гг}`
//!    либо ASSETCODE = BR и LSTTRADE в месяце поставки или предыдущем.
//!    Правило склейки: в день d берётся контракт с наименьшим LSTTRADE строго больше d, т.е. в последний день обращения
//!    уже следующий контракт (переход по экспирации, без сокращённой сессии дня экспирации). Дневная RV считается по барам
//!    самого контракта (ночная доходность — от его же последнего бара окна предыдущего дня), поэтому склейка не даёт ложного скачка;
//!  - 10-минутные бары акций TQBR до начала выгрузки 2014-06-01 (Д24.10): TQBR в ISS отдаёт историю с 2011-12-08 (до 2013-09
//!    это история EQBR); VTBR и PHOR до 2013-02-28 торговались в режиме EQNL (Б-список) — он выгружается отдельно (имя `{T}_EQNL`).
//! Сеть — только GET к iss.moex.com, 4 параллельных, пауза 100 мс (download.rs).
use crate::download::{run_specs, tqbr_base, Spec};
use crate::fetch::Http;
use crate::model::{day_str, parse_day, Bar, TICKERS};
use crate::rv::{daily, DayRv};
use crate::rv1850::{cut, Win, W0950};
use crate::store;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::Path;

pub const USD: &str = "USD000UTSTOM";
/// Основная сессия срочного рынка: begin в [10:00; 18:45) (бары 10:00..18:40; вечер и утро отброшены).
pub const WFUT: Win = Win { from_min: 10 * 60, to_min: 18 * 60 + 45 };
const LETTERS: [char; 12] = ['F', 'G', 'H', 'J', 'K', 'M', 'N', 'Q', 'U', 'V', 'X', 'Z'];

pub fn usd_base() -> String {
    format!("/iss/engines/currency/markets/selt/boards/CETS/securities/{USD}")
}

pub fn fut_base(secid: &str) -> String {
    format!("/iss/engines/futures/markets/forts/securities/{secid}")
}

pub fn eqnl_base(ticker: &str) -> String {
    format!("/iss/engines/stock/markets/shares/boards/EQNL/securities/{ticker}")
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Contract {
    pub month: String,
    pub secid: String,
    pub shortname: String,
    pub frsttrade: String,
    pub lsttrade: String,
}

/// Блок description ответа /iss/securities/{secid}.json -> {name: value}.
pub fn description(text: &str) -> Result<BTreeMap<String, String>> {
    let v: Value = serde_json::from_str(text).context("не JSON")?;
    let t = v.get("description").context("нет description")?;
    let cols = t.get("columns").and_then(Value::as_array).context("columns")?;
    let (mut iname, mut ival) = (None, None);
    for (i, c) in cols.iter().enumerate() {
        match c.as_str() {
            Some("name") => iname = Some(i),
            Some("value") => ival = Some(i),
            _ => {}
        }
    }
    let (iname, ival) = (iname.context("name")?, ival.context("value")?);
    let mut out = BTreeMap::new();
    for r in t.get("data").and_then(Value::as_array).context("data")? {
        if let (Some(n), Some(x)) = (r.get(iname).and_then(Value::as_str), r.get(ival).and_then(Value::as_str)) {
            out.insert(n.to_string(), x.to_string());
        }
    }
    Ok(out)
}

/// Подходит ли описание серии к контракту BR месяца (y, m).
pub fn matches(d: &BTreeMap<String, String>, y: i32, m: u32) -> bool {
    let want = format!("BR-{m}.{:02}", y % 100);
    if d.get("SHORTNAME").map(|s| s == &want).unwrap_or(false) {
        return true;
    }
    let lst = match d.get("LSTTRADE").or_else(|| d.get("LSTDELDATE")) {
        Some(s) if s.len() >= 7 => s.clone(),
        _ => return false,
    };
    let (py, pm) = if m == 1 { (y - 1, 12) } else { (y, m - 1) };
    // у части старых серий (BRF7_2017) нет ASSETCODE и SHORTNAME = SECID — тогда достаточно SECID с BR и LSTTRADE
    let is_br = d.get("ASSETCODE").map(|s| s == "BR").unwrap_or_else(|| d.get("SECID").map_or(false, |s| s.starts_with("BR") && s.matches("BR").count() == 1));
    is_br
        && (lst[..7] == format!("{y:04}-{m:02}") || lst[..7] == format!("{py:04}-{pm:02}"))
}

/// Перебор месяцев поставки [from; till] (ГГГГ-ММ), запрос описаний серий, выбор контракта месяца.
pub async fn resolve_contracts(from: (i32, u32), till: (i32, u32)) -> Result<Vec<Contract>> {
    let http = Http::new()?;
    let mut out = Vec::new();
    let (mut y, mut m) = from;
    while (y, m) <= till {
        let code = format!("BR{}{}", LETTERS[(m - 1) as usize], y.rem_euclid(10));
        let mut found = None;
        for secid in [format!("{code}_{y}"), code.clone()] {
            let txt = match http.get_text(&format!("/iss/securities/{secid}.json?iss.meta=off&iss.only=description")).await {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{secid}: {e:#}");
                    continue;
                }
            };
            tokio::time::sleep(crate::download::PAUSE).await;
            let d = description(&txt).unwrap_or_default();
            if !d.is_empty() && matches(&d, y, m) {
                found = Some(Contract {
                    month: format!("{y:04}-{m:02}"),
                    secid: d.get("SECID").cloned().unwrap_or(secid.clone()),
                    shortname: d.get("SHORTNAME").cloned().unwrap_or_default(),
                    frsttrade: d.get("FRSTTRADE").cloned().unwrap_or_default(),
                    lsttrade: d.get("LSTTRADE").or_else(|| d.get("LSTDELDATE")).cloned().unwrap_or_default(),
                });
                break;
            }
        }
        match found {
            Some(c) => out.push(c),
            None => eprintln!("BR {y}-{m:02}: контракт не найден"),
        }
        (y, m) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    }
    out.sort_by(|a, b| a.lsttrade.cmp(&b.lsttrade));
    Ok(out)
}

/// Окна выгрузки контрактов: от (LSTTRADE предыдущего − 10 дней) до своего LSTTRADE (ближний — с LSTTRADE предыдущего).
pub fn contract_specs(cs: &[Contract], from: &str, till: &str) -> Vec<Spec> {
    let (f0, t0) = (parse_day(from).unwrap_or(0), parse_day(till).unwrap_or(i64::MAX));
    let mut out = Vec::new();
    for (i, c) in cs.iter().enumerate() {
        let lst = parse_day(&c.lsttrade).unwrap_or(0);
        let start = if i == 0 { f0 } else { parse_day(&cs[i - 1].lsttrade).unwrap_or(f0) - 10 };
        let fr = parse_day(&c.frsttrade).unwrap_or(start).max(start).max(f0 - 10);
        if lst < f0 || fr > t0 {
            continue;
        }
        out.push(Spec { name: c.secid.clone(), base: fut_base(&c.secid), from: day_str(fr), till: day_str(lst.min(t0)) });
    }
    out
}

/// Ближний контракт дня d: наименьший LSTTRADE строго больше d.
pub fn front(cs: &[Contract], d: i64) -> Option<&Contract> {
    cs.iter().filter(|c| parse_day(&c.lsttrade).map_or(false, |l| l > d)).min_by_key(|c| parse_day(&c.lsttrade).unwrap_or(i64::MAX))
}

fn read_contracts(path: &Path) -> Result<Vec<Contract>> {
    Ok(serde_json::from_slice(&std::fs::read(path).with_context(|| format!("{}", path.display()))?)?)
}

/// Д24.8: выгрузка USD000UTSTOM и ближнего BR.
pub async fn bars(out: &Path, what: &str, from: &str, till: &str) -> Result<()> {
    let mut specs = Vec::new();
    if what == "usd" || what == "all" {
        specs.push(Spec { name: USD.into(), base: usd_base(), from: from.into(), till: till.into() });
    }
    if what == "br" || what == "all" {
        let cp = out.join("br_contracts.json");
        let cs = if cp.exists() {
            read_contracts(&cp)?
        } else {
            let p = |s: &str| -> (i32, u32) { (s[..4].parse().unwrap_or(2014), s[5..7].parse().unwrap_or(1)) };
            let (fy, fm) = p(from);
            let (ty, tm) = p(till);
            // месяц поставки ближнего контракта лежит в пределах ~2 мес. после даты; запас — с месяца начала до till + 3 мес.
            let till_m = if tm + 3 > 12 { (ty + 1, tm + 3 - 12) } else { (ty, tm + 3) };
            let cs = resolve_contracts((fy, fm), till_m).await?;
            std::fs::create_dir_all(out)?;
            std::fs::write(&cp, serde_json::to_vec_pretty(&cs)?)?;
            cs
        };
        eprintln!("BR: контрактов {}", cs.len());
        specs.extend(contract_specs(&cs, from, till));
    }
    run_specs(&specs, out).await.map(|_| ())
}

/// Д24.10: история 17 тикеров TQBR [from; till] (+ EQNL для VTBR/PHOR до 2013-02-28).
pub async fn hist(out: &Path, from: &str, till: &str) -> Result<()> {
    let mut specs: Vec<Spec> =
        TICKERS.iter().map(|t| Spec { name: t.to_string(), base: tqbr_base(t), from: from.into(), till: till.into() }).collect();
    for t in ["VTBR", "PHOR"] {
        specs.push(Spec { name: format!("{t}_EQNL"), base: eqnl_base(t), from: from.into(), till: "2013-02-28".into() });
    }
    run_specs(&specs, out).await.map(|_| ())
}

/// Строки дневной RV одной серии по окну.
fn rows_for(name: &str, bars: &[Bar], w: Win) -> Vec<(String, i64, DayRv, i64)> {
    daily(&cut(bars, w, None)).into_iter().map(|(d, v)| (name.to_string(), d, v, 0)).collect()
}

/// Д24.8: дневная RV USDRUB (окно [09:50; 18:50)) и ближнего BR (основная сессия [10:00; 18:45), склейка по экспирации)
/// -> {dest}: date, ticker (USDRUB | BRENT), rv, bv, j, rsv_pos, rsv_neg, n_bars, n_quar (= 0, карантина нет);
/// {dest без .parquet}_br_roll.csv: date, secid (ближний контракт дня).
pub fn rv(out: &Path, dest: &Path) -> Result<()> {
    let usd = store::read_bars(&out.join("bars_10m_full").join(format!("{USD}.parquet")))?;
    let mut rows = rows_for("USDRUB", &usd, W0950);
    let cs = read_contracts(&out.join("br_contracts.json"))?;
    let mut per: BTreeMap<String, BTreeMap<i64, DayRv>> = BTreeMap::new();
    for c in &cs {
        let p = out.join("bars_10m_full").join(format!("{}.parquet", c.secid));
        if !p.exists() {
            continue;
        }
        per.insert(c.secid.clone(), daily(&cut(&store::read_bars(&p)?, WFUT, None)));
    }
    if per.is_empty() {
        bail!("нет баров BR");
    }
    let mut days: Vec<i64> = per.values().flat_map(|m| m.keys().copied()).collect();
    days.sort();
    days.dedup();
    let mut roll = String::from("date,secid,n_bars\n");
    let mut nmiss = 0;
    for d in days {
        let Some(c) = front(&cs, d) else { continue };
        match per.get(&c.secid).and_then(|m| m.get(&d)) {
            Some(v) => {
                rows.push(("BRENT".into(), d, *v, 0));
                roll += &format!("{},{},{}\n", day_str(d), c.secid, v.n);
            }
            None => nmiss += 1,
        }
    }
    store::write_rv(dest, &rows)?;
    let rp = dest.with_file_name(format!("{}_br_roll.csv", dest.file_stem().and_then(|s| s.to_str()).unwrap_or("rv")));
    std::fs::write(&rp, roll)?;
    eprintln!("rv_ext: {} строк ({}), дней BR без баров ближнего: {nmiss}", rows.len(), dest.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(secid: &str, lst: &str) -> Contract {
        Contract { month: String::new(), secid: secid.into(), shortname: String::new(), frsttrade: "2014-01-01".into(), lsttrade: lst.into() }
    }

    #[test]
    fn front_switches_on_expiry_day() {
        let cs = vec![c("A", "2014-07-16"), c("B", "2014-08-15")];
        let d = |s: &str| parse_day(s).unwrap_or(0);
        assert_eq!(front(&cs, d("2014-07-15")).map(|c| c.secid.as_str()), Some("A"));
        assert_eq!(front(&cs, d("2014-07-16")).map(|c| c.secid.as_str()), Some("B"));
        assert_eq!(front(&cs, d("2014-08-15")).map(|c| c.secid.as_str()), None);
    }

    #[test]
    fn matches_by_shortname_or_lsttrade() {
        let mut d = BTreeMap::new();
        d.insert("SHORTNAME".to_string(), "BR-8.14".to_string());
        assert!(matches(&d, 2014, 8));
        assert!(!matches(&d, 2024, 8));
        let mut e = BTreeMap::new();
        e.insert("SHORTNAME".to_string(), "BRF7_2017".to_string());
        e.insert("ASSETCODE".to_string(), "BR".to_string());
        e.insert("LSTDELDATE".to_string(), "2017-01-02".to_string());
        assert!(matches(&e, 2017, 1));
        assert!(!matches(&e, 2027, 1));
    }

    #[test]
    fn specs_overlap_previous_expiry() {
        let cs = vec![c("A", "2014-07-16"), c("B", "2014-08-15")];
        let s = contract_specs(&cs, "2014-06-01", "2026-02-27");
        assert_eq!(s.len(), 2);
        assert_eq!(s[1].from, "2014-07-06");
        assert_eq!(s[1].till, "2014-08-15");
    }

    #[test]
    fn description_parses() -> Result<()> {
        let t = r#"{"description":{"columns":["name","title","value"],"data":[["SECID","x","BRQ4_2014"],["LSTTRADE","y","2014-08-15"]]}}"#;
        let d = description(t)?;
        assert_eq!(d.get("SECID").map(String::as_str), Some("BRQ4_2014"));
        Ok(())
    }
}

//! Сверка дневных индексов ISS с тем, что лежит в репозитории:
//! MCFTR/RGBITR — data/mcftr_iss.csv, data/rgbitr_iss.csv (прежняя выгрузка тем же ISS);
//! IMOEX — close_IMOEX из moex_discovery/data/final_v6/master_dataset.parquet.
use crate::model::{day_str, parse_day};
use crate::store;
use crate::stored::Stored;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct IndexStat {
    pub index: String,
    pub rows: usize,
    pub from: String,
    pub to: String,
    pub reference: String,
    pub ref_rows: usize,
    pub common: usize,
    pub off: usize,
    pub max_rel: f64,
    pub only_iss: usize,
    pub only_ref: usize,
}

pub fn read_csv(path: &Path) -> BTreeMap<i64, f64> {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    s.lines()
        .skip(1)
        .filter_map(|l| {
            let (d, c) = l.split_once(',')?;
            Some((parse_day(d)?, c.trim().parse().ok()?))
        })
        .collect()
}

fn read_iss(path: &Path) -> Result<BTreeMap<i64, f64>> {
    let mut m = BTreeMap::new();
    for b in store::read_batches(path)? {
        let (d, c) = (store::col_str(&b, "date")?, store::col_f64(&b, "close")?);
        for (ds, cv) in d.iter().zip(c) {
            if let (Some(day), true) = (parse_day(ds), cv.is_finite()) {
                m.insert(day, cv);
            }
        }
    }
    Ok(m)
}

fn read_master_imoex(path: &Path) -> Result<BTreeMap<i64, f64>> {
    let mut m = BTreeMap::new();
    for b in store::read_batches(path)? {
        let (d, c) = (store::col_str(&b, "date")?, store::col_f64(&b, "close_IMOEX")?);
        for (ds, cv) in d.iter().zip(c) {
            if let (Some(day), true) = (parse_day(ds), cv.is_finite()) {
                m.insert(day, cv);
            }
        }
    }
    Ok(m)
}

pub fn compare(index: &str, reference_name: &str, iss: &BTreeMap<i64, f64>, rf: &BTreeMap<i64, f64>) -> IndexStat {
    let mut s = IndexStat { index: index.into(), rows: iss.len(), reference: reference_name.into(), ref_rows: rf.len(), ..Default::default() };
    s.from = iss.keys().next().map(|d| day_str(*d)).unwrap_or_default();
    s.to = iss.keys().next_back().map(|d| day_str(*d)).unwrap_or_default();
    let (lo, hi) = (rf.keys().next().copied().unwrap_or(0), rf.keys().next_back().copied().unwrap_or(-1));
    for (d, v) in iss {
        match rf.get(d) {
            Some(r) => {
                s.common += 1;
                let rel = (v - r).abs() / r.abs().max(1e-12);
                s.max_rel = s.max_rel.max(rel);
                if rel > 1e-6 {
                    s.off += 1;
                }
            }
            None if (lo..=hi).contains(d) => s.only_iss += 1,
            None => {}
        }
    }
    let (ilo, ihi) = (iss.keys().next().copied().unwrap_or(0), iss.keys().next_back().copied().unwrap_or(-1));
    s.only_ref = rf.keys().filter(|d| (ilo..=ihi).contains(d) && !iss.contains_key(d)).count();
    s
}

pub fn run(out: &Path, st: &Stored, repo: &Path) -> Result<()> {
    let mut all = Vec::new();
    for (ix, csv) in [("MCFTR", "data/mcftr_iss.csv"), ("RGBITR", "data/rgbitr_iss.csv"), ("IMOEX", "")] {
        let iss = read_iss(&out.join("index_d").join(format!("{ix}.parquet")))?;
        let (name, rf) = if csv.is_empty() {
            ("master_dataset.close_IMOEX".to_string(), read_master_imoex(&st.master_path())?)
        } else {
            (csv.to_string(), read_csv(&repo.join(csv)))
        };
        let s = compare(ix, &name, &iss, &rf);
        eprintln!("{ix}: {} дней, общих {}, расхождений {}, макс {:.2e}", s.rows, s.common, s.off, s.max_rel);
        all.push(s);
    }
    std::fs::create_dir_all(out.join("report"))?;
    std::fs::write(out.join("report/index_summary.json"), serde_json::to_vec_pretty(&all)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compares_series() {
        let iss: BTreeMap<i64, f64> = [(1, 10.0), (2, 11.0), (3, 12.0)].into_iter().collect();
        let rf: BTreeMap<i64, f64> = [(1, 10.0), (2, 11.5), (4, 1.0), (9, 1.0)].into_iter().collect();
        let s = compare("X", "r", &iss, &rf);
        assert_eq!((s.common, s.off, s.only_iss, s.only_ref), (2, 1, 1, 0));
    }
}

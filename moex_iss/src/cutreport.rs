//! Раздел отчёта «RV со срезом 18:50» (дописывается в docs/otchety/…proverka_dannyh_iss.md).
use crate::cutstats::median;
use crate::model::day_str;
use crate::rv1850::{VarStat, SPLIT};
use anyhow::Result;
use std::collections::BTreeMap;
use std::path::Path;

const MARK: &str = "## RV со срезом 18:50";
type Post = (i64, bool, f64, f64, bool);

fn pc(x: f64) -> String {
    format!("{:.2}%", x * 100.0).replace('.', ",")
}
fn f3(x: f64) -> String {
    format!("{x:.3}").replace('.', ",")
}
fn share(a: usize, b: usize) -> f64 {
    if b == 0 { 0.0 } else { a as f64 / b as f64 }
}
fn med(v: impl Iterator<Item = f64>) -> f64 {
    median(&mut v.filter(|x| *x > 0.0).collect::<Vec<_>>())
}

fn by_variant(st: &[VarStat]) -> String {
    let mut s = format!("| Вариант | Дней до {SPLIT} | Совпало | Доля | Дней после | Совпало | Доля | Медиана RV новый/старый после | Медиана BV | Медиана RV в днях с вечером |\n|---|---|---|---|---|---|---|---|---|---|\n");
    for v in ["0950_raw", "0950_без_карантина", "1000_raw"] {
        let x: Vec<&VarStat> = st.iter().filter(|a| a.variant == v).collect();
        let sum = |f: fn(&VarStat) -> usize| x.iter().map(|a| f(a)).sum::<usize>();
        let post = || x.iter().flat_map(|a| a.post.iter());
        let (r, b) = (med(post().map(|p: &Post| p.2)), med(post().map(|p: &Post| p.3)));
        let e = med(post().filter(|p| p.4).map(|p| p.2));
        s += &format!(
            "| {v} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            sum(|a| a.pre_days), sum(|a| a.pre_ok), pc(share(sum(|a| a.pre_ok), sum(|a| a.pre_days))),
            sum(|a| a.post_days), sum(|a| a.post_ok), pc(share(sum(|a| a.post_ok), sum(|a| a.post_days))),
            f3(r), f3(b), f3(e)
        );
    }
    s
}

fn by_year(st: &[VarStat]) -> String {
    let mut m: BTreeMap<String, Vec<Post>> = BTreeMap::new();
    for a in st.iter().filter(|a| a.variant == "0950_без_карантина") {
        for p in &a.post {
            m.entry(day_str(p.0)[..4].to_string()).or_default().push(*p);
        }
    }
    let mut s = String::from("| Год | Тикер-дней | Совпало | Доля | Дней, где у старых есть вечер/утро | Медиана RV новый/старый (все дни) | Медиана RV в днях с вечером |\n|---|---|---|---|---|---|---|\n");
    for (y, v) in &m {
        let ok = v.iter().filter(|x| x.1).count();
        let ev = v.iter().filter(|x| x.4).count();
        s += &format!("| {y} | {} | {ok} | {} | {ev} | {} | {} |\n", v.len(), pc(share(ok, v.len())), f3(med(v.iter().map(|x| x.2))), f3(med(v.iter().filter(|x| x.4).map(|x| x.2))));
    }
    s
}

fn csv_table(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let mut s = String::new();
    for (i, l) in text.lines().enumerate() {
        let c: Vec<String> = l
            .split(',')
            .enumerate()
            .map(|(j, x)| if i > 0 && j >= 4 { pc(x.parse().unwrap_or(0.0)) } else { x.replace('_', " ") })
            .collect();
        s += &format!("| {} |\n", c.join(" | "));
        if i == 0 {
            s += &format!("|{}\n", "---|".repeat(c.len()));
        }
    }
    s
}

pub fn run(out: &Path, dest: &Path) -> Result<()> {
    let r = out.join("report");
    let st: Vec<VarStat> = serde_json::from_slice(&std::fs::read(r.join("rv1850_summary.json"))?)?;
    let old = std::fs::read_to_string(dest)?;
    let base = old.split(MARK).next().unwrap_or("").trim_end().to_string();
    let mut md = format!("{base}\n\n{MARK}\n\n");
    md += include_str!("../report_1850_head.md");
    md += "\n### Сверка со старыми хранимыми признаками (допуск 1e-6 по RV, BV и J)\n\n";
    md += &by_variant(&st);
    md += "\n### После 22.06.2020 по годам (окно 09:50–18:50, без карантина)\n\n";
    md += &by_year(&st);
    md += "\n### Число баров до 18:50 по тикер-дням (begin ∈ [10:00; 18:50))\n\n";
    md += &csv_table(&r.join("bars_count_by_year.csv"));
    md += include_str!("../report_1850_tail.md");
    std::fs::write(dest, md + "\n")?;
    eprintln!("раздел дописан: {}", dest.display());
    Ok(())
}

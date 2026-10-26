mod cutreport;
mod cutstats;
mod download;
mod ext;
mod fetch;
mod idxcmp;
mod indices;
mod model;
mod parse;
mod range;
mod report;
mod rv;
mod rv1850;
mod rv5sub;
mod rvcmp;
mod session;
mod store;
mod stored;
mod verify;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Проверка хранимых данных MOEX через ISS (дорожка Г)")]
struct Cli {
    /// Каталог выхода (parquet, прогресс, сводки)
    #[arg(long, default_value = "data/iss_check", global = true)]
    out: PathBuf,
    /// Корень основной копии репозитория (хранимые данные, только чтение)
    #[arg(long, default_value = stored::DEFAULT_ROOT, global = true)]
    stored_root: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Г1: 10-мин бары TQBR (докачка с места обрыва)
    Bars {
        #[arg(long, default_value = "2014-06-01")]
        from: String,
        #[arg(long, default_value = "2026-02-27")]
        till: String,
        /// Список через запятую; по умолчанию все 17
        #[arg(long, value_delimiter = ',')]
        tickers: Vec<String>,
    },
    /// Г1: дневные индексы MCFTR, RGBITR, IMOEX
    Indices {
        #[arg(long, default_value = "2022-01-03")]
        from: String,
        #[arg(long, default_value = "2026-02-27")]
        till: String,
    },
    /// Г2: бар к бару с хранимыми, пересчёт нет
    Verify,
    /// Г3: RV/BV/J по ISS- и хранимым барам против хранимых признаков
    Rv,
    /// Сверка индексов с имеющимися рядами
    IndexCheck,
    /// Пересчёт RV со срезом 18:50 -> rv_1850.parquet, сверка со старыми признаками, число баров
    Rv1850,
    /// Д17.2: реализованный размах (Parkinson, Garman–Klass) 28 тикеров -> rrv_1850.parquet
    Range {
        /// Метки баров карантина (ticker, ts) — функция пайплайна, scripts/analysis/d172_quarantine28.py
        #[arg(long, default_value = "data/iss_check/quarantine_28.parquet")]
        quarantine: PathBuf,
    },
    /// Дописать в отчёт раздел «RV со срезом 18:50»
    Report1850 {
        #[arg(long, default_value = "docs/otchety/2026-10-25_proverka_dannyh_iss.md")]
        dest: PathBuf,
    },
    /// Д24.8: 10-мин бары USD000UTSTOM (CETS) и ближних контрактов BR (FORTS) -> {out}/bars_10m_full/
    ExtBars {
        /// usd | br | all
        #[arg(long, default_value = "all")]
        what: String,
        #[arg(long, default_value = "2014-05-01")]
        from: String,
        #[arg(long, default_value = "2026-02-27")]
        till: String,
    },
    /// Д24.8: дневная RV USDRUB [09:50; 18:50) и ближнего BR [10:00; 18:45) -> parquet + карта склейки
    ExtRv {
        #[arg(long)]
        dest: PathBuf,
    },
    /// Д24.10: история 17 тикеров TQBR (+ EQNL для VTBR/PHOR) до начала основной выгрузки
    HistBars {
        #[arg(long, default_value = "2011-12-01")]
        from: String,
        #[arg(long, default_value = "2014-05-31")]
        till: String,
    },
    /// Г4: собрать отчёт .md из сводок
    Report {
        #[arg(long, default_value = "docs/otchety/2026-10-25_proverka_dannyh_iss.md")]
        dest: PathBuf,
    },
}

fn tickers() -> Vec<String> {
    model::TICKERS.iter().map(|s| s.to_string()).collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Bars { from, till, tickers } => {
            let t = if tickers.is_empty() { self::tickers() } else { tickers };
            download::run(&t, &cli.out, &from, &till).await
        }
        Cmd::Verify => verify::run(&cli.out, &stored::Stored::new(&cli.stored_root), &tickers()),
        Cmd::Rv => rvcmp::run(&cli.out, &stored::Stored::new(&cli.stored_root), &tickers()),
        Cmd::IndexCheck => idxcmp::run(&cli.out, &stored::Stored::new(&cli.stored_root), std::path::Path::new(".")),
        Cmd::Rv1850 => rv1850::run(&cli.out, &stored::Stored::new(&cli.stored_root), &tickers()),
        Cmd::Range { quarantine } => range::run(&cli.out, &quarantine),
        Cmd::Report1850 { dest } => cutreport::run(&cli.out, &dest),
        Cmd::Report { dest } => report::run(&cli.out, &dest),
        Cmd::ExtBars { what, from, till } => ext::bars(&cli.out, &what, &from, &till).await,
        Cmd::ExtRv { dest } => ext::rv(&cli.out, &dest),
        Cmd::HistBars { from, till } => ext::hist(&cli.out, &from, &till).await,
        Cmd::Indices { from, till } => {
            let t: Vec<String> = model::INDICES.iter().map(|s| s.to_string()).collect();
            indices::run(&t, &cli.out, &from, &till).await
        }
    }
}

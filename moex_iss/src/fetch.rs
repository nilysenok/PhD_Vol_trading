//! HTTP: только GET, только iss.moex.com; повтор при 5xx/429/таймауте с экспоненциальной задержкой.
use anyhow::{bail, Result};
use std::time::Duration;

pub const BASE: &str = "https://iss.moex.com";
const MAX_TRIES: u32 = 8;

#[derive(Clone)]
pub struct Http {
    client: reqwest::Client,
}

/// Задержка перед повтором номер `attempt` (0-based): 0,5с · 2^n, не больше 30 с.
pub fn backoff(attempt: u32) -> Duration {
    Duration::from_millis((500u64 << attempt.min(6)).min(30_000))
}

impl Http {
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("moex_iss-check/0.1 (research; GET only)")
            .build()?;
        Ok(Self { client })
    }

    pub async fn get_text(&self, path_and_query: &str) -> Result<String> {
        let url = format!("{BASE}{path_and_query}");
        let mut last = String::new();
        for attempt in 0..MAX_TRIES {
            match self.client.get(&url).send().await {
                Ok(r) if r.status().is_success() => match r.text().await {
                    Ok(t) => return Ok(t),
                    Err(e) => last = format!("тело: {e}"),
                },
                Ok(r) if r.status().is_server_error() || r.status().as_u16() == 429 => {
                    last = format!("HTTP {}", r.status());
                }
                Ok(r) => bail!("HTTP {} для {url}", r.status()),
                Err(e) => last = format!("{e}"),
            }
            eprintln!("повтор {}/{MAX_TRIES} ({last}): {path_and_query}", attempt + 1);
            tokio::time::sleep(backoff(attempt)).await;
        }
        bail!("исчерпаны повторы ({last}): {url}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff(0), Duration::from_millis(500));
        assert_eq!(backoff(1), Duration::from_millis(1000));
        assert_eq!(backoff(3), Duration::from_millis(4000));
        assert_eq!(backoff(20), Duration::from_millis(30_000));
    }
}

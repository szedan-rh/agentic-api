use async_trait::async_trait;
use tracing::debug;

use super::types::{SearchResponse, SearchResult};
use crate::error::Error;

pub struct OgxStore {
    base_url: String,
    client: reqwest::Client,
}

impl OgxStore {
    #[must_use]
    pub fn new(base_url: &str, client: reqwest::Client) -> Self {
        let base_url = base_url.trim_end_matches('/').to_owned();
        Self { base_url, client }
    }
}

#[async_trait]
impl super::VectorSearch for OgxStore {
    async fn search(&self, store_id: &str, query: &str) -> Result<Vec<SearchResult>, Error> {
        let url = format!("{}/v1/vector_stores/{store_id}/search", self.base_url);
        debug!(%url, %query, "searching vector store via OGx");

        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "query": query,
                "max_num_results": 10
            }))
            .send()
            .await
            .map_err(Error::Store)?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::StoreResponse {
                status: status.as_u16(),
                body,
            });
        }

        let search_resp: SearchResponse = resp.json().await.map_err(Error::Store)?;
        Ok(search_resp.data)
    }
}

pub mod ogx;
pub mod types;

use async_trait::async_trait;

use types::SearchResult;

#[async_trait]
pub trait VectorSearch: Send + Sync {
    async fn search(&self, store_id: &str, query: &str) -> Result<Vec<SearchResult>, crate::error::Error>;
}

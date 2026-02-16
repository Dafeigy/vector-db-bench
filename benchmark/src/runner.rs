use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::loader::IndexedVector;

/// Request body for the `/search` endpoint.
#[derive(Debug, Clone, Serialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub top_k: u32,
}

/// A single search result returned by the server.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f64,
}

/// Response from the `/search` endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
}

/// Result of a single query, including latency and returned results.
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub query_index: usize,
    pub results: Vec<SearchResult>,
    pub latency_ms: f64,
}

/// Send warmup queries to the server. Results are discarded.
pub async fn run_warmup(
    client: &reqwest::Client,
    server_url: &str,
    queries: &[IndexedVector],
    warmup_count: usize,
) {
    let url = format!("{}/search", server_url.trim_end_matches('/'));
    let count = warmup_count.min(queries.len());

    for q in queries.iter().take(count) {
        let req = SearchRequest {
            vector: q.vector.clone(),
            top_k: 10,
        };
        // Fire and forget – ignore errors during warmup.
        let _ = client.post(&url).json(&req).send().await;
    }
}

/// Run all queries concurrently with a semaphore-based concurrency limit.
///
/// The query order is shuffled deterministically using the provided `seed`.
/// Each query is sent as a POST `/search` request with `top_k=10`.
/// Returns a `QueryResult` for every query, preserving the original query index.
pub async fn run_queries(
    client: &reqwest::Client,
    server_url: &str,
    queries: &[IndexedVector],
    concurrency: usize,
    seed: u64,
) -> Vec<QueryResult> {
    let url = format!("{}/search", server_url.trim_end_matches('/'));

    // Build (original_index, query) pairs and shuffle deterministically.
    let mut indexed: Vec<(usize, &IndexedVector)> = queries.iter().enumerate().collect();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    indexed.shuffle(&mut rng);

    let semaphore = Arc::new(Semaphore::new(concurrency));
    let client = client.clone();
    let url = Arc::new(url);

    let mut handles = Vec::with_capacity(indexed.len());

    for (idx, query) in indexed {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let url = Arc::clone(&url);
        let vector = query.vector.clone();

        let handle = tokio::spawn(async move {
            let req = SearchRequest {
                vector,
                top_k: 10,
            };

            let start = Instant::now();
            let resp = client.post(url.as_str()).json(&req).send().await;
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

            drop(permit);

            let results = match resp {
                Ok(r) => r
                    .json::<SearchResponse>()
                    .await
                    .map(|sr| sr.results)
                    .unwrap_or_default(),
                Err(_) => Vec::new(),
            };

            QueryResult {
                query_index: idx,
                results,
                latency_ms,
            }
        });

        handles.push(handle);
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        if let Ok(qr) = handle.await {
            results.push(qr);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_request_serialize() {
        let req = SearchRequest {
            vector: vec![1.0, 2.0, 3.0],
            top_k: 10,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["top_k"], 10);
        assert_eq!(json["vector"].as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_search_result_deserialize() {
        let json = r#"{"id": 42, "distance": 1.5}"#;
        let r: SearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(r.id, 42);
        assert!((r.distance - 1.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_search_response_deserialize() {
        let json = r#"{"results": [{"id": 1, "distance": 0.5}, {"id": 2, "distance": 1.0}]}"#;
        let resp: SearchResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.results.len(), 2);
        assert_eq!(resp.results[0].id, 1);
    }

    #[test]
    fn test_deterministic_shuffle() {
        let queries: Vec<IndexedVector> = (0..20)
            .map(|i| IndexedVector {
                id: i,
                vector: vec![i as f32],
            })
            .collect();

        let shuffle = |seed: u64| -> Vec<usize> {
            let mut indexed: Vec<(usize, &IndexedVector)> =
                queries.iter().enumerate().collect();
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indexed.shuffle(&mut rng);
            indexed.iter().map(|(i, _)| *i).collect()
        };

        // Same seed → same order
        assert_eq!(shuffle(42), shuffle(42));
        // Different seed → different order (extremely unlikely to collide)
        assert_ne!(shuffle(42), shuffle(99));
    }
}

use serde::{Deserialize, Serialize};
use std::path::Path;

/// A vector with its ID, as stored in the JSON data files.
#[derive(Debug, Clone, Deserialize)]
pub struct IndexedVector {
    pub id: u64,
    pub vector: Vec<f32>,
}

/// A single item in a bulk insert request.
#[derive(Debug, Clone, Serialize)]
pub struct InsertItem {
    pub id: u64,
    pub vector: Vec<f32>,
}

/// Request body for the `/bulk_insert` endpoint.
#[derive(Debug, Serialize)]
pub struct BulkInsertRequest {
    pub vectors: Vec<InsertItem>,
}

/// Response from the `/bulk_insert` endpoint.
#[derive(Debug, Deserialize)]
pub struct BulkInsertResponse {
    pub status: String,
    pub inserted: usize,
}

/// Load base vectors from a JSON file.
///
/// The file should contain a JSON array of `{"id": u64, "vector": [f32; 128]}` objects.
pub async fn load_vectors_from_file(path: &str) -> Result<Vec<IndexedVector>, Box<dyn std::error::Error>> {
    let path = Path::new(path);
    let content = tokio::fs::read_to_string(path).await?;
    let vectors: Vec<IndexedVector> = serde_json::from_str(&content)?;
    Ok(vectors)
}

/// Bulk insert vectors into the server via HTTP POST `/bulk_insert`.
///
/// Vectors are split into batches of `batch_size` to avoid oversized request bodies.
/// Returns the total number of successfully inserted vectors.
pub async fn bulk_insert_vectors(
    client: &reqwest::Client,
    server_url: &str,
    vectors: &[IndexedVector],
    batch_size: usize,
) -> Result<usize, Box<dyn std::error::Error>> {
    let url = format!("{}/bulk_insert", server_url.trim_end_matches('/'));
    let mut total_inserted: usize = 0;

    for chunk in vectors.chunks(batch_size) {
        let items: Vec<InsertItem> = chunk
            .iter()
            .map(|v| InsertItem {
                id: v.id,
                vector: v.vector.clone(),
            })
            .collect();

        let request = BulkInsertRequest { vectors: items };

        let resp = client
            .post(&url)
            .json(&request)
            .send()
            .await?
            .error_for_status()?
            .json::<BulkInsertResponse>()
            .await?;

        total_inserted += resp.inserted;
    }

    Ok(total_inserted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexed_vector_deserialize() {
        let json = r#"{"id": 42, "vector": [1.0, 2.0, 3.0]}"#;
        let v: IndexedVector = serde_json::from_str(json).unwrap();
        assert_eq!(v.id, 42);
        assert_eq!(v.vector, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bulk_insert_request_serialize() {
        let req = BulkInsertRequest {
            vectors: vec![
                InsertItem { id: 1, vector: vec![0.1, 0.2] },
                InsertItem { id: 2, vector: vec![0.3, 0.4] },
            ],
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["vectors"].as_array().unwrap().len(), 2);
        assert_eq!(json["vectors"][0]["id"], 1);
    }

    #[test]
    fn test_bulk_insert_response_deserialize() {
        let json = r#"{"status": "ok", "inserted": 500}"#;
        let resp: BulkInsertResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.status, "ok");
        assert_eq!(resp.inserted, 500);
    }

    #[test]
    fn test_load_vectors_json_array() {
        let json = r#"[{"id": 0, "vector": [1.0, 2.0]}, {"id": 1, "vector": [3.0, 4.0]}]"#;
        let vectors: Vec<IndexedVector> = serde_json::from_str(json).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].id, 0);
        assert_eq!(vectors[1].vector, vec![3.0, 4.0]);
    }
}

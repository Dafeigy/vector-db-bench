use std::sync::RwLock;
use crate::api::*;
use crate::distance::l2_distance;

pub struct VectorDB {
    // Simple brute-force storage: vectors stored in a flat structure
    vectors: RwLock<Vec<(u64, Vec<f32>)>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vectors = self.vectors.write().unwrap();
        vectors.push((id, vector));
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut db_vectors = self.vectors.write().unwrap();
        let count = vectors.len();
        db_vectors.extend(vectors);
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let vectors = self.vectors.read().unwrap();
        
        // Compute distances to all vectors
        let mut distances: Vec<(u64, f64)> = vectors
            .iter()
            .map(|(id, v)| (*id, l2_distance(vector, v)))
            .collect();
        
        // Partial sort to get top_k smallest distances
        // Using select_nth_unstable for efficiency - we only need to find top_k, not sort everything
        let k = top_k as usize;
        if k >= distances.len() {
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        } else {
            // Use select_nth_unstable to find the k smallest elements
            let mid = distances.as_mut_slice().select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            // Sort only the first k elements
            mid.0.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        
        distances
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}

use crate::api::SearchResult;
use crate::distance::l2_distance_squared;
use rayon::prelude::*;
use std::sync::RwLock;

// Store vectors in a flat array for better cache locality
pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    vectors: RwLock<Vec<f32>>, // flattened: [v0_0, v0_1, ..., v0_127, v1_0, ...]
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            ids: RwLock::new(Vec::new()),
            vectors: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids = self.ids.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        ids.push(id);
        vectors.extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids = self.ids.write().unwrap();
        let mut vecs = self.vectors.write().unwrap();
        let count = vectors.len();
        for (id, v) in vectors {
            ids.push(id);
            vecs.extend(v);
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read().unwrap();
        let vectors = self.vectors.read().unwrap();
        
        let n = ids.len();
        if n == 0 {
            return Vec::new();
        }
        
        let k = top_k as usize;
        
        // Pre-allocate results vector
        let mut results: Vec<(u64, f32)> = Vec::with_capacity(n);
        
        // Parallel distance computation - collect into pre-allocated vec
        (0..n)
            .into_par_iter()
            .for_each(|i| {
                let start = i * 128;
                let dist_sq = l2_distance_squared(query, &vectors[start..start + 128]);
                results[i] = (ids[i], dist_sq);
            });
        
        // Use select_nth_unstable for O(n) partial sort
        if k < results.len() {
            results.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(k);
        }
        
        // Sort the top-k results
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Convert to final results with actual L2 distance
        results
            .into_iter()
            .map(|(id, dist_sq)| SearchResult { 
                id, 
                distance: (dist_sq as f64).sqrt() 
            })
            .collect()
    }
}

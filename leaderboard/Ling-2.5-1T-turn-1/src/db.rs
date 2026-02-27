use crate::api::*;
use parking_lot::RwLock;
use std::collections::HashMap;
use rayon::prelude::*;

pub struct VectorDB {
    data: RwLock<HashMap<u64, Vec<f32>>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != 128 {
            return;
        }
        self.data.write().insert(id, vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut guard = self.data.write();
        let mut count = 0;
        for (id, vec) in vectors {
            if vec.len() == 128 {
                guard.insert(id, vec);
                count += 1;
            }
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if vector.len() != 128 || top_k == 0 {
            return vec![];
        }
        let k = top_k as usize;
        let data = self.data.read();
        let len = data.len();
        if len <= k {
            let mut results: Vec<SearchResult> = data
                .par_iter()
                .map(|(id, v)| {
                    let dist = crate::distance::l2_distance(vector, v);
                    SearchResult { id: *id, distance: dist }
                })
                .collect();
            results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            return results;
        }
        // Parallel distance computation + partial sort
        let mut results: Vec<SearchResult> = data
            .par_iter()
            .map(|(id, v)| {
                let dist = crate::distance::l2_distance(vector, v);
                SearchResult { id: *id, distance: dist }
            })
            .collect();
        if results.len() > k {
            // Partition top-k using nth_element-like selection
            results.select_nth_unstable_by(k - 1, |a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(k);
        }
        // Final sort of top-k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

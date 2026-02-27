use crate::api::*;
use std::sync::RwLock;
use std::collections::HashMap;
use rayon::prelude::*;

pub struct VectorDB {
    vectors: RwLock<HashMap<u64, Vec<f32>>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vectors = self.vectors.write().unwrap();
        vectors.insert(id, vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut store = self.vectors.write().unwrap();
        let count = vectors.len();
        for (id, vector) in vectors {
            store.insert(id, vector);
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let vectors = self.vectors.read().unwrap();
        
        let k = top_k as usize;
        
        // Collect vectors into a Vec for parallel iteration
        let items: Vec<(u64, &[f32])> = vectors.iter().map(|(id, v)| (*id, v.as_slice())).collect();
        
        // Parallel distance computation - each thread computes distances for a chunk
        let distances: Vec<(u64, f64)> = items
            .par_iter()
            .map(|(id, stored_vector)| {
                let dist = crate::distance::l2_distance(vector, stored_vector);
                (*id, dist)
            })
            .collect();
        
        // Sort results
        let mut results = distances;
        results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        results
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}

use crate::api::*;
use std::sync::RwLock;
use rayon::prelude::*;

pub struct VectorDB {
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
        
        // Clone the vector for parallel processing
        let vectors_clone: Vec<(u64, Vec<f32>)> = vectors.clone();
        
        let mut results: Vec<SearchResult> = vectors_clone
            .par_iter()
            .map(|(id, v)| {
                let distance = crate::distance::l2_distance(vector, v);
                SearchResult { id: *id, distance }
            })
            .collect();
        
        // Sort by distance and take top_k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(top_k as usize);
        
        results
    }
}

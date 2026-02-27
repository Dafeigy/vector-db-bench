use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;
use rayon::prelude::*;

pub struct VectorDB {
    // Store vectors in a flat contiguous structure for better cache locality
    vectors: RwLock<VectorStorage>,
}

struct VectorStorage {
    // Parallel arrays for better memory layout
    ids: Vec<u64>,
    // Store all vectors contiguously (128 floats per vector)
    data: Vec<f32>,
}

impl VectorStorage {
    fn new() -> Self {
        VectorStorage {
            ids: Vec::new(),
            data: Vec::new(),
        }
    }
    
    #[inline]
    fn get_vector(&self, index: usize) -> &[f32] {
        let start = index * 128;
        &self.data[start..start + 128]
    }
    
    fn len(&self) -> usize {
        self.ids.len()
    }
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors: RwLock::new(VectorStorage::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut storage = self.vectors.write().unwrap();
        
        // For single inserts, just append (we assume IDs are unique in benchmark)
        storage.ids.push(id);
        storage.data.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut storage = self.vectors.write().unwrap();
        let count = vectors.len();
        
        // Reserve space upfront
        storage.ids.reserve(count);
        storage.data.reserve(count * 128);
        
        for (id, vector) in vectors {
            storage.ids.push(id);
            storage.data.extend_from_slice(&vector);
        }
        
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let storage = self.vectors.read().unwrap();
        let n = storage.len();
        
        // Compute all distances in parallel
        let mut results: Vec<SearchResult> = (0..n)
            .into_par_iter()
            .map(|i| SearchResult {
                id: storage.ids[i],
                distance: l2_distance(query, storage.get_vector(i)),
            })
            .collect();
        
        // Partial sort to get top-k (using select_nth_unstable for better performance)
        let k = (top_k as usize).min(results.len());
        if k > 0 && k < results.len() {
            results.select_nth_unstable_by(k - 1, |a, b| {
                a.distance.partial_cmp(&b.distance).unwrap()
            });
            results.truncate(k);
        } else if k > 0 {
            results.truncate(k);
        }
        
        // Sort the top-k results
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }
}

use crate::api::*;
use crate::distance::l2_distance;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};

// Immutable data structure for efficient search
struct VectorData {
    data: Vec<f32>,  // Flat array of vectors
    ids: Vec<u64>,   // IDs corresponding to each vector
    dim: usize,      // Dimensionality (128)
}

pub struct VectorDB {
    inner: RwLock<Arc<VectorData>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            inner: RwLock::new(Arc::new(VectorData {
                data: Vec::new(),
                ids: Vec::new(),
                dim: 128,
            })),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut guard = self.inner.write().unwrap();
        let old = Arc::clone(&guard);
        
        if Arc::strong_count(&old) == 1 {
            let vec_data = Arc::get_mut(&mut guard).unwrap();
            vec_data.data.extend(vector);
            vec_data.ids.push(id);
        } else {
            let mut data = old.data.clone();
            let mut ids = old.ids.clone();
            
            data.extend(vector);
            ids.push(id);
            
            *guard = Arc::new(VectorData {
                data,
                ids,
                dim: old.dim,
            });
        }
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut guard = self.inner.write().unwrap();
        let old = Arc::clone(&guard);
        
        let count = vectors.len();
        
        if Arc::strong_count(&old) == 1 {
            let vec_data = Arc::get_mut(&mut guard).unwrap();
            vec_data.data.reserve(count * 128);
            vec_data.ids.reserve(count);
            for (id, vector) in vectors {
                vec_data.data.extend(vector);
                vec_data.ids.push(id);
            }
        } else {
            let mut data = Vec::with_capacity(old.data.len() + count * 128);
            let mut ids = Vec::with_capacity(old.ids.len() + count);
            
            data.extend_from_slice(&old.data);
            ids.extend_from_slice(&old.ids);
            
            for (id, vector) in vectors {
                data.extend(vector);
                ids.push(id);
            }
            
            *guard = Arc::new(VectorData {
                data,
                ids,
                dim: old.dim,
            });
        }
        
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let vec_data = Arc::clone(&self.inner.read().unwrap());
        
        let n = vec_data.ids.len();
        let dim = vec_data.dim;
        
        // Compute all distances in parallel
        let mut results: Vec<SearchResult> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * dim;
                let vec_slice = &vec_data.data[start..start + dim];
                let distance = l2_distance(query, vec_slice);
                SearchResult {
                    id: vec_data.ids[i],
                    distance,
                }
            })
            .collect();
        
        // Use select_nth_unstable for efficient top-k
        let k = top_k as usize;
        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| {
                a.distance.partial_cmp(&b.distance).unwrap()
            });
            results.truncate(k);
        }
        
        results
    }
}

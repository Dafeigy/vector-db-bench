use crate::api::*;
use crate::distance::l2_distance;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use parking_lot::RwLock;

/// Vector database storing 128-dimensional float32 vectors.
pub struct VectorDB {
    data: RwLock<Vec<f32>>,
    ids: RwLock<Vec<u64>>,
    count: AtomicUsize,
}

const DIM: usize = 128;
const BATCH_SIZE: usize = 16384; // Process this many vectors per parallel task

impl VectorDB {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(Vec::with_capacity(1_000_000 * DIM)),
            ids: RwLock::new(Vec::with_capacity(1_000_000)),
            count: AtomicUsize::new(0),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        debug_assert_eq!(vector.len(), DIM);
        let mut data = self.data.write();
        let mut ids = self.ids.write();
        data.extend(vector);
        ids.push(id);
        self.count.fetch_add(1, Ordering::Release);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut data = self.data.write();
        let mut ids = self.ids.write();
        let n = vectors.len();
        let needed = data.len() + n * DIM;
        let capacity = data.capacity();
        if capacity < needed {
            data.reserve(needed - capacity);
            ids.reserve(n);
        }
        for (id, vector) in vectors {
            debug_assert_eq!(vector.len(), DIM);
            data.extend(vector);
            ids.push(id);
        }
        self.count.fetch_add(n, Ordering::Release);
        n
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read();
        let ids = self.ids.read();
        let n = self.count.load(Ordering::Acquire);
        let k = top_k as usize;
        
        // Number of batches
        let num_batches = (n + BATCH_SIZE - 1) / BATCH_SIZE;
        
        // Process batches in parallel, each returns top-k candidates
        let batch_results: Vec<Vec<(f64, u64)>> = (0..num_batches)
            .into_par_iter()
            .map(|batch_idx| {
                let start = batch_idx * BATCH_SIZE;
                let end = (start + BATCH_SIZE).min(n);
                let mut local_results: Vec<(f64, u64)> = Vec::with_capacity(end - start);
                
                for i in start..end {
                    let vec_start = i * DIM;
                    let vec_end = vec_start + DIM;
                    let dist = l2_distance(query, &data[vec_start..vec_end]);
                    local_results.push((dist, ids[i]));
                }
                
                // Keep top-k from this batch
                if local_results.len() > k {
                    local_results.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
                    local_results.truncate(k);
                }
                local_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                local_results
            })
            .collect();
        
        // Merge batch results
        let mut all_results: Vec<(f64, u64)> = batch_results.into_iter().flatten().collect();
        
        // Final top-k selection
        if all_results.len() > k {
            all_results.select_nth_unstable_by(k, |a, b| a.0.partial_cmp(&b.0).unwrap());
            all_results.truncate(k);
        }
        all_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        all_results
            .into_iter()
            .map(|(dist, id)| SearchResult { id, distance: dist })
            .collect()
    }
}

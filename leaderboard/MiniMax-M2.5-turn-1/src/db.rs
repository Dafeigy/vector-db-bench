use std::sync::{RwLock, Mutex};
use crate::api::*;
use crate::distance::l2_distance;

const DIM: usize = 128;
const NUM_THREADS: usize = 4;

pub struct VectorDB {
    // Flat storage for better cache locality
    ids: RwLock<Vec<u64>>,
    vectors: RwLock<Vec<f32>>,
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

    pub fn bulk_insert(&self, new_vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids = self.ids.write().unwrap();
        let mut vectors = self.vectors.write().unwrap();
        
        let mut new_ids = Vec::with_capacity(new_vectors.len());
        let mut new_data = Vec::with_capacity(new_vectors.len() * DIM);
        
        for (id, vec) in new_vectors {
            new_ids.push(id);
            new_data.extend(vec);
        }
        
        let inserted = new_ids.len();
        ids.append(&mut new_ids);
        vectors.extend(new_data);
        
        inserted
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read().unwrap();
        let vectors = self.vectors.read().unwrap();
        let count = ids.len();
        
        let k = top_k as usize;
        
        // Parallel distance computation using multiple threads
        let chunk_size = (count + NUM_THREADS - 1) / NUM_THREADS;
        let results: Mutex<Vec<(u64, f64)>> = Mutex::new(Vec::with_capacity(count));
        
        // Create query clone for each thread
        let query_vec: Vec<f32> = query.to_vec();
        
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..NUM_THREADS)
                .map(|t| {
                    let ids = &ids;
                    let vectors = &vectors;
                    let query = &query_vec;
                    let results = &results;
                    
                    s.spawn(move || {
                        let start = t * chunk_size;
                        let end = std::cmp::min(start + chunk_size, count);
                        
                        let mut local_results = Vec::with_capacity(end - start);
                        for i in start..end {
                            let id = ids[i];
                            let vec_start = i * DIM;
                            let vec = &vectors[vec_start..vec_start + DIM];
                            let dist = l2_distance(query, vec);
                            local_results.push((id, dist));
                        }
                        
                        let mut results = results.lock().unwrap();
                        results.extend(local_results);
                    })
                })
                .collect();
            
            // Wait for all threads to complete
            for handle in handles {
                handle.join().unwrap();
            }
        });
        
        let mut distances = results.into_inner().unwrap();
        
        // Use partial sort - only need top k smallest
        if k >= distances.len() {
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            // select_nth_unstable partitions around the k-th element
            let mid = distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            // Sort only the first k elements
            mid.0.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        
        // Take top k
        distances
            .into_iter()
            .take(k)
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}

use crate::api::*;
use std::sync::RwLock;

// Ensure 512-byte alignment for AVX-512
#[repr(align(512))]
struct AlignedVector {
    data: [f32; 128],
}

pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    data: RwLock<Vec<f32>>, // flat array
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            ids: RwLock::new(Vec::new()),
            data: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        self.ids.write().unwrap().push(id);
        self.data.write().unwrap().extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let len = vectors.len();
        let mut ids = self.ids.write().unwrap();
        let mut data = self.data.write().unwrap();
        
        ids.reserve(len);
        data.reserve(len * 128);
        
        for (id, vector) in vectors {
            ids.push(id);
            data.extend(vector);
        }
        len
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let top_k = top_k as usize;
        let ids = self.ids.read().unwrap();
        let data = self.data.read().unwrap();
        
        use rayon::prelude::*;
        
        let chunk_size = 128 * 4096;
        
        let mut results: Vec<SearchResult> = data
            .par_chunks_exact(chunk_size)
            .enumerate()
            .flat_map(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * 4096;
                let mut local_results = Vec::with_capacity(4096);
                let q_ptr = vector.as_ptr();
                for i in 0..4096 {
                    let v_ptr = unsafe { chunk.as_ptr().add(i * 128) };
                    // Just slice, l2_distance is already optimized
                    let v_slice = unsafe { std::slice::from_raw_parts(v_ptr, 128) };
                    local_results.push(SearchResult {
                        id: ids[start_idx + i],
                        distance: crate::distance::l2_distance(vector, v_slice),
                    });
                }
                
                // Return best top_k per chunk early
                if local_results.len() > top_k {
                    let (first, mid, _last) = local_results.select_nth_unstable_by(top_k - 1, |a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
                    let mut top_k_results: Vec<SearchResult> = Vec::with_capacity(top_k);
                    for res in first.iter() {
                        top_k_results.push(SearchResult { id: res.id, distance: res.distance });
                    }
                    top_k_results.push(SearchResult { id: mid.id, distance: mid.distance });
                    top_k_results
                } else {
                    local_results
                }
            })
            .collect();
            
        // Handle remainder
        let remainder = data.chunks_exact(chunk_size).remainder();
        let start_idx = (data.len() / chunk_size) * 4096;
        for (i, v) in remainder.chunks_exact(128).enumerate() {
            results.push(SearchResult {
                id: ids[start_idx + i],
                distance: crate::distance::l2_distance(vector, v),
            });
        }
        
        if top_k >= results.len() {
            results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
            return results;
        }

        let (first, mid, _last) = results.select_nth_unstable_by(top_k - 1, |a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        let mut top_k_results: Vec<SearchResult> = Vec::with_capacity(top_k);
        for res in first.iter() {
             top_k_results.push(SearchResult { id: res.id, distance: res.distance });
        }
        top_k_results.push(SearchResult { id: mid.id, distance: mid.distance });
            
        top_k_results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        top_k_results
    }
}
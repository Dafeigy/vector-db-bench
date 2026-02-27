use crate::api::SearchResult;
use crate::distance::sq_l2_distance;
use rayon::prelude::*;
use std::sync::RwLock;
use std::cmp::Ordering;

#[derive(Debug)]
struct Storage {
    ids: Vec<u64>,
    data: Vec<f32>,
}

pub struct VectorDB {
    storage: RwLock<Storage>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(Storage {
                ids: Vec::new(),
                data: Vec::new(),
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != 128 {
            return;
        }
        let mut storage = self.storage.write().unwrap();
        storage.ids.push(id);
        storage.data.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, mut vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut storage = self.storage.write().unwrap();
        let initial_len = storage.ids.len();
        let reserve_count = vectors.len();
        storage.ids.reserve(reserve_count);
        storage.data.reserve(reserve_count * 128);
        let mut count = 0;
        for (id, vector) in vectors.drain(..) {
            if vector.len() == 128 {
                storage.ids.push(id);
                storage.data.extend_from_slice(&vector);
                count += 1;
            }
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let storage = self.storage.read().unwrap();
        let n = storage.ids.len();
        let k = std::cmp::min(top_k as usize, n);
        if k == 0 || n == 0 {
            return vec![];
        }

        let num_threads = 4;
        let chunk_size = (n + num_threads - 1) / num_threads;
        let partial_topks: Vec<Vec<(f64, u64)>> = (0..num_threads).into_par_iter().map(|tid| {
            let start = tid * chunk_size;
            let end = std::cmp::min(start + chunk_size, n);
            let mut local_topk: Vec<(f64, u64)> = Vec::with_capacity(k);
            for i in start..end {
                let id = storage.ids[i];
                let offset = i * 128;
                let v_slice = unsafe {
                    std::slice::from_raw_parts(storage.data.as_ptr().add(offset), 128)
                };
                let sq_dist = sq_l2_distance(query, v_slice);
                let insert = if local_topk.len() < k {
                    true
                } else {
                    sq_dist < local_topk.last().unwrap().0
                };
                if insert {
                    if local_topk.len() == k {
                        local_topk.pop();
                    }
                    local_topk.push((sq_dist, id));
                    local_topk.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
                }
            }
            local_topk
        }).collect();

        let mut all_candidates: Vec<(f64, u64)> = partial_topks.into_iter().flatten().collect();
        all_candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        if all_candidates.len() > k {
            all_candidates.truncate(k);
        }

        all_candidates.into_iter()
            .map(|(sq_dist, id)| SearchResult {
                id,
                distance: sq_dist.sqrt(),
            })
            .collect()
    }
}

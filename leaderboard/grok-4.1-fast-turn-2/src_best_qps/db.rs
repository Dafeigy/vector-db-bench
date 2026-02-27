use crate::api::SearchResult;
use crate::distance::l2_distance;
use rayon::prelude::*;
use std::sync::RwLock;

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
        storage.data.extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut storage = self.storage.write().unwrap();
        let initial_len = storage.ids.len();
        storage.ids.reserve(vectors.len());
        storage.data.reserve(vectors.len() * 128);
        let mut count = 0;
        for (id, vector) in vectors {
            if vector.len() == 128 {
                storage.ids.push(id);
                storage.data.extend(vector);
                count += 1;
            }
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let storage = self.storage.read().unwrap();
        let n = storage.ids.len();
        let k = top_k as usize;
        if k == 0 {
            return vec![];
        }
        if n == 0 {
            return vec![];
        }

        let candidates: Vec<(f64, u64)> = storage
            .ids
            .par_iter()
            .enumerate()
            .map(|(i, &id)| {
                let offset = i * 128;
                let v_slice = unsafe {
                    std::slice::from_raw_parts(
                        storage.data.as_ptr().add(offset),
                        128,
                    )
                };
                let dist = l2_distance(query, v_slice);
                (dist, id)
            })
            .collect();

        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .map(|(dist, id)| SearchResult { id, distance: dist })
            .collect();
        results.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if results.len() > k {
            results.truncate(k);
        }
        results
    }
}

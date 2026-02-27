use crate::api::*;
use crate::distance::l2_distance;
use std::collections::HashMap;
use std::sync::RwLock;
use rayon::prelude::*;

pub struct VectorDB {
    vectors: RwLock<Vec<(u64, Vec<f32>)>>,
    id_to_index: RwLock<HashMap<u64, usize>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(Vec::new()),
            id_to_index: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vecs = self.vectors.write().unwrap();
        let mut map = self.id_to_index.write().unwrap();
        if let Some(&idx) = map.get(&id) {
            vecs[idx].1 = vector;
        } else {
            let idx = vecs.len();
            vecs.push((id, vector));
            map.insert(id, idx);
        }
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut vecs = self.vectors.write().unwrap();
        let mut map = self.id_to_index.write().unwrap();
        let mut inserted = 0;
        for (id, vector) in vectors {
            if map.contains_key(&id) {
                if let Some(&idx) = map.get(&id) {
                    vecs[idx].1 = vector;
                }
            } else {
                let idx = vecs.len();
                vecs.push((id, vector));
                map.insert(id, idx);
                inserted += 1;
            }
        }
        inserted
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let vecs = self.vectors.read().unwrap();
        let n = vecs.len();
        let k = top_k as usize;
        if n == 0 || k == 0 {
            return Vec::new();
        }

        // Parallel distance computation
        let mut distances: Vec<(u64, f64)> = vecs
            .par_iter()
            .map(|(id, v)| (*id, l2_distance(vector, v)))
            .collect();

        let k = k.min(n);
        // Use select_nth_unstable_by to partition around k-th smallest, then sort only top-k
        if k < n {
            distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(k);
        }
        distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        distances.into_iter().map(|(id, d)| SearchResult { id, distance: d }).collect()
    }
}

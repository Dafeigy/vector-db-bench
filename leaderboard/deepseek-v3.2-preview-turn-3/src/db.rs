use crate::api::*;
use rayon::prelude::*;
use std::sync::RwLock;

pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    data: RwLock<Vec<[f32; 128]>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            ids: RwLock::new(Vec::new()),
            data: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), 128);
        let mut ids = self.ids.write().unwrap();
        let mut data = self.data.write().unwrap();
        ids.push(id);
        let arr: [f32; 128] = vector.try_into().unwrap();
        data.push(arr);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids = self.ids.write().unwrap();
        let mut data = self.data.write().unwrap();
        let start = ids.len();
        for (id, vec) in vectors {
            assert_eq!(vec.len(), 128);
            let arr: [f32; 128] = vec.try_into().unwrap();
            ids.push(id);
            data.push(arr);
        }
        ids.len() - start
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        assert_eq!(vector.len(), 128);
        let ids = self.ids.read().unwrap();
        let data = self.data.read().unwrap();
        let n = ids.len();
        let k = top_k as usize;

        // Parallel compute distances
        let mut candidates: Vec<(u64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let dist = crate::distance::l2_distance(vector, &data[i]);
                (ids[i], dist)
            })
            .collect();

        // If k >= n, just sort all
        if k >= n {
            candidates.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        } else {
            // Select the k-th smallest distance
            let (left, _, _) = candidates.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            // left contains first k elements (unsorted)
            left.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            // truncate candidates to k
            candidates.truncate(k);
        }

        candidates
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}
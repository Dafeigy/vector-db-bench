use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

const DIM: usize = 128;

#[derive(Clone, Copy)]
struct SearchCandidate {
    id: u64,
    distance: f64,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    data: RwLock<Vec<f32>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            ids: RwLock::new(Vec::new()),
            data: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids = self.ids.write().unwrap();
        let mut data = self.data.write().unwrap();
        ids.push(id);
        data.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids = self.ids.write().unwrap();
        let mut data = self.data.write().unwrap();
        let count = vectors.len();
        
        ids.reserve(count);
        data.reserve(count * DIM);
        
        for (id, vec) in vectors {
            ids.push(id);
            data.extend_from_slice(&vec);
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read().unwrap();
        let data = self.data.read().unwrap();
        
        if ids.is_empty() {
            return Vec::new();
        }

        let top_k = top_k as usize;
        let num_vectors = ids.len();
        
        // Parallel distance computation
        let mut all_candidates: Vec<SearchCandidate> = (0..num_vectors)
            .into_par_iter()
            .map(|idx| {
                let start = idx * DIM;
                let vec = &data[start..start + DIM];
                SearchCandidate {
                    id: ids[idx],
                    distance: l2_distance(query, vec),
                }
            })
            .collect();

        // Use select_nth_unstable for top-k
        if all_candidates.len() > top_k {
            all_candidates.select_nth_unstable_by(top_k, |a, b| {
                a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
            });
        }
        
        // Take first top_k and sort them
        let mut results: Vec<SearchCandidate> = all_candidates.into_iter().take(top_k).collect();
        results.sort_by(|a, b| {
            a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal)
        });
        
        results
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect()
    }
}

use crate::api::SearchResult;
use std::cmp::Ordering;
use std::sync::RwLock;
use rayon::prelude::*;

struct Entry {
    id: u64,
    vector: Vec<f32>,
}

pub struct VectorDB {
    entries: RwLock<Vec<Entry>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), 128, "Vector must be 128-dimensional");
        let mut entries = self.entries.write().unwrap();
        entries.push(Entry { id, vector });
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut entries = self.entries.write().unwrap();
        let before = entries.len();
        for (id, vector) in vectors {
            assert_eq!(vector.len(), 128, "Vector must be 128-dimensional");
            entries.push(Entry { id, vector });
        }
        entries.len() - before
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        assert_eq!(vector.len(), 128, "Query vector must be 128-dimensional");
        let entries = self.entries.read().unwrap();
        let k = top_k as usize;
        if entries.is_empty() || k == 0 {
            return Vec::new();
        }

        // Parallel distance computation using rayon
        let mut distances: Vec<(f64, u64)> = entries
            .par_iter()
            .map(|entry| (crate::distance::l2_distance(vector, &entry.vector), entry.id))
            .collect();

        // Use partial sort (select_nth_unstable) to get top-k efficiently
        let k = k.min(distances.len());
        if k < distances.len() {
            distances.select_nth_unstable_by(k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        }
        // Sort the top-k slice
        let top = &mut distances[..k];
        top.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        top.iter()
            .map(|&(d, id)| SearchResult { id, distance: d })
            .collect()
    }
}

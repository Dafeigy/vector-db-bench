use crate::api::*;
use crate::distance::l2_distance;
use arc_swap::ArcSwap;
use rayon::prelude::*;
use std::sync::Arc;

pub struct VectorDB {
    vectors: ArcSwap<FlatStorage>,
}

struct FlatStorage {
    ids: Vec<u64>,
    data: Vec<f32>,
}

impl FlatStorage {
    fn new() -> Self {
        Self {
            ids: Vec::new(),
            data: Vec::new(),
        }
    }
    
    fn push(&mut self, id: u64, vector: Vec<f32>) {
        self.ids.push(id);
        self.data.extend(vector);
    }
    
    fn extend(&mut self, vectors: Vec<(u64, Vec<f32>)>) {
        self.ids.reserve(vectors.len());
        self.data.reserve(vectors.len() * 128);
        for (id, vec) in vectors {
            self.ids.push(id);
            self.data.extend(vec);
        }
    }
    
    #[inline]
    fn len(&self) -> usize {
        self.ids.len()
    }
    
    #[inline]
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * 128;
        &self.data[start..start + 128]
    }
    
    #[inline]
    fn get_id(&self, idx: usize) -> u64 {
        self.ids[idx]
    }
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            vectors: ArcSwap::new(Arc::new(FlatStorage::new())),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let storage = self.vectors.load();
        let mut new_storage = FlatStorage::new();
        new_storage.ids = storage.ids.clone();
        new_storage.data = storage.data.clone();
        new_storage.push(id, vector);
        self.vectors.store(Arc::new(new_storage));
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let storage = self.vectors.load();
        let mut new_storage = FlatStorage::new();
        new_storage.ids = storage.ids.clone();
        new_storage.data = storage.data.clone();
        let count = vectors.len();
        new_storage.extend(vectors);
        self.vectors.store(Arc::new(new_storage));
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let storage = self.vectors.load();
        let n = storage.len();
        let top_k = top_k as usize;
        
        // Parallel computation of distances
        let mut results: Vec<(f64, usize)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = storage.get_vector(i);
                (l2_distance(query, vec), i)
            })
            .collect();
        
        // Partial sort to find top-k smallest
        if results.len() > top_k {
            results.select_nth_unstable_by(top_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            results.truncate(top_k);
        }
        
        results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        
        results
            .into_iter()
            .map(|(dist, idx)| SearchResult {
                id: storage.get_id(idx),
                distance: dist,
            })
            .collect()
    }
}

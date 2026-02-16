use crate::api::*;
use crate::distance::l2_distance_squared;
use std::sync::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

const DIMENSION: usize = 128;
const NUM_BUCKETS: usize = 64;  // Fewer buckets
const BUCKET_MASK: usize = NUM_BUCKETS - 1;  // For modulo operation

/// Result item with ordering by distance (for max-heap of largest to remove)
#[derive(Clone, Copy)]
struct HeapItem {
    id: u64,
    distance: f64,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for HeapItem {}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Max-heap: larger distances stay at top
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

/// Bucket index entry
struct Bucket {
    ids: Vec<u64>,
    vectors: Vec<f32>,  // Flat storage
}

pub struct VectorDB {
    // Single random projection for simple hashing
    projection: [f32; DIMENSION],
    
    // Buckets
    buckets: RwLock<Vec<Bucket>>,
    
    // Linear storage for brute force fallback
    all_ids: RwLock<Vec<u64>>,
    all_vectors: RwLock<Vec<f32>>,
    
    // Whether index is initialized
    initialized: RwLock<bool>,
}

impl VectorDB {
    pub fn new() -> Self {
        // Generate random projection
        let mut projection = [0.0f32; DIMENSION];
        for d in 0..DIMENSION {
            projection[d] = fastrand::f32() * 2.0 - 1.0;
        }
        
        // Initialize empty buckets
        let buckets: Vec<Bucket> = (0..NUM_BUCKETS)
            .map(|_| Bucket { ids: Vec::new(), vectors: Vec::new() })
            .collect();
        
        Self {
            projection,
            buckets: RwLock::new(buckets),
            all_ids: RwLock::new(Vec::with_capacity(1_100_000)),
            all_vectors: RwLock::new(Vec::with_capacity(1_100_000 * DIMENSION)),
            initialized: RwLock::new(false),
        }
    }

    fn compute_bucket(&self, vector: &[f32]) -> usize {
        // Simple dot product hash
        let mut dot = 0.0f32;
        for d in 0..DIMENSION {
            dot += vector[d] * self.projection[d];
        }
        // Map to bucket using bits of the dot product
        dot.to_bits() as usize & BUCKET_MASK
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids_guard = self.all_ids.write().unwrap();
        let mut vecs_guard = self.all_vectors.write().unwrap();
        ids_guard.push(id);
        vecs_guard.extend_from_slice(&vector[..DIMENSION]);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids_guard = self.all_ids.write().unwrap();
        let mut vecs_guard = self.all_vectors.write().unwrap();
        let count = vectors.len();
        ids_guard.reserve(count);
        vecs_guard.reserve(count * DIMENSION);
        for (id, vector) in vectors {
            ids_guard.push(id);
            vecs_guard.extend_from_slice(&vector[..DIMENSION]);
        }
        drop(ids_guard);
        drop(vecs_guard);
        
        self.build_index();
        
        count
    }

    fn build_index(&self) {
        let ids = self.all_ids.read().unwrap();
        let vectors = self.all_vectors.read().unwrap();
        let num_vectors = ids.len();
        
        if num_vectors == 0 {
            return;
        }

        let mut buckets = self.buckets.write().unwrap();
        
        // Reserve space in buckets based on uniform distribution estimate
        let avg_bucket_size = (num_vectors + NUM_BUCKETS - 1) / NUM_BUCKETS;
        for bucket in buckets.iter_mut() {
            bucket.ids.reserve(avg_bucket_size);
            bucket.vectors.reserve(avg_bucket_size * DIMENSION);
        }
        
        // Assign vectors to buckets
        for i in 0..num_vectors {
            let start = i * DIMENSION;
            let vec_slice = &vectors[start..start + DIMENSION];
            let bucket_idx = self.compute_bucket(vec_slice);
            
            buckets[bucket_idx].ids.push(ids[i]);
            buckets[bucket_idx].vectors.extend_from_slice(vec_slice);
        }
        
        *self.initialized.write().unwrap() = true;
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let initialized = *self.initialized.read().unwrap();
        
        if !initialized {
            return self.brute_force_search(query, top_k);
        }
        
        let query_slice = &query[..DIMENSION];
        let query_bucket = self.compute_bucket(query_slice);
        
        let buckets = self.buckets.read().unwrap();
        let top_k = top_k as usize;
        
        // Search the query's bucket and all buckets (for now to ensure recall)
        // We'll optimize this later based on recall
        let mut candidates: Vec<(u64, f64)> = Vec::new();
        
        // Search all buckets (to start with high recall)
        for bucket in buckets.iter() {
            self.search_bucket(bucket, query_slice, &mut candidates);
        }
        
        // Find top-k from candidates
        let mut heap = BinaryHeap::with_capacity(top_k + 1);
        for (id, dist) in candidates {
            heap.push(HeapItem { id, distance: dist });
            if heap.len() > top_k {
                heap.pop();
            }
        }
        
        // Convert to results
        let mut results: Vec<SearchResult> = heap
            .into_vec()
            .into_iter()
            .map(|item| SearchResult {
                id: item.id,
                distance: item.distance.sqrt(),
            })
            .collect();
        
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }

    fn search_bucket(&self, bucket: &Bucket, query: &[f32], candidates: &mut Vec<(u64, f64)>) {
        let num_vecs = bucket.ids.len();
        for i in 0..num_vecs {
            let start = i * DIMENSION;
            let vec_slice = &bucket.vectors[start..start + DIMENSION];
            let dist = l2_distance_squared(query, vec_slice);
            candidates.push((bucket.ids[i], dist));
        }
    }

    fn brute_force_search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.all_ids.read().unwrap();
        let vectors = self.all_vectors.read().unwrap();
        
        if ids.is_empty() {
            return Vec::new();
        }
        
        let top_k = top_k as usize;
        let num_vectors = ids.len();
        let query_slice = &query[..DIMENSION];
        
        let distances: Vec<(u64, f64)> = (0..num_vectors)
            .into_par_iter()
            .map(|i| {
                let vec_start = i * DIMENSION;
                let vec_slice = &vectors[vec_start..vec_start + DIMENSION];
                let dist = l2_distance_squared(query_slice, vec_slice);
                (ids[i], dist)
            })
            .collect();
        
        let mut heap = BinaryHeap::with_capacity(top_k + 1);
        
        for (id, dist) in distances {
            heap.push(HeapItem { id, distance: dist });
            if heap.len() > top_k {
                heap.pop();
            }
        }
        
        let mut results: Vec<SearchResult> = heap
            .into_vec()
            .into_iter()
            .map(|item| SearchResult {
                id: item.id,
                distance: item.distance.sqrt(),
            })
            .collect();
        
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }
}

use crate::api::*;
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;

const DIM: usize = 128;

pub struct VectorDB {
    // flattened vectors, length = len * DIM
    vectors: RwLock<Vec<f32>>,
    ids: RwLock<Vec<u64>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), DIM, "Vector dimension mismatch");
        let mut vectors = self.vectors.write();
        let mut ids = self.ids.write();
        vectors.extend_from_slice(&vector);
        ids.push(id);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut vecs = self.vectors.write();
        let mut ids = self.ids.write();
        // pre-allocate capacity
        vecs.reserve(count * DIM);
        ids.reserve(count);
        for (id, vector) in vectors {
            assert_eq!(vector.len(), DIM, "Vector dimension mismatch");
            vecs.extend_from_slice(&vector);
            ids.push(id);
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        assert_eq!(query.len(), DIM, "Query dimension mismatch");
        let k = top_k as usize;
        let vectors = self.vectors.read();
        let ids = self.ids.read();
        let n = ids.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        // If k >= n, we can just return all vectors sorted.
        // We'll parallelize across chunks.
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n + num_threads - 1) / num_threads;
        // Process each chunk in parallel, produce local top-k (using squared distance).
        let local_results: Vec<Vec<(f64, u64)>> = (0..num_threads)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                if start >= end {
                    return Vec::new();
                }
                let local_ids = &ids[start..end];
                let local_vectors = &vectors[start * DIM..end * DIM];
                let mut heap = BinaryHeap::with_capacity(k + 1);
                for (i, &id) in local_ids.iter().enumerate() {
                    let vec_start = i * DIM;
                    let vec_end = vec_start + DIM;
                    let vec = &local_vectors[vec_start..vec_end];
                    let dist_sq = crate::distance::l2_squared_distance(query, vec);
                    heap.push((OrderedFloat(dist_sq), id));
                    if heap.len() > k {
                        heap.pop(); // removes largest squared distance
                    }
                }
                heap.into_vec()
                    .into_iter()
                    .map(|(OrderedFloat(dist_sq), id)| (dist_sq, id))
                    .collect()
            })
            .collect();
        // Merge local results
        let mut global_heap = BinaryHeap::with_capacity(k + 1);
        for local in local_results {
            for (dist_sq, id) in local {
                global_heap.push((OrderedFloat(dist_sq), id));
                if global_heap.len() > k {
                    global_heap.pop();
                }
            }
        }
        // Convert to sorted vector with actual L2 distance (sqrt)
        let mut results: Vec<SearchResult> = global_heap
            .into_vec()
            .into_iter()
            .map(|(OrderedFloat(dist_sq), id)| SearchResult {
                id,
                distance: dist_sq.sqrt(),
            })
            .collect();
        // Sort by distance ascending
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }
}
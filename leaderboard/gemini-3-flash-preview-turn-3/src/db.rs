use crate::api::*;
use crate::distance::l2_distance_squared;
use parking_lot::RwLock;
use std::sync::Arc;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    vectors: RwLock<Vec<f32>>,
}

#[derive(Debug, Clone, Copy)]
struct DistanceId {
    distance_sq: f32,
    id: u64,
}

impl PartialEq for DistanceId {
    fn eq(&self, other: &Self) -> bool {
        self.distance_sq == other.distance_sq
    }
}

impl Eq for DistanceId {}

impl PartialOrd for DistanceId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance_sq.partial_cmp(&other.distance_sq)
    }
}

impl Ord for DistanceId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            ids: RwLock::new(Vec::with_capacity(1_000_000)),
            vectors: RwLock::new(Vec::with_capacity(1_000_000 * 128)),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids = self.ids.write();
        let mut vectors = self.vectors.write();
        ids.push(id);
        vectors.extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut ids = self.ids.write();
        let mut vectors_data = self.vectors.write();
        for (id, v) in vectors {
            ids.push(id);
            vectors_data.extend(v);
        }
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read();
        let vectors = self.vectors.read();
        let n = ids.len();

        let num_threads = 4;
        let chunk_size = (n + num_threads - 1) / num_threads;

        let results: Vec<Vec<DistanceId>> = (0..num_threads).into_par_iter().map(|t| {
            let start_idx = t * chunk_size;
            let end_idx = std::cmp::min(start_idx + chunk_size, n);
            let mut heap = BinaryHeap::with_capacity(top_k as usize);
            
            let my_ids = &ids[start_idx..end_idx];
            let my_vectors = &vectors[start_idx * 128 .. end_idx * 128];
            
            let mut i = 0;
            while i < my_ids.len() {
                let v = unsafe { my_vectors.get_unchecked(i * 128 .. (i + 1) * 128) };
                let dist_sq = l2_distance_squared(query, v);
                
                if heap.len() < top_k as usize {
                    heap.push(DistanceId { distance_sq: dist_sq, id: my_ids[i] });
                } else if dist_sq < heap.peek().unwrap().distance_sq {
                    let mut root = heap.peek_mut().unwrap();
                    root.distance_sq = dist_sq;
                    root.id = my_ids[i];
                }
                i += 1;
            }
            heap.into_sorted_vec()
        }).collect();

        let mut final_heap = BinaryHeap::with_capacity(top_k as usize);
        for local_res in results {
            for d in local_res {
                if final_heap.len() < top_k as usize {
                    final_heap.push(d);
                } else if d.distance_sq < final_heap.peek().unwrap().distance_sq {
                    let mut root = final_heap.peek_mut().unwrap();
                    root.distance_sq = d.distance_sq;
                    root.id = d.id;
                }
            }
        }

        let mut final_results: Vec<DistanceId> = final_heap.into_sorted_vec();
        final_results.into_iter().map(|d| SearchResult {
            id: d.id,
            distance: (d.distance_sq as f64).sqrt(),
        }).collect()
    }
}

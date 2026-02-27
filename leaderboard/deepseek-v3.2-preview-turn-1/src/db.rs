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
        // We'll parallelize across chunks using rayon's fold.
        let chunk_size = (n + rayon::current_num_threads() - 1) / rayon::current_num_threads();
        // Each thread will maintain a local top-k list as a sorted Vec of (distance_sq, id).
        // We'll use a small Vec and keep it sorted by distance_sq ascending.
        let local_tops: Vec<Vec<(f32, u64)>> = vectors
            .par_chunks(chunk_size * DIM)
            .zip(&ids.par_chunks(chunk_size))
            .fold(
                || Vec::with_capacity(k),
                |mut local, (vec_chunk, id_chunk)| {
                    for (i, &id) in id_chunk.iter().enumerate() {
                        let start = i * DIM;
                        let end = start + DIM;
                        let vec = &vec_chunk[start..end];
                        let dist_sq = crate::distance::l2_squared_distance_f32(query, vec);
                        // Insert into local sorted list (keep size <= k)
                        if local.len() < k {
                            local.push((dist_sq, id));
                            // keep sorted ascending
                            local.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        } else {
                            // If dist_sq smaller than the worst (last) element, replace and resort
                            if dist_sq < local.last().unwrap().0 {
                                local.pop();
                                local.push((dist_sq, id));
                                local.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            }
                        }
                    }
                    local
                },
            )
            .collect();
        // Merge local tops
        let mut global: Vec<(f32, u64)> = Vec::with_capacity(k);
        for local in local_tops {
            for (dist_sq, id) in local {
                if global.len() < k {
                    global.push((dist_sq, id));
                    global.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else if dist_sq < global.last().unwrap().0 {
                    global.pop();
                    global.push((dist_sq, id));
                    global.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                }
            }
        }
        // Convert to SearchResult with sqrt
        global
            .into_iter()
            .map(|(dist_sq, id)| SearchResult {
                id,
                distance: dist_sq.sqrt() as f64,
            })
            .collect()
    }
}
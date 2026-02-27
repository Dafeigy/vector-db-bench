use crate::api::*;
use crate::distance::squared_distance;
use rayon::prelude::*;
use std::sync::RwLock;

const DIM: usize = 128;

struct InnerDB {
    ids: Vec<u64>,
    data: Vec<f32>, // flattened vectors, each consecutive DIM floats
}

pub struct VectorDB {
    inner: RwLock<InnerDB>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(InnerDB {
                ids: Vec::new(),
                data: Vec::new(),
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), DIM);
        let mut inner = self.inner.write().unwrap();
        inner.ids.push(id);
        inner.data.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut inner = self.inner.write().unwrap();
        let start_len = inner.ids.len();
        for (id, vec) in vectors {
            assert_eq!(vec.len(), DIM);
            inner.ids.push(id);
            inner.data.extend_from_slice(&vec);
        }
        inner.ids.len() - start_len
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        assert_eq!(vector.len(), DIM);
        let inner = self.inner.read().unwrap();
        let k = top_k as usize;
        let num_vectors = inner.ids.len();
        let data = &inner.data;
        let ids = &inner.ids;

        // Use a simple array for top-k, sorted ascending (largest distance at end)
        let mut top_ids = vec![0u64; k];
        let mut top_dists = vec![f64::INFINITY; k];
        let mut max_idx = 0; // index of current maximum distance in the top-k array

        // Process vectors in parallel using rayon
        // We'll split the data into chunks and process each chunk independently,
        // then merge results.
        let chunk_size = (num_vectors + rayon::current_num_threads() - 1) / rayon::current_num_threads();
        let chunk_size = chunk_size.max(1);
        let local_tops: Vec<_> = data
            .par_chunks_exact(DIM * chunk_size)
            .zip(ids.par_chunks(chunk_size))
            .map(|(chunk_data, chunk_ids)| {
                let mut local_top_ids = vec![0u64; k];
                let mut local_top_dists = vec![f64::INFINITY; k];
                let mut local_max_idx = 0;
                for (i, (db_vec, &id)) in chunk_data.chunks_exact(DIM).zip(chunk_ids.iter()).enumerate() {
                    let dist_sq = squared_distance(vector, db_vec);
                    if dist_sq < local_top_dists[local_max_idx] {
                        // Insert into top-k array
                        local_top_dists[local_max_idx] = dist_sq;
                        local_top_ids[local_max_idx] = id;
                        // Find new maximum
                        local_max_idx = local_top_dists
                            .iter()
                            .enumerate()
                            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                            .unwrap()
                            .0;
                    }
                }
                (local_top_ids, local_top_dists)
            })
            .collect();

        // Merge local tops
        for (local_ids, local_dists) in local_tops {
            for (&id, &dist_sq) in local_ids.iter().zip(local_dists.iter()) {
                if dist_sq < top_dists[max_idx] {
                    top_dists[max_idx] = dist_sq;
                    top_ids[max_idx] = id;
                    max_idx = top_dists
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap()
                        .0;
                }
            }
        }

        // Sort results by distance ascending
        let mut results: Vec<_> = top_ids
            .into_iter()
            .zip(top_dists.into_iter())
            .map(|(id, dist_sq)| SearchResult {
                id,
                distance: dist_sq.sqrt(),
            })
            .collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }
}
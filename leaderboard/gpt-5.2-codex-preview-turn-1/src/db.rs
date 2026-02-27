use crate::api::*;
use crate::distance::l2_distance_f32;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::Once;

const DIM: usize = 128;
const BLOCK_VECTORS: usize = 1024;
static INIT_RAYON: Once = Once::new();

struct VectorData {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

pub struct VectorDB {
    data: RwLock<VectorData>,
}

struct TopK {
    ids: Vec<u64>,
    dists: Vec<f32>,
    max_index: usize,
    max_value: f32,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            ids: vec![0u64; k],
            dists: vec![f32::INFINITY; k],
            max_index: 0,
            max_value: f32::INFINITY,
        }
    }

    #[inline]
    fn consider(&mut self, id: u64, dist: f32) {
        if dist < self.max_value {
            self.ids[self.max_index] = id;
            self.dists[self.max_index] = dist;
            let mut new_max_index = 0usize;
            let mut new_max_value = self.dists[0];
            for i in 1..self.dists.len() {
                if self.dists[i] > new_max_value {
                    new_max_value = self.dists[i];
                    new_max_index = i;
                }
            }
            self.max_index = new_max_index;
            self.max_value = new_max_value;
        }
    }

    fn merge(&mut self, other: TopK) {
        for (id, dist) in other.ids.into_iter().zip(other.dists.into_iter()) {
            if dist.is_finite() {
                self.consider(id, dist);
            }
        }
    }

    fn into_results(self) -> Vec<SearchResult> {
        let mut results: Vec<SearchResult> = self
            .ids
            .into_iter()
            .zip(self.dists.into_iter())
            .filter(|(_, dist)| dist.is_finite())
            .map(|(id, dist)| SearchResult {
                id,
                distance: dist as f64,
            })
            .collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results
    }
}

impl VectorDB {
    pub fn new() -> Self {
        INIT_RAYON.call_once(|| {
            let _ = rayon::ThreadPoolBuilder::new().num_threads(4).build_global();
        });
        Self {
            data: RwLock::new(VectorData {
                ids: Vec::new(),
                vectors: Vec::new(),
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut guard = self.data.write();
        guard.ids.push(id);
        guard.vectors.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut guard = self.data.write();
        guard.ids.reserve(count);
        guard.vectors.reserve(count * DIM);
        for (id, vector) in vectors {
            guard.ids.push(id);
            guard.vectors.extend_from_slice(&vector);
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let k = top_k as usize;
        if k == 0 {
            return Vec::new();
        }
        let guard = self.data.read();
        if guard.ids.is_empty() {
            return Vec::new();
        }
        let ids = &guard.ids;
        let vectors = &guard.vectors;
        let block_floats = DIM * BLOCK_VECTORS;
        let total_vectors = ids.len();
        let block_count = vectors.len() / block_floats;

        let topk = vectors
            .par_chunks_exact(block_floats)
            .enumerate()
            .fold(|| TopK::new(k), |mut local, (block_idx, block)| {
                let base = block_idx * BLOCK_VECTORS;
                for i in 0..BLOCK_VECTORS {
                    let idx = base + i;
                    let start = i * DIM;
                    let dist = l2_distance_f32(vector, &block[start..start + DIM]);
                    local.consider(ids[idx], dist);
                }
                local
            })
            .reduce(|| TopK::new(k), |mut acc, other| {
                acc.merge(other);
                acc
            });

        let mut final_topk = topk;
        let remainder_start = block_count * BLOCK_VECTORS;
        let mut idx = remainder_start;
        let mut offset = block_count * block_floats;
        while idx < total_vectors {
            let dist = l2_distance_f32(vector, &vectors[offset..offset + DIM]);
            final_topk.consider(ids[idx], dist);
            idx += 1;
            offset += DIM;
        }

        final_topk.into_results()
    }
}

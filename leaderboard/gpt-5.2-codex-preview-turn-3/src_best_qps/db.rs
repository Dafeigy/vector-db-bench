use crate::api::*;
use crate::distance::{l2_distance_with, select_distance_kind, DistanceKind};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::sync::RwLock;

struct Storage {
    ids: Vec<u64>,
    vectors: Vec<f32>,
    dim: usize,
}

pub struct VectorDB {
    data: RwLock<Storage>,
    distance_kind: DistanceKind,
}

impl VectorDB {
    pub fn new() -> Self {
        let _ = rayon::ThreadPoolBuilder::new().num_threads(4).build_global();
        Self {
            data: RwLock::new(Storage {
                ids: Vec::new(),
                vectors: Vec::new(),
                dim: 128,
            }),
            distance_kind: select_distance_kind(),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut data = self.data.write().unwrap();
        data.ids.push(id);
        data.vectors.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut data = self.data.write().unwrap();
        let count = vectors.len();
        let dim = data.dim;
        data.ids.reserve(count);
        data.vectors.reserve(count * dim);
        for (id, vector) in vectors {
            data.ids.push(id);
            data.vectors.extend_from_slice(&vector);
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read().unwrap();
        let k = top_k as usize;
        if k == 0 || data.ids.is_empty() {
            return Vec::new();
        }

        let dim = data.dim;
        let ids = &data.ids;
        let vectors = &data.vectors;
        let kind = self.distance_kind;

        let topk = (0..ids.len())
            .into_par_iter()
            .fold(
                || TopK::new(k),
                |mut topk, i| {
                    let offset = i * dim;
                    let distance = l2_distance_with(kind, vector, &vectors[offset..offset + dim]);
                    topk.insert(ids[i], distance);
                    topk
                },
            )
            .reduce(|| TopK::new(k), |mut left, right| {
                left.merge(&right);
                left
            });

        let mut results: Vec<SearchResult> = topk
            .ids
            .into_iter()
            .zip(topk.dists.into_iter())
            .map(|(id, distance)| SearchResult {
                id,
                distance: distance as f64,
            })
            .collect();
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results
    }
}

#[derive(Clone)]
struct TopK {
    k: usize,
    ids: Vec<u64>,
    dists: Vec<f32>,
    max_dist: f32,
    max_idx: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            ids: Vec::with_capacity(k),
            dists: Vec::with_capacity(k),
            max_dist: f32::NEG_INFINITY,
            max_idx: 0,
        }
    }

    #[inline(always)]
    fn insert(&mut self, id: u64, dist: f32) {
        if self.k == 0 {
            return;
        }
        if self.ids.len() < self.k {
            self.ids.push(id);
            self.dists.push(dist);
            if dist > self.max_dist {
                self.max_dist = dist;
                self.max_idx = self.ids.len() - 1;
            }
            if self.ids.len() == self.k {
                self.recompute_max();
            }
            return;
        }
        if dist < self.max_dist {
            self.ids[self.max_idx] = id;
            self.dists[self.max_idx] = dist;
            self.recompute_max();
        }
    }

    #[inline(always)]
    fn recompute_max(&mut self) {
        if self.dists.is_empty() {
            self.max_dist = f32::NEG_INFINITY;
            self.max_idx = 0;
            return;
        }
        let mut max_dist = self.dists[0];
        let mut max_idx = 0;
        for (i, &dist) in self.dists.iter().enumerate().skip(1) {
            if dist > max_dist {
                max_dist = dist;
                max_idx = i;
            }
        }
        self.max_dist = max_dist;
        self.max_idx = max_idx;
    }

    fn merge(&mut self, other: &TopK) {
        for i in 0..other.ids.len() {
            self.insert(other.ids[i], other.dists[i]);
        }
    }
}

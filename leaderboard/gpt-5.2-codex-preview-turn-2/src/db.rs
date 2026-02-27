use crate::api::*;
use crate::distance::{l2_distance_ptr, select_distance_impl, DistanceImpl};
use parking_lot::RwLock;
use std::cmp::Ordering;

const DIM: usize = 128;

pub struct VectorDB {
    inner: RwLock<DBInner>,
    dist_impl: DistanceImpl,
}

struct DBInner {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(DBInner {
                ids: Vec::new(),
                vectors: Vec::new(),
            }),
            dist_impl: select_distance_impl(),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM {
            return;
        }
        let mut inner = self.inner.write();
        inner.ids.push(id);
        inner.vectors.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut inner = self.inner.write();
        let mut inserted = 0;
        inner.ids.reserve(vectors.len());
        inner.vectors.reserve(vectors.len() * DIM);
        for (id, vector) in vectors {
            if vector.len() != DIM {
                continue;
            }
            inner.ids.push(id);
            inner.vectors.extend_from_slice(&vector);
            inserted += 1;
        }
        inserted
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let k = top_k as usize;
        if k == 0 || vector.len() != DIM {
            return Vec::new();
        }
        let inner = self.inner.read();
        let total = inner.ids.len();
        if total == 0 {
            return Vec::new();
        }

        let ids = &inner.ids;
        let data = &inner.vectors;
        let query_ptr = vector.as_ptr();
        let data_ptr = data.as_ptr();
        let mut top: Vec<HeapItem> = Vec::with_capacity(k);
        let mut max_dist = f32::NEG_INFINITY;
        let mut max_idx = 0usize;

        for idx in 0..total {
            let base_ptr = unsafe { data_ptr.add(idx * DIM) };
            let dist = unsafe { l2_distance_ptr(query_ptr, base_ptr, self.dist_impl) };
            if top.len() < k {
                if dist > max_dist {
                    max_dist = dist;
                    max_idx = top.len();
                }
                top.push(HeapItem { dist, id: ids[idx] });
            } else if dist < max_dist {
                top[max_idx] = HeapItem { dist, id: ids[idx] };
                let mut new_max_dist = f32::NEG_INFINITY;
                let mut new_max_idx = 0usize;
                for (i, item) in top.iter().enumerate() {
                    if item.dist > new_max_dist {
                        new_max_dist = item.dist;
                        new_max_idx = i;
                    }
                }
                max_dist = new_max_dist;
                max_idx = new_max_idx;
            }
        }

        top.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        top.into_iter()
            .map(|item| SearchResult {
                id: item.id,
                distance: item.dist as f64,
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy)]
struct HeapItem {
    dist: f32,
    id: u64,
}

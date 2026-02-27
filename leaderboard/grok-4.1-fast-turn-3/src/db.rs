use std::sync::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use crate::api::SearchResult;

#[derive(Copy, Clone)]
struct DistId {
    dist: f64,
    id: u64,
}

impl Ord for DistId {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.total_cmp(&other.dist).then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for DistId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct Storage {
    ids: Vec<u64>,
    vectors: Vec<f32>,
    dim: usize,
}

pub struct VectorDB {
    storage: RwLock<Storage>,
}

impl VectorDB {
    pub fn new() -> Self {
        // Set rayon to 4 threads
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global();
        Self {
            storage: RwLock::new(Storage {
                ids: Vec::new(),
                vectors: Vec::new(),
                dim: 128,
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut s = self.storage.write().unwrap();
        debug_assert_eq!(vector.len(), s.dim);
        s.ids.push(id);
        s.vectors.extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut s = self.storage.write().unwrap();
        let n = vectors.len();
        let mut total_elements = 0usize;
        for (_, ref v) in &vectors {
            debug_assert_eq!(v.len(), s.dim);
            total_elements += v.len();
        }
        s.vectors.reserve(total_elements);
        s.ids.reserve(n);
        for (id, v) in vectors {
            s.ids.push(id);
            s.vectors.extend(v);
        }
        n
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        debug_assert_eq!(query.len(), 128);
        let s = self.storage.read().unwrap();
        let num_vecs = s.ids.len();
        let k = std::cmp::min(top_k as usize, num_vecs);
        if k == 0 {
            return vec![];
        }
        let dim = s.dim;
        let n_threads = 4usize;
        let chunk_size = (num_vecs + n_threads - 1) / n_threads;

        let partial_topks: Vec<Vec<(f64, u64)>> = (0..n_threads)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                let end = std::cmp::min(start + chunk_size, num_vecs);
                let mut local_heap: BinaryHeap<DistId> = BinaryHeap::with_capacity(k);
                for ii in start..end {
                    let offset = ii * dim;
                    let db_vec = &s.vectors[offset..offset + dim];
                    let dist = crate::distance::l2_distance(query, db_vec);
                    let idd = s.ids[ii];
                    let new_item = DistId { dist, id: idd };
                    if local_heap.len() < k {
                        local_heap.push(new_item);
                    } else if let Some(top) = local_heap.peek() {
                        if dist < top.dist {
                            local_heap.pop();
                            local_heap.push(new_item);
                        }
                    }
                }
                let mut res: Vec<(f64, u64)> = local_heap.into_iter()
                    .map(|di| (di.dist, di.id))
                    .collect();
                res.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                res
            })
            .collect();

        let mut all_candidates: Vec<(f64, u64)> = partial_topks.into_iter().flatten().collect();
        all_candidates.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        all_candidates.truncate(k);
        all_candidates
            .into_iter()
            .map(|(dist, id)| SearchResult { id, distance: dist })
            .collect()
    }
}
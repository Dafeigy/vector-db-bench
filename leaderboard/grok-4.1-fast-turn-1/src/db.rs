use crate::api::SearchResult;
use crate::distance::l2_dist_sq;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::RwLock;
use rayon::prelude::*;

#[derive(Copy, Clone, Eq, PartialEq)]
struct Candidate {
    dist_sq: f64,
    id: u64,
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist_sq.partial_cmp(&other.dist_sq)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq.total_cmp(&other.dist_sq)
    }
}

#[derive(Clone)]
struct Data {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

pub const DIM: usize = 128;
const NUM_THREADS: usize = 4;
const LOCAL_K_FACTOR: usize = 32;

pub struct VectorDB {
    data: RwLock<Data>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(Data { ids: Vec::new(), vectors: Vec::new() }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut data = self.data.write().unwrap();
        data.ids.push(id);
        data.vectors.extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let n = vectors.len();
        let mut data = self.data.write().unwrap();
        data.ids.reserve(n);
        data.vectors.reserve(n * DIM);
        for (id, vector) in vectors {
            data.ids.push(id);
            data.vectors.extend(vector);
        }
        n
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read().unwrap();
        let n = data.ids.len();
        let k = top_k as usize;
        if n == 0 {
            return Vec::new();
        }

        let chunk_size = (n + NUM_THREADS - 1) / NUM_THREADS;
        let local_k = (k * LOCAL_K_FACTOR).min(1024).min(n / NUM_THREADS + 1);

        let partial_tops: Vec<Vec<Candidate>> = (0..NUM_THREADS)
            .into_par_iter()
            .map(|tid| {
                let start = tid * chunk_size;
                let end = std::cmp::min(start + chunk_size, n);
                let mut local_heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(local_k);

                for i in start..end {
                    let base = i * DIM;
                    let dist_sq = l2_dist_sq(query, &data.vectors[base..base + DIM]);
                    let cand = Candidate {
                        dist_sq,
                        id: data.ids[i],
                    };

                    if local_heap.len() < local_k {
                        local_heap.push(cand);
                    } else if let Some(top) = local_heap.peek() {
                        if cand.dist_sq < top.dist_sq {
                            let _ = local_heap.pop();
                            local_heap.push(cand);
                        }
                    }
                }

                let mut locals = Vec::with_capacity(local_heap.len());
                while let Some(cand) = local_heap.pop() {
                    locals.push(cand);
                }
                locals
            })
            .collect();

        let mut all_cands: Vec<Candidate> = partial_tops.into_iter().flatten().collect();
        all_cands.sort_unstable_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap_or(Ordering::Equal));

        all_cands
            .into_iter()
            .take(k)
            .map(|cand| SearchResult {
                id: cand.id,
                distance: cand.dist_sq.sqrt(),
            })
            .collect()
    }
}
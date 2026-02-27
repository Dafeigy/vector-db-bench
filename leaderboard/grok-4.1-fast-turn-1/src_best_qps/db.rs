use crate::api::SearchResult;
use crate::distance::l2_dist_sq;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::RwLock;

#[derive(Copy, Clone)]
struct Candidate {
    dist_sq: f64,
    id: u64,
}

impl Eq for Candidate {}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq.to_bits() == other.dist_sq.to_bits() && self.id == other.id
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq.total_cmp(&other.dist_sq)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.dist_sq.total_cmp(&other.dist_sq))
    }
}

#[derive(Clone)]
struct Data {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

pub const DIM: usize = 128;

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

        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();

        for i in 0..n {
            let base = i * DIM;
            let dist_sq = l2_dist_sq(query, &data.vectors[base..base + DIM]);
            let cand = Candidate { dist_sq, id: data.ids[i] };

            if heap.len() < k {
                heap.push(cand);
            } else if let Some(top) = heap.peek() {
                if cand.dist_sq < top.dist_sq {
                    let _ = heap.pop();
                    heap.push(cand);
                }
            }
        }

        let mut results = Vec::with_capacity(heap.len());
        while let Some(cand) = heap.pop() {
            results.push(SearchResult {
                id: cand.id,
                distance: cand.dist_sq.sqrt(),
            });
        }
        results.reverse();
        results
    }
}
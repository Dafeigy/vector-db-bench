use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// Wrapper for distance to allow ordering
#[derive(Debug, Clone, Copy)]
struct DistanceKey {
    distance: f64,
    id: u64,
}

impl PartialEq for DistanceKey {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance && self.id == other.id
    }
}

impl Eq for DistanceKey {}

impl PartialOrd for DistanceKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Compare by distance descending, then by id ascending for tie-breaking
        if self.distance > other.distance {
            Some(Ordering::Greater)
        } else if self.distance < other.distance {
            Some(Ordering::Less)
        } else {
            // Distances equal, compare ids
            self.id.partial_cmp(&other.id)
        }
    }
}

impl Ord for DistanceKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct VectorDB {
    // Store vectors and ids together for better locality
    data: RwLock<Vec<([f32; 128], u64)>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut data = self.data.write().unwrap();
        let arr: [f32; 128] = vector.try_into().unwrap_or_else(|_| panic!("Vector length must be 128"));
        data.push((arr, id));
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut data = self.data.write().unwrap();
        let start = data.len();
        data.reserve(vectors.len());
        for (id, vector) in vectors {
            if vector.len() != 128 {
                eprintln!("ERROR: vector length is {}, expected 128", vector.len());
                panic!("Vector length must be 128");
            }
            let arr: [f32; 128] = vector.try_into().unwrap();
            data.push((arr, id));
        }
        data.len() - start
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        if query.len() != 128 {
            eprintln!("ERROR: query vector length is {}, expected 128", query.len());
            panic!("Query vector length must be 128");
        }
        let data = self.data.read().unwrap();
        let k = top_k as usize;
        if k == 0 {
            return Vec::new();
        }
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for (vec, id) in data.iter() {
            let dist = l2_distance(query, vec);
            let key = DistanceKey { distance: dist, id: *id };
            if heap.len() < k {
                heap.push(key);
            } else {
                let &max_key = heap.peek().unwrap();
                if dist < max_key.distance {
                    heap.pop();
                    heap.push(key);
                }
            }
        }
        // Convert to sorted results
        let mut results: Vec<_> = heap.into_sorted_vec()
            .into_iter()
            .map(|key| SearchResult { id: key.id, distance: key.distance })
            .collect();
        // Ensure ascending order by distance
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        results
    }
}
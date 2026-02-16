use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Clone, Copy)]
struct SearchCandidate {
    id: u64,
    distance: f64,
}

impl PartialEq for SearchCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Max-heap based on distance - largest distance at top
impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
    }
}

pub struct VectorDB {
    vectors: RwLock<Vec<(u64, Vec<f32>)>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vecs = self.vectors.write().unwrap();
        vecs.push((id, vector));
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut vecs = self.vectors.write().unwrap();
        let count = vectors.len();
        vecs.extend(vectors);
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let vecs = self.vectors.read().unwrap();
        
        if vecs.is_empty() {
            return Vec::new();
        }

        let top_k = top_k as usize;
        
        // Parallel search using rayon
        let results: Vec<SearchCandidate> = vecs
            .par_iter()
            .map(|(id, vec)| {
                SearchCandidate {
                    id: *id,
                    distance: l2_distance(query, vec),
                }
            })
            .collect();

        // Use a max-heap to find top-k (largest distance at top)
        let mut heap = BinaryHeap::with_capacity(top_k);
        
        for candidate in results {
            if heap.len() < top_k {
                heap.push(candidate);
            } else if let Some(&max) = heap.peek() {
                if candidate.distance < max.distance {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }

        // Convert to sorted result (ascending by distance)
        let mut results: Vec<SearchCandidate> = heap.into_sorted_vec();
        results.reverse(); // Now ascending order
        
        results
            .into_iter()
            .map(|c| SearchResult {
                id: c.id,
                distance: c.distance,
            })
            .collect()
    }
}

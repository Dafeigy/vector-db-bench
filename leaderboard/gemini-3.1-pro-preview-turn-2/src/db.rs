use crate::api::*;
use parking_lot::RwLock;
use crate::distance::l2_distance;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use rayon::prelude::*;

pub struct VectorDB {
    data: RwLock<Vec<(u64, Vec<f32>)>>,
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            data: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut data = self.data.write();
        data.push((id, vector));
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let len = vectors.len();
        let mut data = self.data.write();
        data.reserve(len);
        data.extend(vectors);
        len
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read();
        
        #[derive(PartialEq)]
        struct MaxNonNan(f64);

        impl Eq for MaxNonNan {}

        impl PartialOrd for MaxNonNan {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.0.partial_cmp(&other.0)
            }
        }

        impl Ord for MaxNonNan {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        #[derive(Eq, PartialEq)]
        struct HeapElement {
            distance: MaxNonNan,
            id: u64,
        }

        impl PartialOrd for HeapElement {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for HeapElement {
            fn cmp(&self, other: &Self) -> Ordering {
                self.distance.cmp(&other.distance)
            }
        }

        let mut results;
        let chunk_size = data.len() / 4 + 1;
        let threads: Vec<_> = data.par_chunks(chunk_size).map(|chunk| {
            let mut heap = BinaryHeap::with_capacity(top_k as usize + 1);
            for &(id, ref v) in chunk {
                let dist = l2_distance(vector, v);
                if heap.len() < top_k as usize {
                    heap.push(HeapElement { distance: MaxNonNan(dist), id });
                } else if dist < heap.peek().unwrap().distance.0 {
                    heap.push(HeapElement { distance: MaxNonNan(dist), id });
                    heap.pop();
                }
            }
            heap
        }).collect();
        
        let mut final_heap = BinaryHeap::with_capacity(top_k as usize + 1);
        for heap in threads.into_iter() {
            for element in heap {
                if final_heap.len() < top_k as usize {
                    final_heap.push(element);
                } else if element.distance.0 < final_heap.peek().unwrap().distance.0 {
                    final_heap.push(element);
                    final_heap.pop();
                }
            }
        }
        results = final_heap.into_sorted_vec();

        results.into_iter().map(|e| SearchResult {
            id: e.id,
            distance: e.distance.0,
        }).collect()
    }
}

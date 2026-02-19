use crate::api::*;
use parking_lot::RwLock;
use crate::distance::l2_distance;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

#[derive(Clone)]
struct VectorData {
    id: u64,
    vector: [f32; 128],
}

pub struct VectorDB {
    data: RwLock<Vec<VectorData>>,
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
        let mut arr = [0.0; 128];
        arr.copy_from_slice(&vector);
        data.push(VectorData { id, vector: arr });
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let len = vectors.len();
        let mut data = self.data.write();
        data.reserve(len);
        for (id, vec) in vectors {
            let mut arr = [0.0; 128];
            arr.copy_from_slice(&vec);
            data.push(VectorData { id, vector: arr });
        }
        len
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read();
        
        #[derive(PartialEq, Clone, Copy)]
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

        #[derive(Eq, PartialEq, Clone, Copy)]
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

        use rayon::prelude::*;
        
        let chunk_size = std::cmp::max(data.len() / 4, 1);
        let threads: Vec<_> = data.par_chunks(chunk_size).map(|chunk| {
            let mut heap = BinaryHeap::with_capacity(top_k as usize + 1);
            for v in chunk {
                let dist = l2_distance(vector, &v.vector);
                if heap.len() < top_k as usize {
                    heap.push(HeapElement { distance: MaxNonNan(dist), id: v.id });
                } else if dist < heap.peek().unwrap().distance.0 {
                    heap.push(HeapElement { distance: MaxNonNan(dist), id: v.id });
                    heap.pop();
                }
            }
            heap
        }).collect();
        
        let mut final_heap = BinaryHeap::with_capacity(top_k as usize + 1);
        for heap in threads {
            for element in heap {
                if final_heap.len() < top_k as usize {
                    final_heap.push(element);
                } else if element.distance.0 < final_heap.peek().unwrap().distance.0 {
                    final_heap.push(element);
                    final_heap.pop();
                }
            }
        }
        
        let results = final_heap.into_sorted_vec();

        results.into_iter().map(|e| SearchResult {
            id: e.id,
            distance: e.distance.0,
        }).collect()
    }
}

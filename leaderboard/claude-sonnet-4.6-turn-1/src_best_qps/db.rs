use crate::api::SearchResult;
use crate::distance::l2_distance_f32;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;

// Pure brute-force with 4-way parallel search.
// 1M vectors * 128 dims * 4 bytes = 512MB. 
// With AVX-512 & 4 cores, throughput ~32 GB/s effective -> 512MB / 32 GB/s = 16ms/query.
// 4 threads * 1000ms / 16ms = ~62 QPS. Not great, but correct.
// We'll also try to use a simple flat structure.

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    #[inline] fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF32 {
    #[inline] fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct HeapEntry { dist: OrdF32, id: u64 }
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering { self.dist.cmp(&other.dist) }
}

pub struct VectorDB {
    data: RwLock<Vec<f32>>,
    ids: RwLock<Vec<u64>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            data: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        self.data.write().extend_from_slice(&vector);
        self.ids.write().push(id);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut d = self.data.write();
        let mut i = self.ids.write();
        d.reserve(count * 128);
        i.reserve(count);
        for (id, vec) in vectors {
            d.extend_from_slice(&vec);
            i.push(id);
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read();
        let ids = self.ids.read();
        let n = ids.len();
        if n == 0 { return Vec::new(); }
        let k = (top_k as usize).min(n);

        // Split into 4 parallel chunks
        let chunk_size = (n + 3) / 4;

        let results: Vec<Vec<(f32, u64)>> = (0..4).into_par_iter().map(|ci| {
            let start = ci * chunk_size;
            let end = (start + chunk_size).min(n);
            if start >= end { return Vec::new(); }

            let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
            for i in start..end {
                let v = &data[i * 128..(i + 1) * 128];
                let d = l2_distance_f32(vector, v);
                let e = HeapEntry { dist: OrdF32(d), id: ids[i] };
                if heap.len() < k {
                    heap.push(e);
                } else if d < heap.peek().unwrap().dist.0 {
                    heap.pop();
                    heap.push(e);
                }
            }
            heap.into_iter().map(|e| (e.dist.0, e.id)).collect()
        }).collect();

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
        for chunk in results {
            for (d, id) in chunk {
                let e = HeapEntry { dist: OrdF32(d), id };
                if heap.len() < k { heap.push(e); }
                else if d < heap.peek().unwrap().dist.0 { heap.pop(); heap.push(e); }
            }
        }

        let mut out: Vec<SearchResult> = heap.into_iter()
            .map(|e| SearchResult { id: e.id, distance: e.dist.0 as f64 }).collect();
        out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        out
    }
}

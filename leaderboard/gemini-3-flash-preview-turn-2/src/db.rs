use crate::api::*;
use crate::distance::l2_distance;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::cmp::Ordering;

pub struct VectorDB {
    ids: RwLock<Vec<u64>>,
    vectors: RwLock<Vec<f32>>,
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            ids: RwLock::new(Vec::with_capacity(1_000_000)),
            vectors: RwLock::new(Vec::with_capacity(1_000_000 * 128)),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids = self.ids.write();
        let mut vectors = self.vectors.write();
        ids.push(id);
        vectors.extend(vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut ids = self.ids.write();
        let mut vs = self.vectors.write();
        for (id, vector) in vectors {
            ids.push(id);
            vs.extend(vector);
        }
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids = self.ids.read();
        let vectors = self.vectors.read();
        let n = ids.len();

        let k = top_k as usize;

        let mut results: Vec<SearchResult> = (0..n)
            .into_par_iter()
            .fold(
                || Vec::with_capacity(k * 2),
                |mut local_top, i| {
                    let start = i * 128;
                    let v = &vectors[start..start + 128];
                    let dist_sq = l2_distance(v, vector);
                    
                    if local_top.len() < k {
                        local_top.push(SearchResult { id: ids[i], distance: dist_sq });
                        if local_top.len() == k {
                           local_top.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
                        }
                    } else if dist_sq < local_top[k-1].distance {
                        let pos = local_top.binary_search_by(|x| x.distance.partial_cmp(&dist_sq).unwrap_or(Ordering::Equal)).unwrap_or_else(|e| e);
                        local_top.insert(pos, SearchResult { id: ids[i], distance: dist_sq });
                        local_top.pop();
                    }
                    local_top
                }
            )
            .reduce(
                || Vec::new(),
                |mut a, b| {
                    if a.is_empty() { return b; }
                    if b.is_empty() { return a; }
                    
                    let mut merged = Vec::with_capacity(k);
                    let (mut i, mut j) = (0, 0);
                    while merged.len() < k && (i < a.len() || j < b.len()) {
                        if i < a.len() && (j == b.len() || a[i].distance <= b[j].distance) {
                            merged.push(SearchResult { id: a[i].id, distance: a[i].distance });
                            i += 1;
                        } else {
                            merged.push(SearchResult { id: b[j].id, distance: b[j].distance });
                            j += 1;
                        }
                    }
                    merged
                }
            );

        for res in &mut results {
            res.distance = res.distance.sqrt();
        }
        results
    }
}

use crate::api::SearchResult;
use std::sync::{Arc, RwLock};
use rayon::prelude::*;
use std::arch::x86_64::*;

const DIM: usize = 128;
const NUM_CENTROIDS: usize = 2048;
const NUM_PROBE: usize = 30;

pub struct VectorDB {
    data: Arc<RwLock<DBData>>,
}

struct DBData {
    ids: Vec<u64>,
    vectors: Vec<f32>,  // flattened: [v0_0, v0_1, ..., v0_127, v1_0, ...]
    // IVF index
    centroids: Vec<f32>,  // NUM_CENTROIDS * DIM flattened
    inverted_lists: Vec<Vec<usize>>,  // vector indices grouped by centroid
    index_built: bool,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            data: Arc::new(RwLock::new(DBData {
                ids: Vec::new(),
                vectors: Vec::new(),
                centroids: Vec::new(),
                inverted_lists: Vec::new(),
                index_built: false,
            })),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut data = self.data.write().unwrap();
        data.ids.push(id);
        data.vectors.extend(vector);
        data.index_built = false;
    }

    pub fn bulk_insert(&self, vectors_to_insert: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors_to_insert.len();
        let mut data = self.data.write().unwrap();
        
        for (id, vector) in vectors_to_insert {
            data.ids.push(id);
            data.vectors.extend(vector);
        }
        data.index_built = false;
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        // Check if we need to build index
        {
            let data = self.data.read().unwrap();
            if !data.index_built && data.ids.len() >= NUM_CENTROIDS {
                drop(data);
                // Build index with write lock
                let mut data = self.data.write().unwrap();
                if !data.index_built {
                    build_index(&mut data);
                }
            }
        }
        
        let data = self.data.read().unwrap();
        let n = data.ids.len();
        
        if n == 0 {
            return Vec::new();
        }
        
        // If index not built (too few vectors), do brute force
        if !data.index_built {
            drop(data);
            return self.brute_force_search(query, top_k);
        }
        
        let ids = &data.ids;
        let vectors = &data.vectors;
        let centroids = &data.centroids;
        
        // Find nearest NUM_PROBE centroids
        let mut centroid_dists: Vec<(usize, f32)> = (0..NUM_CENTROIDS)
            .map(|c| {
                let c_start = c * DIM;
                let centroid = &centroids[c_start..c_start + DIM];
                let dist = unsafe { l2_distance_squared_avx512(query, centroid) };
                (c, dist)
            })
            .collect();
        
        centroid_dists.select_nth_unstable_by(NUM_PROBE, |a, b| a.1.partial_cmp(&b.1).unwrap());
        centroid_dists.truncate(NUM_PROBE);
        
        // Collect candidate vector indices from selected centroids
        let candidates: Vec<usize> = centroid_dists
            .iter()
            .flat_map(|(c, _)| data.inverted_lists[*c].iter().copied())
            .collect();
        
        // Search only candidates
        let mut results: Vec<(u64, f32)> = candidates
            .into_par_iter()
            .map(|i| {
                let id = ids[i];
                let start = i * DIM;
                let vec = &vectors[start..start + DIM];
                let dist_sq = unsafe { l2_distance_squared_avx512(query, vec) };
                (id, dist_sq)
            })
            .collect();
        
        // Partial sort for top-k
        let k = top_k as usize;
        if k < results.len() {
            results.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(k);
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        } else {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        
        results
            .into_iter()
            .map(|(id, dist_sq)| SearchResult { id, distance: (dist_sq as f64).sqrt() })
            .collect()
    }
    
    fn brute_force_search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let data = self.data.read().unwrap();
        let n = data.ids.len();
        
        if n == 0 {
            return Vec::new();
        }
        
        let ids = &data.ids;
        let vectors = &data.vectors;
        
        let mut results: Vec<(u64, f32)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let id = ids[i];
                let start = i * DIM;
                let vec = &vectors[start..start + DIM];
                let dist_sq = unsafe { l2_distance_squared_avx512(query, vec) };
                (id, dist_sq)
            })
            .collect();
        
        let k = top_k as usize;
        if k < results.len() {
            results.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(k);
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        } else {
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        }
        
        results
            .into_iter()
            .map(|(id, dist_sq)| SearchResult { id, distance: (dist_sq as f64).sqrt() })
            .collect()
    }
}

fn build_index(data: &mut DBData) {
    if data.index_built || data.ids.len() < NUM_CENTROIDS {
        return;
    }

    let n = data.ids.len();
    
    // Select centroids using uniform sampling
    data.centroids = Vec::with_capacity(NUM_CENTROIDS * DIM);
    let step = n / NUM_CENTROIDS;
    for i in 0..NUM_CENTROIDS {
        let idx = i * step;
        let start = idx * DIM;
        data.centroids.extend_from_slice(&data.vectors[start..start + DIM]);
    }

    // Refine centroids using a few iterations of k-means
    for _iter in 0..5 {
        let mut new_centroids = vec![0.0f32; NUM_CENTROIDS * DIM];
        let mut counts = vec![0usize; NUM_CENTROIDS];
        
        // Assign vectors to nearest centroid and accumulate
        for i in 0..n {
            let start = i * DIM;
            let vec = &data.vectors[start..start + DIM];
            let mut best_c = 0;
            let mut best_dist = f32::MAX;
            
            for c in 0..NUM_CENTROIDS {
                let c_start = c * DIM;
                let centroid = &data.centroids[c_start..c_start + DIM];
                let dist = unsafe { l2_distance_squared_avx512(vec, centroid) };
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            
            counts[best_c] += 1;
            for d in 0..DIM {
                new_centroids[best_c * DIM + d] += vec[d];
            }
        }
        
        // Average the centroids
        for c in 0..NUM_CENTROIDS {
            if counts[c] > 0 {
                for d in 0..DIM {
                    new_centroids[c * DIM + d] /= counts[c] as f32;
                }
            }
        }
        
        data.centroids = new_centroids;
    }

    // Build inverted lists
    data.inverted_lists = vec![Vec::new(); NUM_CENTROIDS];
    for i in 0..n {
        let start = i * DIM;
        let vec = &data.vectors[start..start + DIM];
        let mut best_c = 0;
        let mut best_dist = f32::MAX;
        
        for c in 0..NUM_CENTROIDS {
            let c_start = c * DIM;
            let centroid = &data.centroids[c_start..c_start + DIM];
            let dist = unsafe { l2_distance_squared_avx512(vec, centroid) };
            if dist < best_dist {
                best_dist = dist;
                best_c = c;
            }
        }
        
        data.inverted_lists[best_c].push(i);
    }

    data.index_built = true;
}

#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn l2_distance_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut sum_vec = _mm512_setzero_ps();
    
    for i in (0..DIM).step_by(16) {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));
        let diff = _mm512_sub_ps(va, vb);
        sum_vec = _mm512_fmadd_ps(diff, diff, sum_vec);
    }
    
    _mm512_reduce_add_ps(sum_vec)
}

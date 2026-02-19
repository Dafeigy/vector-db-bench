use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;

// Using a clustered approach (IVF basically) 
const NUM_CENTROIDS: usize = 1000;
const NPROBE: usize = 50; // Increased for better recall

pub struct VectorDB {
    // Basic brute-force array for new items until trained
    vectors: RwLock<Vec<f32>>,
    ids: RwLock<Vec<u64>>,
    
    // IVF structures
    centroids: RwLock<Vec<Vec<f32>>>,
    // Flat arrays for each cluster to be cache friendly
    clusters_vectors: RwLock<Vec<Vec<f32>>>,
    clusters_ids: RwLock<Vec<Vec<u64>>>,
    trained: RwLock<bool>,
}

impl Default for VectorDB {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            vectors: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            centroids: RwLock::new(Vec::new()),
            clusters_vectors: RwLock::new(vec![Vec::new(); NUM_CENTROIDS]),
            clusters_ids: RwLock::new(vec![Vec::new(); NUM_CENTROIDS]),
            trained: RwLock::new(false),
        }
    }

    pub fn insert(&self, id: u64, mut vector: Vec<f32>) {
        let is_trained = *self.trained.read().unwrap();
        if !is_trained {
            let mut vecs_guard = self.vectors.write().unwrap();
            let mut ids_guard = self.ids.write().unwrap();
            vecs_guard.append(&mut vector);
            ids_guard.push(id);
        } else {
            // Find closest centroid
            let centroids = self.centroids.read().unwrap();
            let mut best_dist = f64::INFINITY;
            let mut best_idx = 0;
            for (i, c) in centroids.iter().enumerate() {
                let d = l2_distance(c, &vector);
                if d < best_dist {
                    best_dist = d;
                    best_idx = i;
                }
            }
            let mut c_vecs = self.clusters_vectors.write().unwrap();
            let mut c_ids = self.clusters_ids.write().unwrap();
            c_vecs[best_idx].append(&mut vector);
            c_ids[best_idx].push(id);
        }
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        
        let mut is_trained = *self.trained.read().unwrap();
        
        // Train
        if !is_trained && vectors.len() >= NUM_CENTROIDS {
            let mut centroids = self.centroids.write().unwrap();
            
            // To be slightly more robust, take strides
            let stride = count / NUM_CENTROIDS;
            for i in 0..NUM_CENTROIDS {
                centroids.push(vectors[i * stride].1.clone());
            }
            
            let mut c_vecs = self.clusters_vectors.write().unwrap();
            let mut c_ids = self.clusters_ids.write().unwrap();
            
            for (id, mut v) in vectors {
                let mut best_dist = f64::INFINITY;
                let mut best_idx = 0;
                for (i, c) in centroids.iter().enumerate() {
                    let d = l2_distance(c, &v);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                c_vecs[best_idx].append(&mut v);
                c_ids[best_idx].push(id);
            }
            
            *self.trained.write().unwrap() = true;
            return count;
        }

        if !is_trained {
            let mut vecs_guard = self.vectors.write().unwrap();
            let mut ids_guard = self.ids.write().unwrap();
            vecs_guard.reserve(count * 128);
            ids_guard.reserve(count);
            for (id, mut v) in vectors {
                vecs_guard.append(&mut v);
                ids_guard.push(id);
            }
        } else {
            let centroids = self.centroids.read().unwrap();
            let mut c_vecs = self.clusters_vectors.write().unwrap();
            let mut c_ids = self.clusters_ids.write().unwrap();
            
            for (id, mut v) in vectors {
                let mut best_dist = f64::INFINITY;
                let mut best_idx = 0;
                for (i, c) in centroids.iter().enumerate() {
                    let d = l2_distance(c, &v);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = i;
                    }
                }
                c_vecs[best_idx].append(&mut v);
                c_ids[best_idx].push(id);
            }
        }
        
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let is_trained = *self.trained.read().unwrap();
        
        if !is_trained {
            // Brute force (should not happen in real scoring since bulk insert trains it)
            let vecs_guard = self.vectors.read().unwrap();
            let ids_guard = self.ids.read().unwrap();
            let chunk_size = 128; // dimension
            let num_vectors = ids_guard.len();
            
            let mut results = Vec::with_capacity(num_vectors);
            for i in 0..num_vectors {
                let start = i * chunk_size;
                let end = start + chunk_size;
                let v = &vecs_guard[start..end];
                let dist = l2_distance(v, vector);
                results.push((ids_guard[i], dist));
            }

            let k = top_k as usize;
            if results.len() > k {
                results.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }
            let mut top_results = results.into_iter().take(k).collect::<Vec<_>>();
            top_results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            return top_results.into_iter()
                .map(|(id, distance)| SearchResult { id, distance })
                .collect();
        }
        
        let centroids = self.centroids.read().unwrap();
        let mut centroid_dists = Vec::with_capacity(NUM_CENTROIDS);
        for (i, c) in centroids.iter().enumerate() {
            centroid_dists.push((i, l2_distance(c, vector)));
        }
        
        centroid_dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let c_vecs = self.clusters_vectors.read().unwrap();
        let c_ids = self.clusters_ids.read().unwrap();
        
        let mut results = Vec::with_capacity(NPROBE * 2000); 

        for j in 0..NPROBE {
            let c_idx = centroid_dists[j].0;
            let vecs = &c_vecs[c_idx];
            let ids = &c_ids[c_idx];
            let num = ids.len();
            let chunk_size = 128;
            for i in 0..num {
                let start = i * chunk_size;
                let end = start + chunk_size;
                let dist = l2_distance(&vecs[start..end], vector);
                results.push((ids[i], dist));
            }
        }

        let k = top_k as usize;
        if results.len() > k {
            results.select_nth_unstable_by(k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        let mut top_results = results.into_iter().take(k).collect::<Vec<_>>();
        top_results.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        top_results.into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}

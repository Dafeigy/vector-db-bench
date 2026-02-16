use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;
use rayon::prelude::*;

// IVF (Inverted File Index) with optimized parameters
const NUM_CLUSTERS: usize = 200;
const NPROBE: usize = 15;

pub struct VectorDB {
    clusters: RwLock<Vec<Vec<usize>>>,
    centroids: RwLock<Vec<Vec<f32>>>,
    ids: RwLock<Vec<u64>>,
    data: RwLock<Vec<f32>>,
    dim: usize,
}

impl VectorDB {
    pub fn new() -> Self {
        let mut clusters = Vec::with_capacity(NUM_CLUSTERS);
        for _ in 0..NUM_CLUSTERS {
            clusters.push(Vec::with_capacity(10000));
        }
        
        Self {
            clusters: RwLock::new(clusters),
            centroids: RwLock::new(Vec::with_capacity(NUM_CLUSTERS)),
            ids: RwLock::new(Vec::with_capacity(1_100_000)),
            data: RwLock::new(Vec::with_capacity(1_100_000 * 128)),
            dim: 128,
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut ids_guard = self.ids.write().unwrap();
        let mut data_guard = self.data.write().unwrap();
        
        let idx = ids_guard.len();
        ids_guard.push(id);
        data_guard.extend_from_slice(&vector);
        
        let centroids_guard = self.centroids.read().unwrap();
        let mut clusters_guard = self.clusters.write().unwrap();
        
        if centroids_guard.is_empty() {
            let cluster_id = (idx * 7) % NUM_CLUSTERS;
            clusters_guard[cluster_id].push(idx);
        } else {
            let cluster_id = find_nearest_centroid(&vector, &centroids_guard);
            clusters_guard[cluster_id].push(idx);
        }
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut ids_guard = self.ids.write().unwrap();
        let mut data_guard = self.data.write().unwrap();
        let mut clusters_guard = self.clusters.write().unwrap();
        
        let count = vectors.len();
        let start_idx = ids_guard.len();
        
        ids_guard.reserve(count);
        data_guard.reserve(count * self.dim);
        
        // Store all vectors
        for (id, vec) in &vectors {
            ids_guard.push(*id);
            data_guard.extend_from_slice(vec);
        }
        
        let centroids_guard = self.centroids.read().unwrap();
        let need_clustering = centroids_guard.is_empty() && count >= NUM_CLUSTERS;
        drop(centroids_guard);
        
        if need_clustering {
            let new_centroids = self.compute_kmeans(&data_guard, start_idx, count);
            
            let mut centroids_guard = self.centroids.write().unwrap();
            *centroids_guard = new_centroids;
            drop(centroids_guard);
            
            let centroids_guard = self.centroids.read().unwrap();
            for i in 0..count {
                let idx = start_idx + i;
                let vec_start = idx * self.dim;
                let vec = &data_guard[vec_start..vec_start + self.dim];
                let cluster_id = find_nearest_centroid(vec, &centroids_guard);
                clusters_guard[cluster_id].push(idx);
            }
        } else {
            let centroids_guard = self.centroids.read().unwrap();
            if centroids_guard.is_empty() {
                for i in 0..count {
                    let idx = start_idx + i;
                    let cluster_id = (idx * 7) % NUM_CLUSTERS;
                    clusters_guard[cluster_id].push(idx);
                }
            } else {
                for i in 0..count {
                    let idx = start_idx + i;
                    let vec_start = idx * self.dim;
                    let vec = &data_guard[vec_start..vec_start + self.dim];
                    let cluster_id = find_nearest_centroid(vec, &centroids_guard);
                    clusters_guard[cluster_id].push(idx);
                }
            }
        }
        
        count
    }
    
    fn compute_kmeans(&self, data: &[f32], start_idx: usize, count: usize) -> Vec<Vec<f32>> {
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(NUM_CLUSTERS);
        
        // Sample for initial centroids
        let sample_indices: Vec<usize> = (0..count)
            .step_by(count / NUM_CLUSTERS)
            .take(NUM_CLUSTERS)
            .collect();
        
        for &sample_idx in &sample_indices {
            let idx = start_idx + sample_idx;
            let vec_start = idx * self.dim;
            centroids.push(data[vec_start..vec_start + self.dim].to_vec());
        }
        
        // One iteration
        let mut assignments: Vec<usize> = vec![0; count];
        
        for i in 0..count {
            let idx = start_idx + i;
            let vec_start = idx * self.dim;
            let vec = &data[vec_start..vec_start + self.dim];
            assignments[i] = find_nearest_centroid(vec, &centroids);
        }
        
        // Update centroids
        for c in 0..NUM_CLUSTERS {
            let mut new_centroid = vec![0.0f32; self.dim];
            let mut count_in_cluster = 0;
            
            for i in 0..count {
                if assignments[i] == c {
                    let idx = start_idx + i;
                    let vec_start = idx * self.dim;
                    for d in 0..self.dim {
                        new_centroid[d] += data[vec_start + d];
                    }
                    count_in_cluster += 1;
                }
            }
            
            if count_in_cluster > 0 {
                for d in 0..self.dim {
                    new_centroid[d] /= count_in_cluster as f32;
                }
                centroids[c] = new_centroid;
            }
        }
        
        centroids
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let ids_guard = self.ids.read().unwrap();
        let data_guard = self.data.read().unwrap();
        let clusters_guard = self.clusters.read().unwrap();
        let centroids_guard = self.centroids.read().unwrap();
        
        let ids = &*ids_guard;
        let data = &*data_guard;
        let n = ids.len();
        let top_k = top_k as usize;
        
        if n == 0 {
            return Vec::new();
        }
        
        if centroids_guard.is_empty() || centroids_guard.len() != NUM_CLUSTERS {
            return self.brute_force_search(query, top_k, ids, data);
        }
        
        // Find NPROBE nearest centroids
        let mut centroid_distances: Vec<(usize, f64)> = centroids_guard
            .par_iter()
            .enumerate()
            .map(|(i, c)| (i, l2_distance(query, c)))
            .collect();
        
        centroid_distances.select_nth_unstable_by(NPROBE - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        centroid_distances.truncate(NPROBE);
        
        // Collect candidates
        let mut candidate_indices: Vec<usize> = Vec::with_capacity(100000);
        for (cluster_id, _) in &centroid_distances {
            candidate_indices.extend(&clusters_guard[*cluster_id]);
        }
        
        // Search candidates
        let mut results: Vec<(u64, f64)> = candidate_indices
            .par_iter()
            .map(|&idx| {
                let start = idx * self.dim;
                let vector = &data[start..start + self.dim];
                let dist = l2_distance(query, vector);
                (ids[idx], dist)
            })
            .collect();
        
        if results.len() > top_k {
            results.select_nth_unstable_by(top_k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(top_k);
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        results
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
    
    fn brute_force_search(
        &self,
        query: &[f32],
        top_k: usize,
        ids: &[u64],
        data: &[f32],
    ) -> Vec<SearchResult> {
        let n = ids.len();
        
        let mut results: Vec<(u64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = i * self.dim;
                let vector = &data[start..start + self.dim];
                let dist = l2_distance(query, vector);
                (ids[i], dist)
            })
            .collect();
        
        results.select_nth_unstable_by(top_k - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_k);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        results
            .into_iter()
            .map(|(id, distance)| SearchResult { id, distance })
            .collect()
    }
}

fn find_nearest_centroid(vector: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best_id = 0;
    let mut best_dist = l2_distance(vector, &centroids[0]);
    
    for (i, centroid) in centroids.iter().enumerate().skip(1) {
        let dist = l2_distance(vector, centroid);
        if dist < best_dist {
            best_dist = dist;
            best_id = i;
        }
    }
    
    best_id
}

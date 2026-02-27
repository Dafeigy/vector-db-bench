use crate::api::*;
use crate::distance::l2_distance_squared;
use std::sync::RwLock;
use rayon::prelude::*;

const NUM_CLUSTERS: usize = 256;
const NPROBE: usize = 20; // Number of clusters to search (tuned for recall vs speed)

// Store vectors in a contiguous aligned array for better cache performance
pub struct VectorDB {
    // Flat storage: all vectors concatenated in a single Vec
    vectors_data: RwLock<Vec<f32>>,
    // Corresponding IDs
    ids: RwLock<Vec<u64>>,
    // IVF index: cluster centroids and assignments
    centroids: RwLock<Option<Vec<f32>>>, // NUM_CLUSTERS * 128
    cluster_lists: RwLock<Option<Vec<Vec<usize>>>>, // Lists of vector indices per cluster
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors_data: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            centroids: RwLock::new(None),
            cluster_lists: RwLock::new(None),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vectors_data = self.vectors_data.write().unwrap();
        let mut ids = self.ids.write().unwrap();
        vectors_data.extend_from_slice(&vector);
        ids.push(id);
        
        // Invalidate index on insert
        *self.centroids.write().unwrap() = None;
        *self.cluster_lists.write().unwrap() = None;
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut vectors_data = self.vectors_data.write().unwrap();
        let mut ids = self.ids.write().unwrap();
        let count = vectors.len();
        
        for (id, vector) in vectors {
            vectors_data.extend_from_slice(&vector);
            ids.push(id);
        }
        
        // Invalidate index on bulk insert
        *self.centroids.write().unwrap() = None;
        *self.cluster_lists.write().unwrap() = None;
        
        count
    }
    
    fn build_index(&self) {
        let vectors_data = self.vectors_data.read().unwrap();
        let num_vectors = vectors_data.len() / 128;
        
        if num_vectors < NUM_CLUSTERS * 10 {
            // Not enough data for clustering
            return;
        }
        
        // Simple k-means clustering
        let mut centroids = vec![0.0f32; NUM_CLUSTERS * 128];
        let mut assignments = vec![0usize; num_vectors];
        
        // Initialize centroids with random vectors
        for i in 0..NUM_CLUSTERS {
            let src_idx = (i * num_vectors / NUM_CLUSTERS) * 128;
            centroids[i * 128..(i + 1) * 128].copy_from_slice(&vectors_data[src_idx..src_idx + 128]);
        }
        
        // Run k-means for a few iterations
        for _iter in 0..10 {
            // Assignment step
            (0..num_vectors).into_par_iter().map(|i| {
                let offset = i * 128;
                let v = &vectors_data[offset..offset + 128];
                let mut best_cluster = 0;
                let mut best_dist = f32::MAX;
                
                for c in 0..NUM_CLUSTERS {
                    let centroid = &centroids[c * 128..(c + 1) * 128];
                    let dist = l2_distance_squared(v, centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }
                (i, best_cluster)
            }).collect::<Vec<_>>().into_iter().for_each(|(i, cluster)| {
                assignments[i] = cluster;
            });
            
            // Update centroids
            let mut cluster_counts = vec![0usize; NUM_CLUSTERS];
            let mut cluster_sums = vec![0.0f32; NUM_CLUSTERS * 128];
            
            for i in 0..num_vectors {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;
                let offset = i * 128;
                for j in 0..128 {
                    cluster_sums[cluster * 128 + j] += vectors_data[offset + j];
                }
            }
            
            for c in 0..NUM_CLUSTERS {
                if cluster_counts[c] > 0 {
                    let count = cluster_counts[c] as f32;
                    for j in 0..128 {
                        centroids[c * 128 + j] = cluster_sums[c * 128 + j] / count;
                    }
                }
            }
        }
        
        // Build inverted lists
        let mut cluster_lists = vec![Vec::new(); NUM_CLUSTERS];
        for i in 0..num_vectors {
            cluster_lists[assignments[i]].push(i);
        }
        
        *self.centroids.write().unwrap() = Some(centroids);
        *self.cluster_lists.write().unwrap() = Some(cluster_lists);
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let vectors_data = self.vectors_data.read().unwrap();
        let ids = self.ids.read().unwrap();
        
        let num_vectors = ids.len();
        if num_vectors == 0 {
            return Vec::new();
        }
        
        // Build index if needed
        if self.centroids.read().unwrap().is_none() {
            drop(vectors_data);
            drop(ids);
            self.build_index();
            return self.search(vector, top_k);
        }
        
        let centroids_opt = self.centroids.read().unwrap();
        let cluster_lists_opt = self.cluster_lists.read().unwrap();
        
        if let (Some(centroids), Some(cluster_lists)) = (centroids_opt.as_ref(), cluster_lists_opt.as_ref()) {
            // Find nearest clusters to query
            let mut cluster_dists: Vec<(usize, f32)> = (0..NUM_CLUSTERS)
                .map(|c| {
                    let centroid = &centroids[c * 128..(c + 1) * 128];
                    (c, l2_distance_squared(vector, centroid))
                })
                .collect();
            
            // Partial select for nprobe clusters (faster than full sort)
            cluster_dists.select_nth_unstable_by(NPROBE, |a, b| a.1.partial_cmp(&b.1).unwrap());
            
            // Collect all candidate indices from top NPROBE clusters
            let mut candidates = Vec::new();
            for i in 0..NPROBE {
                let cluster = cluster_dists[i].0;
                candidates.extend_from_slice(&cluster_lists[cluster]);
            }
            
            // Parallel distance computation
            let mut results: Vec<(u64, f32)> = candidates
                .into_par_iter()
                .map(|i| {
                    let offset = i * 128;
                    let v = &vectors_data[offset..offset + 128];
                    (ids[i], l2_distance_squared(vector, v))
                })
                .collect();
            
            // Partial sort to get top-k
            let k = (top_k as usize).min(results.len());
            if k > 0 {
                results.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
                results.truncate(k);
                results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            }
            
            // Convert to SearchResult with actual L2 distance
            results.into_iter().map(|(id, dist_sq)| SearchResult {
                id,
                distance: (dist_sq as f64).sqrt(),
            }).collect()
        } else {
            // Fallback to brute force
            drop(centroids_opt);
            drop(cluster_lists_opt);
            
            let mut results: Vec<(u64, f32)> = (0..num_vectors)
                .into_par_iter()
                .map(|i| {
                    let offset = i * 128;
                    let v = &vectors_data[offset..offset + 128];
                    (ids[i], l2_distance_squared(vector, v))
                })
                .collect();
            
            let k = (top_k as usize).min(results.len());
            results.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            results.truncate(k);
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            results.into_iter().map(|(id, dist_sq)| SearchResult {
                id,
                distance: (dist_sq as f64).sqrt(),
            }).collect()
        }
    }
}

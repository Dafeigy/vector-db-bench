use crate::api::*;
use crate::distance::l2_distance;
use std::sync::RwLock;
use rayon::prelude::*;

const NUM_CLUSTERS: usize = 2048;
const PROBE_CLUSTERS: usize = 80; // Number of clusters to search (increased for better recall)

pub struct VectorDB {
    data: RwLock<VectorData>,
}

struct VectorData {
    ids: Vec<u64>,
    vectors: Vec<f32>, // Flattened: 128 floats per vector
    // IVF index
    centroids: Vec<f32>, // NUM_CLUSTERS * 128 floats
    inverted_lists: Vec<Vec<usize>>, // Vector indices for each cluster
    indexed: bool,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            data: RwLock::new(VectorData {
                ids: Vec::new(),
                vectors: Vec::new(),
                centroids: Vec::new(),
                inverted_lists: vec![Vec::new(); NUM_CLUSTERS],
                indexed: false,
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut data = self.data.write().unwrap();
        data.ids.push(id);
        data.vectors.extend_from_slice(&vector);
        data.indexed = false;
    }

    pub fn bulk_insert(&self, new_vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut data = self.data.write().unwrap();
        let count = new_vectors.len();
        
        data.ids.reserve(count);
        data.vectors.reserve(count * 128);
        
        for (id, vector) in new_vectors {
            data.ids.push(id);
            data.vectors.extend_from_slice(&vector);
        }
        
        data.indexed = false;
        count
    }

    pub fn search(&self, query: &[f32], top_k: u32) -> Vec<SearchResult> {
        let mut data = self.data.write().unwrap();
        let k = top_k as usize;
        let num_vectors = data.ids.len();
        
        if num_vectors == 0 {
            return Vec::new();
        }
        
        // Build index if not already built
        if !data.indexed {
            build_ivf_index(&mut data);
        }
        
        // Find nearest centroids to query
        let nearest_clusters = find_nearest_centroids_fast(query, &data.centroids, PROBE_CLUSTERS);
        
        // Collect candidate indices from nearest clusters
        let mut candidate_indices = Vec::new();
        for cluster_id in nearest_clusters {
            candidate_indices.extend_from_slice(&data.inverted_lists[cluster_id]);
        }
        
        // Parallel distance computation for candidates
        let mut results: Vec<SearchResult> = candidate_indices
            .par_iter()
            .map(|&i| {
                let vector_start = i * 128;
                let vector = &data.vectors[vector_start..vector_start + 128];
                let distance = l2_distance(query, vector);
                SearchResult {
                    id: data.ids[i],
                    distance,
                }
            })
            .collect();
        
        // Partial sort to get top-k
        if results.len() > k {
            results.select_nth_unstable_by(k, |a, b| {
                a.distance.partial_cmp(&b.distance).unwrap()
            });
            results.truncate(k);
        }
        
        // Sort the top-k by distance
        results.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        
        results
    }
}

fn build_ivf_index(data: &mut VectorData) {
    let num_vectors = data.ids.len();
    if num_vectors == 0 {
        return;
    }
    
    // Initialize centroids
    data.centroids = vec![0.0; NUM_CLUSTERS * 128];
    
    // Simple initialization: pick evenly spaced vectors as initial centroids
    for i in 0..NUM_CLUSTERS.min(num_vectors) {
        let idx = (i * num_vectors) / NUM_CLUSTERS;
        let src_start = idx * 128;
        let dst_start = i * 128;
        data.centroids[dst_start..dst_start + 128]
            .copy_from_slice(&data.vectors[src_start..src_start + 128]);
    }
    
    // Run k-means for a few iterations with parallel assignment
    for _ in 0..10 {
        // Clear inverted lists
        for list in &mut data.inverted_lists {
            list.clear();
        }
        
        // Parallel assignment to clusters
        let assignments: Vec<usize> = (0..num_vectors)
            .into_par_iter()
            .map(|i| {
                let vector_start = i * 128;
                let vector = &data.vectors[vector_start..vector_start + 128];
                
                let mut min_dist = f64::MAX;
                let mut best_cluster = 0;
                
                for c in 0..NUM_CLUSTERS {
                    let centroid_start = c * 128;
                    let centroid = &data.centroids[centroid_start..centroid_start + 128];
                    let dist = l2_distance(vector, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = c;
                    }
                }
                
                best_cluster
            })
            .collect();
        
        // Build inverted lists from assignments
        for (i, &cluster) in assignments.iter().enumerate() {
            data.inverted_lists[cluster].push(i);
        }
        
        // Update centroids in parallel
        let new_centroids: Vec<f32> = (0..NUM_CLUSTERS)
            .into_par_iter()
            .flat_map(|c| {
                let indices = &data.inverted_lists[c];
                let mut centroid = vec![0.0f32; 128];
                
                if !indices.is_empty() {
                    for &i in indices {
                        let vector_start = i * 128;
                        let vector = &data.vectors[vector_start..vector_start + 128];
                        for d in 0..128 {
                            centroid[d] += vector[d];
                        }
                    }
                    
                    let count = indices.len() as f32;
                    for d in 0..128 {
                        centroid[d] /= count;
                    }
                } else {
                    // Keep old centroid if cluster is empty
                    let old_start = c * 128;
                    centroid.copy_from_slice(&data.centroids[old_start..old_start + 128]);
                }
                
                centroid
            })
            .collect();
        
        data.centroids = new_centroids;
    }
    
    data.indexed = true;
}

fn find_nearest_centroids_fast(query: &[f32], centroids: &[f32], n: usize) -> Vec<usize> {
    let num_centroids = centroids.len() / 128;
    
    // Use a min-heap approach to find top-n without full sort
    let mut heap = Vec::with_capacity(n + 1);
    
    for i in 0..num_centroids {
        let centroid_start = i * 128;
        let centroid = &centroids[centroid_start..centroid_start + 128];
        let dist = l2_distance(query, centroid);
        
        heap.push((i, dist));
        
        if heap.len() > n {
            // Find max and remove it
            let max_idx = heap.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            heap.swap_remove(max_idx);
        }
    }
    
    heap.into_iter().map(|(i, _)| i).collect()
}

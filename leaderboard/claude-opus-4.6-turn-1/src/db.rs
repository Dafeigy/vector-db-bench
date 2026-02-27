use crate::api::*;
use crate::distance::l2_distance_batch;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const DIM: usize = 128;
const NUM_CLUSTERS: usize = 16384;
const NPROBE: usize = 112;
const SAMPLE_SIZE: usize = 150000;
const KMEANS_ITERS: usize = 10;

pub struct VectorDB {
    pending_vectors: RwLock<Vec<f32>>,
    pending_ids: RwLock<Vec<u64>>,
    pending_count: AtomicUsize,

    index: RwLock<Option<Box<IVFIndex>>>,
    indexed: AtomicBool,
}

struct IVFIndex {
    centroids: Vec<f32>,
    cluster_vectors: Vec<f32>,
    cluster_ids: Vec<u64>,
    cluster_offsets: Vec<usize>,
    n_clusters: usize,
}

impl VectorDB {
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .ok();

        Self {
            pending_vectors: RwLock::new(Vec::with_capacity(1_100_000 * DIM)),
            pending_ids: RwLock::new(Vec::with_capacity(1_100_000)),
            pending_count: AtomicUsize::new(0),
            index: RwLock::new(None),
            indexed: AtomicBool::new(false),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vecs = self.pending_vectors.write();
        let mut ids = self.pending_ids.write();
        vecs.extend_from_slice(&vector);
        ids.push(id);
        self.pending_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let mut vecs = self.pending_vectors.write();
        let mut ids = self.pending_ids.write();
        for (id, vector) in &vectors {
            vecs.extend_from_slice(vector);
            ids.push(*id);
        }
        self.pending_count.fetch_add(count, Ordering::Relaxed);
        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if !self.indexed.load(Ordering::Acquire) {
            self.build_index();
        }

        let top_k = top_k as usize;
        let index_guard = self.index.read();
        let idx = index_guard.as_ref().unwrap();

        let n_clusters = idx.n_clusters;
        let nprobe = NPROBE.min(n_clusters);

        // Compute distance to all centroids
        let mut centroid_dists = vec![0.0f32; n_clusters];
        l2_distance_batch(vector, &idx.centroids, n_clusters, &mut centroid_dists);

        // Find top NPROBE nearest clusters
        let mut cluster_indices: Vec<u16> = (0..n_clusters as u16).collect();
        let cd = &centroid_dists;
        cluster_indices.select_nth_unstable_by(nprobe - 1, |&a, &b| {
            cd[a as usize].partial_cmp(&cd[b as usize]).unwrap()
        });

        // Sort top nprobe (closest first)
        cluster_indices[..nprobe].sort_unstable_by(|&a, &b| {
            cd[a as usize].partial_cmp(&cd[b as usize]).unwrap()
        });

        let mut heap_dists = [f32::MAX; 10];
        let mut heap_ids = [0u64; 10];
        let mut heap_size = 0usize;
        let actual_k = top_k.min(10);
        let mut threshold = f32::MAX;

        let mut distances_buf = vec![0.0f32; 512];

        for idx_i in 0..nprobe {
            let ci = cluster_indices[idx_i] as usize;
            let start = idx.cluster_offsets[ci];
            let end = idx.cluster_offsets[ci + 1];
            let n = end - start;
            if n == 0 {
                continue;
            }

            if distances_buf.len() < n {
                distances_buf.resize(n, 0.0);
            }

            let vecs_slice = &idx.cluster_vectors[start * DIM..end * DIM];
            let ids_slice = &idx.cluster_ids[start..end];

            l2_distance_batch(vector, vecs_slice, n, &mut distances_buf);

            for j in 0..n {
                let dist = unsafe { *distances_buf.get_unchecked(j) };

                if dist >= threshold {
                    continue;
                }

                let id = unsafe { *ids_slice.get_unchecked(j) };

                if heap_size < actual_k {
                    heap_dists[heap_size] = dist;
                    heap_ids[heap_size] = id;
                    heap_size += 1;
                    let mut i = heap_size - 1;
                    while i > 0 {
                        let parent = (i - 1) / 2;
                        if heap_dists[i] > heap_dists[parent] {
                            heap_dists.swap(i, parent);
                            heap_ids.swap(i, parent);
                            i = parent;
                        } else {
                            break;
                        }
                    }
                    if heap_size == actual_k {
                        threshold = heap_dists[0];
                    }
                } else {
                    heap_dists[0] = dist;
                    heap_ids[0] = id;
                    let mut i = 0;
                    loop {
                        let left = 2 * i + 1;
                        let right = 2 * i + 2;
                        let mut largest = i;
                        if left < actual_k && heap_dists[left] > heap_dists[largest] {
                            largest = left;
                        }
                        if right < actual_k && heap_dists[right] > heap_dists[largest] {
                            largest = right;
                        }
                        if largest != i {
                            heap_dists.swap(i, largest);
                            heap_ids.swap(i, largest);
                            i = largest;
                        } else {
                            break;
                        }
                    }
                    threshold = heap_dists[0];
                }
            }
        }

        let mut results: Vec<(f32, u64)> = (0..heap_size)
            .map(|i| (heap_dists[i], heap_ids[i]))
            .collect();
        results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        results
            .iter()
            .map(|&(dist, id)| SearchResult {
                id,
                distance: dist as f64,
            })
            .collect()
    }

    fn build_index(&self) {
        if self.indexed.swap(true, Ordering::AcqRel) {
            return;
        }

        let vecs = self.pending_vectors.read();
        let ids = self.pending_ids.read();
        let n = ids.len();

        if n == 0 {
            return;
        }

        let n_clusters = NUM_CLUSTERS.min(n);
        eprintln!("Building index: {} vectors, {} clusters", n, n_clusters);

        let centroids = self.run_kmeans(&vecs, n, n_clusters);

        let chunk_size = 5000;
        let n_chunks = (n + chunk_size - 1) / chunk_size;

        let all_assignments: Vec<usize> = {
            let chunks: Vec<Vec<usize>> = (0..n_chunks)
                .into_par_iter()
                .map(|chunk_idx| {
                    let start = chunk_idx * chunk_size;
                    let end = (start + chunk_size).min(n);
                    let mut centroid_dists = vec![0.0f32; n_clusters];
                    let mut local = Vec::with_capacity(end - start);

                    for i in start..end {
                        let vec_slice = &vecs[i * DIM..(i + 1) * DIM];
                        l2_distance_batch(vec_slice, &centroids, n_clusters, &mut centroid_dists);

                        let mut best_idx = 0;
                        let mut best_dist = f32::MAX;
                        for (ci, &d) in centroid_dists.iter().enumerate() {
                            if d < best_dist {
                                best_dist = d;
                                best_idx = ci;
                            }
                        }
                        local.push(best_idx);
                    }
                    local
                })
                .collect();

            let mut result = Vec::with_capacity(n);
            for chunk in chunks {
                result.extend(chunk);
            }
            result
        };

        let mut cluster_sizes = vec![0usize; n_clusters];
        for &ci in &all_assignments {
            cluster_sizes[ci] += 1;
        }

        let mut offsets = vec![0usize; n_clusters + 1];
        for i in 0..n_clusters {
            offsets[i + 1] = offsets[i] + cluster_sizes[i];
        }

        let mut cluster_vecs = vec![0.0f32; n * DIM];
        let mut cluster_ids_arr = vec![0u64; n];
        let mut write_pos = offsets[..n_clusters].to_vec();

        for i in 0..n {
            let ci = all_assignments[i];
            let pos = write_pos[ci];
            write_pos[ci] += 1;
            cluster_ids_arr[pos] = ids[i];
            let dst_start = pos * DIM;
            cluster_vecs[dst_start..dst_start + DIM]
                .copy_from_slice(&vecs[i * DIM..(i + 1) * DIM]);
        }

        *self.index.write() = Some(Box::new(IVFIndex {
            centroids,
            cluster_vectors: cluster_vecs,
            cluster_ids: cluster_ids_arr,
            cluster_offsets: offsets,
            n_clusters,
        }));

        drop(vecs);
        drop(ids);
        *self.pending_vectors.write() = Vec::new();
        *self.pending_ids.write() = Vec::new();

        eprintln!("Index built successfully");
    }

    fn run_kmeans(&self, vecs: &[f32], n: usize, k: usize) -> Vec<f32> {
        let sample_n = SAMPLE_SIZE.min(n);
        let step = n as f64 / sample_n as f64;
        let sample_indices: Vec<usize> = (0..sample_n)
            .map(|i| ((i as f64 * step) as usize).min(n - 1))
            .collect();

        let mut centroids = vec![0.0f32; k * DIM];
        let init_step = sample_n as f64 / k as f64;
        for i in 0..k {
            let src_idx = sample_indices[((i as f64 * init_step) as usize).min(sample_n - 1)];
            centroids[i * DIM..(i + 1) * DIM]
                .copy_from_slice(&vecs[src_idx * DIM..(src_idx + 1) * DIM]);
        }

        for _iter in 0..KMEANS_ITERS {
            let assignments: Vec<usize> = sample_indices
                .par_chunks(2500)
                .flat_map(|chunk| {
                    let mut centroid_dists = vec![0.0f32; k];
                    chunk
                        .iter()
                        .map(|&sample_idx| {
                            let vec_slice = &vecs[sample_idx * DIM..(sample_idx + 1) * DIM];
                            l2_distance_batch(vec_slice, &centroids, k, &mut centroid_dists);

                            let mut best_idx = 0;
                            let mut best_dist = f32::MAX;
                            for (ci, &d) in centroid_dists.iter().enumerate() {
                                if d < best_dist {
                                    best_dist = d;
                                    best_idx = ci;
                                }
                            }
                            best_idx
                        })
                        .collect::<Vec<_>>()
                })
                .collect();

            let mut new_centroids = vec![0.0f64; k * DIM];
            let mut counts = vec![0usize; k];

            for (si, &sample_idx) in sample_indices.iter().enumerate() {
                let ci = assignments[si];
                counts[ci] += 1;
                let src = &vecs[sample_idx * DIM..(sample_idx + 1) * DIM];
                let dst = &mut new_centroids[ci * DIM..(ci + 1) * DIM];
                for d in 0..DIM {
                    dst[d] += src[d] as f64;
                }
            }

            for ci in 0..k {
                if counts[ci] > 0 {
                    let c = counts[ci] as f64;
                    for d in 0..DIM {
                        centroids[ci * DIM + d] = (new_centroids[ci * DIM + d] / c) as f32;
                    }
                }
            }
        }

        centroids
    }
}

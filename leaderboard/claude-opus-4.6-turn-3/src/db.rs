use crate::api::*;
use crate::distance;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const DIM: usize = 128;
const NUM_CLUSTERS: usize = 8192;
const NPROBE: usize = 80;
const KMEANS_ITERS: usize = 8;
const KMEANS_SAMPLE_SIZE: usize = 160000;

struct Cluster {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

struct IvfIndex {
    centroids: Vec<f32>,
    clusters: Vec<Cluster>,
}

unsafe impl Send for IvfIndex {}
unsafe impl Sync for IvfIndex {}

pub struct VectorDB {
    all_ids: RwLock<Vec<u64>>,
    all_vectors: RwLock<Vec<f32>>,
    index: RwLock<Option<Box<IvfIndex>>>,
    index_built: AtomicBool,
    vector_count: AtomicUsize,
}

impl VectorDB {
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .ok();

        VectorDB {
            all_ids: RwLock::new(Vec::with_capacity(1_100_000)),
            all_vectors: RwLock::new(Vec::with_capacity(1_100_000 * DIM)),
            index: RwLock::new(None),
            index_built: AtomicBool::new(false),
            vector_count: AtomicUsize::new(0),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if self.index_built.load(Ordering::Relaxed) {
            let mut idx = self.index.write();
            if let Some(ref mut ivf) = *idx {
                let best = find_nearest_centroid_fast(&vector, &ivf.centroids, ivf.clusters.len());
                ivf.clusters[best].ids.push(id);
                ivf.clusters[best].vectors.extend_from_slice(&vector);
            }
        } else {
            let mut ids = self.all_ids.write();
            let mut vecs = self.all_vectors.write();
            ids.push(id);
            vecs.extend_from_slice(&vector);
        }
        self.vector_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();

        if self.index_built.load(Ordering::Relaxed) {
            let centroid_copy: Vec<f32> = {
                let idx = self.index.read();
                match idx.as_ref() {
                    Some(ivf) => ivf.centroids.clone(),
                    None => return count,
                }
            };

            let nc = centroid_copy.len() / DIM;
            let assignments: Vec<(usize, u64, Vec<f32>)> = vectors
                .into_iter()
                .map(|(id, vec)| {
                    let best = find_nearest_centroid_fast(&vec, &centroid_copy, nc);
                    (best, id, vec)
                })
                .collect();

            let mut idx = self.index.write();
            if let Some(ref mut ivf) = *idx {
                for (ci, id, vec) in assignments {
                    ivf.clusters[ci].ids.push(id);
                    ivf.clusters[ci].vectors.extend_from_slice(&vec);
                }
            }
            self.vector_count.fetch_add(count, Ordering::Relaxed);
        } else {
            {
                let mut ids = self.all_ids.write();
                let mut vecs = self.all_vectors.write();
                for (id, vector) in &vectors {
                    ids.push(*id);
                    vecs.extend_from_slice(vector);
                }
            }

            let total = self.vector_count.fetch_add(count, Ordering::Relaxed) + count;
            if total >= 900_000 && !self.index_built.load(Ordering::Relaxed) {
                self.build_index();
            }
        }

        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let top_k = top_k as usize;

        if self.index_built.load(Ordering::Acquire) {
            return self.search_ivf(vector, top_k);
        }

        self.search_brute_force(vector, top_k)
    }

    fn build_index(&self) {
        if self.index_built.swap(true, Ordering::SeqCst) {
            return;
        }

        let ids = self.all_ids.read();
        let vectors = self.all_vectors.read();
        let n = ids.len();

        if n == 0 {
            self.index_built.store(false, Ordering::SeqCst);
            return;
        }

        eprintln!("Building IVF index with {} vectors, {} clusters...", n, NUM_CLUSTERS);

        let centroids = kmeans(&vectors, n);

        let assignments: Vec<usize> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vec = &vectors[i * DIM..(i + 1) * DIM];
                find_nearest_centroid_fast(vec, &centroids, NUM_CLUSTERS)
            })
            .collect();

        let mut cluster_ids: Vec<Vec<u64>> = (0..NUM_CLUSTERS).map(|_| Vec::new()).collect();
        let mut cluster_vecs: Vec<Vec<f32>> = (0..NUM_CLUSTERS).map(|_| Vec::new()).collect();

        for i in 0..n {
            let ci = assignments[i];
            cluster_ids[ci].push(ids[i]);
            cluster_vecs[ci].extend_from_slice(&vectors[i * DIM..(i + 1) * DIM]);
        }

        let clusters: Vec<Cluster> = (0..NUM_CLUSTERS)
            .map(|c| {
                Cluster {
                    ids: std::mem::take(&mut cluster_ids[c]),
                    vectors: std::mem::take(&mut cluster_vecs[c]),
                }
            })
            .collect();

        eprintln!("IVF index built. Cluster sizes: min={}, max={}, avg={}",
            clusters.iter().map(|c| c.ids.len()).min().unwrap_or(0),
            clusters.iter().map(|c| c.ids.len()).max().unwrap_or(0),
            n / NUM_CLUSTERS,
        );

        *self.index.write() = Some(Box::new(IvfIndex { centroids, clusters }));
    }

    fn search_ivf(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let idx = self.index.read();
        let ivf = idx.as_ref().unwrap();
        let nc = ivf.clusters.len();

        // Compute distances to all centroids
        let mut centroid_dists = vec![0.0f32; nc];
        unsafe {
            distance::l2_distances_batch(query, &ivf.centroids, nc, &mut centroid_dists);
        }

        // Find NPROBE nearest centroids
        let mut indices: Vec<u16> = (0..nc as u16).collect();
        indices.select_nth_unstable_by(NPROBE - 1, |&a, &b| {
            unsafe {
                centroid_dists.get_unchecked(a as usize)
                    .partial_cmp(centroid_dists.get_unchecked(b as usize))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        let tk = top_k.min(64);
        let mut top_dists = [f32::MAX; 64];
        let mut top_ids = [0u64; 64];
        let mut heap_size = 0usize;
        let mut max_dist = f32::MAX;
        let mut max_idx = 0usize;

        for p in 0..NPROBE {
            let cluster_idx = unsafe { *indices.get_unchecked(p) } as usize;
            let cluster = &ivf.clusters[cluster_idx];
            let n = cluster.ids.len();
            if n == 0 {
                continue;
            }

            let vecs_ptr = cluster.vectors.as_ptr();
            let ids_ptr = cluster.ids.as_ptr();

            for j in 0..n {
                let vec_slice = unsafe {
                    std::slice::from_raw_parts(vecs_ptr.add(j * DIM), DIM)
                };
                let dist = unsafe { distance::l2_distance_fast(query, vec_slice) };

                if heap_size < tk {
                    top_dists[heap_size] = dist;
                    top_ids[heap_size] = unsafe { *ids_ptr.add(j) };
                    if dist > max_dist || heap_size == 0 {
                        max_dist = dist;
                        max_idx = heap_size;
                    }
                    heap_size += 1;
                } else if dist < max_dist {
                    top_dists[max_idx] = dist;
                    top_ids[max_idx] = unsafe { *ids_ptr.add(j) };
                    max_dist = top_dists[0];
                    max_idx = 0;
                    for k in 1..tk {
                        if unsafe { *top_dists.get_unchecked(k) } > max_dist {
                            max_dist = unsafe { *top_dists.get_unchecked(k) };
                            max_idx = k;
                        }
                    }
                }
            }
        }

        let mut result_pairs: Vec<(f32, u64)> = (0..heap_size)
            .map(|i| (top_dists[i], top_ids[i]))
            .collect();
        result_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        result_pairs
            .into_iter()
            .map(|(dist, id)| SearchResult {
                id,
                distance: dist as f64,
            })
            .collect()
    }

    fn search_brute_force(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let ids = self.all_ids.read();
        let vectors = self.all_vectors.read();
        let n = ids.len();

        if n == 0 {
            return Vec::new();
        }

        let tk = top_k.min(64);
        let mut top_dists = [f32::MAX; 64];
        let mut top_ids = [0u64; 64];
        let mut heap_size = 0usize;
        let mut max_dist = f32::MAX;
        let mut max_idx = 0usize;

        for i in 0..n {
            let vec_slice = &vectors[i * DIM..(i + 1) * DIM];
            let dist = unsafe { distance::l2_distance_fast(query, vec_slice) };

            if heap_size < tk {
                top_dists[heap_size] = dist;
                top_ids[heap_size] = ids[i];
                if dist > max_dist || heap_size == 0 {
                    max_dist = dist;
                    max_idx = heap_size;
                }
                heap_size += 1;
            } else if dist < max_dist {
                top_dists[max_idx] = dist;
                top_ids[max_idx] = ids[i];
                max_dist = top_dists[0];
                max_idx = 0;
                for k in 1..tk {
                    if top_dists[k] > max_dist {
                        max_dist = top_dists[k];
                        max_idx = k;
                    }
                }
            }
        }

        let mut result_pairs: Vec<(f32, u64)> = (0..heap_size)
            .map(|i| (top_dists[i], top_ids[i]))
            .collect();
        result_pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result_pairs.truncate(top_k);

        result_pairs
            .into_iter()
            .map(|(dist, id)| SearchResult {
                id,
                distance: dist as f64,
            })
            .collect()
    }
}

#[inline]
fn find_nearest_centroid_fast(vec: &[f32], centroids: &[f32], num_clusters: usize) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::MAX;
    let ptr = centroids.as_ptr();
    for i in 0..num_clusters {
        let centroid = unsafe {
            std::slice::from_raw_parts(ptr.add(i * DIM), DIM)
        };
        let dist = unsafe { distance::l2_distance_fast(vec, centroid) };
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx
}

fn kmeans(vectors: &[f32], n: usize) -> Vec<f32> {
    let sample_size = KMEANS_SAMPLE_SIZE.min(n);
    let step = if n > sample_size { n / sample_size } else { 1 };

    let sample_indices: Vec<usize> = (0..sample_size).map(|i| (i * step).min(n - 1)).collect();

    let mut centroids = vec![0.0f32; NUM_CLUSTERS * DIM];
    let init_step = if sample_size > NUM_CLUSTERS { sample_size / NUM_CLUSTERS } else { 1 };
    for c in 0..NUM_CLUSTERS {
        let idx = sample_indices[(c * init_step).min(sample_size - 1)];
        centroids[c * DIM..(c + 1) * DIM].copy_from_slice(&vectors[idx * DIM..(idx + 1) * DIM]);
    }

    for _iter in 0..KMEANS_ITERS {
        let assignments: Vec<usize> = sample_indices
            .par_iter()
            .map(|&idx| {
                let vec = &vectors[idx * DIM..(idx + 1) * DIM];
                find_nearest_centroid_fast(vec, &centroids, NUM_CLUSTERS)
            })
            .collect();

        let mut new_centroids = vec![0.0f64; NUM_CLUSTERS * DIM];
        let mut counts = vec![0usize; NUM_CLUSTERS];

        for (si, &cluster_idx) in assignments.iter().enumerate() {
            let idx = sample_indices[si];
            counts[cluster_idx] += 1;
            let base = cluster_idx * DIM;
            let vbase = idx * DIM;
            for d in 0..DIM {
                new_centroids[base + d] += vectors[vbase + d] as f64;
            }
        }

        for c in 0..NUM_CLUSTERS {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f64;
                for d in 0..DIM {
                    centroids[c * DIM + d] = (new_centroids[c * DIM + d] * inv) as f32;
                }
            }
        }
    }

    centroids
}

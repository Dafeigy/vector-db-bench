use crate::api::*;
use crate::distance::l2_distance_f32;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};

const DIM: usize = 128;
const NUM_CLUSTERS: usize = 4096;
const NPROBE: usize = 52;
const KMEANS_ITERS: usize = 14;
const KMEANS_SAMPLE_SIZE: usize = 150_000;

pub struct VectorDB {
    vectors: RwLock<Vec<f32>>,
    ids: RwLock<Vec<u64>>,
    index: RwLock<Option<Box<IVFIndex>>>,
    index_built: AtomicBool,
}

struct IVFIndex {
    centroids: Vec<f32>,       // NUM_CLUSTERS * DIM
    cluster_ids: Vec<Vec<u64>>,
    cluster_data: Vec<Vec<f32>>,  // each: len(ids) * DIM
}

unsafe impl Send for IVFIndex {}
unsafe impl Sync for IVFIndex {}

impl VectorDB {
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build_global()
            .ok();
        
        VectorDB {
            vectors: RwLock::new(Vec::new()),
            ids: RwLock::new(Vec::new()),
            index: RwLock::new(None),
            index_built: AtomicBool::new(false),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut vecs = self.vectors.write();
        let mut ids = self.ids.write();
        vecs.extend_from_slice(&vector);
        ids.push(id);
        self.index_built.store(false, Ordering::Release);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let n = vectors.len();
        {
            let mut vecs = self.vectors.write();
            let mut ids = self.ids.write();
            vecs.reserve(n * DIM);
            ids.reserve(n);
            for (id, vector) in &vectors {
                vecs.extend_from_slice(vector);
                ids.push(*id);
            }
        }
        self.index_built.store(false, Ordering::Release);
        n
    }

    fn ensure_index_built(&self) {
        if self.index_built.load(Ordering::Acquire) {
            return;
        }
        self.build_ivf_index();
    }

    fn build_ivf_index(&self) {
        let vecs = self.vectors.read();
        let ids_guard = self.ids.read();
        let n = ids_guard.len();
        
        if n < NUM_CLUSTERS * 2 {
            self.index_built.store(true, Ordering::Release);
            return;
        }
        
        let centroids = kmeans_clustering(&vecs, n);
        
        let assignments: Vec<u16> = (0..n).into_par_iter().map(|i| {
            find_nearest_centroid(&vecs[i * DIM..(i + 1) * DIM], &centroids, NUM_CLUSTERS) as u16
        }).collect();
        
        let mut sizes = vec![0usize; NUM_CLUSTERS];
        for &a in &assignments {
            sizes[a as usize] += 1;
        }
        
        let mut cluster_ids: Vec<Vec<u64>> = sizes.iter().map(|&s| Vec::with_capacity(s)).collect();
        let mut cluster_data: Vec<Vec<f32>> = sizes.iter().map(|&s| Vec::with_capacity(s * DIM)).collect();
        
        for (i, &cluster_id) in assignments.iter().enumerate() {
            let c = cluster_id as usize;
            cluster_ids[c].push(ids_guard[i]);
            cluster_data[c].extend_from_slice(&vecs[i * DIM..(i + 1) * DIM]);
        }
        
        *self.index.write() = Some(Box::new(IVFIndex { centroids, cluster_ids, cluster_data }));
        self.index_built.store(true, Ordering::Release);
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let top_k = top_k as usize;
        self.ensure_index_built();
        
        let index_guard = self.index.read();
        if let Some(ref index) = *index_guard {
            return ivf_search(vector, top_k, index);
        }
        
        drop(index_guard);
        self.brute_force_search(vector, top_k)
    }
    
    fn brute_force_search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        let vecs = self.vectors.read();
        let ids_guard = self.ids.read();
        let n = ids_guard.len();
        if n == 0 { return Vec::new(); }
        
        let mut heap = TopK::new(top_k);
        for i in 0..n {
            let dist = l2_distance_f32(query, &vecs[i * DIM..(i + 1) * DIM]);
            heap.push(dist, ids_guard[i]);
        }
        heap.into_results()
    }
}

#[inline(never)]
fn ivf_search(query: &[f32], top_k: usize, index: &IVFIndex) -> Vec<SearchResult> {
    let centroids = &index.centroids;
    let mut centroid_dists: Vec<(u16, f32)> = Vec::with_capacity(NUM_CLUSTERS);
    
    for c in 0..NUM_CLUSTERS {
        let dist = l2_distance_f32(query, unsafe {
            centroids.get_unchecked(c * DIM..(c + 1) * DIM)
        });
        centroid_dists.push((c as u16, dist));
    }
    
    centroid_dists.select_nth_unstable_by(NPROBE - 1, |a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let mut heap = TopK::new(top_k);
    
    for i in 0..NPROBE {
        let cluster_id = unsafe { centroid_dists.get_unchecked(i) }.0 as usize;
        let data = unsafe { index.cluster_data.get_unchecked(cluster_id) };
        let ids = unsafe { index.cluster_ids.get_unchecked(cluster_id) };
        let count = ids.len();
        if count == 0 { continue; }
        
        let data_ptr = data.as_ptr();
        
        for j in 0..count {
            let dist = l2_distance_f32(query, unsafe {
                std::slice::from_raw_parts(data_ptr.add(j * DIM), DIM)
            });
            if dist < heap.worst || heap.len < heap.k {
                heap.push_unchecked(dist, unsafe { *ids.get_unchecked(j) });
            }
        }
    }
    
    heap.into_results()
}

struct TopK {
    items: [(f32, u64); 16],
    k: usize,
    len: usize,
    worst: f32,
    worst_idx: usize,
}

impl TopK {
    #[inline]
    fn new(k: usize) -> Self {
        TopK {
            items: [(0.0, 0); 16],
            k: k.min(16),
            len: 0,
            worst: f32::INFINITY,
            worst_idx: 0,
        }
    }
    
    #[inline(always)]
    fn push(&mut self, dist: f32, id: u64) {
        if self.len < self.k {
            self.items[self.len] = (dist, id);
            self.len += 1;
            if self.len == self.k {
                self.recompute_worst();
            }
        } else if dist < self.worst {
            self.items[self.worst_idx] = (dist, id);
            self.recompute_worst();
        }
    }
    
    #[inline(always)]
    fn push_unchecked(&mut self, dist: f32, id: u64) {
        if self.len < self.k {
            self.items[self.len] = (dist, id);
            self.len += 1;
            if self.len == self.k {
                self.recompute_worst();
            }
        } else {
            self.items[self.worst_idx] = (dist, id);
            self.recompute_worst();
        }
    }
    
    #[inline(always)]
    fn recompute_worst(&mut self) {
        let mut widx = 0;
        let mut wdist = self.items[0].0;
        for i in 1..self.len {
            let d = self.items[i].0;
            if d > wdist {
                wdist = d;
                widx = i;
            }
        }
        self.worst = wdist;
        self.worst_idx = widx;
    }
    
    fn into_results(mut self) -> Vec<SearchResult> {
        let slice = &mut self.items[..self.len];
        slice.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        slice.iter().map(|&(dist, id)| SearchResult {
            id,
            distance: dist as f64,
        }).collect()
    }
}

#[inline(always)]
fn find_nearest_centroid(vec: &[f32], centroids: &[f32], num_clusters: usize) -> usize {
    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;
    for c in 0..num_clusters {
        let dist = l2_distance_f32(vec, &centroids[c * DIM..(c + 1) * DIM]);
        if dist < best_dist {
            best_dist = dist;
            best_idx = c;
        }
    }
    best_idx
}

fn kmeans_clustering(vecs: &[f32], n: usize) -> Vec<f32> {
    let sample_size = n.min(KMEANS_SAMPLE_SIZE);
    let step = if sample_size < n { n / sample_size } else { 1 };
    
    let mut sample_data = vec![0.0f32; sample_size * DIM];
    for si in 0..sample_size {
        let idx = (si * step).min(n - 1);
        sample_data[si * DIM..(si + 1) * DIM].copy_from_slice(&vecs[idx * DIM..(idx + 1) * DIM]);
    }
    
    let mut centroids = vec![0.0f32; NUM_CLUSTERS * DIM];
    let centroid_step = sample_size / NUM_CLUSTERS;
    for c in 0..NUM_CLUSTERS {
        let src = (c * centroid_step).min(sample_size - 1);
        centroids[c * DIM..(c + 1) * DIM].copy_from_slice(&sample_data[src * DIM..(src + 1) * DIM]);
    }
    
    for _iter in 0..KMEANS_ITERS {
        let assignments: Vec<usize> = (0..sample_size).into_par_iter().map(|si| {
            find_nearest_centroid(&sample_data[si * DIM..(si + 1) * DIM], &centroids, NUM_CLUSTERS)
        }).collect();
        
        let mut new_centroids = vec![0.0f64; NUM_CLUSTERS * DIM];
        let mut counts = vec![0u32; NUM_CLUSTERS];
        
        for (si, &cluster_id) in assignments.iter().enumerate() {
            counts[cluster_id] += 1;
            let src_offset = si * DIM;
            for d in 0..DIM {
                new_centroids[cluster_id * DIM + d] += sample_data[src_offset + d] as f64;
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

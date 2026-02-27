use crate::api::SearchResult;
use crate::distance::l2_distance_f32;
use parking_lot::{Condvar, Mutex, RwLock};
use rayon::prelude::*;
use std::sync::Arc;

const DIM: usize = 128;
const NUM_CENTROIDS: usize = 1024;
const PROBE: usize = 32;
const KMEANS_ITERS: usize = 5;
const KMEANS_SAMPLE: usize = 50_000;

pub struct VectorDB {
    shared: Arc<Shared>,
}

struct Shared {
    inner: RwLock<Inner>,
    building: Mutex<bool>,
    build_done: Condvar,
}

struct Inner {
    data: Vec<f32>,
    ids: Vec<u64>,
    ivf: Option<IvfIndex>,
}

struct IvfIndex {
    centroids: Vec<f32>,
    cells: Vec<Vec<u32>>,
}

impl IvfIndex {
    #[inline(always)]
    fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * DIM..(c + 1) * DIM]
    }
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            shared: Arc::new(Shared {
                inner: RwLock::new(Inner {
                    data: Vec::new(),
                    ids: Vec::new(),
                    ivf: None,
                }),
                building: Mutex::new(false),
                build_done: Condvar::new(),
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut inner = self.shared.inner.write();
        push_vector(&mut inner.data, &vector);
        inner.ids.push(id);
        inner.ivf = None;
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();

        // Insert data
        {
            let mut inner = self.shared.inner.write();
            inner.data.reserve(count * DIM);
            inner.ids.reserve(count);
            for (id, v) in &vectors {
                push_vector(&mut inner.data, v);
                inner.ids.push(*id);
            }
            inner.ivf = None;
        }

        // Mark building
        {
            let mut b = self.shared.building.lock();
            *b = true;
        }

        // Spawn background thread to build IVF index
        let shared = Arc::clone(&self.shared);
        std::thread::spawn(move || {
            run_build(&shared);
        });

        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        // Wait for any in-progress index build
        {
            let mut b = self.shared.building.lock();
            while *b {
                self.shared.build_done.wait(&mut b);
            }
        }

        let inner = self.shared.inner.read();
        let n = inner.ids.len();
        if n == 0 { return vec![]; }
        let top_k = top_k as usize;

        if let Some(ref ivf) = inner.ivf {
            search_ivf(&inner, ivf, vector, top_k)
        } else {
            search_brute(&inner, vector, top_k)
        }
    }
}

fn run_build(shared: &Arc<Shared>) {
    // Snapshot data
    let (data_snap, n) = {
        let inner = shared.inner.read();
        (inner.data.clone(), inner.ids.len())
    };

    let nc = NUM_CENTROIDS;
    if n >= nc * 4 {
        let ivf = build_ivf(data_snap, n, nc);
        let mut inner = shared.inner.write();
        if inner.ids.len() == n {
            inner.ivf = Some(ivf);
        }
    }

    // Signal done
    let mut b = shared.building.lock();
    *b = false;
    shared.build_done.notify_all();
}

fn build_ivf(data_snap: Vec<f32>, n: usize, nc: usize) -> IvfIndex {
    // Build contiguous sample
    let sample_n = KMEANS_SAMPLE.min(n);
    let sample_step = n / sample_n;
    let mut sample_data: Vec<f32> = Vec::with_capacity(sample_n * DIM);
    for i in 0..sample_n {
        let di = i * sample_step;
        sample_data.extend_from_slice(&data_snap[di * DIM..(di + 1) * DIM]);
    }

    // Init centroids
    let init_step = sample_n / nc;
    let mut centroids: Vec<f32> = Vec::with_capacity(nc * DIM);
    for ci in 0..nc {
        let si = ci * init_step;
        centroids.extend_from_slice(&sample_data[si * DIM..(si + 1) * DIM]);
    }

    // K-means
    let mut sample_asgn = vec![0u32; sample_n];
    for _iter in 0..KMEANS_ITERS {
        let new_asgn: Vec<u32> = (0..sample_n)
            .into_par_iter()
            .map(|i| nearest_centroid(&centroids, &sample_data[i * DIM..(i + 1) * DIM]) as u32)
            .collect();
        if new_asgn == sample_asgn { break; }
        sample_asgn = new_asgn;

        let mut sums = vec![0.0f32; nc * DIM];
        let mut cnts = vec![0u32; nc];
        for i in 0..sample_n {
            let c = sample_asgn[i] as usize;
            let v = &sample_data[i * DIM..(i + 1) * DIM];
            let s = &mut sums[c * DIM..(c + 1) * DIM];
            for d in 0..DIM { s[d] += v[d]; }
            cnts[c] += 1;
        }
        for c in 0..nc {
            if cnts[c] > 0 {
                let cf = cnts[c] as f32;
                let cs = &mut centroids[c * DIM..(c + 1) * DIM];
                let ss = &sums[c * DIM..(c + 1) * DIM];
                for d in 0..DIM { cs[d] = ss[d] / cf; }
            }
        }
    }
    drop(sample_data);

    // Assign all
    let all_asgn: Vec<u32> = (0..n)
        .into_par_iter()
        .map(|i| nearest_centroid(&centroids, &data_snap[i * DIM..(i + 1) * DIM]) as u32)
        .collect();
    drop(data_snap);

    // Build cells
    let mut cell_counts = vec![0u32; nc];
    for &a in &all_asgn { cell_counts[a as usize] += 1; }
    let mut cells: Vec<Vec<u32>> = cell_counts
        .iter()
        .map(|&cnt| Vec::with_capacity(cnt as usize))
        .collect();
    for (i, &a) in all_asgn.iter().enumerate() {
        cells[a as usize].push(i as u32);
    }

    IvfIndex { centroids, cells }
}

fn search_ivf(inner: &Inner, ivf: &IvfIndex, vector: &[f32], top_k: usize) -> Vec<SearchResult> {
    let probe = PROBE.min(NUM_CENTROIDS);
    let data = &inner.data;
    let ids = &inner.ids;

    let mut cdists: Vec<(f32, usize)> = (0..NUM_CENTROIDS)
        .map(|c| (l2_distance_f32(vector, ivf.centroid(c)), c))
        .collect();
    cdists.select_nth_unstable_by(probe - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
    cdists.truncate(probe);

    let total: usize = cdists.iter().map(|(_, c)| ivf.cells[*c].len()).sum();
    let mut results: Vec<(f32, u64)> = Vec::with_capacity(total);
    for (_, c) in &cdists {
        for &vi in &ivf.cells[*c] {
            let vi = vi as usize;
            let dist = l2_distance_f32(vector, &data[vi * DIM..(vi + 1) * DIM]);
            results.push((dist, ids[vi]));
        }
    }

    if results.len() > top_k {
        results.select_nth_unstable_by(top_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(top_k);
    }
    results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results.into_iter().map(|(d, id)| SearchResult { id, distance: d as f64 }).collect()
}

fn search_brute(inner: &Inner, vector: &[f32], top_k: usize) -> Vec<SearchResult> {
    let n = inner.ids.len();
    let data = &inner.data;
    let ids = &inner.ids;
    let mut results: Vec<(f32, u64)> = (0..n)
        .into_par_iter()
        .map(|i| (l2_distance_f32(vector, &data[i * DIM..(i + 1) * DIM]), ids[i]))
        .collect();
    if results.len() > top_k {
        results.select_nth_unstable_by(top_k - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
        results.truncate(top_k);
    }
    results.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    results.into_iter().map(|(d, id)| SearchResult { id, distance: d as f64 }).collect()
}

#[inline(always)]
fn push_vector(data: &mut Vec<f32>, v: &[f32]) {
    if v.len() >= DIM {
        data.extend_from_slice(&v[..DIM]);
    } else {
        data.extend_from_slice(v);
        for _ in v.len()..DIM { data.push(0.0); }
    }
}

#[inline(always)]
fn nearest_centroid(centroids: &[f32], v: &[f32]) -> usize {
    let mut best = f32::MAX;
    let mut best_idx = 0usize;
    for c in 0..NUM_CENTROIDS {
        let d = l2_distance_f32(v, &centroids[c * DIM..(c + 1) * DIM]);
        if d < best { best = d; best_idx = c; }
    }
    best_idx
}

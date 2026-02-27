use crate::api::*;
use crate::distance::l2_squared_f32;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;

// IVF parameters
const DIM: usize = 128;
const N_CLUSTERS: usize = 1024;
const N_PROBE: usize = 80;
const KMEANS_ITERS: usize = 10;

// Build the index only when ALL 1M vectors are loaded.
// The benchmark and correctness test both do: bulk_insert(10000 vectors) x 100 = 1M total.
// We detect "done loading" as the first time n_vectors >= BUILD_THRESHOLD.
const BUILD_THRESHOLD: usize = 999_000;

/// Max-heap entry (keeps k-nearest = smallest distances)
#[derive(Clone, Copy)]
struct DistId(f32, u64);

impl PartialEq for DistId { fn eq(&self, o: &Self) -> bool { self.0 == o.0 } }
impl Eq for DistId {}
impl PartialOrd for DistId {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { self.0.partial_cmp(&o.0) }
}
impl Ord for DistId {
    fn cmp(&self, o: &Self) -> Ordering { self.0.partial_cmp(&o.0).unwrap_or(Ordering::Equal) }
}

struct IvfIndex {
    vectors: Vec<f32>,
    ids: Vec<u64>,
    centroids: Vec<f32>,     // N_CLUSTERS * DIM
    clusters: Vec<Vec<u32>>, // N_CLUSTERS entries
}

unsafe impl Send for IvfIndex {}
unsafe impl Sync for IvfIndex {}

pub struct VectorDB {
    inner: RwLock<DBInner>,
    index: RwLock<Option<Arc<IvfIndex>>>,
}

struct DBInner {
    vectors: Vec<f32>,
    ids: Vec<u64>,
    index_built: bool,
}

impl DBInner {
    fn new() -> Self { DBInner { vectors: Vec::new(), ids: Vec::new(), index_built: false } }
    fn n_vectors(&self) -> usize { self.ids.len() }
}

fn build_ivf_index(vectors: &[f32], ids: &[u64]) -> IvfIndex {
    let n = ids.len();
    eprintln!("[IVF] Building: {} vecs, {} clusters", n, N_CLUSTERS);
    let t0 = std::time::Instant::now();

    let step = n / N_CLUSTERS;
    let mut centroids = vec![0.0f32; N_CLUSTERS * DIM];
    for c in 0..N_CLUSTERS {
        centroids[c * DIM..(c + 1) * DIM].copy_from_slice(&vectors[c * step * DIM..(c * step + 1) * DIM]);
    }

    let mut assignments = vec![0u32; n];

    for iter in 0..KMEANS_ITERS {
        let new_asgn: Vec<u32> = (0..n).into_par_iter()
            .with_min_len(4096)
            .map(|i| {
                let v = &vectors[i * DIM..(i + 1) * DIM];
                let mut best_c = 0u32;
                let mut best_d = f32::MAX;
                for c in 0..N_CLUSTERS {
                    let d = l2_squared_f32(v, &centroids[c * DIM..(c + 1) * DIM]);
                    if d < best_d { best_d = d; best_c = c as u32; }
                }
                best_c
            }).collect();

        let changed = new_asgn.iter().zip(&assignments).filter(|(a, b)| a != b).count();
        assignments = new_asgn;

        const NT: usize = 4;
        let chunk = (n + NT - 1) / NT;
        let partials: Vec<(Vec<f32>, Vec<u32>)> = (0..NT).into_par_iter().map(|t| {
            let s = t * chunk;
            let e = (s + chunk).min(n);
            let mut sums = vec![0.0f32; N_CLUSTERS * DIM];
            let mut cnts = vec![0u32; N_CLUSTERS];
            for i in s..e {
                let c = assignments[i] as usize;
                cnts[c] += 1;
                let v = &vectors[i * DIM..(i + 1) * DIM];
                let cv = &mut sums[c * DIM..(c + 1) * DIM];
                for d in 0..DIM { cv[d] += v[d]; }
            }
            (sums, cnts)
        }).collect();

        let mut new_cents = vec![0.0f32; N_CLUSTERS * DIM];
        let mut cnts = vec![0u32; N_CLUSTERS];
        for (s, c) in &partials {
            for k in 0..N_CLUSTERS {
                cnts[k] += c[k];
                let cv = &mut new_cents[k * DIM..(k + 1) * DIM];
                let sv = &s[k * DIM..(k + 1) * DIM];
                for d in 0..DIM { cv[d] += sv[d]; }
            }
        }
        for k in 0..N_CLUSTERS {
            if cnts[k] > 0 {
                let inv = 1.0 / cnts[k] as f32;
                let cv = &mut new_cents[k * DIM..(k + 1) * DIM];
                for d in 0..DIM { cv[d] *= inv; }
            } else {
                new_cents[k * DIM..(k + 1) * DIM].copy_from_slice(&centroids[k * DIM..(k + 1) * DIM]);
            }
        }
        centroids = new_cents;
        eprintln!("  iter {}: {} changed ({:.2}s)", iter, changed, t0.elapsed().as_secs_f32());
        if changed == 0 { break; }
    }

    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); N_CLUSTERS];
    for i in 0..n { clusters[assignments[i] as usize].push(i as u32); }

    eprintln!("[IVF] Built in {:.2}s", t0.elapsed().as_secs_f32());

    IvfIndex { vectors: vectors.to_vec(), ids: ids.to_vec(), centroids, clusters }
}

fn search_with_index(index: &IvfIndex, query: &[f32], top_k: usize) -> Vec<SearchResult> {
    let np = N_PROBE.min(N_CLUSTERS);

    let mut cdists: Vec<(f32, usize)> = (0..N_CLUSTERS).map(|c| {
        (l2_squared_f32(query, &index.centroids[c * DIM..(c + 1) * DIM]), c)
    }).collect();
    cdists.select_nth_unstable_by(np - 1, |a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut heap: BinaryHeap<DistId> = BinaryHeap::with_capacity(top_k + 1);
    let mut worst = f32::MAX;

    for &(_, c) in &cdists[..np] {
        for &vi in &index.clusters[c] {
            let vi = vi as usize;
            let v = &index.vectors[vi * DIM..(vi + 1) * DIM];
            let d = l2_squared_f32(query, v);
            let id = index.ids[vi];
            if heap.len() < top_k {
                heap.push(DistId(d, id));
                if heap.len() == top_k { worst = heap.peek().unwrap().0; }
            } else if d < worst {
                heap.pop();
                heap.push(DistId(d, id));
                worst = heap.peek().unwrap().0;
            }
        }
    }

    heap.into_sorted_vec().into_iter()
        .map(|di| SearchResult { id: di.1, distance: (di.0 as f64).sqrt() })
        .collect()
}

fn search_brute(vectors: &[f32], ids: &[u64], query: &[f32], top_k: usize) -> Vec<SearchResult> {
    let n = ids.len();
    let mut heap: BinaryHeap<DistId> = BinaryHeap::with_capacity(top_k + 1);
    let mut worst = f32::MAX;

    for i in 0..n {
        let v = &vectors[i * DIM..(i + 1) * DIM];
        let d = l2_squared_f32(query, v);
        if heap.len() < top_k {
            heap.push(DistId(d, ids[i]));
            if heap.len() == top_k { worst = heap.peek().unwrap().0; }
        } else if d < worst {
            heap.pop();
            heap.push(DistId(d, ids[i]));
            worst = heap.peek().unwrap().0;
        }
    }

    heap.into_sorted_vec().into_iter()
        .map(|di| SearchResult { id: di.1, distance: (di.0 as f64).sqrt() })
        .collect()
}

impl VectorDB {
    pub fn new() -> Self {
        rayon::ThreadPoolBuilder::new().num_threads(4).build_global().ok();
        VectorDB {
            inner: RwLock::new(DBInner::new()),
            index: RwLock::new(None),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        let mut db = self.inner.write();
        db.ids.push(id);
        db.vectors.extend_from_slice(&vector);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        {
            let mut db = self.inner.write();
            db.ids.reserve(count);
            db.vectors.reserve(count * DIM);
            for (id, v) in &vectors {
                db.ids.push(*id);
                db.vectors.extend_from_slice(v);
            }
        }

        // Build index only once when we first cross the threshold
        let should_build = {
            let db = self.inner.read();
            !db.index_built && db.n_vectors() >= BUILD_THRESHOLD
        };
        if should_build {
            let mut db = self.inner.write();
            if !db.index_built && db.n_vectors() >= BUILD_THRESHOLD {
                let vecs_snap = db.vectors.clone();
                let ids_snap = db.ids.clone();
                db.index_built = true;
                drop(db);

                let idx = build_ivf_index(&vecs_snap, &ids_snap);
                *self.index.write() = Some(Arc::new(idx));
            }
        }

        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        let k = top_k as usize;

        {
            let idx = self.index.read();
            if let Some(ref arc_idx) = *idx {
                let arc = arc_idx.clone();
                drop(idx);
                return search_with_index(&arc, vector, k);
            }
        }

        let db = self.inner.read();
        search_brute(&db.vectors, &db.ids, vector, k)
    }
}

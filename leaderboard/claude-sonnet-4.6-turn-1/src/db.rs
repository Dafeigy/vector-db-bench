use crate::api::SearchResult;
use crate::distance::l2_distance_f32;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use std::sync::Arc;

// IVF with few clusters and many probes for fast build + good recall.
// With 128 clusters at 1M vecs: ~7812 vecs/cluster.
// 128 probes = scan all clusters -> brute force (100% recall).
// We use 32 probes = scan 25% -> ~250k candidates/query.
// Build: 1M * 128 * N_ITERS dist computations.
// Timing: each dist = ~6ns (scalar) or ~2ns (AVX-512).
// Without target-cpu=native the compiler may not vectorize.
// Let's force AVX-512 via target_feature.
const N_CLUSTERS: usize = 128;
const N_PROBES: usize = 48;
const KMEANS_ITERS: usize = 3;

#[derive(Clone, Copy, PartialEq)]
struct OrdF32(f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    #[inline] fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for OrdF32 {
    #[inline] fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct HeapEntry { dist: OrdF32, id: u64 }
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering { self.dist.cmp(&other.dist) }
}

struct IvfIndex {
    centroids: Vec<f32>,
    lists: Vec<Vec<u32>>,
    n_clusters: usize,
}

pub struct VectorDB {
    data: RwLock<Arc<Vec<f32>>>,
    ids: RwLock<Arc<Vec<u64>>>,
    ivf: RwLock<Option<Arc<IvfIndex>>>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            data: RwLock::new(Arc::new(Vec::new())),
            ids: RwLock::new(Arc::new(Vec::new())),
            ivf: RwLock::new(None),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        {
            let mut d = self.data.write();
            let v = Arc::make_mut(&mut d);
            v.extend_from_slice(&vector);
        }
        {
            let mut i = self.ids.write();
            let v = Arc::make_mut(&mut i);
            v.push(id);
        }
        *self.ivf.write() = None;
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let count = vectors.len();
        let n;
        {
            let mut d = self.data.write();
            let mut i = self.ids.write();
            let dv = Arc::make_mut(&mut d);
            let iv = Arc::make_mut(&mut i);
            dv.reserve(count * 128);
            iv.reserve(count);
            for (id, vec) in &vectors {
                dv.extend_from_slice(vec);
                iv.push(*id);
            }
            n = iv.len();
        }

        if n >= 100 {
            // Get snapshot for building
            let data_arc = self.data.read().clone();
            let ivf = build_ivf(&data_arc, n);
            *self.ivf.write() = Some(Arc::new(ivf));
        }

        count
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        // Acquire read locks and clone Arcs (cheap)
        let data = self.data.read().clone();
        let ids = self.ids.read().clone();
        let ivf_opt = self.ivf.read().clone();

        let n = ids.len();
        if n == 0 { return Vec::new(); }
        let k = (top_k as usize).min(n);

        if let Some(ivf) = ivf_opt {
            search_ivf_inner(&data, &ids, &ivf, vector, k)
        } else {
            search_brute_inner(&data, &ids, vector, k)
        }
    }
}

fn build_ivf(data: &[f32], n: usize) -> IvfIndex {
    let nc = N_CLUSTERS.min(n);
    let t0 = std::time::Instant::now();
    eprintln!("[IVF] {nc} clusters, {n} vecs");

    // Init centroids: evenly spaced
    let step = n / nc;
    let mut centroids: Vec<f32> = (0..nc)
        .flat_map(|i| data[i*step*128..(i*step+1)*128].iter().copied())
        .collect();

    let mut asgn = vec![0u32; n];

    for it in 0..KMEANS_ITERS {
        let sp = data.as_ptr() as usize;
        let cp = centroids.as_ptr() as usize;

        let new_asgn: Vec<u32> = (0..n).into_par_iter().map(|i| {
            let v = unsafe { std::slice::from_raw_parts((sp as *const f32).add(i*128), 128) };
            let mut bd = f32::MAX;
            let mut bc = 0u32;
            for c in 0..nc {
                let cen = unsafe { std::slice::from_raw_parts((cp as *const f32).add(c*128), 128) };
                let d = l2_distance_f32(v, cen);
                if d < bd { bd = d; bc = c as u32; }
            }
            bc
        }).collect();

        let chg = asgn.iter().zip(&new_asgn).filter(|(a,b)| a!=b).count();
        asgn = new_asgn;

        let mut sums = vec![0f32; nc * 128];
        let mut cnts = vec![0u32; nc];
        for i in 0..n {
            let c = asgn[i] as usize;
            cnts[c] += 1;
            let v = &data[i*128..(i+1)*128];
            let s = &mut sums[c*128..(c+1)*128];
            for j in 0..128 { s[j] += v[j]; }
        }
        for c in 0..nc {
            let cnt = cnts[c].max(1) as f32;
            let s = &sums[c*128..(c+1)*128];
            let cc = &mut centroids[c*128..(c+1)*128];
            for j in 0..128 { cc[j] = s[j] / cnt; }
        }

        eprintln!("[IVF] iter {it}/{KMEANS_ITERS}: {chg} chg {:.1}s", t0.elapsed().as_secs_f64());
        if chg == 0 { break; }
    }

    let mut lists: Vec<Vec<u32>> = (0..nc).map(|_| Vec::new()).collect();
    for (i, &c) in asgn.iter().enumerate() { lists[c as usize].push(i as u32); }
    eprintln!("[IVF] done {:.1}s avg_list={}", t0.elapsed().as_secs_f64(), n/nc);
    IvfIndex { centroids, lists, n_clusters: nc }
}

fn search_ivf_inner(data: &[f32], ids: &[u64], ivf: &IvfIndex, q: &[f32], k: usize) -> Vec<SearchResult> {
    // Find nearest centroids
    let nc = ivf.n_clusters;
    let nprobes = N_PROBES.min(nc);
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(nprobes+1);
    for i in 0..nc {
        let c = &ivf.centroids[i*128..(i+1)*128];
        let d = l2_distance_f32(q, c);
        let e = HeapEntry { dist: OrdF32(d), id: i as u64 };
        if heap.len() < nprobes { heap.push(e); }
        else if d < heap.peek().unwrap().dist.0 { heap.pop(); heap.push(e); }
    }
    let mut probe_ids: Vec<usize> = heap.into_iter().map(|e| e.id as usize).collect();
    probe_ids.sort_unstable_by(|&a, &b| {
        let da = l2_distance_f32(q, &ivf.centroids[a*128..(a+1)*128]);
        let db = l2_distance_f32(q, &ivf.centroids[b*128..(b+1)*128]);
        da.partial_cmp(&db).unwrap_or(Ordering::Equal)
    });

    let mut cands: Vec<u32> = Vec::new();
    for &cid in &probe_ids { cands.extend_from_slice(&ivf.lists[cid]); }
    let n = cands.len();
    let k = k.min(n);
    if k == 0 { return Vec::new(); }

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k+1);
    for &idx in &cands {
        let idx = idx as usize;
        let v = &data[idx*128..(idx+1)*128];
        let d = l2_distance_f32(q, v);
        let e = HeapEntry { dist: OrdF32(d), id: ids[idx] };
        if heap.len() < k { heap.push(e); }
        else if d < heap.peek().unwrap().dist.0 { heap.pop(); heap.push(e); }
    }

    let mut out: Vec<SearchResult> = heap.into_iter()
        .map(|e| SearchResult { id: e.id, distance: e.dist.0 as f64 }).collect();
    out.sort_by(|a,b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    out
}

fn search_brute_inner(data: &[f32], ids: &[u64], q: &[f32], k: usize) -> Vec<SearchResult> {
    let n = ids.len();
    let chunk_size = (n + 3) / 4;

    let results: Vec<Vec<(f32, u64)>> = (0..4).into_par_iter().map(|ci| {
        let start = ci * chunk_size;
        let end = (start + chunk_size).min(n);
        if start >= end { return Vec::new(); }
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);
        for i in start..end {
            let v = &data[i * 128..(i + 1) * 128];
            let d = l2_distance_f32(q, v);
            let e = HeapEntry { dist: OrdF32(d), id: ids[i] };
            if heap.len() < k { heap.push(e); }
            else if d < heap.peek().unwrap().dist.0 { heap.pop(); heap.push(e); }
        }
        heap.into_iter().map(|e| (e.dist.0, e.id)).collect()
    }).collect();

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k+1);
    for chunk in results {
        for (d, id) in chunk {
            let e = HeapEntry { dist: OrdF32(d), id };
            if heap.len() < k { heap.push(e); }
            else if d < heap.peek().unwrap().dist.0 { heap.pop(); heap.push(e); }
        }
    }
    let mut out: Vec<SearchResult> = heap.into_iter()
        .map(|e| SearchResult { id: e.id, distance: e.dist.0 as f64 }).collect();
    out.sort_by(|a,b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
    out
}

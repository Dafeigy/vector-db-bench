use crate::api::SearchResult;
use crate::distance::l2_distance_squared;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::{
    atomic::{AtomicBool, Ordering as AtomicOrdering},
    Mutex, RwLock,
};

const DIM: usize = 128;
const NLIST: usize = 64;
const NPROBE: usize = 6;
const TRAINING_SAMPLE: usize = 2048;
const KMEANS_ITERS: usize = 4;

struct Storage {
    ids: Vec<u64>,
    vectors: Vec<f32>,
}

struct IvfIndex {
    centroids: Vec<f32>,
    lists: Vec<Vec<usize>>,
    nprobe: usize,
}

pub struct VectorDB {
    inner: RwLock<Storage>,
    index: RwLock<Option<IvfIndex>>,
    index_dirty: AtomicBool,
    build_lock: Mutex<()>,
}

#[derive(Debug, Clone, Copy)]
struct HeapItem {
    dist_sq: f64,
    id: u64,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq == other.dist_sq && self.id == other.id
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Storage {
                ids: Vec::new(),
                vectors: Vec::new(),
            }),
            index: RwLock::new(None),
            index_dirty: AtomicBool::new(false),
            build_lock: Mutex::new(()),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM {
            return;
        }

        let mut guard = self.inner.write().unwrap();
        guard.ids.push(id);
        guard.vectors.extend_from_slice(&vector);
        self.index_dirty.store(true, AtomicOrdering::Release);
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut guard = self.inner.write().unwrap();

        let valid_count = vectors.iter().filter(|(_, v)| v.len() == DIM).count();
        guard.ids.reserve(valid_count);
        guard.vectors.reserve(valid_count * DIM);

        let mut inserted = 0usize;
        for (id, vector) in vectors {
            if vector.len() != DIM {
                continue;
            }
            guard.ids.push(id);
            guard.vectors.extend_from_slice(&vector);
            inserted += 1;
        }

        self.index_dirty.store(true, AtomicOrdering::Release);
        drop(guard);
        self.ensure_index();

        inserted
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if vector.len() != DIM || top_k == 0 {
            return Vec::new();
        }

        self.ensure_index();

        let storage = self.inner.read().unwrap();
        let n = storage.ids.len();
        if n == 0 {
            return Vec::new();
        }

        let k = (top_k as usize).min(n);
        let index_guard = self.index.read().unwrap();

        if let Some(index) = index_guard.as_ref() {
            self.search_ivf(&storage, index, vector, k)
        } else {
            self.search_bruteforce(&storage, vector, k)
        }
    }

    fn ensure_index(&self) {
        if !self.index_dirty.load(AtomicOrdering::Acquire) {
            return;
        }

        let _build_guard = self.build_lock.lock().unwrap();
        if !self.index_dirty.load(AtomicOrdering::Acquire) {
            return;
        }

        let storage = self.inner.read().unwrap();
        let n = storage.ids.len();
        if n < 50_000 {
            *self.index.write().unwrap() = None;
            self.index_dirty.store(false, AtomicOrdering::Release);
            return;
        }

        let nlist = NLIST.min(n.max(1));
        let sample_size = TRAINING_SAMPLE.min(n);

        let mut sample_indices = Vec::with_capacity(sample_size);
        let step = (n / sample_size).max(1);
        let mut idx = 0usize;
        while sample_indices.len() < sample_size && idx < n {
            sample_indices.push(idx);
            idx += step;
        }

        let mut centroids = vec![0.0f32; nlist * DIM];
        for c in 0..nlist {
            let src_idx = sample_indices[c % sample_size];
            let src = &storage.vectors[src_idx * DIM..(src_idx + 1) * DIM];
            centroids[c * DIM..(c + 1) * DIM].copy_from_slice(src);
        }

        for _ in 0..KMEANS_ITERS {
            let mut sums = vec![0.0f32; nlist * DIM];
            let mut counts = vec![0u32; nlist];

            for &vi in &sample_indices {
                let v = &storage.vectors[vi * DIM..(vi + 1) * DIM];
                let cidx = nearest_centroid(v, &centroids, nlist);
                counts[cidx] += 1;
                let base = cidx * DIM;
                for d in 0..DIM {
                    sums[base + d] += v[d];
                }
            }

            for c in 0..nlist {
                let base = c * DIM;
                let cnt = counts[c];
                if cnt > 0 {
                    let inv = 1.0f32 / (cnt as f32);
                    for d in 0..DIM {
                        centroids[base + d] = sums[base + d] * inv;
                    }
                }
            }
        }

        let mut lists = (0..nlist).map(|_| Vec::<usize>::new()).collect::<Vec<_>>();
        for i in 0..n {
            let v = &storage.vectors[i * DIM..(i + 1) * DIM];
            let cidx = nearest_centroid(v, &centroids, nlist);
            lists[cidx].push(i);
        }

        *self.index.write().unwrap() = Some(IvfIndex {
            centroids,
            lists,
            nprobe: NPROBE.min(nlist),
        });
        self.index_dirty.store(false, AtomicOrdering::Release);
    }

    fn search_bruteforce(&self, storage: &Storage, query: &[f32], k: usize) -> Vec<SearchResult> {
        let mut heap = BinaryHeap::with_capacity(k);
        for (i, candidate) in storage.vectors.chunks_exact(DIM).enumerate() {
            push_topk(
                &mut heap,
                HeapItem {
                    dist_sq: l2_distance_squared(query, candidate),
                    id: storage.ids[i],
                },
                k,
            );
        }
        heap_to_sorted_results(heap)
    }

    fn search_ivf(&self, storage: &Storage, index: &IvfIndex, query: &[f32], k: usize) -> Vec<SearchResult> {
        let nlist = index.lists.len();
        let mut centroid_dists: Vec<(f64, usize)> = Vec::with_capacity(nlist);
        for c in 0..nlist {
            let centroid = &index.centroids[c * DIM..(c + 1) * DIM];
            centroid_dists.push((l2_distance_squared(query, centroid), c));
        }

        let nprobe = index.nprobe.min(nlist);
        centroid_dists.select_nth_unstable_by(nprobe - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)
        });

        let mut heap = BinaryHeap::with_capacity(k);
        for &(_, cid) in &centroid_dists[..nprobe] {
            for &vidx in &index.lists[cid] {
                let candidate = &storage.vectors[vidx * DIM..(vidx + 1) * DIM];
                push_topk(
                    &mut heap,
                    HeapItem {
                        dist_sq: l2_distance_squared(query, candidate),
                        id: storage.ids[vidx],
                    },
                    k,
                );
            }
        }

        heap_to_sorted_results(heap)
    }
}

#[inline]
fn nearest_centroid(v: &[f32], centroids: &[f32], nlist: usize) -> usize {
    let mut best = 0usize;
    let mut best_dist = f64::INFINITY;
    for c in 0..nlist {
        let d = l2_distance_squared(v, &centroids[c * DIM..(c + 1) * DIM]);
        if d < best_dist {
            best_dist = d;
            best = c;
        }
    }
    best
}

#[inline]
fn push_topk(heap: &mut BinaryHeap<HeapItem>, item: HeapItem, k: usize) {
    if heap.len() < k {
        heap.push(item);
    } else if let Some(worst) = heap.peek() {
        if item.dist_sq < worst.dist_sq {
            heap.pop();
            heap.push(item);
        }
    }
}

#[inline]
fn heap_to_sorted_results(heap: BinaryHeap<HeapItem>) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = heap
        .into_iter()
        .map(|x| SearchResult {
            id: x.id,
            distance: x.dist_sq.sqrt(),
        })
        .collect();

    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });

    results
}

use crate::api::*;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

const DIM: usize = 128;
const NLIST: usize = 512;
const NPROBE: usize = 16;
const TRAIN_SAMPLES: usize = 65_536;
const KMEANS_ITERS: usize = 5;

pub struct VectorDB {
    storage: RwLock<Storage>,
}

struct Storage {
    ids: Vec<u64>,
    vectors: Vec<f32>, // flat layout [n * DIM]
    index: Option<IvfIndex>,
}

struct IvfIndex {
    centroids: Vec<f32>,      // [NLIST * DIM]
    lists: Vec<Vec<usize>>,   // vector indices per list
}

#[derive(Debug)]
struct HeapItem {
    distance2: f32,
    id: u64,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance2 == other.distance2 && self.id == other.id
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance2.partial_cmp(&other.distance2)
    }
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

#[derive(Debug)]
struct CentroidItem {
    distance2: f32,
    idx: usize,
}
impl PartialEq for CentroidItem {
    fn eq(&self, other: &Self) -> bool {
        self.distance2 == other.distance2 && self.idx == other.idx
    }
}
impl Eq for CentroidItem {}
impl PartialOrd for CentroidItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance2.partial_cmp(&other.distance2)
    }
}
impl Ord for CentroidItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            storage: RwLock::new(Storage {
                ids: Vec::new(),
                vectors: Vec::new(),
                index: None,
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM {
            return;
        }
        let mut s = self.storage.write();
        s.ids.push(id);
        s.vectors.extend_from_slice(&vector);
        s.index = None;
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut s = self.storage.write();
        s.ids.reserve(vectors.len());
        s.vectors.reserve(vectors.len() * DIM);

        let mut inserted = 0usize;
        for (id, v) in vectors {
            if v.len() != DIM {
                continue;
            }
            s.ids.push(id);
            s.vectors.extend_from_slice(&v);
            inserted += 1;
        }
        if inserted > 0 {
            s.index = None;
        }
        inserted
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if vector.len() != DIM || top_k == 0 {
            return Vec::new();
        }

        self.ensure_index();

        let s = self.storage.read();
        let n = s.ids.len();
        if n == 0 {
            return Vec::new();
        }

        let k = (top_k as usize).min(n);
        let index = match &s.index {
            Some(idx) => idx,
            None => return Vec::new(),
        };

        let mut probe_heap: BinaryHeap<CentroidItem> = BinaryHeap::with_capacity(NPROBE + 1);
        for c in 0..NLIST {
            let start = c * DIM;
            let centroid = &index.centroids[start..start + DIM];
            let d2 = l2_distance_squared_128(vector, centroid);

            if probe_heap.len() < NPROBE {
                probe_heap.push(CentroidItem { distance2: d2, idx: c });
            } else if let Some(worst) = probe_heap.peek() {
                if d2 < worst.distance2 {
                    probe_heap.pop();
                    probe_heap.push(CentroidItem { distance2: d2, idx: c });
                }
            }
        }

        let mut probes = Vec::with_capacity(probe_heap.len());
        while let Some(item) = probe_heap.pop() {
            probes.push(item.idx);
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);

        for &list_idx in &probes {
            for &vec_idx in &index.lists[list_idx] {
                let start = vec_idx * DIM;
                let candidate = &s.vectors[start..start + DIM];
                let d2 = l2_distance_squared_128(vector, candidate);

                if heap.len() < k {
                    heap.push(HeapItem {
                        distance2: d2,
                        id: s.ids[vec_idx],
                    });
                } else if let Some(worst) = heap.peek() {
                    if d2 < worst.distance2 {
                        heap.pop();
                        heap.push(HeapItem {
                            distance2: d2,
                            id: s.ids[vec_idx],
                        });
                    }
                }
            }
        }

        let mut out = Vec::with_capacity(heap.len());
        while let Some(item) = heap.pop() {
            out.push(SearchResult {
                id: item.id,
                distance: (item.distance2 as f64).sqrt(),
            });
        }

        out.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
        out
    }

    fn ensure_index(&self) {
        if self.storage.read().index.is_some() {
            return;
        }

        let mut s = self.storage.write();
        if s.index.is_some() || s.ids.is_empty() {
            return;
        }

        s.index = Some(build_ivf_index(&s.vectors));
    }
}

fn build_ivf_index(vectors: &[f32]) -> IvfIndex {
    let n = vectors.len() / DIM;
    let sample_n = n.min(TRAIN_SAMPLES.max(NLIST));

    let mut sample_indices = Vec::with_capacity(sample_n);
    let stride = (n / sample_n).max(1);
    let mut i = 0usize;
    while sample_indices.len() < sample_n && i < n {
        sample_indices.push(i);
        i += stride;
    }
    while sample_indices.len() < sample_n {
        sample_indices.push(sample_indices.len() % n);
    }

    let mut centroids = vec![0.0f32; NLIST * DIM];
    for c in 0..NLIST {
        let src_idx = sample_indices[c % sample_indices.len()];
        let src_start = src_idx * DIM;
        let dst_start = c * DIM;
        centroids[dst_start..dst_start + DIM]
            .copy_from_slice(&vectors[src_start..src_start + DIM]);
    }

    let mut sums = vec![0.0f32; NLIST * DIM];
    let mut counts = vec![0u32; NLIST];

    for _ in 0..KMEANS_ITERS {
        sums.fill(0.0);
        counts.fill(0);

        for &idx in &sample_indices {
            let vstart = idx * DIM;
            let v = &vectors[vstart..vstart + DIM];
            let mut best_c = 0usize;
            let mut best_d = f32::INFINITY;

            for c in 0..NLIST {
                let cstart = c * DIM;
                let d2 = l2_distance_squared_128(v, &centroids[cstart..cstart + DIM]);
                if d2 < best_d {
                    best_d = d2;
                    best_c = c;
                }
            }

            counts[best_c] += 1;
            let sstart = best_c * DIM;
            for d in 0..DIM {
                sums[sstart + d] += v[d];
            }
        }

        for c in 0..NLIST {
            let cnt = counts[c];
            if cnt == 0 {
                continue;
            }
            let inv = 1.0f32 / cnt as f32;
            let cstart = c * DIM;
            for d in 0..DIM {
                centroids[cstart + d] = sums[cstart + d] * inv;
            }
        }
    }

    let mut lists: Vec<Vec<usize>> = (0..NLIST).map(|_| Vec::new()).collect();

    for idx in 0..n {
        let vstart = idx * DIM;
        let v = &vectors[vstart..vstart + DIM];
        let mut best_c = 0usize;
        let mut best_d = f32::INFINITY;

        for c in 0..NLIST {
            let cstart = c * DIM;
            let d2 = l2_distance_squared_128(v, &centroids[cstart..cstart + DIM]);
            if d2 < best_d {
                best_d = d2;
                best_c = c;
            }
        }

        lists[best_c].push(idx);
    }

    IvfIndex { centroids, lists }
}

#[inline]
fn l2_distance_squared_128(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: Feature checked at runtime.
            return unsafe { l2_distance_squared_128_avx512(a, b) };
        }
    }

    let mut sum = 0.0f32;
    for i in 0..DIM {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_squared_128_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_ps();
    let pa = a.as_ptr();
    let pb = b.as_ptr();

    for i in (0..DIM).step_by(16) {
        let va = _mm512_loadu_ps(pa.add(i));
        let vb = _mm512_loadu_ps(pb.add(i));
        let diff = _mm512_sub_ps(va, vb);
        let sq = _mm512_mul_ps(diff, diff);
        acc = _mm512_add_ps(acc, sq);
    }

    _mm512_reduce_add_ps(acc)
}

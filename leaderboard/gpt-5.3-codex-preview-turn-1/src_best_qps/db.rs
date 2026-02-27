use crate::api::*;
use parking_lot::RwLock;

const DIM: usize = 128;
const BRUTE_FORCE_THRESHOLD: usize = 20_000;

struct IVFIndex {
    nlist: usize,
    nprobe: usize,
    centroids: Vec<f32>,      // flattened [nlist * DIM]
    lists: Vec<Vec<usize>>,   // vector indices per list
}

struct Inner {
    ids: Vec<u64>,
    vectors: Vec<f32>, // flattened [n * DIM]
    index: Option<IVFIndex>,
    dirty: bool,
}

pub struct VectorDB {
    inner: RwLock<Inner>,
}

#[inline(always)]
fn l2_distance_sq_scalar_128(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut i = 0usize;
    while i < DIM {
        let d0 = a[i] - b[i];
        let d1 = a[i + 1] - b[i + 1];
        let d2 = a[i + 2] - b[i + 2];
        let d3 = a[i + 3] - b[i + 3];
        let d4 = a[i + 4] - b[i + 4];
        let d5 = a[i + 5] - b[i + 5];
        let d6 = a[i + 6] - b[i + 6];
        let d7 = a[i + 7] - b[i + 7];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 + d6 * d6 + d7 * d7;
        i += 8;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn l2_distance_sq_avx512_128(a: *const f32, b: *const f32) -> f32 {
    use std::arch::x86_64::*;

    let mut acc = _mm512_setzero_ps();
    let mut offset = 0isize;
    while offset < DIM as isize {
        let va = _mm512_loadu_ps(a.offset(offset));
        let vb = _mm512_loadu_ps(b.offset(offset));
        let d = _mm512_sub_ps(va, vb);
        acc = _mm512_fmadd_ps(d, d, acc);
        offset += 16;
    }

    _mm512_reduce_add_ps(acc)
}

#[inline(always)]
fn use_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[inline(always)]
fn l2_distance_sq_128(a: &[f32], b: &[f32], avx512: bool) -> f32 {
    if avx512 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            return l2_distance_sq_avx512_128(a.as_ptr(), b.as_ptr());
        }
    }
    l2_distance_sq_scalar_128(a, b)
}

#[inline(always)]
fn update_top_k(best: &mut Vec<(f32, u64)>, k: usize, worst_idx: &mut usize, worst_dist: &mut f32, dist_sq: f32, id: u64) {
    if best.len() < k {
        best.push((dist_sq, id));
        if dist_sq > *worst_dist {
            *worst_dist = dist_sq;
            *worst_idx = best.len() - 1;
        }
    } else if dist_sq < *worst_dist {
        best[*worst_idx] = (dist_sq, id);
        let mut wi = 0usize;
        let mut wd = best[0].0;
        let mut j = 1usize;
        while j < best.len() {
            if best[j].0 > wd {
                wd = best[j].0;
                wi = j;
            }
            j += 1;
        }
        *worst_idx = wi;
        *worst_dist = wd;
    }
}

impl VectorDB {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                ids: Vec::new(),
                vectors: Vec::new(),
                index: None,
                dirty: false,
            }),
        }
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        if vector.len() != DIM {
            return;
        }

        let mut db = self.inner.write();
        db.ids.push(id);
        db.vectors.extend_from_slice(&vector);
        db.dirty = true;
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        let mut db = self.inner.write();
        db.ids.reserve(vectors.len());
        db.vectors.reserve(vectors.len() * DIM);

        let mut inserted = 0usize;
        for (id, vector) in vectors {
            if vector.len() != DIM {
                continue;
            }
            db.ids.push(id);
            db.vectors.extend_from_slice(&vector);
            inserted += 1;
        }

        if inserted > 0 {
            db.dirty = true;
        }

        inserted
    }

    fn maybe_build_index(db: &mut Inner) {
        let n = db.ids.len();
        if n < BRUTE_FORCE_THRESHOLD {
            db.index = None;
            db.dirty = false;
            return;
        }

        let mut nlist = (n / 2048).clamp(128, 512);
        nlist = nlist.next_power_of_two().min(512);
        let nprobe = if nlist >= 512 { 16 } else if nlist >= 256 { 14 } else if nlist >= 128 { 12 } else { 10 };

        let sample_size = n.min(nlist * 128).max(nlist);
        let step = (n / sample_size).max(1);

        let mut centroids = vec![0.0f32; nlist * DIM];
        for c in 0..nlist {
            let src_idx = ((c * step) % n) * DIM;
            let dst = c * DIM;
            centroids[dst..dst + DIM].copy_from_slice(&db.vectors[src_idx..src_idx + DIM]);
        }

        let avx512 = use_avx512();

        let mut assign = vec![0usize; sample_size];
        let iters = 5usize;
        for _ in 0..iters {
            let mut counts = vec![0u32; nlist];
            let mut sums = vec![0.0f32; nlist * DIM];

            let mut si = 0usize;
            let mut vec_idx = 0usize;
            while si < sample_size {
                let vbase = vec_idx * DIM;
                let v = &db.vectors[vbase..vbase + DIM];

                let mut best_c = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..nlist {
                    let cbase = c * DIM;
                    let d = l2_distance_sq_128(v, &centroids[cbase..cbase + DIM], avx512);
                    if d < best_d {
                        best_d = d;
                        best_c = c;
                    }
                }
                assign[si] = best_c;
                counts[best_c] += 1;

                let sbase = best_c * DIM;
                let mut d = 0usize;
                while d < DIM {
                    sums[sbase + d] += v[d];
                    d += 1;
                }

                si += 1;
                vec_idx += step;
                if vec_idx >= n {
                    vec_idx -= n;
                }
            }

            for c in 0..nlist {
                let cbase = c * DIM;
                if counts[c] == 0 {
                    let src = ((c * 9973 + 17) % n) * DIM;
                    centroids[cbase..cbase + DIM].copy_from_slice(&db.vectors[src..src + DIM]);
                    continue;
                }
                let inv = 1.0f32 / counts[c] as f32;
                let mut d = 0usize;
                while d < DIM {
                    centroids[cbase + d] = sums[cbase + d] * inv;
                    d += 1;
                }
            }
        }

        let mut lists = vec![Vec::<usize>::new(); nlist];
        let avg_per_list = (n / nlist).max(1);
        for list in &mut lists {
            list.reserve(avg_per_list);
        }

        let mut i = 0usize;
        let mut base = 0usize;
        while i < n {
            let v = &db.vectors[base..base + DIM];
            let mut best_c = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..nlist {
                let cbase = c * DIM;
                let d = l2_distance_sq_128(v, &centroids[cbase..cbase + DIM], avx512);
                if d < best_d {
                    best_d = d;
                    best_c = c;
                }
            }
            lists[best_c].push(i);
            i += 1;
            base += DIM;
        }

        db.index = Some(IVFIndex {
            nlist,
            nprobe,
            centroids,
            lists,
        });
        db.dirty = false;
    }

    fn brute_force_search(db: &Inner, query: &[f32], k: usize, avx512: bool) -> Vec<SearchResult> {
        let n = db.ids.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }

        let mut best: Vec<(f32, u64)> = Vec::with_capacity(k);
        let mut worst_idx = 0usize;
        let mut worst_dist = f32::NEG_INFINITY;

        let mut i = 0usize;
        let mut base = 0usize;
        while i < n {
            let cand = &db.vectors[base..base + DIM];
            let dist_sq = l2_distance_sq_128(query, cand, avx512);
            update_top_k(&mut best, k, &mut worst_idx, &mut worst_dist, dist_sq, db.ids[i]);
            i += 1;
            base += DIM;
        }

        best.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        best.into_iter()
            .map(|(dist_sq, id)| SearchResult {
                id,
                distance: (dist_sq as f64).sqrt(),
            })
            .collect()
    }

    fn ivf_search(db: &Inner, index: &IVFIndex, query: &[f32], k: usize, avx512: bool) -> Vec<SearchResult> {
        let mut probe_lists: Vec<(f32, usize)> = Vec::with_capacity(index.nprobe);
        let mut worst_idx = 0usize;
        let mut worst_dist = f32::NEG_INFINITY;

        for c in 0..index.nlist {
            let cbase = c * DIM;
            let dist = l2_distance_sq_128(query, &index.centroids[cbase..cbase + DIM], avx512);
            if probe_lists.len() < index.nprobe {
                probe_lists.push((dist, c));
                if dist > worst_dist {
                    worst_dist = dist;
                    worst_idx = probe_lists.len() - 1;
                }
            } else if dist < worst_dist {
                probe_lists[worst_idx] = (dist, c);
                let mut wi = 0usize;
                let mut wd = probe_lists[0].0;
                let mut j = 1usize;
                while j < probe_lists.len() {
                    if probe_lists[j].0 > wd {
                        wd = probe_lists[j].0;
                        wi = j;
                    }
                    j += 1;
                }
                worst_idx = wi;
                worst_dist = wd;
            }
        }

        let mut best: Vec<(f32, u64)> = Vec::with_capacity(k);
        let mut best_worst_idx = 0usize;
        let mut best_worst_dist = f32::NEG_INFINITY;

        for &(_, list_id) in &probe_lists {
            for &vec_idx in &index.lists[list_id] {
                let base = vec_idx * DIM;
                let cand = &db.vectors[base..base + DIM];
                let dist_sq = l2_distance_sq_128(query, cand, avx512);
                update_top_k(
                    &mut best,
                    k,
                    &mut best_worst_idx,
                    &mut best_worst_dist,
                    dist_sq,
                    db.ids[vec_idx],
                );
            }
        }

        best.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        best.into_iter()
            .map(|(dist_sq, id)| SearchResult {
                id,
                distance: (dist_sq as f64).sqrt(),
            })
            .collect()
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        if vector.len() != DIM || top_k == 0 {
            return Vec::new();
        }

        {
            let db = self.inner.read();
            if !db.dirty {
                let n = db.ids.len();
                if n == 0 {
                    return Vec::new();
                }
                let k = (top_k as usize).min(n);
                let avx512 = use_avx512();
                if let Some(index) = &db.index {
                    return Self::ivf_search(&db, index, vector, k, avx512);
                }
                return Self::brute_force_search(&db, vector, k, avx512);
            }
        }

        {
            let mut db = self.inner.write();
            if db.dirty {
                Self::maybe_build_index(&mut db);
            }
        }

        let db = self.inner.read();
        let n = db.ids.len();
        if n == 0 {
            return Vec::new();
        }
        let k = (top_k as usize).min(n);
        let avx512 = use_avx512();
        if let Some(index) = &db.index {
            Self::ivf_search(&db, index, vector, k, avx512)
        } else {
            Self::brute_force_search(&db, vector, k, avx512)
        }
    }
}

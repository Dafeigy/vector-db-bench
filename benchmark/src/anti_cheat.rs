use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Result of anti-cheat analysis on benchmark query results.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AntiCheatResult {
    /// Whether the results passed anti-cheat checks.
    pub passed: bool,
    /// Average Jaccard similarity between sampled pairs of query results.
    pub avg_jaccard_similarity: f64,
    /// Number of unique IDs across all query results.
    pub unique_ids: usize,
    /// Total number of result IDs across all queries.
    pub total_results: usize,
    /// Human-readable summary message.
    pub message: String,
}

/// Compute the Jaccard similarity between two sets of IDs.
///
/// Jaccard(A, B) = |A ∩ B| / |A ∪ B|. Returns 0.0 when both sets are empty.
fn jaccard_similarity(a: &HashSet<u64>, b: &HashSet<u64>) -> f64 {
    let union_size = a.union(b).count();
    if union_size == 0 {
        return 0.0;
    }
    let intersection_size = a.intersection(b).count();
    intersection_size as f64 / union_size as f64
}

/// Detect hardcoded / low-diversity results across queries.
///
/// `results[i]` contains the top-K IDs returned for query `i`.
///
/// Two checks are performed:
/// 1. **Pairwise Jaccard similarity**: sample up to `max_pairs` pairs of queries
///    and compute average Jaccard similarity. If it exceeds `similarity_threshold`
///    (default 0.8), flag as suspicious.
/// 2. **Unique ID count**: count distinct IDs across all results. If the ratio
///    `unique_ids / total_results` is below a minimum diversity ratio, flag.
pub fn detect_hardcoded_results(results: &[Vec<u64>]) -> AntiCheatResult {
    detect_hardcoded_results_with_config(results, 0.8, 500)
}

/// Configurable version of [`detect_hardcoded_results`].
pub fn detect_hardcoded_results_with_config(
    results: &[Vec<u64>],
    similarity_threshold: f64,
    max_pairs: usize,
) -> AntiCheatResult {
    // Handle degenerate cases.
    if results.is_empty() {
        return AntiCheatResult {
            passed: true,
            avg_jaccard_similarity: 0.0,
            unique_ids: 0,
            total_results: 0,
            message: "No results to analyze.".to_string(),
        };
    }

    let all_empty = results.iter().all(|r| r.is_empty());
    if all_empty {
        return AntiCheatResult {
            passed: true,
            avg_jaccard_similarity: 0.0,
            unique_ids: 0,
            total_results: 0,
            message: "All result sets are empty.".to_string(),
        };
    }

    // --- Check 1: Pairwise Jaccard similarity ---
    let sets: Vec<HashSet<u64>> = results
        .iter()
        .map(|ids| ids.iter().copied().collect())
        .collect();

    let n = sets.len();
    let total_possible_pairs = n * (n - 1) / 2;
    let pairs_to_sample = total_possible_pairs.min(max_pairs);

    let avg_similarity = if n < 2 {
        0.0
    } else if pairs_to_sample == total_possible_pairs {
        // Enumerate all pairs.
        let mut sum = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                sum += jaccard_similarity(&sets[i], &sets[j]);
            }
        }
        sum / total_possible_pairs as f64
    } else {
        // Deterministic sampling: evenly spaced pairs from the pair index space.
        let step = total_possible_pairs as f64 / pairs_to_sample as f64;
        let mut sum = 0.0;
        for s in 0..pairs_to_sample {
            let pair_idx = (s as f64 * step) as usize;
            let (i, j) = pair_index_to_ij(pair_idx, n);
            sum += jaccard_similarity(&sets[i], &sets[j]);
        }
        sum / pairs_to_sample as f64
    };

    // --- Check 2: Unique ID count ---
    let all_ids: HashSet<u64> = results.iter().flat_map(|ids| ids.iter().copied()).collect();
    let unique_ids = all_ids.len();
    let total_results: usize = results.iter().map(|r| r.len()).sum();

    // --- Verdict ---
    let similarity_flag = avg_similarity > similarity_threshold;
    let passed = !similarity_flag;

    let message = if similarity_flag {
        format!(
            "SUSPICIOUS: Average Jaccard similarity {:.4} exceeds threshold {:.2}. \
             Unique IDs: {}/{} total. Possible hardcoded results detected.",
            avg_similarity, similarity_threshold, unique_ids, total_results
        )
    } else {
        format!(
            "OK: Average Jaccard similarity {:.4} is within threshold {:.2}. \
             Unique IDs: {}/{} total.",
            avg_similarity, similarity_threshold, unique_ids, total_results
        )
    };

    AntiCheatResult {
        passed,
        avg_jaccard_similarity: avg_similarity,
        unique_ids,
        total_results,
        message,
    }
}

/// Convert a linear pair index into (i, j) where i < j < n.
///
/// Pairs are enumerated as: (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1).
fn pair_index_to_ij(idx: usize, n: usize) -> (usize, usize) {
    // Row i starts at cumulative offset: i*n - i*(i+1)/2
    // We solve for i from: idx < i*n - i*(i+1)/2 + (n - i - 1)
    let mut i = 0;
    let mut offset = 0;
    while i < n - 1 {
        let row_len = n - i - 1;
        if idx < offset + row_len {
            let j = i + 1 + (idx - offset);
            return (i, j);
        }
        offset += row_len;
        i += 1;
    }
    // Fallback (should not happen for valid idx).
    (n - 2, n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- jaccard_similarity tests ---

    #[test]
    fn test_jaccard_identical_sets() {
        let a: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let sim = jaccard_similarity(&a, &a);
        assert!((sim - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_disjoint_sets() {
        let a: HashSet<u64> = [1, 2, 3].into_iter().collect();
        let b: HashSet<u64> = [4, 5, 6].into_iter().collect();
        let sim = jaccard_similarity(&a, &b);
        assert!(sim.abs() < f64::EPSILON);
    }

    #[test]
    fn test_jaccard_partial_overlap() {
        let a: HashSet<u64> = [1, 2, 3, 4].into_iter().collect();
        let b: HashSet<u64> = [3, 4, 5, 6].into_iter().collect();
        // intersection = {3,4} = 2, union = {1,2,3,4,5,6} = 6
        let sim = jaccard_similarity(&a, &b);
        assert!((sim - 2.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_both_empty() {
        let a: HashSet<u64> = HashSet::new();
        let b: HashSet<u64> = HashSet::new();
        assert_eq!(jaccard_similarity(&a, &b), 0.0);
    }

    // --- pair_index_to_ij tests ---

    #[test]
    fn test_pair_index_mapping() {
        // n=4 → pairs: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        assert_eq!(pair_index_to_ij(0, 4), (0, 1));
        assert_eq!(pair_index_to_ij(1, 4), (0, 2));
        assert_eq!(pair_index_to_ij(2, 4), (0, 3));
        assert_eq!(pair_index_to_ij(3, 4), (1, 2));
        assert_eq!(pair_index_to_ij(4, 4), (1, 3));
        assert_eq!(pair_index_to_ij(5, 4), (2, 3));
    }

    // --- detect_hardcoded_results tests ---

    #[test]
    fn test_all_identical_results_detected_as_cheat() {
        // Every query returns the exact same IDs → Jaccard = 1.0 → flagged.
        let same_ids: Vec<u64> = (0..10).collect();
        let results: Vec<Vec<u64>> = (0..20).map(|_| same_ids.clone()).collect();

        let ac = detect_hardcoded_results(&results);
        assert!(!ac.passed, "Identical results should be flagged");
        assert!((ac.avg_jaccard_similarity - 1.0).abs() < f64::EPSILON);
        assert_eq!(ac.unique_ids, 10);
        assert_eq!(ac.total_results, 200);
    }

    #[test]
    fn test_all_different_results_pass() {
        // Each query returns completely disjoint IDs → Jaccard ≈ 0.
        let results: Vec<Vec<u64>> = (0..20)
            .map(|i| {
                let base = i as u64 * 10;
                (base..base + 10).collect()
            })
            .collect();

        let ac = detect_hardcoded_results(&results);
        assert!(ac.passed, "Disjoint results should pass");
        assert!(ac.avg_jaccard_similarity < 0.01);
        assert_eq!(ac.unique_ids, 200);
        assert_eq!(ac.total_results, 200);
    }

    #[test]
    fn test_partially_similar_results_pass() {
        // Some overlap is normal for nearest-neighbor search.
        // Each query shares ~3 IDs with its neighbor but differs otherwise.
        let results: Vec<Vec<u64>> = (0..20)
            .map(|i| {
                let base = i as u64 * 7; // stride of 7 with 10 IDs → ~3 overlap
                (base..base + 10).collect()
            })
            .collect();

        let ac = detect_hardcoded_results(&results);
        assert!(ac.passed, "Partial overlap should pass");
        assert!(ac.avg_jaccard_similarity < 0.8);
    }

    #[test]
    fn test_empty_results_handled_gracefully() {
        let ac = detect_hardcoded_results(&[]);
        assert!(ac.passed);
        assert_eq!(ac.unique_ids, 0);
        assert_eq!(ac.total_results, 0);
    }

    #[test]
    fn test_all_empty_result_sets() {
        let results: Vec<Vec<u64>> = vec![vec![], vec![], vec![]];
        let ac = detect_hardcoded_results(&results);
        assert!(ac.passed);
        assert_eq!(ac.unique_ids, 0);
        assert_eq!(ac.total_results, 0);
    }

    #[test]
    fn test_single_query_passes() {
        // Only one query → no pairs to compare → similarity = 0 → passes.
        let results = vec![vec![1, 2, 3, 4, 5]];
        let ac = detect_hardcoded_results(&results);
        assert!(ac.passed);
        assert_eq!(ac.avg_jaccard_similarity, 0.0);
        assert_eq!(ac.unique_ids, 5);
    }

    #[test]
    fn test_threshold_boundary() {
        // Craft results where similarity is exactly at the boundary.
        // Two identical sets → similarity = 1.0 > 0.8 → flagged.
        let results = vec![vec![1, 2, 3], vec![1, 2, 3]];
        let ac = detect_hardcoded_results(&results);
        assert!(!ac.passed);

        // Two completely different sets → similarity = 0.0 ≤ 0.8 → passes.
        let results = vec![vec![1, 2, 3], vec![4, 5, 6]];
        let ac = detect_hardcoded_results(&results);
        assert!(ac.passed);
    }

    #[test]
    fn test_custom_threshold() {
        // All queries share 8 out of 10 IDs (common base) with 2 unique each.
        // Jaccard for any pair ≈ 8/12 ≈ 0.67 — below 0.8 but above 0.5.
        let results: Vec<Vec<u64>> = (0..10)
            .map(|i| {
                let mut ids: Vec<u64> = (0..8).collect(); // shared base
                ids.push(100 + i as u64 * 2);
                ids.push(100 + i as u64 * 2 + 1);
                ids
            })
            .collect();

        // Default threshold (0.8) should pass since similarity ≈ 0.67.
        let ac_default = detect_hardcoded_results(&results);
        assert!(ac_default.passed);

        // Strict threshold (0.5) should flag since similarity ≈ 0.67 > 0.5.
        let ac_strict = detect_hardcoded_results_with_config(&results, 0.5, 500);
        assert!(!ac_strict.passed);
    }

    #[test]
    fn test_anti_cheat_result_json_roundtrip() {
        let result = AntiCheatResult {
            passed: false,
            avg_jaccard_similarity: 0.95,
            unique_ids: 10,
            total_results: 200,
            message: "SUSPICIOUS".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: AntiCheatResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }
}

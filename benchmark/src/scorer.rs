use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

use crate::runner::QueryResult;

/// A single ground truth entry: the pre-computed nearest neighbors for a query.
#[derive(Debug, Clone, Deserialize)]
pub struct GroundTruthEntry {
    pub query_id: usize,
    pub neighbors: Vec<u64>, // Top-100 nearest neighbor IDs
}

/// Complete benchmark result with all metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResult {
    pub qps: f64,
    pub total_queries: usize,
    pub duration_secs: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub recall: f64,
    pub recall_threshold: f64,
    pub recall_passed: bool,
    pub concurrency: usize,
}

/// Calculate queries per second.
pub fn calculate_qps(total_queries: usize, duration_secs: f64) -> f64 {
    if duration_secs <= 0.0 {
        return 0.0;
    }
    total_queries as f64 / duration_secs
}

/// Calculate the p-th percentile from a mutable slice of latencies.
///
/// Sorts the slice in place and returns the value at the given percentile.
/// `p` should be in [0.0, 100.0].
pub fn calculate_percentile(latencies: &mut [f64], p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rank = (p / 100.0) * (latencies.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        latencies[lower]
    } else {
        let frac = rank - lower as f64;
        latencies[lower] * (1.0 - frac) + latencies[upper] * frac
    }
}

/// Calculate average recall across all queries.
///
/// For each query, computes the intersection of the model's top-10 result IDs
/// with the ground truth top-10 IDs, divided by 10. Returns the average across
/// all queries.
pub fn calculate_recall(
    query_results: &[QueryResult],
    ground_truth: &[GroundTruthEntry],
) -> f64 {
    if query_results.is_empty() || ground_truth.is_empty() {
        return 0.0;
    }

    // Build a lookup from query_id -> ground truth neighbors
    let gt_map: std::collections::HashMap<usize, &Vec<u64>> = ground_truth
        .iter()
        .map(|gt| (gt.query_id, &gt.neighbors))
        .collect();

    let total_recall: f64 = query_results
        .iter()
        .filter_map(|qr| {
            let gt_neighbors = gt_map.get(&qr.query_index)?;
            let gt_top10: HashSet<u64> = gt_neighbors.iter().take(10).copied().collect();
            let model_top10: HashSet<u64> = qr.results.iter().take(10).map(|r| r.id).collect();
            let intersection = model_top10.intersection(&gt_top10).count();
            Some(intersection as f64 / 10.0)
        })
        .sum();

    let matched_count = query_results
        .iter()
        .filter(|qr| gt_map.contains_key(&qr.query_index))
        .count();

    if matched_count == 0 {
        return 0.0;
    }

    total_recall / matched_count as f64
}

/// Combine all calculations into a single BenchmarkResult.
pub fn compute_benchmark_result(
    query_results: &[QueryResult],
    ground_truth: &[GroundTruthEntry],
    duration_secs: f64,
    concurrency: usize,
    recall_threshold: f64,
) -> BenchmarkResult {
    let total_queries = query_results.len();
    let qps = calculate_qps(total_queries, duration_secs);

    let mut latencies: Vec<f64> = query_results.iter().map(|qr| qr.latency_ms).collect();
    let avg_latency_ms = if latencies.is_empty() {
        0.0
    } else {
        latencies.iter().sum::<f64>() / latencies.len() as f64
    };

    let p50 = calculate_percentile(&mut latencies, 50.0);
    let p95 = calculate_percentile(&mut latencies, 95.0);
    let p99 = calculate_percentile(&mut latencies, 99.0);

    let recall = calculate_recall(query_results, ground_truth);
    let recall_passed = recall >= recall_threshold;

    BenchmarkResult {
        qps,
        total_queries,
        duration_secs,
        avg_latency_ms,
        p50_latency_ms: p50,
        p95_latency_ms: p95,
        p99_latency_ms: p99,
        recall,
        recall_threshold,
        recall_passed,
        concurrency,
    }
}

/// Load ground truth entries from a JSON file.
pub async fn load_ground_truth(
    path: &str,
) -> Result<Vec<GroundTruthEntry>, Box<dyn std::error::Error>> {
    let path = Path::new(path);
    let content = tokio::fs::read_to_string(path).await?;
    let entries: Vec<GroundTruthEntry> = serde_json::from_str(&content)?;
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::SearchResult;

    // --- QPS tests ---

    #[test]
    fn test_qps_basic() {
        let qps = calculate_qps(1000, 2.0);
        assert!((qps - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_qps_zero_duration() {
        assert_eq!(calculate_qps(100, 0.0), 0.0);
    }

    #[test]
    fn test_qps_negative_duration() {
        assert_eq!(calculate_qps(100, -1.0), 0.0);
    }

    #[test]
    fn test_qps_zero_queries() {
        assert_eq!(calculate_qps(0, 5.0), 0.0);
    }

    // --- Percentile tests ---

    #[test]
    fn test_percentile_single_value() {
        let mut data = vec![42.0];
        assert!((calculate_percentile(&mut data, 50.0) - 42.0).abs() < f64::EPSILON);
        assert!((calculate_percentile(&mut data, 99.0) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_empty() {
        let mut data: Vec<f64> = vec![];
        assert_eq!(calculate_percentile(&mut data, 50.0), 0.0);
    }

    #[test]
    fn test_percentile_known_values() {
        // 10 values: 1..=10
        let mut data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let p50 = calculate_percentile(&mut data, 50.0);
        // rank = 0.5 * 9 = 4.5 → interpolate between data[4]=5 and data[5]=6 → 5.5
        assert!((p50 - 5.5).abs() < 1e-9);

        let p0 = calculate_percentile(&mut data, 0.0);
        assert!((p0 - 1.0).abs() < 1e-9);

        let p100 = calculate_percentile(&mut data, 100.0);
        assert!((p100 - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_ordering() {
        let mut data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p50 = calculate_percentile(&mut data, 50.0);
        let p95 = calculate_percentile(&mut data, 95.0);
        let p99 = calculate_percentile(&mut data, 99.0);
        assert!(p50 <= p95);
        assert!(p95 <= p99);
    }

    // --- Recall tests ---

    fn make_query_result(query_index: usize, ids: Vec<u64>) -> QueryResult {
        QueryResult {
            query_index,
            results: ids
                .into_iter()
                .map(|id| SearchResult {
                    id,
                    distance: 0.0,
                })
                .collect(),
            latency_ms: 1.0,
        }
    }

    fn make_gt(query_id: usize, neighbors: Vec<u64>) -> GroundTruthEntry {
        GroundTruthEntry {
            query_id,
            neighbors,
        }
    }

    #[test]
    fn test_recall_perfect() {
        let ids: Vec<u64> = (0..10).collect();
        let qr = vec![make_query_result(0, ids.clone())];
        let gt = vec![make_gt(0, ids)];
        let recall = calculate_recall(&qr, &gt);
        assert!((recall - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recall_zero() {
        let qr = vec![make_query_result(0, (0..10).collect())];
        let gt = vec![make_gt(0, (100..110).collect())];
        let recall = calculate_recall(&qr, &gt);
        assert!(recall.abs() < f64::EPSILON);
    }

    #[test]
    fn test_recall_partial() {
        // 5 out of 10 match
        let model_ids: Vec<u64> = (0..10).collect();
        let gt_ids: Vec<u64> = (5..15).collect();
        let qr = vec![make_query_result(0, model_ids)];
        let gt = vec![make_gt(0, gt_ids)];
        let recall = calculate_recall(&qr, &gt);
        assert!((recall - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_recall_empty_inputs() {
        assert_eq!(calculate_recall(&[], &[]), 0.0);
        let gt = vec![make_gt(0, (0..10).collect())];
        assert_eq!(calculate_recall(&[], &gt), 0.0);
        let qr = vec![make_query_result(0, (0..10).collect())];
        assert_eq!(calculate_recall(&qr, &[]), 0.0);
    }

    #[test]
    fn test_recall_multiple_queries() {
        let qr = vec![
            make_query_result(0, (0..10).collect()),  // perfect
            make_query_result(1, (100..110).collect()), // zero
        ];
        let gt = vec![
            make_gt(0, (0..10).collect()),
            make_gt(1, (0..10).collect()),
        ];
        let recall = calculate_recall(&qr, &gt);
        assert!((recall - 0.5).abs() < f64::EPSILON);
    }

    // --- compute_benchmark_result test ---

    #[test]
    fn test_compute_benchmark_result() {
        let ids: Vec<u64> = (0..10).collect();
        let qr = vec![
            QueryResult {
                query_index: 0,
                results: ids.iter().map(|&id| SearchResult { id, distance: 0.0 }).collect(),
                latency_ms: 10.0,
            },
            QueryResult {
                query_index: 1,
                results: ids.iter().map(|&id| SearchResult { id, distance: 0.0 }).collect(),
                latency_ms: 20.0,
            },
        ];
        let gt = vec![
            make_gt(0, ids.clone()),
            make_gt(1, ids.clone()),
        ];

        let result = compute_benchmark_result(&qr, &gt, 1.0, 4, 0.95);

        assert!((result.qps - 2.0).abs() < f64::EPSILON);
        assert_eq!(result.total_queries, 2);
        assert!((result.duration_secs - 1.0).abs() < f64::EPSILON);
        assert!((result.avg_latency_ms - 15.0).abs() < f64::EPSILON);
        assert!((result.recall - 1.0).abs() < f64::EPSILON);
        assert!(result.recall_passed);
        assert_eq!(result.concurrency, 4);
    }

    // --- JSON round-trip test ---

    #[test]
    fn test_benchmark_result_json_roundtrip() {
        let result = BenchmarkResult {
            qps: 1234.5,
            total_queries: 10000,
            duration_secs: 8.1,
            avg_latency_ms: 3.2,
            p50_latency_ms: 2.5,
            p95_latency_ms: 8.0,
            p99_latency_ms: 15.0,
            recall: 0.97,
            recall_threshold: 0.95,
            recall_passed: true,
            concurrency: 4,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, deserialized);
    }

    // --- Ground truth deserialization test ---

    #[test]
    fn test_ground_truth_deserialize() {
        let json = r#"[{"query_id": 0, "neighbors": [1, 2, 3]}]"#;
        let entries: Vec<GroundTruthEntry> = serde_json::from_str(json).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].query_id, 0);
        assert_eq!(entries[0].neighbors, vec![1, 2, 3]);
    }
}

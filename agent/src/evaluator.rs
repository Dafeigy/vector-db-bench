// 评分和排行榜逻辑

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::tools::BenchmarkResult;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardEntry {
    pub model_name: String,
    pub qps: f64,
    pub recall: f64,
    pub recall_passed: bool,
    pub tool_calls_used: u32,
    pub total_time_secs: f64,
    pub optimization_summary: String,
    pub timestamp: DateTime<Utc>,
}

/// Compute the final score for a benchmark result.
/// If recall is below the threshold, the score is 0.0 (QPS zeroed out).
/// Otherwise, the score equals the QPS value.
pub fn compute_final_score(benchmark: &BenchmarkResult, recall_threshold: f64) -> f64 {
    if benchmark.recall < recall_threshold {
        0.0
    } else {
        benchmark.qps
    }
}

/// Sort leaderboard entries by QPS descending; ties broken by recall descending.
pub fn sort_leaderboard(entries: &mut Vec<LeaderboardEntry>) {
    entries.sort_by(|a, b| {
        b.qps
            .partial_cmp(&a.qps)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                b.recall
                    .partial_cmp(&a.recall)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
}

/// Serialize leaderboard entries to JSON and write to the given path.
pub fn save_leaderboard(path: &Path, entries: &[LeaderboardEntry]) -> Result<(), String> {
    let json =
        serde_json::to_string_pretty(entries).map_err(|e| format!("序列化失败: {}", e))?;
    fs::write(path, json).map_err(|e| format!("写入文件失败: {}", e))
}

/// Read a JSON leaderboard file and deserialize into entries.
pub fn load_leaderboard(path: &Path) -> Result<Vec<LeaderboardEntry>, String> {
    let data = fs::read_to_string(path).map_err(|e| format!("读取文件失败: {}", e))?;
    serde_json::from_str(&data).map_err(|e| format!("反序列化失败: {}", e))
}

/// Load existing leaderboard (or start empty), add an entry, sort, save, and return.
pub fn add_to_leaderboard(
    path: &Path,
    entry: LeaderboardEntry,
) -> Result<Vec<LeaderboardEntry>, String> {
    let mut entries = if path.exists() {
        load_leaderboard(path)?
    } else {
        Vec::new()
    };
    entries.push(entry);
    sort_leaderboard(&mut entries);
    save_leaderboard(path, &entries)?;
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::BenchmarkResult;

    fn make_benchmark(qps: f64, recall: f64) -> BenchmarkResult {
        BenchmarkResult {
            qps,
            total_queries: 10000,
            duration_secs: 10.0,
            avg_latency_ms: 4.0,
            p50_latency_ms: 3.0,
            p95_latency_ms: 8.0,
            p99_latency_ms: 15.0,
            recall,
            recall_threshold: 0.95,
            recall_passed: recall >= 0.95,
            concurrency: 4,
            comparison: None,
        }
    }

    fn make_entry(model: &str, qps: f64, recall: f64) -> LeaderboardEntry {
        LeaderboardEntry {
            model_name: model.to_string(),
            qps,
            recall,
            recall_passed: recall >= 0.95,
            tool_calls_used: 30,
            total_time_secs: 120.0,
            optimization_summary: "test".to_string(),
            timestamp: Utc::now(),
        }
    }

    // --- compute_final_score tests ---

    #[test]
    fn test_score_zero_when_recall_below_threshold() {
        let br = make_benchmark(5000.0, 0.80);
        assert_eq!(compute_final_score(&br, 0.95), 0.0);
    }

    #[test]
    fn test_score_equals_qps_when_recall_meets_threshold() {
        let br = make_benchmark(5000.0, 0.96);
        assert!((compute_final_score(&br, 0.95) - 5000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_equals_qps_when_recall_exactly_at_threshold() {
        let br = make_benchmark(3000.0, 0.95);
        assert!((compute_final_score(&br, 0.95) - 3000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_score_zero_when_recall_just_below_threshold() {
        let br = make_benchmark(9999.0, 0.9499);
        assert_eq!(compute_final_score(&br, 0.95), 0.0);
    }

    // --- sort_leaderboard tests ---

    #[test]
    fn test_sort_by_qps_descending() {
        let mut entries = vec![
            make_entry("low", 1000.0, 0.96),
            make_entry("high", 5000.0, 0.96),
            make_entry("mid", 3000.0, 0.96),
        ];
        sort_leaderboard(&mut entries);
        assert_eq!(entries[0].model_name, "high");
        assert_eq!(entries[1].model_name, "mid");
        assert_eq!(entries[2].model_name, "low");
    }

    #[test]
    fn test_sort_tiebreak_by_recall_descending() {
        let mut entries = vec![
            make_entry("lower_recall", 5000.0, 0.96),
            make_entry("higher_recall", 5000.0, 0.99),
        ];
        sort_leaderboard(&mut entries);
        assert_eq!(entries[0].model_name, "higher_recall");
        assert_eq!(entries[1].model_name, "lower_recall");
    }

    #[test]
    fn test_sort_empty_leaderboard() {
        let mut entries: Vec<LeaderboardEntry> = vec![];
        sort_leaderboard(&mut entries);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_sort_single_entry() {
        let mut entries = vec![make_entry("solo", 2000.0, 0.97)];
        sort_leaderboard(&mut entries);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].model_name, "solo");
    }

    // --- serialization roundtrip ---

    #[test]
    fn test_leaderboard_entry_serialization_roundtrip() {
        let entry = make_entry("test-model", 4500.0, 0.98);
        let json = serde_json::to_string(&entry).unwrap();
        let deserialized: LeaderboardEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_name, entry.model_name);
        assert!((deserialized.qps - entry.qps).abs() < f64::EPSILON);
        assert!((deserialized.recall - entry.recall).abs() < f64::EPSILON);
        assert_eq!(deserialized.recall_passed, entry.recall_passed);
        assert_eq!(deserialized.tool_calls_used, entry.tool_calls_used);
        assert!((deserialized.total_time_secs - entry.total_time_secs).abs() < f64::EPSILON);
        assert_eq!(deserialized.optimization_summary, entry.optimization_summary);
    }

    // --- save / load roundtrip ---

    #[test]
    fn test_save_and_load_leaderboard_roundtrip() {
        let dir = std::env::temp_dir().join("evaluator_test_save_load");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("leaderboard.json");

        let entries = vec![
            make_entry("model-a", 5000.0, 0.99),
            make_entry("model-b", 3000.0, 0.96),
        ];

        save_leaderboard(&path, &entries).unwrap();
        let loaded = load_leaderboard(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].model_name, "model-a");
        assert_eq!(loaded[1].model_name, "model-b");
        assert!((loaded[0].qps - 5000.0).abs() < f64::EPSILON);
        assert!((loaded[1].qps - 3000.0).abs() < f64::EPSILON);

        let _ = fs::remove_dir_all(&dir);
    }

    // --- add_to_leaderboard ---

    #[test]
    fn test_add_to_leaderboard_creates_new_file() {
        let dir = std::env::temp_dir().join("evaluator_test_add_new");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("leaderboard.json");
        let _ = fs::remove_file(&path); // ensure clean

        let entry = make_entry("first", 4000.0, 0.97);
        let result = add_to_leaderboard(&path, entry).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].model_name, "first");
        assert!(path.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_add_to_leaderboard_appends_and_sorts() {
        let dir = std::env::temp_dir().join("evaluator_test_add_sort");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("leaderboard.json");
        let _ = fs::remove_file(&path);

        add_to_leaderboard(&path, make_entry("slow", 1000.0, 0.96)).unwrap();
        let result = add_to_leaderboard(&path, make_entry("fast", 8000.0, 0.99)).unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].model_name, "fast");
        assert_eq!(result[1].model_name, "slow");

        let _ = fs::remove_dir_all(&dir);
    }
}

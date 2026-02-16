// Agent state management - tracks tool call counts, timing, logs, and session lifecycle.

use crate::tools::{AgentStatus, BenchmarkResult, ToolResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Default maximum number of tool calls allowed per evaluation session.
const DEFAULT_TOOL_CALL_LIMIT: u32 = 50;

/// A log entry for a single tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallLog {
    pub index: u32,
    pub tool: String,
    pub input: serde_json::Value,
    pub output: serde_json::Value,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

/// Tracks the state of an agent evaluation session.
pub struct AgentState {
    pub tool_calls_used: u32,
    pub tool_calls_total: u32,
    #[allow(dead_code)]
    start_time: Instant,
    pub server_running: bool,
    pub last_benchmark: Option<BenchmarkResult>,
    /// Tracks the best benchmark result (highest QPS with passing recall) across the entire session.
    pub best_benchmark: Option<BenchmarkResult>,
    pub call_log: Vec<ToolCallLog>,
}

impl AgentState {
    /// Create a new agent state with the given tool call limit.
    pub fn new(max_tool_calls: Option<u32>) -> Self {
        Self {
            tool_calls_used: 0,
            tool_calls_total: max_tool_calls.unwrap_or(DEFAULT_TOOL_CALL_LIMIT),
            start_time: Instant::now(),
            server_running: false,
            last_benchmark: None,
            best_benchmark: None,
            call_log: Vec::new(),
        }
    }

    /// Return the current session status.
    pub fn get_status(&self) -> AgentStatus {
        AgentStatus {
            tool_calls_used: self.tool_calls_used,
            tool_calls_remaining: self.tool_calls_remaining(),
            tool_calls_total: self.tool_calls_total,
            elapsed_time_secs: self.start_time.elapsed().as_secs_f64(),
            server_running: self.server_running,
            last_benchmark: self.last_benchmark.clone(),
            best_benchmark: self.best_benchmark.clone(),
        }
    }

    /// Record a completed tool call in the log and increment the counter.
    pub fn record_call(
        &mut self,
        tool: String,
        input: serde_json::Value,
        output: serde_json::Value,
        duration_ms: u64,
    ) {
        self.tool_calls_used += 1;
        self.call_log.push(ToolCallLog {
            index: self.tool_calls_used,
            tool,
            input,
            output,
            duration_ms,
            timestamp: Utc::now(),
        });
    }

    /// Check whether the tool call limit has been reached.
    pub fn is_limit_reached(&self) -> bool {
        self.tool_calls_used >= self.tool_calls_total
    }

    /// Return the number of tool calls remaining.
    pub fn tool_calls_remaining(&self) -> u32 {
        self.tool_calls_total.saturating_sub(self.tool_calls_used)
    }

    /// Trigger the final benchmark and return the finish result.
    ///
    /// Runs `bench_tools::run_benchmark` with default settings and records the
    /// result in `last_benchmark`.
    pub async fn finish(
        &mut self,
        base_dir: &std::path::Path,
        config: &crate::bench_tools::BenchConfig,
        summary: &str,
    ) -> ToolResult {
        let bench_result =
            crate::bench_tools::run_benchmark(base_dir, config, None, None, Some(0)).await;

        match &bench_result {
            ToolResult::RunBenchmark(br) => {
                self.last_benchmark = Some(br.clone());
                // Also update best_benchmark if this is better
                let is_new_best = match &self.best_benchmark {
                    Some(prev) => br.recall_passed && br.qps > prev.qps,
                    None => br.recall_passed,
                };
                if is_new_best {
                    self.best_benchmark = Some(br.clone());
                }
                ToolResult::Finish {
                    status: format!(
                        "Evaluation complete. Summary: {}. Final QPS: {:.2}, Recall: {:.4}",
                        summary, br.qps, br.recall
                    ),
                    final_benchmark: Some(br.clone()),
                }
            }
            ToolResult::Error { message } => {
                // Final build/benchmark failed — fall back to best_benchmark if available
                if let Some(ref best) = self.best_benchmark {
                    eprintln!(
                        "[agent] Final benchmark failed, using best recorded QPS: {:.2} (recall: {:.4})",
                        best.qps, best.recall
                    );
                    self.last_benchmark = Some(best.clone());
                    ToolResult::Finish {
                        status: format!(
                            "Evaluation complete with final benchmark error: {}. Using best recorded QPS: {:.2}, Recall: {:.4}. Summary: {}",
                            message, best.qps, best.recall, summary
                        ),
                        final_benchmark: Some(best.clone()),
                    }
                } else {
                    ToolResult::Finish {
                        status: format!(
                            "Evaluation complete with benchmark error: {}. Summary: {}",
                            message, summary
                        ),
                        final_benchmark: None,
                    }
                }
            }
            _ => ToolResult::Finish {
                status: format!("Evaluation complete. Summary: {}", summary),
                final_benchmark: self.best_benchmark.clone(),
            },
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_initializes_correctly() {
        let state = AgentState::new(None);
        assert_eq!(state.tool_calls_used, 0);
        assert_eq!(state.tool_calls_total, 50);
        assert!(!state.server_running);
        assert!(state.last_benchmark.is_none());
        assert!(state.call_log.is_empty());
    }

    #[test]
    fn test_new_with_custom_limit() {
        let state = AgentState::new(Some(200));
        assert_eq!(state.tool_calls_total, 200);
    }

    #[test]
    fn test_record_call_increments_counter() {
        let mut state = AgentState::new(None);
        state.record_call(
            "read_file".to_string(),
            serde_json::json!({"path": "src/main.rs"}),
            serde_json::json!({"content": "fn main() {}"}),
            15,
        );
        assert_eq!(state.tool_calls_used, 1);
        assert_eq!(state.call_log.len(), 1);
        assert_eq!(state.call_log[0].index, 1);
        assert_eq!(state.call_log[0].tool, "read_file");
        assert_eq!(state.call_log[0].duration_ms, 15);
    }

    #[test]
    fn test_record_call_multiple() {
        let mut state = AgentState::new(None);
        for i in 0..5 {
            state.record_call(
                format!("tool_{}", i),
                serde_json::json!({}),
                serde_json::json!({}),
                10,
            );
        }
        assert_eq!(state.tool_calls_used, 5);
        assert_eq!(state.call_log.len(), 5);
        // Indices should be 1..=5
        for (i, log) in state.call_log.iter().enumerate() {
            assert_eq!(log.index, (i + 1) as u32);
        }
    }

    #[test]
    fn test_is_limit_reached_at_50() {
        let mut state = AgentState::new(None);
        assert!(!state.is_limit_reached());

        for _ in 0..49 {
            state.record_call(
                "t".to_string(),
                serde_json::json!({}),
                serde_json::json!({}),
                1,
            );
        }
        assert!(!state.is_limit_reached());

        // 50th call
        state.record_call(
            "t".to_string(),
            serde_json::json!({}),
            serde_json::json!({}),
            1,
        );
        assert!(state.is_limit_reached());
        assert_eq!(state.tool_calls_used, 50);
    }

    #[test]
    fn test_tool_calls_remaining() {
        let mut state = AgentState::new(None);
        assert_eq!(state.tool_calls_remaining(), 50);

        state.record_call(
            "t".to_string(),
            serde_json::json!({}),
            serde_json::json!({}),
            1,
        );
        assert_eq!(state.tool_calls_remaining(), 49);
    }

    #[test]
    fn test_invariant_used_plus_remaining_equals_total() {
        let mut state = AgentState::new(None);
        for _ in 0..50 {
            assert_eq!(
                state.tool_calls_used + state.tool_calls_remaining(),
                state.tool_calls_total
            );
            state.record_call(
                "t".to_string(),
                serde_json::json!({}),
                serde_json::json!({}),
                1,
            );
        }
        // Also holds at the limit
        assert_eq!(
            state.tool_calls_used + state.tool_calls_remaining(),
            state.tool_calls_total
        );
    }

    #[test]
    fn test_get_status_returns_correct_values() {
        let mut state = AgentState::new(None);
        state.server_running = true;

        for _ in 0..10 {
            state.record_call(
                "t".to_string(),
                serde_json::json!({}),
                serde_json::json!({}),
                1,
            );
        }

        let status = state.get_status();
        assert_eq!(status.tool_calls_used, 10);
        assert_eq!(status.tool_calls_remaining, 40);
        assert_eq!(status.tool_calls_total, 50);
        assert!(status.server_running);
        assert!(status.last_benchmark.is_none());
        assert!(status.elapsed_time_secs >= 0.0);
    }

    #[test]
    fn test_get_status_with_benchmark() {
        let mut state = AgentState::new(None);
        state.last_benchmark = Some(BenchmarkResult {
            qps: 500.0,
            total_queries: 10000,
            duration_secs: 20.0,
            avg_latency_ms: 8.0,
            p50_latency_ms: 6.0,
            p95_latency_ms: 15.0,
            p99_latency_ms: 25.0,
            recall: 0.96,
            recall_threshold: 0.95,
            recall_passed: true,
            concurrency: 4,
            comparison: None,
        });

        let status = state.get_status();
        assert!(status.last_benchmark.is_some());
        let br = status.last_benchmark.unwrap();
        assert!((br.qps - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log_entries_equal_used_count() {
        let mut state = AgentState::new(None);
        for _ in 0..25 {
            state.record_call(
                "t".to_string(),
                serde_json::json!({}),
                serde_json::json!({}),
                1,
            );
            assert_eq!(state.call_log.len() as u32, state.tool_calls_used);
        }
    }

    #[test]
    fn test_tool_call_log_serialization() {
        let log = ToolCallLog {
            index: 1,
            tool: "read_file".to_string(),
            input: serde_json::json!({"path": "test.rs"}),
            output: serde_json::json!({"content": "hello"}),
            duration_ms: 42,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&log).unwrap();
        let deserialized: ToolCallLog = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.index, 1);
        assert_eq!(deserialized.tool, "read_file");
        assert_eq!(deserialized.duration_ms, 42);
    }
}

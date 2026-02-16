use serde::{Deserialize, Serialize};

// ─── ToolCall enum ───────────────────────────────────────────────────────────

/// Represents a tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "tool", content = "params")]
pub enum ToolCall {
    #[serde(rename = "read_file")]
    ReadFile { path: String },

    #[serde(rename = "write_file")]
    WriteFile { path: String, content: String },

    #[serde(rename = "list_files")]
    ListFiles { path: String },

    #[serde(rename = "run_benchmark")]
    RunBenchmark {
        concurrency: Option<usize>,
        warmup: Option<usize>,
        max_queries: Option<usize>,
    },

    #[serde(rename = "run_profiling")]
    RunProfiling { duration: Option<u64> },

    #[serde(rename = "run_correctness_test")]
    RunCorrectnessTest,

    #[serde(rename = "build_project")]
    BuildProject,

    #[serde(rename = "get_status")]
    GetStatus,

    #[serde(rename = "finish")]
    Finish { summary: String },
}

// ─── ToolResult enum ─────────────────────────────────────────────────────────

/// Represents the result of executing a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ToolResult {
    ReadFile {
        content: String,
    },
    WriteFile {
        status: String,
        bytes_written: usize,
    },
    ListFiles {
        files: Vec<String>,
    },
    RunCommand {
        exit_code: i32,
        stdout: String,
        stderr: String,
        duration_ms: u64,
    },
    RunBenchmark(BenchmarkResult),
    RunProfiling {
        flamegraph_svg_path: String,
        top_functions: Vec<FunctionProfile>,
        total_samples: u64,
    },
    RunCorrectnessTest {
        passed: bool,
        total_queries: usize,
        recall: f64,
        recall_threshold: f64,
        failed_queries: Vec<u64>,
        message: String,
    },
    BuildProject {
        success: bool,
        message: String,
    },
    GetStatus(AgentStatus),
    Finish {
        status: String,
        final_benchmark: Option<BenchmarkResult>,
    },
    Error {
        message: String,
    },
}

// ─── Supporting structs ──────────────────────────────────────────────────────

/// Benchmark result mirroring the fields from benchmark/src/scorer.rs.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comparison: Option<BenchmarkComparison>,
}

/// Comparison with the previous benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub previous_qps: f64,
    pub qps_change_pct: f64,
    pub previous_recall: f64,
    pub recall_change_pct: f64,
}

/// A single entry from profiling output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionProfile {
    pub function: String,
    pub percentage: f64,
}

/// Current agent session status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub tool_calls_used: u32,
    pub tool_calls_remaining: u32,
    pub tool_calls_total: u32,
    pub elapsed_time_secs: f64,
    pub server_running: bool,
    pub last_benchmark: Option<BenchmarkResult>,
    pub best_benchmark: Option<BenchmarkResult>,
}

// ─── Tool definitions for LLM function calling ──────────────────────────────

/// Returns JSON Schema tool definitions in OpenAI function calling format.
pub fn get_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file. Accessible paths: src/*, Cargo.toml, benchmarks/*, profiling/*.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read"
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Writable paths: src/* (except read-only src/main.rs and src/api.rs) and Cargo.toml.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write into the file"
                        }
                    },
                    "required": ["path", "content"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files in a directory. Accessible directories: src/, benchmarks/, profiling/.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path to list"
                        }
                    },
                    "required": ["path"]
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "run_benchmark",
                "description": "Run the full benchmark: builds project, starts server, loads 1M vectors, runs queries, reports QPS/latency/recall, stops server. Results are saved to benchmarks/ with round numbers. Includes comparison with previous run if available. Default 1000 queries; use max_queries=0 for full 10K.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "concurrency": {
                            "type": "integer",
                            "description": "Number of concurrent query threads (default 4)"
                        },
                        "warmup": {
                            "type": "integer",
                            "description": "Number of warmup queries (default 100)"
                        },
                        "max_queries": {
                            "type": "integer",
                            "description": "Maximum number of queries to run (default 1000, 0 = all 10000)"
                        }
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "run_profiling",
                "description": "Run performance profiling with real benchmark workload: builds project, starts server, runs perf record while benchmark client sends real queries, stops server. Returns top 10 hottest functions with CPU percentage and flamegraph SVG path. Results saved to profiling/ with round numbers. You can list_files('profiling') to see historical flamegraphs.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration": {
                            "type": "integer",
                            "description": "Profiling duration in seconds (default 30)"
                        }
                    },
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "run_correctness_test",
                "description": "Run correctness validation: builds project, starts server, runs test, stops server. Returns recall and pass/fail status.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "build_project",
                "description": "Build the project with cargo build --release. Returns success/failure with compiler error messages. Use this to quickly check if your code compiles before running benchmark or profiling.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Get current agent session status: tool call counts, elapsed time, server status, and last benchmark result.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }),
        serde_json::json!({
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Signal that optimization is complete. Triggers the final benchmark run and records the score.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "Summary of optimizations performed"
                        }
                    },
                    "required": ["summary"]
                }
            }
        }),
    ]
}

// ─── Tool call dispatch ──────────────────────────────────────────────────────

/// Dispatch a tool call to the appropriate handler.
///
/// `base_dir` is the project root directory used for file operations, command
/// execution, and benchmark/profiling tools.
///
/// `state` is the mutable agent session state used by `get_status` and `finish`.
pub async fn dispatch_tool_call(
    call: &ToolCall,
    base_dir: &std::path::Path,
    config: &crate::bench_tools::BenchConfig,
    state: &mut crate::state::AgentState,
) -> ToolResult {
    let result = match call {
        ToolCall::ReadFile { path } => crate::sandbox::read_file(base_dir, path),
        ToolCall::WriteFile { path, content } => {
            crate::sandbox::write_file(base_dir, path, content)
        }
        ToolCall::ListFiles { path } => crate::sandbox::list_files(base_dir, path),
        ToolCall::RunBenchmark {
            concurrency,
            warmup,
            max_queries,
        } => crate::bench_tools::run_benchmark(base_dir, config, *concurrency, *warmup, *max_queries).await,
        ToolCall::RunProfiling { duration } => {
            crate::bench_tools::run_profiling(base_dir, config, *duration).await
        }
        ToolCall::RunCorrectnessTest => crate::bench_tools::run_correctness_test(base_dir, config).await,
        ToolCall::BuildProject => crate::bench_tools::build_project_tool(base_dir).await,
        ToolCall::GetStatus => ToolResult::GetStatus(state.get_status()),
        ToolCall::Finish { summary } => state.finish(base_dir, config, summary).await,
    };

    // Track best benchmark result and backup src when a new best QPS is achieved
    if let ToolResult::RunBenchmark(ref br) = result {
        if br.recall_passed {
            let is_new_best = match &state.best_benchmark {
                Some(prev) => br.qps > prev.qps,
                None => true,
            };
            if is_new_best {
                eprintln!(
                    "[agent] New best QPS: {:.2} (recall: {:.4}). Backing up src to src_best_qps/",
                    br.qps, br.recall
                );
                state.best_benchmark = Some(br.clone());
                // Backup src/ to src_best_qps/
                let src_dir = base_dir.join("src");
                let backup_dir = base_dir.join("src_best_qps");
                if backup_dir.exists() {
                    let _ = std::fs::remove_dir_all(&backup_dir);
                }
                if let Err(e) = copy_dir_recursive(&src_dir, &backup_dir) {
                    eprintln!("[agent] Warning: failed to backup src to src_best_qps: {}", e);
                }
            }
        }
    }

    result
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_call_serialize_read_file() {
        let call = ToolCall::ReadFile {
            path: "src/main.rs".to_string(),
        };
        let json = serde_json::to_value(&call).unwrap();
        assert_eq!(json["tool"], "read_file");
        assert_eq!(json["params"]["path"], "src/main.rs");
    }

    #[test]
    fn test_tool_call_serialize_write_file() {
        let call = ToolCall::WriteFile {
            path: "src/db.rs".to_string(),
            content: "fn main() {}".to_string(),
        };
        let json = serde_json::to_value(&call).unwrap();
        assert_eq!(json["tool"], "write_file");
        assert_eq!(json["params"]["path"], "src/db.rs");
        assert_eq!(json["params"]["content"], "fn main() {}");
    }

    #[test]
    fn test_tool_call_serialize_run_benchmark_defaults() {
        let call = ToolCall::RunBenchmark {
            concurrency: None,
            warmup: None,
            max_queries: None,
        };
        let json = serde_json::to_value(&call).unwrap();
        assert_eq!(json["tool"], "run_benchmark");
    }

    #[test]
    fn test_tool_call_serialize_finish() {
        let call = ToolCall::Finish {
            summary: "Optimized with SIMD".to_string(),
        };
        let json = serde_json::to_value(&call).unwrap();
        assert_eq!(json["tool"], "finish");
        assert_eq!(json["params"]["summary"], "Optimized with SIMD");
    }

    #[test]
    fn test_tool_call_deserialize_read_file() {
        let json = r#"{"tool": "read_file", "params": {"path": "Cargo.toml"}}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();
        match call {
            ToolCall::ReadFile { path } => assert_eq!(path, "Cargo.toml"),
            _ => panic!("Expected ReadFile"),
        }
    }

    #[test]
    fn test_tool_call_deserialize_get_status() {
        let json = r#"{"tool": "get_status"}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();
        assert!(matches!(call, ToolCall::GetStatus));
    }

    #[test]
    fn test_tool_call_deserialize_run_correctness_test() {
        let json = r#"{"tool": "run_correctness_test"}"#;
        let call: ToolCall = serde_json::from_str(json).unwrap();
        assert!(matches!(call, ToolCall::RunCorrectnessTest));
    }

    #[test]
    fn test_tool_result_serialize_error() {
        let result = ToolResult::Error {
            message: "something went wrong".to_string(),
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["type"], "Error");
        assert_eq!(json["message"], "something went wrong");
    }

    #[test]
    fn test_tool_result_serialize_read_file() {
        let result = ToolResult::ReadFile {
            content: "hello world".to_string(),
        };
        let json = serde_json::to_value(&result).unwrap();
        assert_eq!(json["type"], "ReadFile");
        assert_eq!(json["content"], "hello world");
    }

    #[test]
    fn test_benchmark_result_roundtrip() {
        let br = BenchmarkResult {
            qps: 1000.0,
            total_queries: 10000,
            duration_secs: 10.0,
            avg_latency_ms: 4.0,
            p50_latency_ms: 3.0,
            p95_latency_ms: 8.0,
            p99_latency_ms: 15.0,
            recall: 0.97,
            recall_threshold: 0.95,
            recall_passed: true,
            concurrency: 4,
            comparison: None,
        };
        let json = serde_json::to_string(&br).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();
        assert!((deserialized.qps - br.qps).abs() < f64::EPSILON);
        assert_eq!(deserialized.total_queries, br.total_queries);
        assert_eq!(deserialized.recall_passed, br.recall_passed);
    }

    #[test]
    fn test_agent_status_serialize() {
        let status = AgentStatus {
            tool_calls_used: 10,
            tool_calls_remaining: 40,
            tool_calls_total: 50,
            elapsed_time_secs: 120.5,
            server_running: true,
            last_benchmark: None,
            best_benchmark: None,
        };
        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["tool_calls_used"], 10);
        assert_eq!(json["tool_calls_remaining"], 40);
        assert_eq!(json["tool_calls_total"], 50);
        assert!(json["last_benchmark"].is_null());
    }

    #[test]
    fn test_get_tool_definitions_count() {
        let defs = get_tool_definitions();
        // 9 tools: read_file, write_file, list_files,
        // run_benchmark, run_profiling, run_correctness_test, build_project, get_status, finish
        assert_eq!(defs.len(), 9);
    }

    #[test]
    fn test_get_tool_definitions_structure() {
        let defs = get_tool_definitions();
        for def in &defs {
            assert_eq!(def["type"], "function");
            assert!(def["function"]["name"].is_string());
            assert!(def["function"]["description"].is_string());
            assert!(def["function"]["parameters"].is_object());
        }
    }

    #[test]
    fn test_tool_call_all_variants_roundtrip() {
        let calls = vec![
            ToolCall::ReadFile { path: "a.txt".into() },
            ToolCall::WriteFile { path: "b.txt".into(), content: "data".into() },
            ToolCall::ListFiles { path: ".".into() },
            ToolCall::RunBenchmark { concurrency: Some(8), warmup: Some(500), max_queries: None },
            ToolCall::RunProfiling { duration: Some(60) },
            ToolCall::RunCorrectnessTest,
            ToolCall::BuildProject,
            ToolCall::GetStatus,
            ToolCall::Finish { summary: "done".into() },
        ];
        for call in &calls {
            let json = serde_json::to_string(call).unwrap();
            let deserialized: ToolCall = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2);
        }
    }

    #[tokio::test]
    async fn test_dispatch_get_status() {
        let dir = std::env::temp_dir();
        let config = crate::bench_tools::BenchConfig {
            benchmark_bin: dir.join("nonexistent_binary"),
            data_dir: dir.join("nonexistent_data"),
            cpu_cores: None,
        };
    let mut state = crate::state::AgentState::new(None);
        let call = ToolCall::GetStatus;
        let result = dispatch_tool_call(&call, &dir, &config, &mut state).await;
        match result {
            ToolResult::GetStatus(status) => {
                assert_eq!(status.tool_calls_used, 0);
                assert_eq!(status.tool_calls_remaining, 50);
                assert_eq!(status.tool_calls_total, 50);
            }
            _ => panic!("Expected GetStatus result"),
        }
    }
}

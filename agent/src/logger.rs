// Real-time JSONL logger for agent sessions.
//
// Appends one JSON object per line to `agent_log.jsonl` in the work directory.
// Each line is flushed immediately so the log survives crashes.

use chrono::Utc;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Event types written to the log.
#[derive(Debug, Serialize)]
#[serde(tag = "event")]
pub enum LogEvent<'a> {
    #[serde(rename = "session_start")]
    SessionStart {
        model: &'a str,
        work_dir: &'a str,
        thinking_mode: &'a str,
        timestamp: String,
    },
    #[serde(rename = "llm_request")]
    LlmRequest {
        iteration: u32,
        message_count: usize,
        timestamp: String,
    },
    #[serde(rename = "llm_response")]
    LlmResponse {
        iteration: u32,
        has_tool_calls: bool,
        tool_call_count: usize,
        content: Option<String>,
        thinking_content: Option<String>,
        duration_ms: u64,
        timestamp: String,
    },
    #[serde(rename = "tool_call")]
    ToolCallEvent {
        index: u32,
        tool: &'a str,
        arguments: &'a str,
        tool_call_id: &'a str,
        timestamp: String,
    },
    #[serde(rename = "tool_result")]
    ToolResultEvent {
        index: u32,
        tool: &'a str,
        tool_call_id: &'a str,
        result: serde_json::Value,
        duration_ms: u64,
        timestamp: String,
    },
    #[serde(rename = "session_end")]
    SessionEnd {
        tool_calls_used: u32,
        tool_calls_total: u32,
        elapsed_secs: f64,
        reason: &'a str,
        timestamp: String,
    },
    #[serde(rename = "error")]
    Error {
        message: String,
        timestamp: String,
    },
}

/// Real-time JSONL logger. Each write is flushed immediately.
pub struct AgentLogger {
    writer: BufWriter<File>,
    log_path: PathBuf,
    iteration: u32,
}

impl AgentLogger {
    /// Create a new logger writing to `<work_dir>/agent_log.jsonl`.
    /// Appends to existing file if present (supports resume inspection).
    pub fn new(work_dir: &Path) -> Result<Self, String> {
        let log_path = work_dir.join("agent_log.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_path)
            .map_err(|e| format!("Failed to open log file {}: {}", log_path.display(), e))?;

        Ok(Self {
            writer: BufWriter::new(file),
            log_path,
            iteration: 0,
        })
    }

    /// Write a log event as a single JSON line, flushed immediately.
    pub fn log(&mut self, event: &LogEvent) {
        let line = match serde_json::to_string(event) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[logger] Failed to serialize log event: {}", e);
                return;
            }
        };
        if let Err(e) = writeln!(self.writer, "{}", line) {
            eprintln!("[logger] Failed to write log: {}", e);
            return;
        }
        if let Err(e) = self.writer.flush() {
            eprintln!("[logger] Failed to flush log: {}", e);
        }
    }

    /// Log session start.
    pub fn log_session_start(&mut self, model: &str, work_dir: &str, thinking_mode: &str) {
        self.log(&LogEvent::SessionStart {
            model,
            work_dir,
            thinking_mode,
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log an LLM request (before sending).
    pub fn log_llm_request(&mut self, message_count: usize) {
        self.iteration += 1;
        self.log(&LogEvent::LlmRequest {
            iteration: self.iteration,
            message_count,
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log an LLM response (after receiving).
    pub fn log_llm_response(
        &mut self,
        has_tool_calls: bool,
        tool_call_count: usize,
        content: Option<&str>,
        thinking_content: Option<&str>,
        duration_ms: u64,
    ) {
        self.log(&LogEvent::LlmResponse {
            iteration: self.iteration,
            has_tool_calls,
            tool_call_count,
            content: content.map(|c| c.to_string()),
            thinking_content: thinking_content.map(|c| c.to_string()),
            duration_ms,
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log a tool call (before execution).
    pub fn log_tool_call(
        &mut self,
        index: u32,
        tool: &str,
        arguments: &str,
        tool_call_id: &str,
    ) {
        self.log(&LogEvent::ToolCallEvent {
            index,
            tool,
            arguments,
            tool_call_id,
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log a tool result (after execution).
    pub fn log_tool_result(
        &mut self,
        index: u32,
        tool: &str,
        tool_call_id: &str,
        result: &serde_json::Value,
        duration_ms: u64,
    ) {
        self.log(&LogEvent::ToolResultEvent {
            index,
            tool,
            tool_call_id,
            result: result.clone(),
            duration_ms,
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log session end.
    pub fn log_session_end(
        &mut self,
        tool_calls_used: u32,
        tool_calls_total: u32,
        elapsed_secs: f64,
        reason: &str,
    ) {
        self.log(&LogEvent::SessionEnd {
            tool_calls_used,
            tool_calls_total,
            elapsed_secs,
            reason,
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Log an error event.
    pub fn log_error(&mut self, message: &str) {
        self.log(&LogEvent::Error {
            message: message.to_string(),
            timestamp: Utc::now().to_rfc3339(),
        });
    }

    /// Return the log file path.
    pub fn path(&self) -> &Path {
        &self.log_path
    }
}


// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("agent_logger_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_logger_creates_file() {
        let dir = temp_dir();
        let logger = AgentLogger::new(&dir).unwrap();
        assert!(logger.path().exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_session_start_writes_jsonl() {
        let dir = temp_dir();
        let mut logger = AgentLogger::new(&dir).unwrap();
        logger.log_session_start("gpt-4o", "/tmp/work", "false");

        let content = std::fs::read_to_string(logger.path()).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 1);

        let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(v["event"], "session_start");
        assert_eq!(v["model"], "gpt-4o");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multiple_events_append() {
        let dir = temp_dir();
        let mut logger = AgentLogger::new(&dir).unwrap();
        logger.log_session_start("test", "/tmp", "false");
        logger.log_llm_request(3);
        logger.log_llm_response(true, 2, None, None, 150);
        logger.log_tool_call(1, "read_file", r#"{"path":"a.rs"}"#, "call_1");
        logger.log_tool_result(1, "read_file", "call_1", &serde_json::json!({"content": "ok"}), 10);
        logger.log_error("something went wrong");
        logger.log_session_end(5, 50, 30.0, "finished");

        let content = std::fs::read_to_string(logger.path()).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        assert_eq!(lines.len(), 7);

        // Each line should be valid JSON
        for line in &lines {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert!(v.get("event").is_some());
            assert!(v.get("timestamp").is_some());
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_iteration_increments() {
        let dir = temp_dir();
        let mut logger = AgentLogger::new(&dir).unwrap();
        logger.log_llm_request(1);
        logger.log_llm_request(2);

        let content = std::fs::read_to_string(logger.path()).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        let v1: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        let v2: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(v1["iteration"], 1);
        assert_eq!(v2["iteration"], 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_content_full_and_thinking() {
        let dir = temp_dir();
        let mut logger = AgentLogger::new(&dir).unwrap();
        let long_content = "x".repeat(1000);
        let thinking = "Let me think about this...";
        logger.log_llm_request(1);
        logger.log_llm_response(false, 0, Some(&long_content), Some(thinking), 50);

        let content = std::fs::read_to_string(logger.path()).unwrap();
        let lines: Vec<&str> = content.trim().lines().collect();
        let v: serde_json::Value = serde_json::from_str(lines[1]).unwrap();
        // Full content, no truncation
        let logged_content = v["content"].as_str().unwrap();
        assert_eq!(logged_content.len(), 1000);
        // Thinking content is present
        let logged_thinking = v["thinking_content"].as_str().unwrap();
        assert_eq!(logged_thinking, thinking);

        let _ = std::fs::remove_dir_all(&dir);
    }
}

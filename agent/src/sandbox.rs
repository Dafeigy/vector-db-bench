// 沙箱环境管理 - 文件操作、命令执行、权限控制

use crate::tools::ToolResult;
use std::fs;
use std::path::Path;
use tokio::process::Command;
use tokio::time::{timeout, Duration};

/// Allowed root paths for file access.
/// The model can only read/write/list within these prefixes.
/// `Cargo.toml` is additionally allowed for reading (but not writing).
const ALLOWED_PREFIXES: &[&str] = &["src/"];

/// Paths that are writable outside of the normal `src/` scope.
const EXTRA_WRITABLE: &[&str] = &["Cargo.toml"];

/// Paths that are readable outside of the normal `src/` scope.
const EXTRA_READABLE: &[&str] = &["Cargo.toml", "benchmarks/", "profiling/"];

/// Check whether a path is within the allowed `src/` scope.
/// Normalizes `./`, `\`, and rejects `..` traversal.
fn is_allowed_path(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim_start_matches("./");

    // Reject any parent-directory traversal
    if normalized.contains("..") {
        return false;
    }

    // Check against allowed prefixes
    for prefix in ALLOWED_PREFIXES {
        if normalized.starts_with(prefix) || normalized == prefix.trim_end_matches('/') {
            return true;
        }
    }

    false
}

/// Check whether a path is allowed for reading.
/// Same as `is_allowed_path` but also permits `Cargo.toml`.
fn is_read_allowed(path: &str) -> bool {
    if is_allowed_path(path) {
        return true;
    }
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim_start_matches("./");
    for &extra in EXTRA_READABLE {
        if extra.ends_with('/') {
            if normalized.starts_with(extra) || normalized == extra.trim_end_matches('/') {
                return true;
            }
        } else if normalized == extra {
            return true;
        }
    }
    false
}

/// Readonly paths that the model is not allowed to modify.
/// Paths ending with '/' protect the entire directory.
const READONLY_PATHS: &[&str] = &[
    "src/main.rs",
    "src/api.rs",
    "benchmark/",
    "scripts/load_data.py",
];

/// Check whether a given path is readonly.
pub fn is_readonly(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim_start_matches("./");
    for &ro in READONLY_PATHS {
        if ro.ends_with('/') {
            // Directory protection: any path under this prefix is readonly
            if normalized.starts_with(ro) || normalized == ro.trim_end_matches('/') {
                return true;
            }
        } else if normalized == ro {
            return true;
        }
    }
    false
}

/// Read a file relative to `base_dir` and return its content.
/// Only paths within `src/` (and `Cargo.toml`) are allowed.
pub fn read_file(base_dir: &Path, path: &str) -> ToolResult {
    if !is_read_allowed(path) {
        return ToolResult::Error {
            message: format!(
                "Access denied: '{}' is outside the allowed scope. You can only read files under src/ and Cargo.toml.",
                path
            ),
        };
    }
    let full_path = base_dir.join(path);
    match fs::read_to_string(&full_path) {
        Ok(content) => ToolResult::ReadFile { content },
        Err(e) => ToolResult::Error {
            message: format!("Failed to read file '{}': {}", path, e),
        },
    }
}

/// Write content to a file relative to `base_dir`.
/// Only paths within `src/` are allowed. Also checks the readonly list.
pub fn write_file(base_dir: &Path, path: &str, content: &str) -> ToolResult {
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim_start_matches("./");

    // Check if path is in the normal allowed scope or in extra writable list
    let writable = is_allowed_path(path) || EXTRA_WRITABLE.iter().any(|&w| normalized == w);
    if !writable {
        return ToolResult::Error {
            message: format!(
                "Access denied: '{}' is outside the allowed scope. You can write files under src/ and Cargo.toml.",
                path
            ),
        };
    }

    if is_readonly(path) {
        return ToolResult::Error {
            message: format!("Permission denied: '{}' is read-only", path),
        };
    }

    let full_path = base_dir.join(path);

    // Create parent directories if needed
    if let Some(parent) = full_path.parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return ToolResult::Error {
                message: format!("Failed to create directories for '{}': {}", path, e),
            };
        }
    }

    let bytes_written = content.len();
    match fs::write(&full_path, content) {
        Ok(()) => ToolResult::WriteFile {
            status: "ok".to_string(),
            bytes_written,
        },
        Err(e) => ToolResult::Error {
            message: format!("Failed to write file '{}': {}", path, e),
        },
    }
}

/// List files in a directory relative to `base_dir`.
/// Only paths within `src/` are allowed. Returns a sorted list of filenames.
pub fn list_files(base_dir: &Path, path: &str) -> ToolResult {
    if !is_read_allowed(path) {
        return ToolResult::Error {
            message: format!(
                "Access denied: '{}' is outside the allowed scope. You can list files under src/, benchmarks/, and profiling/.",
                path
            ),
        };
    }
    let full_path = base_dir.join(path);
    match fs::read_dir(&full_path) {
        Ok(entries) => {
            let mut files: Vec<String> = entries
                .filter_map(|entry| entry.ok())
                .filter_map(|entry| {
                    entry.file_name().into_string().ok()
                })
                .collect();
            files.sort();
            ToolResult::ListFiles { files }
        }
        Err(e) => ToolResult::Error {
            message: format!("Failed to list directory '{}': {}", path, e),
        },
    }
}

/// Maximum bytes to keep from stdout/stderr to avoid blowing up context.
const OUTPUT_LIMIT: usize = 100 * 1024; // 100 KB

/// Truncate a string to at most `limit` bytes (on a char boundary) and append
/// a notice when truncation occurs.
fn truncate_output(s: String, limit: usize) -> String {
    if s.len() <= limit {
        return s;
    }
    // Find the last char boundary at or before `limit`
    let mut end = limit;
    while !s.is_char_boundary(end) && end > 0 {
        end -= 1;
    }
    let mut truncated = s[..end].to_string();
    truncated.push_str("\n... [truncated]");
    truncated
}

/// Execute a shell command inside `base_dir` with an optional timeout.
///
/// * Uses `sh -c` on unix and `cmd /C` on windows.
/// * Captures stdout, stderr, exit code and wall-clock duration.
/// * Kills the child process on timeout.
/// * Truncates stdout/stderr to 100 KB.
pub async fn run_command(base_dir: &Path, command: &str, timeout_secs: Option<u64>) -> ToolResult {
    let timeout_dur = Duration::from_secs(timeout_secs.unwrap_or(120));
    let start = std::time::Instant::now();

    #[cfg(unix)]
    let mut child = match Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(base_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return ToolResult::Error {
                message: format!("Failed to spawn command: {}", e),
            };
        }
    };

    #[cfg(windows)]
    let mut child = match Command::new("cmd")
        .arg("/C")
        .arg(command)
        .current_dir(base_dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            return ToolResult::Error {
                message: format!("Failed to spawn command: {}", e),
            };
        }
    };

    // Take stdout/stderr handles so we can still kill the child on timeout.
    let child_stdout = child.stdout.take();
    let child_stderr = child.stderr.take();

    match timeout(timeout_dur, child.wait()).await {
        Ok(Ok(status)) => {
            let duration_ms = start.elapsed().as_millis() as u64;
            let exit_code = status.code().unwrap_or(-1);

            let stdout_bytes = match child_stdout {
                Some(mut out) => {
                    let mut buf = Vec::new();
                    let _ = tokio::io::AsyncReadExt::read_to_end(&mut out, &mut buf).await;
                    buf
                }
                None => Vec::new(),
            };
            let stderr_bytes = match child_stderr {
                Some(mut err) => {
                    let mut buf = Vec::new();
                    let _ = tokio::io::AsyncReadExt::read_to_end(&mut err, &mut buf).await;
                    buf
                }
                None => Vec::new(),
            };

            let stdout = truncate_output(String::from_utf8_lossy(&stdout_bytes).into_owned(), OUTPUT_LIMIT);
            let stderr = truncate_output(String::from_utf8_lossy(&stderr_bytes).into_owned(), OUTPUT_LIMIT);

            ToolResult::RunCommand {
                exit_code,
                stdout,
                stderr,
                duration_ms,
            }
        }
        Ok(Err(e)) => ToolResult::Error {
            message: format!("Command execution error: {}", e),
        },
        Err(_) => {
            // Timeout – kill the child process
            let _ = child.kill().await;
            let duration_ms = start.elapsed().as_millis() as u64;
            ToolResult::Error {
                message: format!(
                    "Command timed out after {} seconds ({}ms elapsed)",
                    timeout_dur.as_secs(),
                    duration_ms
                ),
            }
        }
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    // ─── is_allowed_path / is_read_allowed tests ─────────────────────────────

    #[test]
    fn test_is_allowed_path_src_files() {
        assert!(is_allowed_path("src/db.rs"));
        assert!(is_allowed_path("src/distance.rs"));
        assert!(is_allowed_path("src/sub/mod.rs"));
        assert!(is_allowed_path("src"));
    }

    #[test]
    fn test_is_allowed_path_rejects_outside_src() {
        assert!(!is_allowed_path("Cargo.toml"));
        assert!(!is_allowed_path("target/release/binary"));
        assert!(!is_allowed_path("."));
        assert!(!is_allowed_path("benchmark/src/main.rs"));
        assert!(!is_allowed_path("hello.txt"));
    }

    #[test]
    fn test_is_allowed_path_rejects_traversal() {
        assert!(!is_allowed_path("src/../Cargo.toml"));
        assert!(!is_allowed_path("../etc/passwd"));
        assert!(!is_allowed_path("src/../../secret"));
    }

    #[test]
    fn test_is_allowed_path_dot_slash_prefix() {
        assert!(is_allowed_path("./src/db.rs"));
        assert!(!is_allowed_path("./Cargo.toml"));
    }

    #[test]
    fn test_is_read_allowed_cargo_toml() {
        assert!(is_read_allowed("Cargo.toml"));
        assert!(is_read_allowed("./Cargo.toml"));
        assert!(is_read_allowed("src/db.rs"));
        assert!(is_read_allowed("benchmarks/benchmark_001.json"));
        assert!(is_read_allowed("profiling/flamegraph_001.svg"));
        assert!(is_read_allowed("benchmarks"));
        assert!(is_read_allowed("profiling"));
        assert!(!is_read_allowed("target/debug/binary"));
        assert!(!is_read_allowed("."));
    }

    // ─── is_readonly tests ───────────────────────────────────────────────────

    #[test]
    fn test_is_readonly_exact_match() {
        assert!(is_readonly("src/main.rs"));
        assert!(is_readonly("src/api.rs"));
        assert!(is_readonly("scripts/load_data.py"));
    }

    #[test]
    fn test_is_readonly_directory_prefix() {
        assert!(is_readonly("benchmark/src/main.rs"));
        assert!(is_readonly("benchmark/Cargo.toml"));
        assert!(is_readonly("benchmark/"));
    }

    #[test]
    fn test_is_readonly_bare_directory_name() {
        assert!(is_readonly("benchmark"));
    }

    #[test]
    fn test_is_readonly_non_readonly_paths() {
        assert!(!is_readonly("src/db.rs"));
        assert!(!is_readonly("src/distance.rs"));
        assert!(!is_readonly("Cargo.toml"));
    }

    #[test]
    fn test_is_readonly_with_dot_slash_prefix() {
        assert!(is_readonly("./src/main.rs"));
        assert!(is_readonly("./benchmark/foo.rs"));
        assert!(!is_readonly("./src/db.rs"));
    }

    // ─── read_file tests ─────────────────────────────────────────────────────

    #[test]
    fn test_read_file_src() {
        let dir = tempdir();
        let src = dir.join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("db.rs"), "fn main() {}").unwrap();

        match read_file(&dir, "src/db.rs") {
            ToolResult::ReadFile { content } => assert_eq!(content, "fn main() {}"),
            other => panic!("Expected ReadFile, got {:?}", other),
        }
    }

    #[test]
    fn test_read_file_cargo_toml() {
        let dir = tempdir();
        fs::write(dir.join("Cargo.toml"), "[package]").unwrap();

        match read_file(&dir, "Cargo.toml") {
            ToolResult::ReadFile { content } => assert_eq!(content, "[package]"),
            other => panic!("Expected ReadFile, got {:?}", other),
        }
    }

    #[test]
    fn test_read_file_blocked_outside_src() {
        let dir = tempdir();
        fs::write(dir.join("secret.txt"), "secret").unwrap();

        match read_file(&dir, "secret.txt") {
            ToolResult::Error { message } => {
                assert!(message.contains("Access denied"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_read_file_blocked_traversal() {
        let dir = tempdir();
        match read_file(&dir, "src/../Cargo.lock") {
            ToolResult::Error { message } => {
                assert!(message.contains("Access denied"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_read_file_nonexistent() {
        let dir = tempdir();
        fs::create_dir_all(dir.join("src")).unwrap();
        match read_file(&dir, "src/nope.rs") {
            ToolResult::Error { message } => {
                assert!(message.contains("nope.rs"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    // ─── write_file tests ────────────────────────────────────────────────────

    #[test]
    fn test_write_file_src() {
        let dir = tempdir();
        fs::create_dir_all(dir.join("src")).unwrap();
        match write_file(&dir, "src/db.rs", "fn db() {}") {
            ToolResult::WriteFile { status, bytes_written } => {
                assert_eq!(status, "ok");
                assert_eq!(bytes_written, 10);
            }
            other => panic!("Expected WriteFile, got {:?}", other),
        }
        assert_eq!(fs::read_to_string(dir.join("src/db.rs")).unwrap(), "fn db() {}");
    }

    #[test]
    fn test_write_file_creates_subdirs_in_src() {
        let dir = tempdir();
        match write_file(&dir, "src/sub/mod.rs", "mod sub;") {
            ToolResult::WriteFile { status, bytes_written } => {
                assert_eq!(status, "ok");
                assert_eq!(bytes_written, 8);
            }
            other => panic!("Expected WriteFile, got {:?}", other),
        }
        assert_eq!(
            fs::read_to_string(dir.join("src/sub/mod.rs")).unwrap(),
            "mod sub;"
        );
    }

    #[test]
    fn test_write_file_cargo_toml_allowed() {
        let dir = tempdir();
        match write_file(&dir, "Cargo.toml", "[package]") {
            ToolResult::WriteFile { status, bytes_written } => {
                assert_eq!(status, "ok");
                assert_eq!(bytes_written, 9);
            }
            other => panic!("Expected WriteFile, got {:?}", other),
        }
    }

    #[test]
    fn test_write_file_blocked_traversal() {
        let dir = tempdir();
        match write_file(&dir, "src/../evil.rs", "bad") {
            ToolResult::Error { message } => {
                assert!(message.contains("Access denied"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_write_file_readonly_exact() {
        let dir = tempdir();
        match write_file(&dir, "src/main.rs", "hacked") {
            ToolResult::Error { message } => {
                assert!(message.contains("read-only"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_write_file_readonly_api() {
        let dir = tempdir();
        match write_file(&dir, "src/api.rs", "hacked") {
            ToolResult::Error { message } => {
                assert!(message.contains("read-only"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    // ─── list_files tests ────────────────────────────────────────────────────

    #[test]
    fn test_list_files_src() {
        let dir = tempdir();
        let src = dir.join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("db.rs"), "").unwrap();
        fs::write(src.join("api.rs"), "").unwrap();
        fs::write(src.join("main.rs"), "").unwrap();

        match list_files(&dir, "src") {
            ToolResult::ListFiles { files } => {
                assert_eq!(files, vec!["api.rs", "db.rs", "main.rs"]);
            }
            other => panic!("Expected ListFiles, got {:?}", other),
        }
    }

    #[test]
    fn test_list_files_blocked_root() {
        let dir = tempdir();
        match list_files(&dir, ".") {
            ToolResult::Error { message } => {
                assert!(message.contains("Access denied"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_list_files_blocked_parent() {
        let dir = tempdir();
        match list_files(&dir, "..") {
            ToolResult::Error { message } => {
                assert!(message.contains("Access denied"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_list_files_blocked_target() {
        let dir = tempdir();
        match list_files(&dir, "target") {
            ToolResult::Error { message } => {
                assert!(message.contains("Access denied"));
            }
            other => panic!("Expected Error, got {:?}", other),
        }
    }

    #[test]
    fn test_list_files_src_subdir() {
        let dir = tempdir();
        let sub = dir.join("src").join("utils");
        fs::create_dir_all(&sub).unwrap();
        fs::write(sub.join("helper.rs"), "").unwrap();

        match list_files(&dir, "src/utils") {
            ToolResult::ListFiles { files } => {
                assert_eq!(files, vec!["helper.rs"]);
            }
            other => panic!("Expected ListFiles, got {:?}", other),
        }
    }

    // ─── run_command tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_run_command_echo() {
        let dir = tempdir();
        let cmd = if cfg!(windows) { "echo hello" } else { "echo hello" };
        match run_command(&dir, cmd, None).await {
            ToolResult::RunCommand { exit_code, stdout, stderr: _, duration_ms } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stdout.trim(), "hello");
                assert!(duration_ms < 10_000);
            }
            other => panic!("Expected RunCommand, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_run_command_captures_stderr() {
        let dir = tempdir();
        let cmd = if cfg!(windows) { "echo err 1>&2" } else { "echo err >&2" };
        match run_command(&dir, cmd, None).await {
            ToolResult::RunCommand { exit_code, stdout: _, stderr, duration_ms: _ } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stderr.trim(), "err");
            }
            other => panic!("Expected RunCommand, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_run_command_nonzero_exit() {
        let dir = tempdir();
        let cmd = if cfg!(windows) { "exit /b 42" } else { "exit 42" };
        match run_command(&dir, cmd, None).await {
            ToolResult::RunCommand { exit_code, .. } => {
                assert_eq!(exit_code, 42);
            }
            other => panic!("Expected RunCommand, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_run_command_timeout() {
        let dir = tempdir();
        let cmd = if cfg!(windows) { "ping -n 60 127.0.0.1" } else { "sleep 60" };
        match run_command(&dir, cmd, Some(1)).await {
            ToolResult::Error { message } => {
                assert!(message.contains("timed out"));
            }
            other => panic!("Expected timeout Error, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_run_command_working_directory() {
        let dir = tempdir();
        fs::write(dir.join("marker.txt"), "found_it").unwrap();
        let cmd = if cfg!(windows) { "type marker.txt" } else { "cat marker.txt" };
        match run_command(&dir, cmd, None).await {
            ToolResult::RunCommand { exit_code, stdout, .. } => {
                assert_eq!(exit_code, 0);
                assert_eq!(stdout.trim(), "found_it");
            }
            other => panic!("Expected RunCommand, got {:?}", other),
        }
    }

    // ─── truncate_output tests ───────────────────────────────────────────────

    #[test]
    fn test_truncate_output_short() {
        let s = "hello".to_string();
        assert_eq!(truncate_output(s, 100), "hello");
    }

    #[test]
    fn test_truncate_output_long() {
        let s = "a".repeat(200);
        let result = truncate_output(s, 100);
        assert!(result.len() < 200);
        assert!(result.ends_with("... [truncated]"));
    }

    // ─── helper ──────────────────────────────────────────────────────────────

    fn tempdir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sandbox_test_{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}

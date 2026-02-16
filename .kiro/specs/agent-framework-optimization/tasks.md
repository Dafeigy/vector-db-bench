# 实现计划: Agent 框架优化

## 概述

将设计分解为增量式编码任务，每步构建在前一步基础上。核心改动集中在 `bench_tools.rs`（服务器生命周期管理 + 路径配置化）、`main.rs`（CLI 参数）、`tools.rs`（dispatch 适配）、`state.rs`（finish 适配）和 `system_prompt.txt`。

## Tasks

- [x] 1. 新增 BenchConfig 结构体和 CLI 参数
  - [x] 1.1 在 `agent/src/bench_tools.rs` 中定义 `BenchConfig` 结构体（含 `benchmark_bin: PathBuf` 和 `data_dir: PathBuf` 字段），派生 Debug 和 Clone
    - _Requirements: 1.1, 1.2_
  - [x] 1.2 在 `agent/src/main.rs` 的 `Args` 结构体中新增 `--data-dir` 和 `--benchmark-bin` 可选参数
    - _Requirements: 1.1, 1.2_
  - [x] 1.3 在 `main()` 函数中实现默认路径解析逻辑：未指定时分别默认为 `{work_dir}/data` 和 `{work_dir}/benchmark/target/release/vector-db-benchmark`
    - _Requirements: 1.3, 1.4_
  - [x] 1.4 在 `main()` 函数中实现路径存在性验证：data_dir 不存在或 benchmark_bin 不存在时打印错误并退出
    - _Requirements: 1.5, 1.6_
  - [ ]* 1.5 为 BenchConfig 构造和默认路径解析编写属性测试
    - **Property 1: BenchConfig 路径传播**
    - **Property 2: 默认路径解析**
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4**
  - [ ]* 1.6 为无效路径验证编写属性测试
    - **Property 3: 无效路径验证**
    - **Validates: Requirements 1.5, 1.6**

- [x] 2. 实现服务器生命周期管理函数
  - [x] 2.1 在 `agent/src/bench_tools.rs` 中实现 `kill_process_on_port(port: u16)` 函数，使用 lsof/ss 查找并终止占用端口的进程
    - _Requirements: 2.7_
  - [x] 2.2 实现 `build_project(work_dir: &Path) -> Result<(), String>` 函数，执行 `cargo build --release`，超时 300 秒，失败时返回 stderr 内容
    - _Requirements: 2.5_
  - [x] 2.3 实现 `start_server(work_dir: &Path) -> Result<tokio::process::Child, String>` 函数，启动 `target/release/vector-db-skeleton` 二进制
    - _Requirements: 2.1, 2.2_
  - [x] 2.4 实现 `wait_for_server_ready(port: u16, timeout_secs: u64, poll_interval_ms: u64) -> Result<(), String>` 函数，通过 TCP 连接轮询端口就绪状态
    - _Requirements: 2.3, 2.4_
  - [x] 2.5 实现 `kill_server(child: &mut tokio::process::Child)` 函数，终止服务器进程
    - _Requirements: 2.6_
  - [ ]* 2.6 为 `wait_for_server_ready` 编写属性测试
    - **Property 4: 服务器健康检查轮询**
    - **Validates: Requirements 2.3, 2.4**

- [x] 3. Checkpoint - 确保生命周期管理函数编译通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 4. 重构 bench_tools 核心函数以集成生命周期管理和路径配置
  - [x] 4.1 实现数据文件发现函数 `find_base_vectors(data_dir: &Path, work_dir: &Path) -> Result<PathBuf, String>`，支持单文件和分片文件两种布局
    - _Requirements: 3.4, 3.5_
  - [ ]* 4.2 为数据文件发现编写属性测试
    - **Property 6: 数据文件发现**
    - **Validates: Requirements 3.4, 3.5**
  - [x] 4.3 重构 `run_benchmark` 函数：签名改为 `(work_dir, config, concurrency, warmup)`，内部集成完整的服务器生命周期管理流程（kill_port → build → start → wait → benchmark → kill）
    - _Requirements: 2.1, 3.1, 3.2_
  - [x] 4.4 重构 `run_correctness_test` 函数：签名改为 `(work_dir, config)`，内部集成相同的服务器生命周期管理流程
    - _Requirements: 2.2, 3.3_
  - [x] 4.5 更新 `run_profiling` 函数，移除手动查找服务器 PID 的逻辑（profiling 仍需服务器已运行，但可以在文档中说明）
    - _Requirements: 2.1_

- [x] 5. 适配 dispatch 层和 state 模块
  - [x] 5.1 修改 `tools.rs` 中 `dispatch_tool_call` 的签名，新增 `config: &BenchConfig` 参数，并将其传递给 `run_benchmark`、`run_correctness_test` 和 `state.finish`
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 5.2 修改 `state.rs` 中 `AgentState::finish` 的签名，新增 `config: &BenchConfig` 参数，传递给内部的 `run_benchmark` 调用
    - _Requirements: 6.3_
  - [x] 5.3 更新 `main.rs` 中的 agent 主循环，构造 `BenchConfig` 并传递给 `dispatch_tool_call`
    - _Requirements: 1.1, 1.2, 6.1, 6.2, 6.3_

- [x] 6. Checkpoint - 确保完整编译通过和现有测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 7. 更新系统提示词和评测脚本
  - [x] 7.1 更新 `agent/system_prompt.txt`：添加 work_dir 说明、工具自动管理说明、工具调用分配建议、初始文件结构列表
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  - [x] 7.2 更新 `scripts/run_eval.sh`：在 Agent 启动命令中添加 `--data-dir "${DATA_DIR}"` 和 `--benchmark-bin "${BENCHMARK_BIN}"` 参数，并在启动前验证路径存在
    - _Requirements: 5.1, 5.2, 5.3_

- [x] 8. 更新现有测试以适配新签名
  - [x] 8.1 更新 `bench_tools.rs` 中的现有测试，传入 `BenchConfig` 参数
    - _Requirements: 3.1, 3.2, 3.3_
  - [x] 8.2 更新 `tools.rs` 中 `dispatch_tool_call` 的测试，传入 `BenchConfig` 参数
    - _Requirements: 6.1, 6.2, 6.3_
  - [x] 8.3 更新 `state.rs` 中 `finish` 相关的测试（如有）
    - _Requirements: 6.3_

- [x] 9. Final checkpoint - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP
- 每个任务引用了具体的需求编号以确保可追溯性
- Checkpoint 任务确保增量验证
- 属性测试需要在 `agent/Cargo.toml` 的 `[dev-dependencies]` 中添加 `proptest`
- `run_profiling` 的改动较小，因为 profiling 本身需要服务器已在运行状态（由模型通过 run_command 启动），这与 benchmark/correctness 的自动管理模式不同

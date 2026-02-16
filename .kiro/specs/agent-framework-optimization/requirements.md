# 需求文档

## 简介

优化 Vector DB Bench Agent 框架，解决模型在评测过程中浪费大量工具调用在环境探索（定位 benchmark 二进制文件、数据目录、管理服务器生命周期）上的问题。通过让工具自动处理服务器生命周期管理、通过 CLI 参数配置外部资源路径、以及改进系统提示词，使模型能够将有限的 50 次工具调用集中在实际的实现和优化工作上。

## 术语表

- **Agent**: 基于 LLM 的自动化评测代理，通过工具调用与环境交互来实现和优化向量数据库
- **work_dir**: Agent 的工作目录，包含 skeleton 项目代码，所有文件操作和命令执行都相对于此目录
- **benchmark_bin**: 预编译的基准测试二进制文件路径（`vector-db-benchmark`），用于测量 QPS 和 recall
- **data_dir**: 包含 `base_vectors_*.json`、`query_vectors.json`、`ground_truth.json` 的数据目录
- **skeleton_server**: 模型在 work_dir 中实现的向量数据库 HTTP 服务，监听 8080 端口
- **bench_tools**: Agent 框架中负责执行 benchmark、correctness test 和 profiling 的工具模块
- **System_Prompt**: 发送给 LLM 的系统提示词，描述任务、环境和可用工具

## 需求

### 需求 1: CLI 参数扩展

**用户故事:** 作为评测运维人员，我希望通过 CLI 参数配置 data_dir 和 benchmark_bin 路径，以便在不同环境中灵活运行 Agent 而无需修改代码。

#### 验收标准

1. WHEN Agent 启动时指定 `--data-dir` 参数, THE Agent SHALL 将该路径传递给 bench_tools 模块用于定位数据文件
2. WHEN Agent 启动时指定 `--benchmark-bin` 参数, THE Agent SHALL 将该路径传递给 bench_tools 模块用于定位 benchmark 二进制文件
3. WHEN `--data-dir` 参数未指定, THE Agent SHALL 使用 `{work_dir}/data` 作为默认数据目录
4. WHEN `--benchmark-bin` 参数未指定, THE Agent SHALL 使用 `{work_dir}/benchmark/target/release/vector-db-benchmark` 作为默认 benchmark 二进制路径
5. IF 指定的 data_dir 路径不存在, THEN THE Agent SHALL 在启动时输出错误信息并以非零退出码退出
6. IF 指定的 benchmark_bin 路径不存在, THEN THE Agent SHALL 在启动时输出错误信息并以非零退出码退出

### 需求 2: 服务器自动生命周期管理

**用户故事:** 作为 LLM 模型，我希望 `run_benchmark` 和 `run_correctness_test` 工具自动处理服务器的构建、启动、等待就绪、测试和停止，以便我不需要手动管理服务器进程。

#### 验收标准

1. WHEN `run_benchmark` 被调用, THE bench_tools SHALL 自动执行以下完整流程：在 work_dir 中构建项目（`cargo build --release`）、启动服务器二进制文件、等待服务器就绪、执行基准测试、停止服务器进程
2. WHEN `run_correctness_test` 被调用, THE bench_tools SHALL 自动执行与 `run_benchmark` 相同的服务器生命周期管理流程
3. WHEN 服务器启动后, THE bench_tools SHALL 通过轮询 HTTP 端口（8080）来检测服务器是否就绪，轮询间隔为 200 毫秒，超时时间为 30 秒
4. IF 服务器在超时时间内未就绪, THEN THE bench_tools SHALL 终止服务器进程并返回包含超时信息的错误结果
5. IF `cargo build --release` 构建失败, THEN THE bench_tools SHALL 返回包含编译错误输出的错误结果，不启动服务器
6. WHEN 测试完成后（无论成功或失败）, THE bench_tools SHALL 终止服务器进程并确保端口 8080 被释放
7. IF 端口 8080 已被占用, THEN THE bench_tools SHALL 先终止占用该端口的进程，再启动新的服务器

### 需求 3: bench_tools 路径配置化

**用户故事:** 作为开发者，我希望 bench_tools 模块通过参数接收 data_dir 和 benchmark_bin 路径，而不是使用硬编码的相对路径常量，以便支持灵活的部署配置。

#### 验收标准

1. THE `run_benchmark` 函数 SHALL 接受 `data_dir` 参数用于定位 base_vectors、query_vectors 和 ground_truth 文件
2. THE `run_benchmark` 函数 SHALL 接受 `benchmark_bin` 参数用于定位 benchmark 二进制文件
3. THE `run_correctness_test` 函数 SHALL 接受与 `run_benchmark` 相同的 `data_dir` 和 `benchmark_bin` 参数
4. WHEN `data_dir` 中包含分片的 base_vectors 文件（`base_vectors_0.json` 到 `base_vectors_9.json`）, THE bench_tools SHALL 将所有分片文件路径传递给 benchmark 二进制文件
5. WHEN `data_dir` 中包含单个 `base_vectors.json` 文件, THE bench_tools SHALL 将该单一文件路径传递给 benchmark 二进制文件

### 需求 4: 系统提示词优化

**用户故事:** 作为 LLM 模型，我希望系统提示词清楚地告诉我工作环境的关键信息，以便我不需要浪费工具调用来探索环境。

#### 验收标准

1. THE System_Prompt SHALL 明确说明所有文件操作和命令执行都相对于 work_dir 进行
2. THE System_Prompt SHALL 明确说明 `run_benchmark`、`run_correctness_test` 工具会自动处理项目构建和服务器生命周期管理，模型无需手动启动或停止服务器
3. THE System_Prompt SHALL 明确说明 benchmark 数据和二进制文件由工具自动管理，模型无需定位这些外部资源
4. THE System_Prompt SHALL 包含建议的工具调用分配策略，引导模型将工具调用集中在实现和优化上
5. THE System_Prompt SHALL 明确列出 work_dir 中的初始文件结构（src/main.rs、src/api.rs、src/db.rs、src/distance.rs、Cargo.toml）

### 需求 5: run_eval.sh 脚本适配

**用户故事:** 作为评测运维人员，我希望 run_eval.sh 脚本在启动 Agent 时传递正确的 `--data-dir` 和 `--benchmark-bin` 参数，以便评测流程能够正确运行。

#### 验收标准

1. WHEN run_eval.sh 启动 Agent, THE 脚本 SHALL 传递 `--data-dir` 参数指向项目的 data 目录
2. WHEN run_eval.sh 启动 Agent, THE 脚本 SHALL 传递 `--benchmark-bin` 参数指向已构建的 benchmark 二进制文件路径
3. THE run_eval.sh SHALL 在启动 Agent 之前验证 data_dir 和 benchmark_bin 路径存在

### 需求 6: dispatch 层适配

**用户故事:** 作为开发者，我希望工具调度层能够将配置好的 data_dir 和 benchmark_bin 路径正确传递给 bench_tools 函数，以便整个调用链路完整。

#### 验收标准

1. THE `dispatch_tool_call` 函数 SHALL 将 data_dir 和 benchmark_bin 配置传递给 `run_benchmark` 调用
2. THE `dispatch_tool_call` 函数 SHALL 将 data_dir 和 benchmark_bin 配置传递给 `run_correctness_test` 调用
3. THE `dispatch_tool_call` 函数 SHALL 将 data_dir 和 benchmark_bin 配置传递给 `state.finish` 调用（因为 finish 内部会调用 run_benchmark）

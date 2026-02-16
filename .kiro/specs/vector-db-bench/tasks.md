# 实现计划：Vector DB Bench

## 概述

将评测系统分为 5 个主要阶段实现：骨架代码、Benchmark Client、数据预处理脚本、Agent 框架、评测流程编排。每个阶段包含核心实现和可选的测试任务。

## 任务

- [x] 1. 搭建项目结构和骨架代码
  - [x] 1.1 创建 skeleton 项目结构和 Cargo.toml
    - 创建 `skeleton/Cargo.toml`，添加 axum、tokio、serde、serde_json 依赖
    - 配置 release profile（默认 LTO、codegen-units=1）
    - 创建 `skeleton/src/` 目录结构
    - _Requirements: 1.1, 1.8_

  - [x] 1.2 实现 API 类型定义（api.rs）
    - 创建 `skeleton/src/api.rs`
    - 定义 InsertRequest、InsertResponse、BulkInsertRequest、BulkInsertResponse、SearchRequest、SearchResult、SearchResponse 结构体
    - 使用 serde 的 Serialize/Deserialize 派生宏
    - _Requirements: 1.2, 1.3, 1.4_

  - [x] 1.3 实现 VectorDB 桩代码（db.rs）
    - 创建 `skeleton/src/db.rs`
    - 定义 VectorDB 结构体，包含 new()、insert()、bulk_insert()、search() 方法签名
    - 所有方法体使用 `todo!("模型实现")` 占位
    - _Requirements: 1.6_

  - [x] 1.4 实现距离计算桩代码（distance.rs）
    - 创建 `skeleton/src/distance.rs`
    - 定义 `l2_distance(a: &[f32], b: &[f32]) -> f64` 函数签名
    - 方法体使用 `todo!("模型实现")` 占位
    - _Requirements: 1.7_

  - [x] 1.5 实现 HTTP 服务入口（main.rs）
    - 创建 `skeleton/src/main.rs`
    - 使用 axum 定义路由：POST /insert、POST /bulk_insert、POST /search
    - 实现 handler 函数，调用 VectorDB 的对应方法
    - 使用 Arc<VectorDB> 作为共享状态
    - 监听 0.0.0.0:8080
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 1.6 编写骨架代码单元测试
    - 测试 API 类型的 JSON 序列化/反序列化
    - 测试请求体格式是否符合规范
    - _Requirements: 1.2, 1.3, 1.4_

- [x] 2. 实现 Benchmark Client
  - [x] 2.1 创建 benchmark 项目结构
    - 创建 `benchmark/Cargo.toml`，添加 tokio、reqwest、serde、serde_json、rand、clap 依赖
    - 创建 `benchmark/src/main.rs` 入口
    - _Requirements: 2.1_

  - [x] 2.2 实现数据加载模块
    - 实现从 JSON 文件读取 base vectors 的功能
    - 实现通过 HTTP `/bulk_insert` 接口批量加载向量的功能
    - 支持分批发送（每批 1000-5000 条）以避免请求体过大
    - _Requirements: 2.1_

  - [x] 2.3 实现并发查询运行器
    - 实现线程池并发查询逻辑（默认 4 线程，可配置）
    - 实现 query 顺序随机打乱（使用可配置的随机种子）
    - 实现预热查询逻辑（默认 1000 条，不计入评分）
    - 记录每个请求的延迟和返回结果
    - _Requirements: 2.2, 2.3, 2.6_

  - [x] 2.4 实现评分计算模块
    - 实现 QPS 计算：总查询数 / 总耗时秒数
    - 实现延迟分位数计算：P50、P95、P99
    - 实现 recall 计算：模型 Top-10 与 Ground Truth Top-10 的交集 / 10
    - 实现 JSON 格式的 BenchmarkResult 输出
    - _Requirements: 2.4, 2.5, 2.7_

  - [ ]* 2.5 编写 Benchmark Client 属性测试
    - **Property 4: Recall 计算正确性**
    - **Validates: Requirements 2.5**
    - **Property 5: 延迟分位数有序性**
    - **Validates: Requirements 2.4**
    - **Property 6: Benchmark 结果序列化 Round Trip**
    - **Validates: Requirements 2.7**

  - [x] 2.6 实现 Benchmark Client CLI 入口
    - 使用 clap 解析命令行参数（server_url、concurrency、warmup、数据文件路径等）
    - 串联数据加载 → 预热 → 并发查询 → 评分计算的完整流程
    - 输出 JSON 格式的 benchmark 结果到 stdout
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

- [x] 3. Checkpoint - 确保骨架代码和 Benchmark Client 编译通过
  - 确保 `cargo build --release` 在 skeleton 和 benchmark 项目中都能成功
  - 确保所有测试通过，如有问题请询问用户

- [x] 4. 实现数据预处理脚本
  - [x] 4.1 实现数据集下载脚本
    - 创建 `scripts/download_dataset.py`
    - 从 HuggingFace 或 corpus-texmex.irisa.fr 下载 SIFT1M 数据集
    - 下载 sift_base.fvecs、sift_query.fvecs、sift_groundtruth.ivecs
    - 保存到 `data/` 目录
    - _Requirements: 4.5_

  - [x] 4.2 实现 fvecs/ivecs 格式解析和 JSON 转换
    - 创建 `scripts/convert_data.py`
    - 实现 fvecs 格式解析：读取 [dim: i32][v0: f32]...[v_{dim-1}: f32] 序列
    - 实现 ivecs 格式解析：读取 [dim: i32][v0: i32]...[v_{dim-1}: i32] 序列
    - 将 base vectors 转换为 JSON 格式（支持分片输出以控制文件大小）
    - 将 query vectors 转换为 JSON 格式
    - 将 ground truth 转换为 JSON 格式
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [x] 4.3 实现 Ground Truth 生成脚本
    - 创建 `scripts/generate_ground_truth.py`
    - 读取 base vectors 和 query vectors
    - 对每条 query 暴力计算与所有 base vectors 的 L2 距离
    - 取 Top-100 最近邻 ID，输出为 ground_truth.json
    - _Requirements: 4.6_

  - [ ]* 4.4 编写数据预处理属性测试
    - **Property 11: fvecs/ivecs 解析 Round Trip**
    - **Validates: Requirements 4.1, 4.2, 4.3**
    - **Property 12: 数据格式转换 Round Trip**
    - **Validates: Requirements 4.4**
    - **Property 13: 暴力搜索 Ground Truth 正确性**
    - **Validates: Requirements 4.6**
    - 使用 pytest + hypothesis 进行 Python 属性测试

  - [x] 4.5 实现数据加载脚本
    - 创建 `scripts/load_data.py`（只读脚本）
    - 读取 JSON 格式的 base vectors
    - 通过 HTTP `/bulk_insert` 接口批量加载到被测服务
    - 支持配置 batch size 和 server URL
    - _Requirements: 2.1_

- [x] 5. Checkpoint - 确保数据预处理脚本正常工作
  - 确保所有 Python 脚本语法正确
  - 确保所有测试通过，如有问题请询问用户

- [x] 6. 实现 Agent 框架
  - [x] 6.1 创建 Agent 项目结构
    - 创建 `agent/Cargo.toml`，添加 tokio、serde、serde_json、reqwest、clap 依赖
    - 创建 `agent/src/` 目录结构（main.rs、tools.rs、sandbox.rs、evaluator.rs）
    - _Requirements: 3.1_

  - [x] 6.2 实现 Tool Call 类型定义和路由
    - 在 `agent/src/tools.rs` 中定义 ToolCall 和 ToolResult 枚举
    - 实现 JSON Schema 格式的工具描述（供 LLM function calling 使用）
    - 实现 tool call 路由分发逻辑
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9_

  - [x] 6.3 实现文件操作工具（read_file、write_file、list_files）
    - 实现 read_file：读取指定路径文件内容
    - 实现 write_file：写入内容到指定路径，检查只读文件列表
    - 实现 list_files：列出指定目录下的文件
    - 实现只读文件权限检查逻辑
    - _Requirements: 3.1, 3.2, 3.3, 3.12_

  - [ ]* 6.4 编写文件操作属性测试
    - **Property 7: 文件读写 Round Trip**
    - **Validates: Requirements 3.1, 3.2**
    - **Property 8: 文件列表包含性**
    - **Validates: Requirements 3.3**
    - **Property 10: 只读文件写入保护**
    - **Validates: Requirements 3.12, 7.2**

  - [x] 6.5 实现命令执行工具（run_command）
    - 实现 shell 命令执行，支持超时控制（默认 120 秒）
    - 捕获 stdout、stderr、退出码、执行时长
    - 超时时终止子进程
    - _Requirements: 3.4_

  - [x] 6.6 实现 Benchmark 和 Profiling 工具
    - 实现 run_benchmark：启动 Benchmark Client 进程，解析 JSON 输出
    - 实现 run_profiling：使用 perf/flamegraph 工具运行性能分析
    - 实现 run_correctness_test：运行正确性校验
    - _Requirements: 3.5, 3.6, 3.7_

  - [x] 6.7 实现状态管理和 finish 工具
    - 实现 AgentState 结构体，跟踪 tool call 计数、时间、日志
    - 实现 get_status：返回当前状态
    - 实现 finish：触发最终 benchmark，记录成绩
    - 实现 50 次 tool call 上限检查，超限自动触发 finish
    - _Requirements: 3.8, 3.9, 3.10, 3.11, 3.13_

  - [ ]* 6.8 编写 Agent 状态管理属性测试
    - **Property 9: Tool Call 计数不变量**
    - **Validates: Requirements 3.8, 3.10, 3.13**

  - [x] 6.9 实现 Agent 主循环（main.rs）
    - 实现 LLM API 客户端（支持 OpenAI 兼容接口）
    - 实现 Tool Call 循环：发送上下文 → 接收 tool call → 执行 → 返回结果
    - 实现系统 Prompt 加载和会话管理
    - 实现 tool call 日志记录
    - _Requirements: 3.1, 3.10, 3.13_

- [x] 7. Checkpoint - 确保 Agent 框架编译通过
  - 确保 `cargo build --release` 在 agent 项目中成功
  - 确保所有测试通过，如有问题请询问用户

- [x] 8. 实现评测流程和评分逻辑
  - [x] 8.1 实现评分逻辑（evaluator.rs）
    - 实现 recall 不达标时 QPS 归零逻辑
    - 实现排行榜排序：QPS 降序，相同 QPS 时 recall 降序
    - 实现排行榜条目的序列化和持久化
    - _Requirements: 5.2, 5.3_

  - [ ]* 8.2 编写评分逻辑属性测试
    - **Property 14: Recall 不达标时 QPS 归零**
    - **Validates: Requirements 5.2**
    - **Property 15: 排行榜排序正确性**
    - **Validates: Requirements 5.3**

  - [x] 8.3 实现系统 Prompt 和公平性保障
    - 创建系统 Prompt 文本文件
    - 确保 Prompt 仅包含功能需求和工具说明
    - 确保不包含任何具体算法名称或优化技术名称
    - _Requirements: 5.4, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5_

  - [ ]* 8.4 编写 Prompt 公平性属性测试
    - **Property 16: Prompt 黑名单关键词检查**
    - **Validates: Requirements 5.5, 6.3, 6.4**

  - [x] 8.5 实现反作弊检测
    - 实现硬编码结果检测：分析不同 query 返回结果的多样性
    - 集成到 Benchmark Client 的结果分析中
    - _Requirements: 5.7, 7.4_

- [x] 9. 实现评测流程脚本
  - [x] 9.1 创建完整评测流程脚本（run_eval.sh）
    - 实现数据集下载检查和下载
    - 实现数据预处理（fvecs → JSON）
    - 实现 ground truth 生成检查
    - 实现骨架代码初始化到工作目录
    - 实现 Agent 框架启动和参数传递
    - 实现最终结果收集和排行榜更新
    - _Requirements: 5.1, 5.8_

- [ ] 10. 集成测试和最终验证
  - [ ]* 10.1 编写向量插入-搜索集成属性测试
    - **Property 1: 向量插入-搜索 Round Trip**
    - **Validates: Requirements 1.2, 1.4**
    - **Property 2: 搜索结果距离有序性**
    - **Validates: Requirements 1.4**
    - **Property 3: Bulk Insert 计数一致性**
    - **Validates: Requirements 1.3**
    - 需要启动骨架服务（使用暴力搜索的简单实现）进行集成测试

  - [ ]* 10.2 编写 Query 随机化属性测试
    - **Property 17: Query 顺序随机化**
    - **Validates: Requirements 2.6, 5.6, 7.1**

- [x] 11. Final Checkpoint - 确保所有组件编译通过且测试通过
  - 确保所有 Rust 项目（skeleton、benchmark、agent）编译通过
  - 确保所有 Python 脚本语法正确
  - 确保所有测试通过，如有问题请询问用户

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加速 MVP 开发
- 每个任务引用了具体的需求编号以确保可追溯性
- Checkpoint 任务确保增量验证
- 属性测试验证通用正确性属性，单元测试验证具体示例和边界条件
- 骨架代码中的 `main.rs` 和 `api.rs` 为只读文件，实现时需注意标记
- Benchmark Client 使用 Rust (tokio + reqwest) 实现，数据预处理使用 Python
- Agent 框架使用 Rust 实现，支持 OpenAI 兼容的 LLM API

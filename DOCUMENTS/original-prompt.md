# Vector DB Bench — 大模型后端代码能力评测项目

## 项目概述

通过让大模型从零实现一个简单的向量数据库（仅支持插入和搜索），并进行极限性能优化，以最终的搜索 QPS 作为核心指标，量化评估大模型的后端工程能力。

项目以天梯排行榜形式呈现结果，观众只需关注一个数字：QPS。

## 评测流程

```
给出骨架代码 + Prompt
        ↓
模型实现基础版本（插入 + 搜索）
        ↓
模型自主提出 5 个优化方向并排序
        ↓
模型自主迭代优化（共 50 次 tool call 机会）
        ↓
最终 benchmark：正确性校验 + QPS 测试
        ↓
归一化为 QPS 分数，进入天梯
```

## 核心设计原则

### 约束（固定不可修改）

- 语言：Rust
- API 接口：HTTP，固定 `/insert` 和 `/search` 两个端点
- 请求/响应 JSON Schema 固定
- 向量维度：128 维 float32
- 距离度量：L2 欧氏距离
- Top-K：10
- Benchmark Client：只读，模型不可修改
- 数据加载脚本：只读，模型不可修改
- 启动方式：监听 `0.0.0.0:8080`，通过 `cargo run --release` 启动
- 编译模式：release（模型可调整 Cargo.toml 中的 release profile 参数，如 LTO、codegen-units）

### 自由（模型可自由发挥）

- 索引结构（暴力搜索、HNSW、IVF、KD-Tree、PQ 等）
- 内存管理策略
- 并发模型（线程池、async、lock-free 等）
- SIMD 指令集优化（SSE、AVX2、AVX-512）
- 内联汇编
- 数据结构与算法选择
- Cargo.toml 中的 release profile 配置
- 任何不依赖外部向量数据库库的优化手段

### 禁止

- 使用现成的向量搜索库（如 faiss、annoy、hnsw_rs、usearch 等）
- 基础数学运算库允许使用（如 BLAS 绑定）
- 网络访问（沙箱环境中隔离）
- 修改 benchmark client 和数据加载脚本

## 数据集

### SIFT1M

来源：[qbo-odp/sift1m (HuggingFace)](https://huggingface.co/datasets/qbo-odp/sift1m)，原始数据来自 [corpus-texmex.irisa.fr](http://corpus-texmex.irisa.fr/)

引用：Jégou H, Douze M, Schmid C. Improving bag-of-features for large scale image search. International Journal of Computer Vision, 2010, 87(3): 316-336.

数据集构成：
- Base vectors：1,000,000 条 128 维 float32 向量（用于插入）
- Query vectors：10,000 条 128 维 float32 向量（用于搜索 benchmark）
- Ground truth：每条 query 对应的精确 Top-100 最近邻 ID（用于 recall 校验）

### 数据格式转换

原始 SIFT1M 使用 `.fvecs` / `.ivecs` 二进制格式。我们需要编写一个预处理脚本，将其转换为项目使用的 JSON 格式，供数据加载脚本和 benchmark client 使用。

### 数据规模

| 项目 | 数量 |
|------|------|
| 插入向量数 | 1,000,000 |
| 查询向量数 | 10,000 |
| 向量维度 | 128 |
| Top-K | 10 |

## API 接口定义

### POST /insert

请求体：
```json
{
  "id": 42,
  "vector": [0.1, 0.2, ..., 0.128]
}
```

响应体：
```json
{
  "status": "ok"
}
```

### POST /search

请求体：
```json
{
  "vector": [0.1, 0.2, ..., 0.128],
  "top_k": 10
}
```

响应体：
```json
{
  "results": [
    {"id": 42, "distance": 0.123},
    {"id": 7, "distance": 0.456},
    ...
  ]
}
```

results 按 distance 升序排列（距离越小越相似），返回 top_k 条结果。

### POST /bulk_insert

为了加速数据加载阶段（非 benchmark 计分），提供批量插入接口：

请求体：
```json
{
  "vectors": [
    {"id": 0, "vector": [0.1, 0.2, ...]},
    {"id": 1, "vector": [0.3, 0.4, ...]},
    ...
  ]
}
```

响应体：
```json
{
  "status": "ok",
  "inserted": 1000
}
```

## Agent 框架设计

### 交互模式

模型以 Tool Call Agent 的形式与评测环境交互，模拟一个 terminal coding agent。

### 单次会话限制

- 单次会话上下文窗口：128K tokens
- 总 tool call 次数：50 次
- 模型自主管理轮次分配

### Tool Call 定义

#### 1. 文件操作

**read_file**
```
输入：{ "path": "src/main.rs" }
输出：{ "content": "..." }
```

**write_file**
```
输入：{ "path": "src/main.rs", "content": "..." }
输出：{ "status": "ok", "bytes_written": 1234 }
```

**list_files**
```
输入：{ "path": "src/" }
输出：{ "files": ["main.rs", "lib.rs", "index.rs"] }
```

#### 2. 编译与执行

**run_command**
```
输入：{ "command": "cargo build --release", "timeout": 120 }
输出：{ "exit_code": 0, "stdout": "...", "stderr": "...", "duration_ms": 45000 }
```

timeout 单位为秒，默认 120 秒。用于编译、运行测试、执行自定义脚本等。

#### 3. Benchmark 与 Profiling

**run_benchmark**
```
输入：{ "concurrency": 4, "warmup": 1000 }
输出：{
  "qps": 12345.6,
  "total_queries": 10000,
  "duration_secs": 0.81,
  "avg_latency_ms": 0.32,
  "p50_latency_ms": 0.28,
  "p95_latency_ms": 0.55,
  "p99_latency_ms": 1.2,
  "recall": 0.967,
  "recall_threshold": 0.95,
  "recall_passed": true,
  "concurrency": 4
}
```

concurrency 默认为 4，可调节。warmup 默认为 1000。

**run_profiling**
```
输入：{ "duration": 30 }
输出：{
  "flamegraph_svg_path": "/tmp/flamegraph.svg",
  "top_functions": [
    {"function": "vector_db::index::search", "percentage": 35.2},
    {"function": "vector_db::distance::l2", "percentage": 28.1},
    {"function": "core::slice::sort", "percentage": 12.4},
    ...
  ],
  "total_samples": 50000
}
```

返回文本化的热点函数摘要 + SVG 火焰图路径。模型可通过 read_file 读取 SVG 源码进行深入分析。

**run_correctness_test**
```
输入：{}
输出：{
  "passed": true,
  "total_queries": 10000,
  "recall": 0.967,
  "recall_threshold": 0.95,
  "failed_queries": [],
  "message": "Correctness test passed: recall 96.7% >= 95.0%"
}
```

#### 4. 状态管理

**get_status**
```
输入：{}
输出：{
  "tool_calls_used": 12,
  "tool_calls_remaining": 38,
  "tool_calls_total": 50,
  "elapsed_time_secs": 320,
  "server_running": true,
  "last_benchmark": {
    "qps": 12345.6,
    "recall": 0.967
  }
}
```

**finish**
```
输入：{ "summary": "实现了 HNSW 索引 + SIMD L2 距离计算 + 内存池优化" }
输出：{
  "status": "completed",
  "final_benchmark": {
    "qps": 23456.7,
    "recall": 0.971,
    "recall_passed": true
  }
}
```

模型调用 finish 后，评测结束。系统自动运行最终 benchmark 并记录成绩。

## Prompt 设计

### 系统 Prompt（对所有模型完全相同）

```
你是一个后端工程师。你的任务是实现一个高性能的向量搜索引擎。

项目骨架已经准备好，包含固定的 HTTP API 接口定义和项目结构。
你需要实现向量的插入和搜索功能，然后尽可能优化搜索性能。

要求：
1. 实现 /insert、/bulk_insert 和 /search 三个 API 端点
2. 向量维度为 128，数据类型为 float32
3. 距离度量为 L2 欧氏距离
4. /search 返回距离最近的 top-10 结果，按距离升序排列
5. 搜索结果的 recall 必须 >= 95%（与暴力搜索的精确结果对比）
6. 在保证 recall 的前提下，尽可能提高搜索 QPS
7. 不允许使用任何现成的向量搜索库
8. 你有 50 次 tool call 机会，请合理规划

建议的工作流程：
1. 阅读项目骨架代码，理解接口定义
2. 实现基础版本，确保正确性
3. 运行 benchmark 获取基线性能
4. 分析性能瓶颈，提出优化方案
5. 逐步实施优化，每次优化后验证正确性和性能
6. 完成后调用 finish

你可以使用以下工具：
- read_file / write_file / list_files：文件操作
- run_command：执行 shell 命令（编译、运行等）
- run_benchmark：运行性能测试
- run_profiling：运行性能分析，生成火焰图
- run_correctness_test：运行正确性测试
- get_status：查看剩余 tool call 次数和当前状态
- finish：完成评测
```

### Prompt 公平性原则

- 不提及任何具体的索引算法名称（HNSW、IVF、PQ 等）
- 不提及任何具体的优化技术（SIMD、内存池、cache-line 等）
- 不暗示任何优化方向
- 只描述功能需求和性能目标
- 所有模型使用完全相同的 prompt

## 评分规则

### 正确性门槛

- 最终 benchmark 的 recall 必须 >= 95%（默认阈值，可配置）
- recall 不达标的模型，QPS 记为 0，排名垫底
- recall 计算方式：对每条 query，模型返回的 top-10 结果与 ground truth top-10 的交集大小 / 10，取所有 query 的平均值

### QPS 计算

- 使用固定并发数（默认 4 线程）的 benchmark client
- 预热 1000 次查询后开始计时
- 对全部 10,000 条 query 进行搜索
- QPS = 总查询数 / 总耗时（秒）

### 天梯排名

按最终 QPS 降序排列。相同 QPS 时，recall 更高的排名靠前。

## 项目骨架结构

```
vector-db-bench/
├── DOCUMENTS/
│   └── original-prompt.md          # 本文档
├── skeleton/                        # 骨架代码（提供给模型的初始代码）
│   ├── Cargo.toml                   # 项目配置（模型可修改 release profile）
│   ├── src/
│   │   ├── main.rs                  # HTTP 服务入口（固定路由，模型实现 handler）
│   │   ├── api.rs                   # API 请求/响应类型定义（固定）
│   │   ├── db.rs                    # 向量数据库核心（模型主要实现区域）
│   │   └── distance.rs              # 距离计算（模型实现区域）
│   └── benches/                     # （可选）模型可添加自己的 micro-benchmark
├── benchmark/                       # Benchmark Client（只读）
│   ├── Cargo.toml
│   └── src/
│       └── main.rs                  # benchmark 主程序
├── scripts/
│   ├── load_data.py                 # 数据加载脚本（只读）
│   ├── download_dataset.py          # 数据集下载脚本
│   ├── generate_ground_truth.py     # Ground truth 生成脚本
│   └── run_eval.sh                  # 完整评测流程脚本
├── data/                            # 数据目录（运行时生成）
│   ├── sift_base.fvecs              # 100 万条 base vectors
│   ├── sift_query.fvecs             # 1 万条 query vectors
│   └── ground_truth.json            # 预计算的 ground truth
└── agent/                           # Agent 框架
    ├── Cargo.toml
    └── src/
        ├── main.rs                  # Agent 主程序
        ├── tools.rs                 # Tool Call 实现
        ├── sandbox.rs               # 沙箱管理
        └── evaluator.rs             # 评分逻辑
```

## Benchmark Client 设计

### 工作流程

1. 启动被测向量数据库服务
2. 通过 `/bulk_insert` 加载全部 100 万条 base vectors
3. 等待服务就绪
4. 预热：发送 1000 条随机 query（不计分）
5. 正式测试：使用 N 个并发线程，发送全部 10,000 条 query
6. 收集结果：计算 QPS、延迟分位数、recall

### 并发模型

- 使用线程池，默认 4 个工作线程
- 每个线程从 query 队列中取任务，发送 HTTP 请求
- 记录每个请求的延迟和返回结果

### Recall 计算

```
对每条 query q:
  model_results = 模型返回的 top-10 ID 集合
  truth_results = ground truth 的 top-10 ID 集合
  recall_q = |model_results ∩ truth_results| / 10

overall_recall = mean(recall_q for all q)
```

## 反作弊措施

- Benchmark client 只读，模型无法修改
- Query 顺序在 benchmark 时随机打乱（防止缓存预测）
- 每次 benchmark 使用不同的随机种子打乱 query 顺序
- Ground truth 由暴力搜索预计算，独立于模型实现
- 检测模型是否返回了硬编码结果（通过对比不同 query 的返回模式）

## 沙箱环境（后续容器化时实现）

- Docker 容器运行
- 固定 CPU 核数和内存上限
- 网络隔离（无外网访问）
- 包安装白名单（仅允许 Rust crates，禁止向量搜索相关 crate）
- 文件系统权限控制（骨架代码中标记为只读的文件不可修改）

## 技术选型

| 组件 | 技术 |
|------|------|
| 被测服务 | Rust |
| HTTP 框架（骨架） | actix-web 或 axum |
| Benchmark Client | Rust（tokio + reqwest） |
| 数据预处理脚本 | Python |
| Agent 框架 | Rust 或 Python |
| 容器化 | Docker |

## 后续扩展

- 支持不同维度的数据集（256、768 等）
- 支持不同距离度量（余弦相似度、内积）
- 支持不同数据规模（10 万、100 万、1000 万）
- 多轮评测：同一模型多次运行取最优
- 增加内存效率评分维度

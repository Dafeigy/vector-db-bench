# 需求文档

## 简介

Vector DB Bench 是一个大模型后端代码能力评测项目。通过让大模型从零实现一个简单的向量数据库（仅支持插入和搜索），并进行极限性能优化，以最终的搜索 QPS 作为核心指标，量化评估大模型的后端工程能力。项目以天梯排行榜形式呈现结果。

评测流程为：提供骨架代码和 Prompt → 模型实现基础版本 → 模型自主迭代优化（50 次 tool call 机会）→ 最终 benchmark 评分 → 进入天梯排行榜。

## 术语表

- **Skeleton（骨架代码）**：提供给被测模型的初始 Rust 项目代码，包含固定的 HTTP API 路由和类型定义，模型在此基础上实现核心逻辑
- **Benchmark_Client（基准测试客户端）**：只读的性能测试程序，负责向被测服务发送查询请求并收集 QPS、延迟、recall 等指标
- **Agent_Framework（Agent 框架）**：Tool Call Agent 运行框架，管理模型与评测环境的交互，执行 tool call 并记录结果
- **SIFT1M**：标准向量检索评测数据集，包含 100 万条 128 维 float32 base vectors、1 万条 query vectors 和预计算的 ground truth
- **QPS（Queries Per Second）**：每秒查询数，核心评分指标
- **Recall（召回率）**：模型返回的 Top-10 结果与 ground truth Top-10 的交集比例，正确性门槛为 >= 95%
- **Ground_Truth**：通过暴力搜索预计算的每条 query 的精确 Top-100 最近邻 ID
- **Tool_Call**：模型通过 Agent 框架与环境交互的一次操作调用
- **L2_Distance（L2 欧氏距离）**：两个向量之间的欧氏距离，作为相似度度量
- **Data_Preprocessor（数据预处理器）**：将 SIFT1M 原始 .fvecs/.ivecs 二进制格式转换为项目使用的 JSON 格式的脚本

## 需求

### 需求 1：骨架代码

**用户故事：** 作为评测组织者，我希望提供一套结构清晰的 Rust 骨架代码，以便被测模型能在固定接口约束下实现向量数据库核心逻辑。

#### 验收标准

1. THE Skeleton SHALL 包含一个可通过 `cargo build --release` 编译的 Rust 项目，监听 `0.0.0.0:8080`
2. THE Skeleton SHALL 定义 POST `/insert` 端点，接受 `{"id": u64, "vector": [f32; 128]}` 格式的 JSON 请求体，返回 `{"status": "ok"}`
3. THE Skeleton SHALL 定义 POST `/bulk_insert` 端点，接受 `{"vectors": [{"id": u64, "vector": [f32; 128]}, ...]}` 格式的 JSON 请求体，返回 `{"status": "ok", "inserted": N}`
4. THE Skeleton SHALL 定义 POST `/search` 端点，接受 `{"vector": [f32; 128], "top_k": u32}` 格式的 JSON 请求体，返回 `{"results": [{"id": u64, "distance": f64}, ...]}` 按 L2_Distance 升序排列的结果
5. THE Skeleton SHALL 将 HTTP 路由和 API 类型定义标记为只读（模型不可修改），将 `db.rs` 和 `distance.rs` 标记为模型实现区域
6. THE Skeleton SHALL 在 `db.rs` 中提供 `VectorDB` trait 或结构体的空实现桩，包含 `insert`、`bulk_insert`、`search` 方法签名
7. THE Skeleton SHALL 在 `distance.rs` 中提供 L2 距离计算函数的签名桩
8. THE Skeleton SHALL 允许模型修改 `Cargo.toml` 中的 release profile 参数（如 LTO、codegen-units）

### 需求 2：Benchmark Client

**用户故事：** 作为评测组织者，我希望有一个标准化的基准测试客户端，以便公平、可重复地测量被测向量数据库的性能和正确性。

#### 验收标准

1. THE Benchmark_Client SHALL 通过 `/bulk_insert` 接口将全部 1,000,000 条 base vectors 加载到被测服务中
2. THE Benchmark_Client SHALL 在正式测试前发送可配置数量的预热查询（默认 1000 条），预热查询不计入评分
3. THE Benchmark_Client SHALL 使用可配置数量的并发线程（默认 4 个）对全部 10,000 条 query vectors 执行搜索
4. THE Benchmark_Client SHALL 计算并报告 QPS（总查询数 / 总耗时秒数）、平均延迟、P50/P95/P99 延迟分位数
5. THE Benchmark_Client SHALL 对每条 query 计算 recall：模型返回的 Top-10 ID 集合与 Ground_Truth Top-10 ID 集合的交集大小除以 10，并报告所有 query 的平均 recall
6. WHEN 运行 benchmark 时，THE Benchmark_Client SHALL 使用随机种子打乱 query 顺序，防止缓存预测
7. THE Benchmark_Client SHALL 以 JSON 格式输出完整的 benchmark 结果，包含 QPS、延迟统计、recall、并发数等字段
8. THE Benchmark_Client SHALL 作为只读组件，被测模型不可修改其代码

### 需求 3：Agent 框架

**用户故事：** 作为评测组织者，我希望有一个 Tool Call Agent 框架，以便被测模型能通过标准化的工具调用与评测环境交互，完成代码编写和优化。

#### 验收标准

1. THE Agent_Framework SHALL 提供 `read_file` 工具，接受文件路径参数，返回文件内容
2. THE Agent_Framework SHALL 提供 `write_file` 工具，接受文件路径和内容参数，将内容写入指定文件并返回写入字节数
3. THE Agent_Framework SHALL 提供 `list_files` 工具，接受目录路径参数，返回该目录下的文件列表
4. THE Agent_Framework SHALL 提供 `run_command` 工具，接受命令字符串和可选超时参数（默认 120 秒），执行 shell 命令并返回退出码、stdout、stderr 和执行时长
5. THE Agent_Framework SHALL 提供 `run_benchmark` 工具，接受可选的并发数（默认 4）和预热数（默认 1000）参数，启动 Benchmark_Client 并返回完整的性能测试结果
6. THE Agent_Framework SHALL 提供 `run_profiling` 工具，接受可选的持续时间参数（默认 30 秒），运行性能分析并返回热点函数摘要和火焰图 SVG 路径
7. THE Agent_Framework SHALL 提供 `run_correctness_test` 工具，运行正确性校验并返回 recall 值、是否通过阈值、失败的 query 列表
8. THE Agent_Framework SHALL 提供 `get_status` 工具，返回已用 tool call 次数、剩余次数、已用时间、服务运行状态和最近一次 benchmark 结果
9. THE Agent_Framework SHALL 提供 `finish` 工具，接受模型的优化总结，触发最终 benchmark 并记录成绩
10. THE Agent_Framework SHALL 将每次会话的总 tool call 次数限制为 50 次
11. WHEN 模型的 tool call 次数达到 50 次上限时，THE Agent_Framework SHALL 自动触发最终 benchmark 并结束评测
12. THE Agent_Framework SHALL 对只读文件（benchmark client、数据加载脚本、API 类型定义）的写入操作返回权限错误
13. THE Agent_Framework SHALL 记录每次 tool call 的输入、输出和耗时，用于评测回放和分析

### 需求 4：数据预处理

**用户故事：** 作为评测组织者，我希望有数据预处理脚本，以便将 SIFT1M 数据集从原始二进制格式转换为项目可用的格式。

#### 验收标准

1. THE Data_Preprocessor SHALL 从 SIFT1M 数据集读取 `.fvecs` 格式的 base vectors 文件，解析出 1,000,000 条 128 维 float32 向量
2. THE Data_Preprocessor SHALL 从 SIFT1M 数据集读取 `.fvecs` 格式的 query vectors 文件，解析出 10,000 条 128 维 float32 向量
3. THE Data_Preprocessor SHALL 从 SIFT1M 数据集读取 `.ivecs` 格式的 ground truth 文件，解析出每条 query 对应的 Top-100 最近邻 ID
4. THE Data_Preprocessor SHALL 将解析后的数据转换为 JSON 格式文件，供 Benchmark_Client 和数据加载脚本使用
5. THE Data_Preprocessor SHALL 提供数据集下载脚本，从 HuggingFace 数据源下载 SIFT1M 数据集
6. THE Data_Preprocessor SHALL 提供 ground truth 生成脚本，通过暴力搜索计算每条 query 的精确 Top-100 最近邻，作为 recall 校验的基准
7. IF 输入文件格式不符合 `.fvecs`/`.ivecs` 规范，THEN THE Data_Preprocessor SHALL 报告描述性错误信息并终止处理

### 需求 5：评测流程与评分

**用户故事：** 作为评测组织者，我希望有一套完整的自动化评测流程，以便端到端地运行评测并生成可比较的分数。

#### 验收标准

1. THE Agent_Framework SHALL 按以下顺序执行评测流程：提供骨架代码和 Prompt → 模型实现基础版本 → 模型自主迭代优化 → 最终 benchmark → 记录成绩
2. WHEN 最终 benchmark 的 recall 低于 95% 阈值时，THE Agent_Framework SHALL 将该模型的 QPS 记为 0
3. THE Agent_Framework SHALL 以 QPS 降序排列生成天梯排行榜，相同 QPS 时 recall 更高的模型排名靠前
4. THE Agent_Framework SHALL 使用固定的系统 Prompt，对所有被测模型完全相同
5. THE Agent_Framework SHALL 确保系统 Prompt 不提及任何具体的索引算法名称、优化技术或优化方向，仅描述功能需求和性能目标
6. THE Benchmark_Client SHALL 在每次 benchmark 时使用不同的随机种子打乱 query 顺序，防止作弊
7. THE Agent_Framework SHALL 检测模型是否返回硬编码结果，通过对比不同 query 的返回模式进行识别
8. THE Agent_Framework SHALL 提供完整的评测流程脚本（`run_eval.sh`），一键执行从数据准备到最终评分的全部步骤

### 需求 6：Prompt 公平性

**用户故事：** 作为评测组织者，我希望确保所有被测模型收到完全相同且公平的 Prompt，以便评测结果反映模型的真实工程能力而非 Prompt 引导。

#### 验收标准

1. THE Agent_Framework SHALL 对所有被测模型使用完全相同的系统 Prompt 文本
2. THE Agent_Framework SHALL 确保系统 Prompt 仅包含功能需求（插入、搜索、recall 阈值、QPS 目标）和工具使用说明
3. THE Agent_Framework SHALL 确保系统 Prompt 不包含任何具体索引算法名称（如 HNSW、IVF、PQ、KD-Tree）
4. THE Agent_Framework SHALL 确保系统 Prompt 不包含任何具体优化技术名称（如 SIMD、内存池、cache-line 对齐、lock-free）
5. THE Agent_Framework SHALL 确保系统 Prompt 不暗示任何优化方向或实现策略

### 需求 7：反作弊机制

**用户故事：** 作为评测组织者，我希望有反作弊机制，以便确保评测结果的真实性和公正性。

#### 验收标准

1. THE Benchmark_Client SHALL 在每次 benchmark 运行时使用不同的随机种子打乱 query 顺序
2. THE Agent_Framework SHALL 禁止模型修改 Benchmark_Client 代码和数据加载脚本
3. THE Agent_Framework SHALL 在沙箱环境中隔离网络访问，禁止被测模型访问外部网络
4. THE Agent_Framework SHALL 检测模型返回结果中的硬编码模式，通过分析不同 query 返回结果的多样性进行识别
5. THE Ground_Truth SHALL 由独立的暴力搜索预计算生成，与模型实现完全无关

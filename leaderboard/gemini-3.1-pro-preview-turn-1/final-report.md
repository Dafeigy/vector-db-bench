# 向量数据库基准测试最终分析报告

## 一、总览

本报告分析 KCores LLM Arena 向量数据库基准测试中三个模型最佳实现的性能差异：

| 模型 | 轮次 | QPS | Recall | 平均延迟 (ms) | P99 延迟 (ms) |
|------|------|-----|--------|---------------|---------------|
| **gemini-3-pro-preview** | turn-3 | **1970.50** | 0.9534 | 1.95 | 3.53 |
| qwen3.5-plus | turn-3 | 1405.30 | 0.9558 | 2.75 | 5.23 |
| gemini-3.1-pro-preview | turn-1 | 657.99 | 0.9728 | 5.97 | 11.10 |

gemini-3-pro-preview-turn-3 的 QPS 是 qwen3.5-plus-turn-3 的 **1.40 倍**，是 gemini-3.1-pro-preview-turn-1 的 **2.99 倍**。

## 二、核心架构差异

### 2.1 搜索算法

三个模型均采用了 IVF（倒排文件索引）作为核心搜索算法，但参数配置差异显著。

| 特性 | gemini-3-pro-preview-turn-3 | qwen3.5-plus-turn-3 | gemini-3.1-pro-preview-turn-1 |
|------|---------------------------|---------------------|-------------------------------|
| 搜索算法 | **IVF (倒排文件索引)** | **IVF (倒排文件索引)** | **IVF (倒排文件索引)** |
| 聚类数 | 2048 | 2048 | 1000 |
| 探测数 (nprobe) | 48 | 30 | 50 |
| K-Means 迭代 | **0 轮（无精炼）** | **5 轮** | **0 轮（无精炼）** |
| 搜索空间比例 | 48/2048 ≈ 2.34% | 30/2048 ≈ 1.46% | 50/1000 = 5.0% |
| 估算候选数 (~1M 向量) | ~23,400 | ~14,600 | ~50,000 |

**关键分析：**

- **gemini-3-pro-preview** 使用 2048 聚类 + 48 探测，搜索空间约 2.34%。虽然比 qwen 扫描更多数据（约 1.6 倍），但通过其他方面的优化（在线 Top-K heap、更高效的并行策略）弥补了这一差距，最终 QPS 反超。

- **qwen3.5-plus** 使用 2048 聚类 + 30 探测，搜索空间最小（约 1.46%），是三者中最激进的剪枝策略。配合 5 轮 K-Means 精炼聚类质量，以最少的候选集达到了 0.9558 的 recall。

- **gemini-3.1-pro-preview** 使用 1000 聚类 + 50 探测，搜索空间高达 5.0%，候选集约 50,000 条，是 qwen 的 3.4 倍、gemini-3-pro 的 2.1 倍。这是其 QPS 较低的主要原因之一。

### 2.2 K-Means 聚类质量

| 特性 | gemini-3-pro-preview-turn-3 | qwen3.5-plus-turn-3 | gemini-3.1-pro-preview-turn-1 |
|------|---------------------------|---------------------|-------------------------------|
| 初始化方式 | 随机采样 | 均匀步长采样 | 均匀步长采样 |
| 迭代次数 | **0 轮** | **5 轮** | **0 轮** |
| 聚类质量 | 较低（纯随机） | **较高** | 较低（纯随机） |

- **qwen** 是唯一执行 K-Means 迭代精炼的实现。5 轮迭代在全量数据上运行，虽然索引构建时间较长，但聚类质量显著更高。这使得 qwen 能以最少的 nprobe（30）达到足够的 recall，从而最大化搜索效率。

- **gemini-3-pro** 和 **gemini-3.1-pro** 都跳过了 K-Means 精炼，直接使用随机采样的初始聚类中心。这意味着向量分配质量较低，需要更大的 nprobe 来补偿 recall 损失。gemini-3-pro 通过将 nprobe 提高到 48 来弥补，而 gemini-3.1-pro 则用了 50。

- **gemini-3-pro** 的注释 `// No K-Means refinement for speed. Rely on N_PROBE.` 明确表达了这一策略：牺牲聚类质量，依赖更多探测来保证 recall。

### 2.3 距离计算

| 特性 | gemini-3-pro-preview-turn-3 | qwen3.5-plus-turn-3 | gemini-3.1-pro-preview-turn-1 |
|------|---------------------------|---------------------|-------------------------------|
| SIMD 指令集 | AVX-512 + AVX2 回退 | AVX-512（仅） | AVX-512（仅） |
| 计算精度 | f32 → f64 返回 | f32（平方距离） | f32 → f64 返回 |
| 返回值 | f64 平方距离 | f32 平方距离 | f64 平方距离 |
| 循环展开 | while 循环 + mask 尾处理 | for step_by(16) 循环 | **完全手动展开 8 路 + 4 累加器** |
| 运行时检测 | `is_x86_feature_detected!` | 无（直接 unsafe 调用） | 无（直接调用 + `target_feature`） |
| FMA 指令 | `_mm512_fmadd_ps` | `_mm512_fmadd_ps` | `_mm512_fmadd_ps` |

**关键分析：**

- **gemini-3.1-pro** 的距离函数实现最为精细：完全手动展开 128 维为 8 条独立的 load+sub+fmadd 指令链，使用 4 个独立累加器（sum_vec, sum_vec2, sum_vec3, sum_vec4）消除迭代间数据依赖，最大化 CPU 指令流水线利用率。最终通过两级归约（sum12 = sum_vec + sum_vec2, sum34 = sum_vec3 + sum_vec4, sum = sum12 + sum34）合并结果。这是三者中最优的距离计算实现。

- **qwen** 在 `db.rs` 中内联了 AVX-512 距离函数，使用 `for i in (0..DIM).step_by(16)` 循环 8 次，每次处理 16 个 float，单累加器。虽然简洁，但存在迭代间数据依赖（每次 fmadd 都依赖上一次的 sum_vec），理论上流水线利用率不如 gemini-3.1-pro 的 4 路累加器。

- **gemini-3-pro** 的距离函数在独立的 `distance.rs` 中，使用 `is_x86_feature_detected!` 运行时检测（虽然编译器可能优化掉），while 循环处理 + mask 尾部处理。返回 f64 类型，存在 f32→f64 转换开销。

### 2.4 Top-K 选择策略

| 特性 | gemini-3-pro-preview-turn-3 | qwen3.5-plus-turn-3 | gemini-3.1-pro-preview-turn-1 |
|------|---------------------------|---------------------|-------------------------------|
| Top-K 策略 | **在线 BinaryHeap** | Vec + select_nth_unstable | Vec + select_nth_unstable |
| 内存分配 | 固定 k+1 大小 heap | ~15,000 元素 Vec | ~50,000 元素 Vec |
| 排序开销 | 无需全量排序 | partial sort + truncate | partial sort + truncate |

**关键分析：**

- **gemini-3-pro** 使用 `BinaryHeap`（最大堆）在扫描过程中在线维护 top-k 结果。每个 rayon 线程维护自己的 heap，最后通过 reduce 合并。对于 top_k=10 的场景，heap 操作的常数因子极小，避免了大 Vec 的分配和 partial sort 开销。这是其 QPS 领先的重要因素之一。

- **qwen** 和 **gemini-3.1-pro** 都先收集所有候选距离到 Vec，再用 `select_nth_unstable_by` 做 partial sort。对于 qwen 的 ~15,000 候选和 gemini-3.1-pro 的 ~50,000 候选，这意味着额外的内存分配和排序开销。

### 2.5 并发与数据结构

| 特性 | gemini-3-pro-preview-turn-3 | qwen3.5-plus-turn-3 | gemini-3.1-pro-preview-turn-1 |
|------|---------------------------|---------------------|-------------------------------|
| 锁实现 | `parking_lot::RwLock` | `std::sync::RwLock` | `std::sync::RwLock` |
| 数据布局 | 扁平 Vec<f32>（倒排列表内） | 扁平 Vec<f32> | 扁平 Vec<f32>（按聚类分组） |
| 并行搜索 | rayon `par_iter` (fold+reduce) | rayon `par_iter` | 无（单线程遍历） |
| 索引构建并行 | rayon 并行分配 | 单线程 K-Means | 单线程分配 |
| Staging 区 | 有（未索引数据暂存） | 无 | 无 |

**关键分析：**

- **gemini-3-pro** 使用 `parking_lot::RwLock`，比标准库的 `RwLock` 更轻量（无 poisoning 检查，更小的内存占用）。搜索时使用 rayon 的 `fold + reduce` 模式并行扫描倒排列表，每个线程维护独立的 BinaryHeap，最后合并。还实现了 staging 区机制，未索引的新数据暂存在 `data` 中，搜索时同时检查索引和 staging 区。

- **qwen** 使用标准 `RwLock`，搜索时将候选集通过 `into_par_iter` 并行计算距离。索引构建（5 轮 K-Means）为单线程，但由于构建不计入 QPS，影响有限。

- **gemini-3.1-pro** 的搜索是**完全单线程**的——遍历选中聚类的所有向量时使用顺序循环而非 rayon 并行。这在 4 并发查询的场景下实际上是合理的（4 个查询各占 1 个核心），但由于候选集较大（~50,000），单线程扫描的绝对耗时仍然较高。

### 2.6 索引构建策略

| 特性 | gemini-3-pro-preview-turn-3 | qwen3.5-plus-turn-3 | gemini-3.1-pro-preview-turn-1 |
|------|---------------------------|---------------------|-------------------------------|
| 构建触发 | 首次搜索时（data > 10,000） | 首次搜索时（n ≥ NUM_CENTROIDS） | bulk_insert 时直接构建 |
| 锁升级 | `RwLockUpgradableReadGuard` | 先 read 检查，再 write 构建 | 无（写锁内直接构建） |
| 数据迁移 | data → index，清空 data | 原地构建，不迁移 | 直接分配到聚类 |

- **gemini-3-pro** 使用 `parking_lot` 的 `upgradable_read` 锁升级机制，避免了先释放读锁再获取写锁之间的竞态条件。构建完成后清空 staging 区。

- **qwen** 在搜索时检查索引状态，先获取读锁检查，若需构建则释放读锁、获取写锁、再次检查（双重检查锁定模式）。

- **gemini-3.1-pro** 在 `bulk_insert` 中直接构建索引，无延迟构建机制。当 bulk_insert 的数据量 ≥ NUM_CENTROIDS 时立即训练并分配。


## 三、性能瓶颈分析

### 3.1 gemini-3-pro-preview-turn-3 (1970 QPS) ✅ 冠军

根据 profiling 数据：
- 79.03% 时间在 `l2_distance_avx512` — 距离计算
- 7.61% 时间在函数调用开销（`FnMut::call_mut`）
- 4.78% 时间在 `l2_distance` 分发函数（运行时特性检测）

IVF 索引将搜索空间缩小到约 2.34% 的数据，每次查询约计算 23,400 次距离。配合在线 BinaryHeap 的 top-k 维护，避免了大 Vec 分配和 partial sort 开销。平均延迟仅 1.95ms。

值得注意的是，`l2_distance` 分发函数占 4.78%，说明 `is_x86_feature_detected!` 运行时检测虽然被编译器缓存，但在热路径上仍有可测量的开销。如果像 qwen 一样直接内联 AVX-512 调用，可能还有 ~5% 的提升空间。

### 3.2 qwen3.5-plus-turn-3 (1405 QPS)

根据 profiling 数据：
- 82.14% 时间在 `l2_distance_squared_avx512` — 距离计算
- 14.52% 时间在 `VectorDB::search` — IVF 索引查找 + 排序

IVF 索引将搜索空间缩小到约 1.46% 的数据，每次查询仅需计算约 14,600 次距离。AVX-512 直接内联调用，零分发开销。但 `VectorDB::search` 占 14.52% 较高，其中包含：
- 候选集的 `par_iter` 并行计算（rayon 调度开销）
- `select_nth_unstable_by` partial sort
- Vec 分配和 truncate

相比 gemini-3-pro 的在线 heap 方案，这部分开销是 qwen QPS 较低的次要原因。主要原因仍是 qwen 的候选集虽小（14,600 vs 23,400），但 gemini-3-pro 在 top-k 选择和并行策略上的优化弥补了候选集更大的劣势。

### 3.3 gemini-3.1-pro-preview-turn-1 (658 QPS)

根据 profiling 数据：
- 58.23% 时间在 `l2_distance_avx512` — 距离计算
- 12.21% 时间在 `VectorDB::bulk_insert` — 索引构建
- 4.51% 时间在 JSON 反序列化

距离计算仅占 58.23%（远低于其他两个模型的 ~80%），说明有大量时间花在了非计算路径上。12.21% 的 `bulk_insert` 开销表明索引构建（聚类分配）在 profiling 期间仍在执行或其开销被计入。

gemini-3.1-pro 的 QPS 较低的根本原因是：
1. **候选集过大**：50,000 条（qwen 的 3.4 倍），直接导致距离计算量大
2. **搜索单线程**：未使用 rayon 并行化聚类内扫描
3. **聚类数不足**：1000 个聚类导致每个倒排列表平均 1000 条向量，过长
4. **无 K-Means 精炼**：聚类质量低，需要更大 nprobe 补偿

但其 recall 高达 0.9728，远超 0.95 阈值，说明有大量 recall 余量可以用来换取 QPS 提升。

## 四、Recall 与 QPS 的权衡

| 模型 | QPS | Recall | 策略 |
|------|-----|--------|------|
| gemini-3-pro-preview-turn-3 | 1970.50 | 0.9534 | 激进剪枝 + 在线 heap，recall 紧贴阈值 |
| qwen3.5-plus-turn-3 | 1405.30 | 0.9558 | 精细 K-Means + 最小搜索空间，recall 略有余量 |
| gemini-3.1-pro-preview-turn-1 | 657.99 | 0.9728 | 保守参数，recall 余量大 |

三个模型展现了不同的 recall-QPS 权衡策略：

- **gemini-3-pro** 最为激进，recall 仅 0.9534，距阈值 0.95 仅 0.0034 的余量。通过跳过 K-Means 精炼、使用较大 nprobe（48）配合在线 heap 优化，将 QPS 推到最高。

- **qwen** 通过 5 轮 K-Means 精炼获得高质量聚类，使得仅需 30 个探测就能达到 0.9558 的 recall。策略是"用构建时间换搜索效率"。

- **gemini-3.1-pro** 的 recall 0.9728 远超阈值，存在约 2.3% 的 recall 余量。如果将 nprobe 从 50 降低到 ~25，或将聚类数从 1000 提高到 2048，QPS 有望大幅提升。

## 五、各实现优劣总结

### gemini-3-pro-preview-turn-3 ✅ 冠军

**优势：**
- IVF 索引 + 在线 BinaryHeap top-k 维护，避免大 Vec 分配和排序开销
- rayon fold+reduce 并行模式，每线程独立 heap，合并开销极小
- `parking_lot::RwLock` 更轻量的锁实现
- Staging 区机制，支持索引构建后的增量插入
- AVX-512 + AVX2 + 标量三级回退链，兼容性好
- recall 精确控制在阈值附近，最大化吞吐

**劣势：**
- recall 仅 0.9534，距阈值极近（0.0034 余量），稳定性风险最高
- 无 K-Means 精炼，聚类质量依赖随机采样，不同数据分布下 recall 可能波动
- `is_x86_feature_detected!` 运行时检测在热路径上有 ~5% 的可测量开销
- 距离函数返回 f64，存在 f32→f64 转换开销

### qwen3.5-plus-turn-3 🥈 亚军

**优势：**
- 5 轮 K-Means 精炼保证聚类质量，搜索空间最小（1.46%）
- AVX-512 直接内联调用，零分发开销
- f32 精度计算，避免不必要的类型转换
- recall 0.9558，比 gemini-3-pro 更稳定

**劣势：**
- 候选集使用 Vec + `select_nth_unstable_by`，不如在线 heap 高效
- `par_iter` 对 ~15,000 候选集的 rayon 调度开销可能超过并行收益
- 无 AVX2 回退路径，不支持无 AVX-512 的 CPU
- 索引构建时间较长（5 轮 K-Means × 2048 聚类 × 全量数据）

### gemini-3.1-pro-preview-turn-1 🥉 季军

**优势：**
- **距离计算实现最优**：完全手动展开 + 4 路累加器，最大化指令流水线利用率
- recall 高达 0.9728，稳定性最好，有大量余量可优化
- `target-cpu = native` 编译选项，充分利用本机 CPU 特性
- 代码结构清晰，距离函数与数据库逻辑分离

**劣势：**
- **聚类数不足**（1000），每个倒排列表过长，搜索空间 5.0%
- **搜索完全单线程**，未使用 rayon 并行化聚类内扫描
- **无 K-Means 精炼**，聚类质量低
- nprobe=50 过高，大量 recall 余量未转化为 QPS
- bulk_insert 中直接构建索引，无延迟构建优化

## 六、结论

gemini-3-pro-preview-turn-3 以 1970 QPS 领先的核心原因是**在线 BinaryHeap top-k 维护 + rayon fold/reduce 并行模式**的组合优化，避免了大 Vec 分配和 partial sort 的开销。虽然其搜索空间（2.34%）大于 qwen（1.46%），但 top-k 选择策略的优势弥补了候选集更大的劣势。

qwen3.5-plus-turn-3 以 1405 QPS 位居第二，其核心优势是**5 轮 K-Means 精炼带来的高质量聚类**，使得搜索空间最小化。但 Vec + partial sort 的 top-k 策略和 rayon 在小候选集上的调度开销限制了其进一步提升。

gemini-3.1-pro-preview-turn-1 以 658 QPS 位居第三，虽然拥有三者中**最优的距离计算实现**（完全展开 + 4 路累加器），但**聚类参数保守**（1000 聚类 + 50 探测）和**单线程搜索**严重制约了整体性能。其 0.9728 的高 recall 表明有巨大的优化空间。

在向量数据库场景中，性能优化是一个系统工程：**索引参数调优**（聚类数、探测数）决定搜索空间大小，**top-k 选择策略**（在线 heap vs partial sort）决定后处理效率，**并行策略**决定 CPU 利用率，而**距离计算优化**（SIMD 展开、累加器数量）则决定单次计算的效率。三个模型各有侧重，最终性能取决于这些因素的综合表现。

## 七、gemini-3.1-pro-preview-turn-1 进一步优化空间分析

当前 658 QPS 虽已通过 recall 阈值，但与冠军差距明显。以下是按优先级排列的优化方向：

### 7.1 增加聚类数 + 降低 nprobe（收益：极高，风险：低）

当前 1000 聚类 + 50 探测 = 5% 搜索空间。改为 2048 聚类 + 30 探测 = 1.46% 搜索空间，候选集从 ~50,000 降至 ~14,600，距离计算量减少约 3.4 倍。这是最直接的优化，预计可将 QPS 提升 2-3 倍。

### 7.2 添加 K-Means 精炼（收益：高，风险：低）

当前无 K-Means 迭代，聚类质量低。添加 3-5 轮 K-Means 迭代可显著提高聚类质量，使得更小的 nprobe 就能达到足够 recall。配合 7.1 的参数调整，可以进一步降低 nprobe 到 20-25。

### 7.3 在线 Top-K BinaryHeap（收益：中-高，风险：低）

当前使用 Vec + `select_nth_unstable_by`，对 ~50,000 候选需要分配大 Vec 并做 partial sort。改为在线 BinaryHeap（参考 gemini-3-pro 的实现），在扫描过程中维护固定大小 max-heap，避免大 Vec 分配。对 top_k=10 的场景，heap 操作开销极小。

### 7.4 搜索并行化（收益：中，风险：低）

当前搜索为单线程顺序遍历。在 4 并发查询场景下，每个查询独占 1 个核心是合理的。但如果候选集较大，可以考虑使用 rayon 的 fold+reduce 模式并行扫描多个倒排列表，每线程维护独立 heap。需注意在高并发下 rayon 线程池的竞争问题。

### 7.5 距离函数内联到 db.rs（收益：低-中，风险：低）

当前距离函数在独立的 `distance.rs` 中，虽然 LTO 可能会内联，但直接在 `db.rs` 中内联 AVX-512 距离函数（参考 qwen 的做法）可以确保零分发开销。当前实现已经是完全展开 + 4 路累加器，距离计算本身已经很优秀。

### 7.6 使用 parking_lot::RwLock（收益：低，风险：低）

标准库的 `RwLock` 有 poisoning 检查开销。`parking_lot::RwLock` 更轻量，在高并发读场景下性能更好。

### 7.7 优化优先级建议

| 优先级 | 优化项 | 预期 QPS 提升 | 实现难度 | 风险 |
|--------|--------|--------------|---------|------|
| P0 | 增加聚类数 (2048) + 降低 nprobe (30) | 2-3x | 低 | 低 |
| P0 | 添加 K-Means 精炼 (3-5 轮) | 配合 P0 进一步提升 | 中 | 低 |
| P1 | 在线 Top-K BinaryHeap | 10-20% | 低 | 低 |
| P1 | 搜索并行化 (rayon fold+reduce) | 10-30% | 中 | 中 |
| P2 | 距离函数内联 | 3-5% | 低 | 低 |
| P2 | parking_lot::RwLock | 2-5% | 低 | 低 |

保守估计，P0 优化组合可将 QPS 从 658 推到 1500-2000 区间，接近甚至超过当前冠军水平。P1 优化可进一步推到 2000-2500。gemini-3.1-pro 拥有三者中最优的距离计算实现，一旦索引参数和搜索策略优化到位，其潜力不容小觑。

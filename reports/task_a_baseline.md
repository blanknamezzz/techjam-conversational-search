# 任务 A：BM25 基线与数据结论

> 状态：已完成。官方数据校验、3 个单元测试、200 会话 baseline、数据分析和 160/40 划分均已复现。

## 1. Baseline 结果

环境：Miniconda Python 3.14.6、SQLite FTS5、无第三方依赖、无需 API Key。完整运行约 13 秒。

| 指标 | 结果 |
|---|---:|
| 会话数 | 200 |
| Hit Rate@10 | 0.125000 |
| MRR | 0.068034 |
| MTTC | 9.810000 |
| Efficiency | 0.119000 |
| TechnicalScore | 0.106710 |

结果与 `docs/baseline_results.json` 完全一致。

| 场景 | 样本数 | Hit Rate@10 | MRR | MTTC |
|---|---:|---:|---:|---:|
| Buying | 80 | 0.237500 | 0.126508 | 8.625000 |
| Browsing | 80 | 0.025000 | 0.004514 | 10.750000 |
| Intent Override | 30 | 0.133333 | 0.104167 | 10.066667 |
| Boundary | 10 | 0.000000 | 0.000000 | 11.000000 |

## 2. 当前 Baseline 的主要问题

| 问题 | 当前行为 | 直接影响 |
|---|---|---|
| 只处理当前一句话 | 不保存历轮条件，`user_profile` 也未使用 | 多轮需求丢失，Intent Override 容易保留错误意图 |
| 不主动澄清 | `ask_attribute` 始终为 `null` | 模拟用户无法提供有针对性的下一条属性，MTTC 很高 |
| 纯关键词匹配 | 无同义词和语义理解 | Browsing Hit Rate 仅 2.5% |
| 没有结构化约束 | 价格、颜色、材料、尺寸都作为普通文本 | 明确购买条件无法稳定满足 |
| 固定 OR 查询 | 所有有效词用 OR 连接，未区分品类、硬条件和软偏好 | 候选噪声较多，精确率低 |
| 固定字段权重 | 标题、品类等权重未经本地验证集调优 | 不同查询类型无法自适应 |
| 没有候选融合 | 只有 SQLite FTS5/BM25 | 目标未被 BM25 召回时无法补救 |
| 没有重排序 | BM25 顺序直接作为最终 Top 10 | 目标即使被召回也可能排名靠后，MRR 低 |
| 没有场景路由 | Buying、Browsing、Override、Boundary 使用同一策略 | 各场景无法针对性优化 |
| 没有评分明细 | 只返回 ASIN，不记录为什么命中或失败 | 难以进行错误分析和消融实验 |

## 3. 后续改进优先级

| 优先级 | 改进内容 | 主要目标 |
|---|---|---|
| P0 多轮行为 | 保存结构化状态；支持条件追加、覆盖和清除；主动选择 `ask_attribute`；按完整状态改写查询 | 降低 MTTC，改善 Override 和 Boundary |
| P1 召回 | 调整 BM25 字段权重；加入结构化软匹配和 Dense 语义召回；用 RRF 融合候选 | 提升 Hit Rate@10，尤其是 Browsing |
| P2 排序 | 规则粗排 + 可选语义精排；约束采用 `MATCH / VIOLATION / UNKNOWN`；最终只过滤明确违反显式硬约束的商品 | 提升 MRR 和 Top 10 精度 |
| P3 工程与评估 | 缓存索引和向量；按场景记录召回／重排位置；开发集调参、验证集选方案；补充异常测试 | 提高速度、稳定性和可分析性 |

评分与热度只能作为弱质量特征，例如 `average_rating` 和 `log1p(rating_number)`，不能覆盖需求相关性。约束策略应对比“全软”“全硬”和“软约束 + 最终显式硬门控”三组实验。

## 4. 关键数据结论

目录包含 50,000 个唯一 `parent_asin`，没有重复商品 ID。

| 字段 | 缺失率 | 对方案的影响 |
|---|---:|---|
| `title` | 0.004% | 最稳定的检索字段，可保持较高权重 |
| `features` | 10.438% | 适合属性和语义召回，但需要缺失容错 |
| `description` | 47.774% | 不能依赖它覆盖全部商品 |
| `price` | 78.946% | 缺失必须视为 `UNKNOWN`，不能直接硬过滤 |
| `details` | 3.340% | 适合提取材料、尺寸、部门等属性 |
| `store` | 0.628% | 可用于品牌／店铺匹配 |

标题词数中位数为 11，适合作为 BM25 和 Embedding 主文本；`features` 和 `description` 长度差异较大，需要独立权重和截断。`rating_number` 最大为 408,371，进入排序前应使用 `log1p`。

完整统计见 `reports/task_a_data_summary.json`。

## 5. 开发／验证划分

| 场景 | 总数 | 开发集 | 验证集 |
|---|---:|---:|---:|
| Buying | 80 | 64 | 16 |
| Browsing | 80 | 64 | 16 |
| Intent Override | 30 | 24 | 6 |
| Boundary | 10 | 8 | 2 |
| **总计** | **200** | **160** | **40** |

划分使用固定种子 `techjam-task-a-v1`，按场景通过 SHA-256 稳定生成；两个集合目标无交叉。验证集仅 40 条且 Boundary 只有 2 条，因此还需结合完整 200 条结果和消融趋势判断方案。

## 6. 数据校验与复现

两个官方压缩包均通过 SHA-256 校验。当前目录为 50,000 行官方数据，文件 SHA-256 是 `da979b05a68af864cb0dcf9ee6a81c010c7e66a57978ad286c7a2e005fc69a67`。

```bash
# 数据分析与稳定划分
/Users/charlie/miniconda3/bin/python scripts/task_a_analysis.py

# 单元测试
/Users/charlie/miniconda3/bin/python -m unittest discover -v -s tests -p "test_*.py"

# 官方 BM25 baseline
/Users/charlie/miniconda3/bin/python -m evaluator.local_evaluator --output results.json
```

相关产物：

- `reports/task_a_data_summary.json`
- `data/splits/task_a_dev_ids.txt`
- `data/splits/task_a_validation_ids.txt`
- `results.json`（运行生成，已被 Git 忽略）

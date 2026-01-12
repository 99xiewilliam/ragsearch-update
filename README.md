## autorag_offline_search

离线对比多种“RAG 组合/超参搜索”算法的小项目：在 `train` 上最多评测 N 个配置（默认 10），选出最优配置；再把该配置固定到 `validation` 上评测一次。

### 数据格式

每个数据集目录下包含：

- `train/corpus.parquet`
- `train/qa.parquet`
- `validation/corpus.parquet`
- `validation/qa.parquet`

与 `three_dataset_sample_split/` 的结构一致。

`qa.parquet` 需包含至少：

- `qid`
- `query`
- `generation_gt`（参考答案；允许是 string 或 list[string]）

`corpus.parquet` 需包含至少：

- `doc_id`
- `contents`

### Pipeline（重构后的三大类）

项目现在按三类 pipeline 组织（由配置 `pipeline` 选择）：

- **common**：常用 RAG（rewriter → chunking → embedding → retriever → reranker → pruner → generator）
- **graph**：GraphRAG（目前为骨架：结构已就位，后续可在此基础上加入图构建/扩展检索）
- **multimodal**：多模态 RAG（目前为骨架：结构已就位，后续可从 `Doc.metadata` 引入图像/音频等）

### 可选组件与超参（只保留你指定的那些）

- **rewriter**
  - `rewriter_enabled`: 是否启用
  - `rewriter_model`: `qwen3|llama3`（也支持直接传模型名/路径）
  - `rewriter_prompt_id`: `rewrite_v1|rewrite_v2_keywords`
  - `rewriter_max_tokens`
- **chunking**
  - `chunking_enabled`: 是否启用（关闭则退化为“每个 doc 当成一个 chunk”）
  - `chunking_method`: `semantic|token`
  - `chunk_size`
  - `chunk_overlap`
  - `min_chunk_words`
- **embedding**
  - `embedding_enabled`: 是否启用（关闭时要求 `retriever=bm25`）
  - `embedder_model`: 支持多选（默认用 CUDA，失败自动回退 CPU；不作为超参暴露）
- **retriever**
  - `retriever`: `cosine|bm25|hybrid`
  - `retriever_topk`
  - `hybrid_alpha`（仅 `hybrid` 生效）
- **reranker**
  - `reranker_enabled`: 是否启用（关闭则不 rerank）
  - `reranker_model`: `none` 或 cross-encoder 模型名
  - `rerank_topk`
- **pruner**
  - `pruner_enabled`
  - `pruner_model`: `qwen3|llama3`
  - `pruner_prompt_id`: `prune_v1`
  - `pruner_max_tokens`
- **generator（固定 prompt，不作为超参）**
  - `generator_model`：默认 `qwen3`（可通过 CLI 覆盖）
  - `generator_max_tokens`：可通过 CLI 覆盖

补充（pipeline 专属开关，不引入额外超参逻辑）：
- **graph**
  - `graph_expand_enabled`: 是否启用 1-hop 图扩展
- **multimodal**
  - `multimodal_metadata_enabled`: 是否把 `Doc.metadata` 中的 caption/OCR 等文本拼入检索语料

### 算法

- Random Search
- Greedy / Coordinate Descent
- TPE（Optuna）
- UCB1 Multi-Armed Bandit（把“完整 config”视为 arm）
- Thompson Sampling（连续 reward 的近似 TS）
- GRPO-lite（简化版策略梯度：对离散选项分布做优势更新）

### 安装

建议使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 运行

对单个数据集做算法对比（默认 train 10 次 trial，validation 评 1 次）：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /home/xwh/three_dataset_sample_split/bioasq \
  --train_trials 10 \
  --metrics_weights "rougeL:0.34,bertscore_f1:0.33,meteor:0.33" \
  --llm_base_url http://localhost:9000/v1 \
  --generator_model qwen3 \
  --generator_max_tokens 128 \
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq
```

注意：当前版本 **强制依赖 vLLM/OpenAI-compatible endpoint**，必须提供 `llm_base_url`，并且 pipeline 总会走 generator（固定 prompt）。

推荐：把配置写进 YAML（更清晰、更容易复现），然后用 `--config` 加载：

```yaml
# config.yaml（示例：用于“钉死/覆盖”某些配置；未写的字段仍可由搜索空间生成）
pipeline: common
rewriter_enabled: false
chunking_enabled: true
chunking_method: token
chunk_size: 256
chunk_overlap: 64
min_chunk_words: 8

embedding_enabled: true
embedder_model: BAAI/bge-m3
retriever: hybrid
retriever_topk: 10
hybrid_alpha: 0.5

reranker_enabled: false
pruner_enabled: false

generator_model: qwen3
generator_max_tokens: 128
```

然后运行：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /home/xwh/three_dataset_sample_split/bioasq \
  --train_trials 10 \
  --config /path/to/config.yaml \
  --llm_base_url http://localhost:9000/v1 \
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq
```

你也可以用 `--algo` 只跑某个算法，或用 `--all_datasets_root` 一次跑三个数据集。

### 可插拔：替换成你自己的 RAG

只要你提供一个 Python 类，满足：

- `__init__(docs, config)`
- `answer(query) -> str`

就可以通过 `--rag_plugin module:Class` 接入。

项目内置默认插件（新架构入口）：

```bash
--rag_plugin autorag_offline_search.plugins.rag:UnifiedRag
```

（历史示例里曾包含多个旧插件；已随重构移除。）

### 可插拔：自定义搜索空间（可选）

你可以通过 `--space_plugin module:ObjOrClass` 替换搜索空间；对象/类需要提供：

- `all_configs()`（给 random/UCB/TS 用）
- `space_dict()`（给 greedy/TPE/GRPO 用）

（旧的 bits 空间实验 / sweep 功能已移除；如果需要可以在新架构上再加回一个“预算调度 + sweep”包装层。）

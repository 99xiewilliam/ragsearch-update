# ragsearch-update（Offline RAG 参数搜索 + Pipeline 对比）

这个项目做一件事：**在离线数据集上，用多种搜索算法（random/greedy/tpe/ucb1/ts/grpo…）搜索 RAG pipeline 的配置组合**，并用你指定的指标权重选出 **train 上最优 config**，再在 **validation 上复评一次**，输出 `results.json` 和 `summary.tsv` 方便对比。

支持两类 pipeline：
- **common**：纯文本 RAG（rewriter → chunking → embedding → retrieval → rerank → prune → generate）
- **multimodal**：多模态 RAG（同上，但能读取 `Doc.metadata` 里的图片路径，并可用 CLIP/Qwen3-VL embedding、VL reranker、VL generator）

---

## 快速开始：用现有脚本跑通 pipeline

### 1) 先确保你有一个 OpenAI-compatible 的本地 LLM 服务（vLLM）

本项目所有 LLM 调用都走 `--llm_base_url`（例如 `http://localhost:9000/v1`）。如服务需要鉴权，设置：

```bash
export OPENAI_API_KEY="你的key"
```

### 2) 安装依赖

`requirements.txt` **不包含 torch**（避免 CUDA 版本装错）。请先按你的机器装 torch，再装项目依赖：

```bash
# 例：CUDA 12.8 (cu128)
pip install -r requirements-torch-cu128.txt
pip install -r requirements.txt
```

### 3) 一键 smoke（common）

这个脚本会：
- 抽样构造一个 `datasets/hotpotqa_distractor_20`
- 用 **BM25-only** 跑一遍（避免下载 embedding/reranker 模型），确保 pipeline 端到端能跑

```bash
LLM_BASE_URL="http://localhost:9000/v1" bash scripts/demo_run_hotpotqa_common_20.sh
```

输出在 `runs/hotpotqa_common_20_smoke/`：
- `results.json`
- `summary.tsv`
- `val_predictions_random.jsonl`（因为脚本开了 `--dump_val_generations`）

### 4) 一键 smoke（multimodal）

这个脚本会构造 `datasets/m2rag_mmqa_test10` 并用 multimodal pipeline 跑一遍（同样默认 BM25-only，更稳定）：

```bash
LLM_BASE_URL="http://localhost:9000/v1" bash scripts/demo_run_m2rag_mmqa_10_multimodal.sh
```

### 5) 只想验证“单次调用”是否能跑（不做搜索）

```bash
python scripts/run_common_pipeline.py --demo --llm_base_url "http://localhost:9000/v1"
```

它会生成一个极小 demo 数据集到 `/tmp/autorag_demo`，并打印 `answer_with_trace()` 的 JSON（包含检索/最终上下文/答案等关键信息）。

---

## 数据集格式（必须满足）

磁盘布局（固定）：
- `<dataset_dir>/train/corpus.parquet`
- `<dataset_dir>/train/qa.parquet`
- `<dataset_dir>/validation/corpus.parquet`
- `<dataset_dir>/validation/qa.parquet`

`corpus.parquet` 字段：
- `doc_id`：string
- `contents`：string
- `metadata`：可选（dict 或 JSON string）。multimodal 时可在这里放图片路径/URL。

`qa.parquet` 字段：
- `qid`：string
- `query`：string
- `generation_gt`：list[str]（允许多参考答案；也兼容某些数据集把 list 写成字符串的情况）

现成的数据集适配脚本在 `scripts/datasets/`（例如 HotPotQA、M2RAG-MMQA 抽样构造）。

---

## 最核心的命令：跑“搜索/对比”实验（CLI）

入口：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir "<你的dataset_dir>" \
  --out_dir "<输出目录>" \
  --llm_base_url "http://localhost:9000/v1"
```

它会对 `--algo` 指定的每个算法：
- 在 **train split** 上做 `--train_trials` 次 trial（每个 trial=评估一个 config）
- 选 train 最优 config
- 在 **validation split** 上用该 config 评估一次并记录指标

### 输出文件说明

在 `--out_dir` 下：
- **`results.json`**：完整结构化结果（每个 algo 的每个 trial、best_train、validation 指标、best config）
- **`summary.tsv`**：面向人看的对比表（每个 algo 一行，含 val 指标和 best_config）
- （可选）**`val_predictions_<algo>.jsonl`**：每条样本的预测/trace（需要 `--dump_val_generations`）

---

## 重要概念：train_trials 是什么？设多少合适？

`--train_trials` 是 **每个算法在 train 上可调用 objective() 的预算**（也就是“要评估多少个 config”）。

经验建议：
- **1**：只验证 pipeline 跑通（smoke）
- **10（默认）**：快速对比/小预算搜索
- **20~50**：想让算法差异更稳定一些（代价：更慢、更贵）

注意：组合算法（如 `portfolio_grpo_ts_tpe` / `two_stage_tpe_then_grpo`）**不会增加总预算**，只是在内部切分 `train_trials`。

---

## CLI 参数说明（逐个解释）

### 数据集输入（二选一，必填其一）
- **`--dataset_dir`**：单个数据集目录（包含 train/validation 四个 parquet）
- **`--all_datasets_root`**：一个目录，下面每个子目录都当成一个 dataset，CLI 会逐个跑并把结果写到 `out_dir/<dataset_name>/`

### 输出
- **`--out_dir`**（必填）：输出目录

### 搜索算法
- **`--algo`**：逗号分隔算法列表。内置支持：
  - `random`、`greedy`、`tpe`（Optuna）、`ucb1`、`ts`、`grpo`（以及 `grpo_a2/grpo_a3`）
  - 组合：`portfolio_grpo_ts_tpe`、`portfolio_grpo_greedy`、`two_stage_tpe_then_grpo`、`two_stage_tpe_then_grpo_a2`
  - **插件算法**：`module.sub:ClassOrObj`（必须实现 `run(SearchInput)->SearchOutput`）
- **`--train_trials`**：每个 algo 的 trial 预算（见上文）
- **`--seed`**：随机种子
- **`--gpus`**：并行跑多个算法时用的 GPU 列表，例如 `0,1`（每个算法单独起进程，设置对应 `CUDA_VISIBLE_DEVICES`）

### 指标与 reward 聚合
- **`--metrics_weights`**：reward 的加权方式，例如：
  - `qa_f1:1,em:1`（HotPotQA 常用）
  - `em:1,rougeL:1,accuracy:1`
  支持指标键：`rouge1/rouge2/rougeL/meteor/bleu/bertscore_f1/em/qa_f1/chrf/similarity/accuracy`
- **`--bertscore_model`**：当 weights 里包含 `bertscore_f1` 时使用的模型（默认 `microsoft/deberta-xlarge-mnli`）

### GRPO / TPE（可选控制）
- **`--grpo_group_size`**：GRPO 每轮采样 group 大小（`0=自动`）。group size 会消耗预算
- **`--tpe_patience`**：TPE early stop：最佳 reward 连续多少次不提升就停（`0=禁用`）
- **`--tpe_min_delta`**：TPE early stop：最小提升阈值
- **`--tpe_warmup`**：TPE early stop：至少收集多少个有效 trial 后才允许早停

### 调试与日志（建议先用这个定位“为什么效果差/为什么慢”）
- **`--module_logs`**：打开 pipeline 内部模块日志（rewriter/chunking/retrieval/rerank/prune/generate）
- **`--verbose`**：打印每个 trial 的简要日志（reward、best-so-far）
- **`--log_every`**：配合 `--verbose`，每 N 次 objective 打印一次（默认 1）
- **`--debug_trace_n`**：在一次 eval 中，对前 N 条样本打印完整 trace（0=关闭）
- **`--debug_trace_every`**：只在每第 N 个 trial 打印 trace（0=不做频控）
- **`--debug_trace_max_chars`**：trace 打印字段的最大字符数（默认 400）
- **`--debug_trace_split`**：trace 打印哪个 split：`train|validation|both`
- **`--show_eval_progress`**：在每次 eval 内显示 tqdm（trial 多时会很吵/略慢）

### 预测/trace 落盘（用于人工检查）
- **`--dump_val_generations`**：把 validation 的逐样本预测与 trace 写到 `val_predictions_<algo>.jsonl`
- **`--dump_val_limit`**：最多 dump 多少条（0=不限制）

### 插件机制（可选）
- **`--rag_plugin`**：自定义 RAG 实现：`module:Class`（构造签名：`__init__(docs, config)`，并实现 `answer(query)->str`）
- **`--space_plugin`**：自定义搜索空间：`module:ObjOrClass`（需提供 `all_configs()`；可选 `space_dict()`）

### 配置注入/覆盖（强烈建议用 YAML 管住默认值）
- **`--config`**：YAML 配置文件路径（详见下一节）
- **`--cache_dir`**：缓存目录（默认 `<out_dir>/.cache`，缓存 chunk/embedding/index）
- **`--llm_base_url`**：覆盖 config 里的 `llm_base_url`
- **`--pipeline`**：强制指定 `common|multimodal`（覆盖 config/space）
- **`--embedder_model`**：强制指定 embedder（覆盖 config/space）
- **`--generator_model`**：强制指定 generator 模型（覆盖 config/space）
- **`--generator_max_tokens`**：强制指定 generator 输出 token 上限（`>0` 才生效；注意这是 *输出* token，不是上下文长度）
- **`--bm25_weight`**：统一检索旋钮 \([0,1]\)：`0=cosine`，`1=bm25`，中间=hybrid（覆盖 config/space）

---

## YAML 配置怎么写（base / space / range）

CLI 会把每个 trial 的 config 按如下顺序合并（后者覆盖前者）：
1) 搜索空间/算法提案的 `cfg`
2) **YAML 的 base config**（`--config`）
3) 运行时注入（`dataset_id/cache_dir` + CLI 覆盖项）

### 推荐写法：结构化 `base` + `space`

```yaml
base:
  pipeline: common
  llm_base_url: http://localhost:9000/v1
  module_logs: true
  rewriter_enabled: false
  pruner_enabled: false
  reranker_enabled: false
  generator_model: /path/or/served/model/id
  generator_max_tokens: 128

space:
  # 只允许搜索这些候选值（相当于“裁剪”默认 SearchSpace）
  bm25_weight: [1.0]          # 纯 BM25：不会算 embedding
  retriever_topk: [3, 5, 10]
  chunking_enabled: [false]
```

### 兼容写法：顶层 list 会被当成 space 约束

```yaml
pipeline: common
bm25_weight: [0.0, 0.5, 1.0]   # 这是 space 约束
retriever_topk: [3, 10]        # 这是 space 约束
generator_model: qwen3         # 这是 base（标量）
```

### range 展开（把连续区间离散成网格）

```yaml
space:
  bm25_weight:
    low: 0
    high: 1
    steps: 11   # 生成 0.0,0.1,...,1.0
```

---

## Pipeline 配置字段（哪些 key 真正会生效）

这些字段由 `autorag_offline_search/pipelines/config.py::normalize_config()` 读取并生效：

### 必填
- **`llm_base_url`**：OpenAI-compatible endpoint（例如 `http://localhost:9000/v1`）
- **`pipeline`**：`common|multimodal`（不写默认 common）

### rewriter（query 改写）
- **`rewriter_enabled`**：是否启用
- **`rewriter_model`**：模型名/别名（见下方“模型名/路径”）
- **`rewriter_prompt_id`**：`rewrite_v1|rewrite_v2_keywords`（优先于 `rewriter_prompt`）
- **`rewriter_prompt`**：自定义 prompt（不推荐频繁改，影响可比性）
- **`rewriter_max_tokens`**：输出 token 上限；约定：`>=32000` 表示“不传 max_tokens，让服务端自行决定”

### chunking（切分文档）
- **`chunking_enabled`**：是否启用；false 时每个 doc 当一个 chunk
- **`chunking_method`**：`semantic|token`
- **`chunk_size`**：chunk 大小（token chunking 会用 `embedder_model` 的 tokenizer）
- **`chunk_overlap`**：重叠（必须 `< chunk_size`）
- **`min_chunk_words`**：最小词数（过大可能导致“没有 chunk 产生”）

### embedding + retrieval（检索）
推荐只用一个旋钮：**`bm25_weight`**：
- `bm25_weight=0`：cosine（向量检索）
- `bm25_weight=1`：bm25（会自动 `embedding_enabled=false`，更适合 smoke）
- `0<bm25_weight<1`：hybrid（`hybrid_alpha=bm25_weight`）

相关字段：
- **`embedding_enabled`**：通常不用手填；bm25 时会自动关掉
- **`embedder_model`**：向量模型（如 `BAAI/bge-m3`；multimodal 可用 `clip` 或 `Qwen3-VL-Embedding-*`）
- **`retriever`** / **`hybrid_alpha`**：兼容老写法（不填 bm25_weight 时才用）
- **`retriever_topk`**：召回候选数

### reranker（重排）
- **`reranker_enabled`**
- **`reranker_model`**：`none` 或 HF CrossEncoder 模型名；multimodal 可尝试 `Qwen/Qwen3-VL-Reranker-*`
- **`rerank_topk`**：重排前 K（必须 `<= retriever_topk`）

### pruner（裁剪上下文）
- **`pruner_enabled`**
- **`pruner_model`**
- **`pruner_mode`**：`select|compress`（默认 select；compress 会用不同 prompt）
- **`pruner_max_tokens`**：同 rewriter 的 max_tokens 约定

### generator（最终回答）
- **`generator_model`**：用于 vLLM / OpenAI-compatible 的 `model` 字段
- **`generator_max_tokens`**：输出 token 上限（同样是输出，不是上下文长度）

### multimodal 专属
- **`multimodal_metadata_enabled`**：是否把 `Doc.metadata` 的 caption/ocr 等文本拼进 `contents`（默认 true）

### 日志
- **`module_logs`**：打开模块日志（或 `verbose=true` 也会触发日志打印）

---

## 默认搜索空间（SearchSpace）里会被“搜索”的参数有哪些？

默认搜索空间在 `autorag_offline_search/search_space.py::SearchSpace`。会被算法枚举/搜索的核心 knob（可用 YAML 的 `space:` 裁剪）：
- `pipeline`：`common|multimodal`
- `rewriter_enabled`、`rewriter_model`（按 pipeline 自动切换候选）、`rewriter_prompt_id`
- `chunking_enabled`、`chunk_size`
- `embedder_model`（按 pipeline 切换到 multimodal embedder 候选）
- `retriever_topk`
- `bm25_weight`（统一检索旋钮）
- `reranker_enabled`、`reranker_model`（按 pipeline 切换到 multimodal reranker 候选）、`rerank_topk`
- `pruner_enabled`、`pruner_model`
- `multimodal_metadata_enabled`（multimodal 时才有意义）

像 `chunk_overlap/min_chunk_words/generator_model/generator_max_tokens` 等 **默认不作为搜索维度**，但你可以在 YAML `base:` 里固定它们，或用 CLI 覆盖。

---

## 模型名/路径怎么填？（你提到的“模型怎么配/怎么记录”）

这个 repo 里有少量 **model preset**（只是字符串别名到路径的映射）：
- `qwen3` → `/home/xwh/models/Qwen3-4B-Instruct-2507`
- `qwen3_vl_4b` → `/home/xwh/models/Qwen3-VL-4B-Instruct`

它们用于 `rewriter_model/pruner_model/generator_model` 的解析（最终会把解析后的字符串当作 OpenAI-compatible 的 `model` 传给服务端）。

更推荐的做法：
- **如果你用 vLLM**：`generator_model` 填 vLLM `/v1/models` 返回的 `id`
- **如果你用本地路径作为 vLLM 的 model id**：也可以直接填路径

想“记录用了哪个模型”，直接看：
- `results.json` 中每个 trial 的 `config.generator_model / config.embedder_model / ...`
- `summary.tsv` 的 `best_config`（`config_to_str()` 会把所有 key=value 串起来）

---

## 常见配方（可直接抄）

### A) 最稳的 smoke：纯 BM25（不算 embedding）

```bash
python -m autorag_offline_search.cli \
  --dataset_dir "<dataset_dir>" \
  --out_dir "<out_dir>" \
  --llm_base_url "http://localhost:9000/v1" \
  --algo random \
  --train_trials 1 \
  --config tmp/hotpotqa_common_smoke.yaml \
  --metrics_weights "qa_f1:1,em:1" \
  --verbose --module_logs
```

### B) 固定 pipeline=common，只比较检索/重排/裁剪开关

在 YAML `base:` 里写 `pipeline: common`，并在 `space:` 里只开放你关心的候选。

### C) 多 GPU 并行跑多个算法（每个算法一个进程）

```bash
python -m autorag_offline_search.cli \
  --dataset_dir "<dataset_dir>" \
  --out_dir "<out_dir>" \
  --gpus "0,1" \
  --algo "random,tpe,grpo" \
  --train_trials 20 \
  --llm_base_url "http://localhost:9000/v1"
```

---

## 参考：数据集/任务清单

见 `rag_search_dataset.txt`（列了常见文本/多模态 RAG 数据集、评估建议与链接）。


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

### Pipeline（可搜索空间示例）

- **chunk_size**（word-based）：可设为任意候选（支持二元空间实验，例如 `128/512`）
- **chunk_overlap**：`0 / 32`（并保证 overlap < chunk_size）
- **retriever_topk**：可设为任意候选（支持二元空间实验，例如 `3/10`）
- **generator**（不依赖外部 LLM，保证离线可复现）：
  - `best_sentence`: 从 topk chunks 中挑选与 query 最相近的句子作为回答
  - `concat_chunks`: 拼接 topk chunks 的前若干词作为回答

可选二元扩展超参（用于把搜索空间扩到 \(2^6\sim 2^{14}\)）：
- `ngram_range`：`(1,1)` vs `(1,2)`
- `stop_words`：`english` vs `none`
- `max_features`：`20k` vs `200k`
- `concat_word_limit`：`80` vs `200`
- `min_chunk_words`：`4` vs `16`
- `lowercase`：`True` vs `False`

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
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq
```

你也可以用 `--algo` 只跑某个算法，或用 `--all_datasets_root` 一次跑三个数据集。

### 可插拔：替换成你自己的 RAG

只要你提供一个 Python 类，满足：

- `__init__(docs, config)`
- `answer(query) -> str`

就可以通过 `--rag_plugin module:Class` 接入。

项目内置了两个示例：

- 默认 baseline（TF-IDF）：

```bash
--rag_plugin autorag_offline_search.plugins.rag_baseline:TfidfRag
```

- 自定义变体：字符 n-gram 检索（对名字/拼写更鲁棒）：

```bash
--rag_plugin autorag_offline_search.plugins.rag_variant:CharNgramRag
```

完整示例（对比同一搜索算法在两套 RAG 上的结果）：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /home/xwh/three_dataset_sample_split/bioasq \
  --train_trials 10 \
  --algo random \
  --rag_plugin autorag_offline_search.plugins.rag_variant:CharNgramRag \
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq_charngram
```

### 可插拔：自定义搜索空间

你可以通过 `--space_plugin module:ObjOrClass` 替换搜索空间；对象/类需要提供：

- `all_configs()`（给 random/UCB/TS 用）
- `space_dict()`（给 greedy/TPE/GRPO 用）

项目内置示例：

- 固定 \(2^{14}\) 二元空间：

```bash
--space_plugin autorag_offline_search.plugins.spaces:Bits14Space
```

### 2^bits 搜索空间规模实验

用 `--space_bits` 构造严格的二元空间（总配置数 \(=2^{bits}\)），例如：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /home/xwh/three_dataset_sample_split/bioasq \
  --space_bits 10 \
  --train_trials 10 \
  --metrics_weights "rougeL:0.34,bertscore_f1:0.33,meteor:0.33" \
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq_bits10
```

### 组件级 + 超参级混合搜索（本地 / OpenAI 可独立控制）

你可以使用组件化 RAG 插件 `ComponentRag`，它支持：

- Chunker：`fixed_window` / `semantic_split`
- Embedder：本地 `sentence-transformers` 或 OpenAI embedding API（通过 `embedder_backend` 控制）
- Retriever：`vector_topk` / `hybrid`（BM25+vector）
- Reranker：`none` / `cross_encoder`
- Generator：`best_sentence` / `concat_chunks`
- LLM Generator（可选）：本地 vLLM（OpenAI 兼容）/ OpenAI / Gemini（通过 `llm_backend` 控制）
- Prompt（可搜索超参）：`prompt_id=factoid_short | evidence_then_answer`（可扩展成更多模板）

通过 CLI 选择：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /home/xwh/three_dataset_sample_split/bioasq \
  --space_bits 14 \
  --train_trials 40 \
  --rag_plugin autorag_offline_search.plugins.component_rag:ComponentRag \
  --cache_dir /home/xwh/autorag_offline_search_runs/.cache \
  --llm_base_url http://localhost:9000/v1 \
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq_component_bits14
```

说明：
- `--space_bits` 使用的是“二元组件+超参混合空间”，每一位都是真正影响行为的 knob（见 `BinarySpace`）。
- OpenAI embedding/LLM 需要设置 `OPENAI_API_KEY`；并且将某些 config 中的 `embedder_backend` 或 `llm_backend` 选到 `openai/openai_compat` 才会调用。
- 本地 vLLM：设置 `llm_backend=openai_compat`，并提供 `llm_base_url`（例如 `http://localhost:9000/v1`）以及 `llm_model`（可用本地模型路径或 vLLM 里注册的模型名）。
- 如果你要提升 BioASQ 这类 factoid 的 ROUGE-L/METEOR：建议优先让搜索空间里包含 `prompt_id=factoid_short`（强制输出短答案）。
- Gemini：设置 `llm_backend=gemini` 且提供 `GEMINI_API_KEY`（或 `GOOGLE_API_KEY`）以及 `llm_model`（例如 `gemini-1.5-flash`）。

### 一键跑：三数据集 × 多空间规模 × 多算法（自动调 train_trials）

你想要的对比可以用 `--sweep_bits` 一次跑完，并且 `train_trials` 会随空间规模自动增长（默认线性：`5*bits`，并且 min/max 截断）。

```bash
python -m autorag_offline_search.cli \
  --all_datasets_root /home/xwh/three_dataset_sample_split \
  --sweep_bits "4,6,8,10,12,14" \
  --algo "random,greedy,tpe,ucb1,ts,grpo" \
  --metrics_weights "rougeL:0.34,bertscore_f1:0.33,meteor:0.33" \
  --auto_trials_mode linear \
  --auto_trials_scale 5 \
  --auto_trials_min 10 \
  --auto_trials_max 80 \
  --rag_plugin autorag_offline_search.plugins.component_rag:ComponentRag \
  --cache_dir /home/xwh/autorag_offline_search_runs/.cache \
  --out_dir /home/xwh/autorag_offline_search_runs/sweep_all
```

输出：
- `sweep_all/sweep.tsv`：总对比表（dataset/bits/algo → validation 指标与 best_config）
- `sweep_all/<dataset>/bitsX/`：每个子实验的 `results.json` 与 `summary.tsv`



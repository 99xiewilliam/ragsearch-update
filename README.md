## autorag_offline_search（RAG 超参搜索/算法对比框架）

这个项目用来**离线对比不同搜索算法**在同一套 RAG pipeline 上的效果：

- 在 `train` split 上最多评测 \(N=\)`--train_trials` 个配置（trial）
- 选出 `train` 最优配置后，在 `validation` split 上用**同一配置**再评一次
- 产出：`results.json / summary.tsv / val_predictions_<algo>.jsonl`

---

### 0 分钟跑通（建议先跑这些）

- **common（文本）+ miniwiki 抽样 30 条**：

```bash
bash scripts/demo_run_miniwiki_common_30.sh
```

- **multimodal（含图片）+ M2RAG/mmqa 抽样 10 条**：

```bash
bash scripts/demo_run_m2rag_mmqa_10_multimodal.sh
```

这些脚本会在 `runs/<...>/` 下生成输出文件，并把终端输出写入 `run.log`。

---

### 数据集目录格式（必须）

你传给 `--dataset_dir` 的目录必须包含 4 个 parquet：

- `train/corpus.parquet`
- `train/qa.parquet`
- `validation/corpus.parquet`
- `validation/qa.parquet`

`qa.parquet` 至少包含：

- `qid`：样本 id
- `query`：问题
- `generation_gt`：参考答案（string 或 list[string]）

`corpus.parquet` 至少包含：

- `doc_id`
- `contents`
- （可选）`metadata`：dict 或 JSON 字符串（multimodal 会用到）

---

### Pipeline（两类）

- **`common`**：文本 RAG（rewriter → chunking → embedding → retriever → reranker → pruner → generator）
- **`multimodal`**：多模态 RAG  
  - 会把 `metadata` 里的 `caption/ocr/alt_text/...` 拼进 `contents`（如果 `multimodal_metadata_enabled=true`）
  - 如果检索到的 chunk 携带 `metadata.image_path`，会把图片读成本地 base64 data URL，一起发给 vLLM（需要 vision 模型）

---

### 算法怎么停（收敛）与 `--train_trials` 怎么选

在这个项目里，**大多数算法不会“自动收敛停下来”**，而是**按预算停止**：

- **`--train_trials`**：训练阶段最多评测多少个配置（objective 调用次数）。越大越慢，但更可能找到更好配置。
  - **跑通/排错**：`--train_trials 1`
  - **小规模对比**：`--train_trials 10~30`
  - **认真找最优**：`--train_trials 50~200+`

目前唯一内置“早停”的是 **TPE**（需要你显式开启）：

- `--tpe_patience K`：best reward 连续 K 个有效 trial 没提升就停
- `--tpe_min_delta D`：把“提升”定义为至少提升 D
- `--tpe_warmup W`：至少先跑 W 个有效 trial 才允许早停

---

### 用内置算法跑 pipeline（最常用命令）

#### 1) 最小可跑（推荐从这里开始）

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --out_dir /tmp/autorag_out \
  --pipeline common \
  --algo random \
  --train_trials 1 \
  --llm_base_url http://localhost:9000/v1 \
  --module_logs \
  --debug_trace_n 1 \
  --dump_val_generations \
  --dump_val_limit 20
```

#### 2) 同时对比多种算法

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --out_dir /tmp/autorag_out \
  --algo random,greedy,tpe,ucb1,ts,grpo \
  --train_trials 30 \
  --llm_base_url http://localhost:9000/v1 \
  --verbose
```

---

### 输出文件怎么读（排查/评估最有用）

在 `--out_dir` 下：

- **`results.json`**：每个算法的 best config（train 最优）+ train/val 指标
- **`summary.tsv`**：汇总表
- **`val_predictions_<algo>.jsonl`**（需要 `--dump_val_generations`）：
  - `pred`：模型回答（generator 输出）
  - `refs`：参考答案（gold）
  - `metrics`：该样本的 em/qa_f1/rouge 等
  - `trace`：检索 chunks、是否用图（`used_images`）、最终上下文等

---

### 所有 CLI 参数（逐条解释，来自 `autorag_offline_search/cli.py`）

你也可以随时运行 `python -m autorag_offline_search.cli -h` 查看。

#### 数据输入 / 输出

- **`--dataset_dir PATH`**：单个数据集目录（与 `--all_datasets_root` 二选一）
- **`--all_datasets_root PATH`**：一次跑 root 下所有子目录数据集
- **`--out_dir PATH`**：输出目录（必填）
- **`--cache_dir PATH`**：缓存目录（默认 `<out_dir>/.cache`，chunks/embeddings/chroma 等都在这）

#### 算法与预算

- **`--algo STR`**：算法列表（逗号分隔）。默认：`random,greedy,tpe,ucb1,ts,grpo`
  - 也支持插件：`module.sub:ClassOrObj`（需实现 `run(SearchInput)->SearchOutput`）
- **`--train_trials N`**：训练阶段预算（评测 config 次数）
- **`--gpus "0,1"`**：并行算法用的 GPU（会写环境变量 `AUTORAG_GPUS`）
- **`--seed INT`**：随机种子
- **`--grpo_group_size INT`**：GRPO group size（0=自动；group 会消耗预算）

#### 指标（reward）

- **`--metrics_weights "k:v,k:v"`**：reward 权重（默认含多项；常用 QA：`"qa_f1:1,em:1"`）
- **`--bertscore_model NAME`**：当 weights 里包含 `bertscore_f1` 时使用

#### TPE 早停（可选）

- **`--tpe_patience K`**：早停 patience（0=关闭）
- **`--tpe_min_delta D`**：最小提升阈值
- **`--tpe_warmup W`**：warmup trial 数

#### Pipeline / 模型覆盖（CLI 会覆盖 YAML/space）

- **`--pipeline common|multimodal`**：选择 pipeline（覆盖 YAML/space）
- **`--config PATH`**：YAML base config（也可用于约束搜索空间）
- **`--llm_base_url URL`**：OpenAI-compatible endpoint（vLLM），如 `http://localhost:9000/v1`
- **`--embedder_model NAME`**：覆盖 embedder（覆盖 YAML/space）
- **`--generator_model ID_OR_PATH`**：覆盖 generator 模型（覆盖 YAML/space；必须是 vLLM `/v1/models` 里存在的 id）
- **`--generator_max_tokens INT`**：覆盖 generator 输出 token 上限（>0 才生效；注意是**输出** token，不是上下文长度）
- **`--bm25_weight FLOAT`**：覆盖检索旋钮（>=0 才生效：0=cosine，1=bm25，中间=hybrid）

#### 日志 / 调试

- **`--module_logs`**：打印 pipeline 各模块日志（rewriter/retrieval/pruner/generator…）
- **`--verbose`**：打印每个 trial 的日志（reward、best so far…）
- **`--log_every N`**：配合 `--verbose`，每 N 次 objective 打一次
- **`--show_eval_progress`**：每次 eval 显示 tqdm（很吵，通常不建议开）

#### Trace / dump（强烈推荐用于排查）

- **`--dump_val_generations`**：生成 `val_predictions_<algo>.jsonl`
- **`--dump_val_limit N`**：dump 前 N 条 validation（0=不限制）
- **`--debug_trace_n N`**：打印前 N 条样本的 gold/pred/检索/生成 trace
- **`--debug_trace_every N`**：只在第 N、2N… 个 trial 打 trace（0=每个 trial 都打）
- **`--debug_trace_split train|validation|both`**：trace 打哪个 split
- **`--debug_trace_max_chars N`**：trace 字段截断长度

#### 插件（可选）

- **`--rag_plugin module:Class`**：替换 RAG 实现（需 `__init__(docs, config)` + `answer(query)->str`）
- **`--space_plugin module:ObjOrClass`**：替换 SearchSpace（提供 `all_configs()` 和/或 `space_dict()`）

---

### multimodal 数据最小要求（让图片真的被用上）

要让 `multimodal` pipeline 真的把图片发给模型：

- `corpus.parquet.metadata` 里要能解析出 **本地路径**：`image_path`
- vLLM 的 `generator_model` 必须是**支持 vision 输入**的模型（否则会 404/不支持）
- 跑完看 `val_predictions_<algo>.jsonl` 的 `trace.used_images`：非空即表示“确实用图”

## autorag_offline_search

离线对比多种“RAG 组合/超参搜索”算法的小项目：在 `train` 上最多评测 N 个配置（默认 10），选出最优配置；再把该配置固定到 `validation` 上评测一次。

### 数据格式

每个数据集目录下包含：

- `train/corpus.parquet`
- `train/qa.parquet`
- `validation/corpus.parquet`
- `validation/qa.parquet`

说明：`three_dataset_sample_split/` **不是本仓库必带目录**。只要你的数据集目录长这样（四个 parquet），就可以直接跑。

如果你现在“找不到这些文件”，通常是因为你还没生成任何 sample dataset。你可以先跑本仓库自带的 demo 脚本生成一个最小数据集：

```bash
# 生成一个可跑通的 multimodal 示例数据集（从 M2RAG/mmqa 抽 10 条）
bash scripts/demo_run_m2rag_mmqa_10_multimodal.sh

# 生成一个可跑通的 common 示例数据集（miniwiki 抽样）
bash scripts/demo_run_miniwiki_common_30.sh
```

生成后的数据集一般会出现在 `ragsearch-update/datasets/<name>/train/*.parquet`。

`qa.parquet` 需包含至少：

- `qid`
- `query`
- `generation_gt`（参考答案；允许是 string 或 list[string]）

`corpus.parquet` 需包含至少：

- `doc_id`
- `contents`

### Pipeline（两大类）

项目现在按两类 pipeline 组织（由配置 `pipeline` 选择）：

- **common**：常用 RAG（rewriter → chunking → embedding → retriever → reranker → pruner → generator）
- **multimodal**：多模态 RAG（支持从 `Doc.metadata` 引入图像等信息，走多模态 embedding/rerank/generator）

### 配置项（新手只需要记住的最少集合）

这个项目的“核心可调旋钮”其实就几类：

- **检索方式**：`bm25_weight`（0=cosine，1=bm25，(0,1)=hybrid）+ `retriever_topk`
- **是否启用模块**：`rewriter_enabled / chunking_enabled / reranker_enabled / pruner_enabled`
- **模型选择**：`embedder_model / reranker_model / generator_model`（以及 `rewriter_model/pruner_model`）
- **multimodal 专属**：`multimodal_metadata_enabled`（是否把 caption/ocr 等元数据文本拼进语料）

完整字段与默认值见 `autorag_offline_search/pipelines/config.py::normalize_config`。

### 算法

- Random Search
- Greedy / Coordinate Descent
- TPE（Optuna）
- UCB1 Multi-Armed Bandit（把“完整 config”视为 arm）
- Thompson Sampling（连续 reward 的近似 TS）
- GRPO-lite（简化版策略梯度：对离散选项分布做优势更新）

### 算法“收敛”与 `--train_trials`（非常重要）

先说结论：**在这个项目里，大多数算法不会“自动收敛到停下来”**，而是**按预算停止**。

- **`--train_trials` 的含义**：在 `train` split 上，算法最多调用多少次 `objective(cfg)`（也就是“评测多少个配置”）。
  - `--train_trials 1`：只评测 1 个配置（常用来 smoke 跑通、看日志、验证 pipeline 没报错）
  - `--train_trials 10/30/100...`：预算越大，越有机会找到更好的配置，但耗时线性增加

那“收敛”通常指什么？

- **对搜索算法来说**：连续很多次 trial 都没有让 best reward 变好（或改善小于某个阈值），就可以认为“基本收敛了”
- **对本项目来说**：是否提前停，取决于算法实现是否支持早停

目前的行为（按代码）：

- **random / ucb1 / ts / grpo / greedy**：默认都以 `--train_trials` 为硬预算（或者候选全无效时提前结束），不会因为“看起来收敛了”就自动停。
- **tpe（Optuna）**：支持“早停”，需要你显式开启：
  - `--tpe_patience K`：如果 best reward 连续 K 个**有效 trial**都没提升（提升幅度至少 `--tpe_min_delta`），就提前停止
  - `--tpe_min_delta D`：把“提升”的门槛设为 D（默认 0）
  - `--tpe_warmup W`：至少先跑 W 个有效 trial 再允许早停

一个“会自动早停”的例子：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --out_dir /tmp/out \
  --algo tpe \
  --train_trials 100 \
  --tpe_warmup 10 \
  --tpe_patience 15 \
  --tpe_min_delta 0.001 \
  --llm_base_url http://localhost:9000/v1
```

如何选 `--train_trials`（经验法）：

- **只想跑通/确认环境**：`--train_trials 1` + `--module_logs --debug_trace_n 1`
- **小规模对比算法**：`--train_trials 10~30`
- **真的想“调参找到更好组合”**：`--train_trials 50~200+`（并建议 tpe 开 `--tpe_patience` 做早停）

### 安装

建议使用 venv：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 快速上手（给第一次接触的人）

你只需要准备两样东西：

- **一个数据集目录**：满足本文档“数据格式”的 parquet 结构
- **一个 OpenAI-compatible 的推理服务**：例如 vLLM，提供 `llm_base_url`（默认示例 `http://localhost:9000/v1`）

最小可跑（common pipeline + random search，跑 1 个 trial）：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --out_dir /tmp/autorag_out \
  --train_trials 1 \
  --algo random \
  --pipeline common \
  --llm_base_url http://localhost:9000/v1 \
  --module_logs \
  --verbose
```

跑完你主要看这三个文件：

- `results.json`：每个算法的最优配置与指标
- `summary.tsv`：便于快速 grep/Excel 的汇总表
- `val_predictions_<algo>.jsonl`：逐样本的 `pred/refs/metrics/trace`（最适合排查“答对没”）

### 模型准备（必需 / 可选）

本项目的定位是“在同一套 RAG pipeline 下，对比不同搜索算法/组合效果”，所以为了让外部用户**一键复现**，建议把模型依赖分成两类：

- **必需（所有 pipeline 都需要）**
  - **vLLM/OpenAI-compatible endpoint**：用于 `generator`（以及你若开启了 `rewriter/pruner` 也会走同一个 endpoint）
  - 你需要自己启动一个 vLLM 服务，并把 `--llm_base_url` 指到它（例如 `http://localhost:9000/v1`）
  - 本仓库默认示例使用本地模型路径：`/home/xwh/models/Qwen3-4B-Instruct-2507`（你也可以换成任何 vLLM 已加载的模型 id/path）

- **可选（取决于你是否开启模块 / 是否跑 multimodal）**
  - **Text embedding（cosine/hybrid 检索需要）**
    - `BAAI/bge-m3`
    - `intfloat/e5-large-v2`
  - **Text reranker（reranker_enabled=true 时需要）**
    - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - **Multimodal embedding / reranker（multimodal pipeline + embedding/rerank 时需要）**
    - `Qwen/Qwen3-VL-Embedding-2B`
    - `Qwen/Qwen3-VL-Embedding-8B`
    - `Qwen/Qwen3-VL-Reranker-2B`
    - `Qwen/Qwen3-VL-Reranker-8B`
    - CLIP（用于最小可用的 image+text embedding 支持）：`openai/clip-vit-base-patch32`

#### 一键下载（HuggingFace）

如果你的环境需要 HuggingFace 模型下载（embedding / reranker / CLIP），可以用仓库内脚本批量下载到本地：

```bash
# 如果模型需要权限/同意协议，请先设置：
# export HF_TOKEN=xxxx

python scripts/download_hf_models.py \
  --models "Qwen/Qwen3-VL-Reranker-2B" \
  --models "Qwen/Qwen3-VL-Reranker-8B" \
  --models "Qwen/Qwen3-VL-Embedding-2B" \
  --models "Qwen/Qwen3-VL-Embedding-8B" \
  --models "openai/clip-vit-base-patch32" \
  --local_root /home/xwh/models/hf \
  --cache_dir /home/xwh/.cache/huggingface \
  --max_retries 8
```

#### PyTorch（Blackwell / sm_120 用户注意）

如果你是 Blackwell（`sm_120`）显卡，建议先安装 **CUDA 12.8+ 的 PyTorch**，否则会看到 `sm_120 is not compatible...`。
我们把 torch 从 `requirements.txt` 中移除了，避免 pip/conda “不小心降级”你的 torch。你可以先装对应 torch，再安装项目依赖：

```bash
pip install -r requirements-torch-cu128.txt
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

补充：你可以按 pipeline 选择不同的 generator 模型（仍然是固定 prompt，只换模型）。例如 multimodal 想用 VL 指令模型：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --train_trials 1 \
  --pipeline multimodal \
  --llm_base_url http://localhost:9000/v1 \
  --generator_model /home/xwh/models/Qwen3-VL-4B-Instruct \
  --out_dir /tmp/out
```

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

补充：YAML 也支持把某些字段写成 **list**，表示“候选集合/约束搜索空间”（用于让算法在候选里挑最优）。
推荐写法是显式区分：

```yaml
base:
  pipeline: multimodal
  llm_base_url: http://localhost:9000/v1
  chunking_enabled: false
  generator_model: /home/xwh/models/Qwen3-VL-4B-Instruct

space:
  # 只在这些候选里搜索
  embedder_model:
    - /home/xwh/models/Qwen3-VL-Embedding-2B
    - /home/xwh/models/Qwen3-VL-Embedding-8B
  reranker_model:
    - /home/xwh/models/Qwen3-VL-Reranker-2B
    - /home/xwh/models/Qwen3-VL-Reranker-8B
  retriever_topk: [3, 5, 10]
  # bm25_weight 既可以写成离散候选，也可以写成“范围”（会自动展开成网格）
  bm25_weight:
    low: 0
    high: 1
    steps: 11
```

（兼容简写：如果你直接把 `embedder_model: [.., ..]` 写在顶层，也会被当作 space 约束；顶层的标量仍然是固定配置。）

然后运行：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /home/xwh/three_dataset_sample_split/bioasq \
  --train_trials 10 \
  --config /path/to/config.yaml \
  --llm_base_url http://localhost:9000/v1 \
  --module_logs \
  --out_dir /home/xwh/autorag_offline_search_runs/bioasq
```

你也可以用 `--algo` 只跑某个算法，或用 `--all_datasets_root` 一次跑三个数据集。

### 常用脚本（建议先跑这些确认环境 OK）

- **common + miniwiki（30 条 smoke）**：`scripts/demo_run_miniwiki_common_30.sh`
- **common + hotpotqa（20 条 smoke）**：`scripts/demo_run_hotpotqa_common_20.sh`
- **multimodal demo（本地图片构造数据集）**：`scripts/demo_run_multimodal.sh`
- **M2RAG(mmqa) 抽 10 条跑通 multimodal**：`scripts/demo_run_m2rag_mmqa_10_multimodal.sh`

这些脚本会把关键输出写到 `runs/<...>/` 下，并且把终端输出 `tee` 到 `run.log`，方便回溯。

### CLI 参数说明（新人向，按“你最常用的顺序”）

完整参数列表永远以 `python -m autorag_offline_search.cli -h` 为准；下面是最常用/最容易踩坑的部分。

#### 数据与输出

- **`--dataset_dir`**：单个数据集目录（必须满足 `train/` + `validation/` 四个 parquet）
- **`--all_datasets_root`**：一次跑一个 root 下的多个数据集目录（和 `--dataset_dir` 二选一）
- **`--out_dir`**：输出目录（必填）。会生成 `results.json/summary.tsv/...`
- **`--cache_dir`**：缓存目录（默认 `<out_dir>/.cache`）。chunk/embedding/chroma 等都会放这里；你想复用缓存就保持不变
- **`--seed`**：随机种子（影响抽样/算法随机性）

#### Pipeline 选择与配置

- **`--pipeline`**：`common|multimodal`（可覆盖 YAML）
- **`--config`**：YAML 路径。用于：
  - **固定某些模块开关/模型**（例如 `bm25_weight=1.0` 强制 BM25）
  - **约束搜索空间**（写 `base:` + `space:`，只在候选里搜索）
- **`--bm25_weight`**：命令行强制覆盖检索旋钮（0=cosine，1=bm25，中间=hybrid）；优先级高于 YAML/space
- **`--embedder_model`**：命令行强制覆盖 embedder（用于你只想测某个 embedder 的情况）
- **`--generator_model` / `--generator_max_tokens`**：命令行强制覆盖 generator（注意 `max_tokens` 是**输出 token 上限**，不是上下文长度）

#### 算法（怎么切换/怎么一起跑）

- **`--algo`**：要跑的搜索算法，逗号分隔，例如：
  - 只跑一种：`--algo random`
  - 同时对比多种：`--algo random,greedy,tpe,ucb1,ts,grpo`
  - 跑你自己的算法：`--algo scripts.template_algo:ExampleAlgo`
- **`--train_trials`**：训练阶段最多评测多少个 config（预算）；越大越慢
- **`--gpus`**：并行执行算法用哪些 GPU（例如 `0,1`）。仅在你启用了多算法并行时有用

#### TPE（Optuna）早停参数（可选）

- **`--tpe_patience`**：启用早停；best reward 连续多少个有效 trial 没提升就停（0=关闭）
- **`--tpe_min_delta`**：把“提升”的最小幅度设为一个阈值（避免抖动导致一直不早停）
- **`--tpe_warmup`**：至少先跑多少个有效 trial 再允许早停

#### 指标（reward 怎么算）

- **`--metrics_weights`**：reward 的加权方式，例如：
  - QA：`"qa_f1:1,em:1"`
  - 偏鲁棒：`"rougeL:0.34,bertscore_f1:0.33,meteor:0.33"`
- **`--bertscore_model`**：只有当你把 `bertscore_f1` 放进 weights 时才会用到

#### 日志 / 调试（排查“到底有没有跑对”）

- **`--module_logs`**：打开 pipeline 各模块日志（rewriter/retriever/pruner/generator 等），适合确认“到底走了哪个分支”
- **`--verbose`**：打开每个 trial 的日志（trial idx、reward、best so far）
- **`--log_every`**：配合 `--verbose`，控制“每 N 次 objective 打一次日志”
- **`--dump_val_generations`**：把 validation 的逐样本预测写到 `val_predictions_<algo>.jsonl`
- **`--dump_val_limit`**：只 dump 前 N 条 validation（避免文件太大）
- **`--debug_trace_n`**：在运行时直接打印前 N 条样本的 `gold/pred/trace`（适合快速确认 generator 是否在胡答）
- **`--debug_trace_every`**：只在第 N、2N、3N... 个 trial 打 debug trace（大搜索时降噪）
- **`--debug_trace_split`**：`train|validation|both`
- **`--debug_trace_max_chars`**：trace 打印截断长度
- **`--show_eval_progress`**：每次 eval 打 tqdm（很吵，通常不建议开）

#### 插件（自定义 RAG / 自定义搜索空间）

- **`--rag_plugin module:Class`**：替换 RAG 实现（需 `__init__(docs, config)` + `answer(query)->str`）
- **`--space_plugin module:ObjOrClass`**：替换搜索空间（需 `all_configs()` 和/或 `space_dict()`）

### multimodal 数据怎么写（最小可用）

multimodal pipeline 仍然读取同样的 `corpus.parquet/qa.parquet`，区别在于 `corpus.parquet` 的 `metadata` 里可以带图像信息：

- `metadata.image_path`：本地图片路径（推荐）
- `metadata.image_url`：如果你自己实现了“下载成 data URL”的逻辑也可以；当前内置生成器是读本地文件
- `metadata.caption/ocr/...`：会被拼进 `contents`，提升纯文本检索的可用性

本仓库示例：`scripts/datasets/make_m2rag_mmqa_sample_dataset.py` 会把 M2RAG 的 `pos_image_path` 下载成本地文件，并写进 `metadata.image_path`。

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

### 可插拔：自定义搜索算法（推荐）

你的方法如果想在本框架下复现/对比，只需要实现一个“算法类”，遵循统一的 **输入/输出标准**，就能被 `autorag_offline_search` 调用并在相同 RAG pipeline + 指标下评测。

- **输入标准**：`autorag_offline_search.algorithms.base.SearchInput`
  - `objective(cfg: Dict) -> Trial`：评测函数（内部会跑 RAG + metrics，返回 reward/metrics/seconds）
  - `budget: int`：最多评测多少个 config（每调用一次 objective 消耗 1）
  - `seed: int`：随机种子提示
  - `space: Dict[str, Sequence] | None`：离散搜索空间（给 TPE/GRPO/greedy 这类算法）
  - `configs: List[Dict] | None`：预枚举的 configs（给 random/UCB/TS 这类把 config 当 arm 的算法）
- **输出标准**：`autorag_offline_search.algorithms.base.SearchOutput`
  - `trials: List[Trial]`：所有评测过的 trial
  - `best: Trial | None`：最优 trial（不填也可以，框架会自动从 trials 里挑）

最小示例（新接口，推荐）：

```python
from dataclasses import dataclass
from autorag_offline_search.algorithms.base import SearchAlgorithm, SearchInput, SearchOutput, best_trial


@dataclass
class MyAlgo:
    name: str = "my_algo"

    def run(self, inp: SearchInput) -> SearchOutput:
        # 例：最简单的“只评测一次空 config”
        tr = inp.objective({})
        return SearchOutput(algo=self.name, trials=[tr], best=best_trial([tr]))
```

运行时，把它作为 `--algo` 传入（可与内置算法混用，用逗号分隔）：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --out_dir /tmp/out \
  --train_trials 10 \
  --config /path/to/config.yaml \
  --llm_base_url http://localhost:9000/v1 \
  --algo random,my_pkg.my_algo:MyAlgo
```

（说明：本项目只接受新接口 `run(SearchInput)->SearchOutput`，不再保留旧接口兼容层，以保证代码简洁易读。）

#### 外部算法作者 checklist（建议照这个来）

- **必须实现**：`run(self, inp: SearchInput) -> SearchOutput`
- **必须有**：`name: str`（会出现在 `results.json/summary.tsv` 里）
- **不要直接访问**：数据集、RAG pipeline 内部对象（都通过 `inp.objective(cfg)` 间接评测）
- **候选来源二选一**：
  - **configs 模式**：用 `inp.configs` 作为候选池（random/UCB/TS 这类）
  - **space 模式**：用 `inp.space` 作为离散搜索空间（greedy/TPE/GRPO 这类）
- **合法性校验**：优先走 `evaluate(inp, cfg)`（会自动处理 `inp.validate` + `inp.on_trial`）
- **预算约束**：把 `inp.budget` 当成“最多 objective 次数”，不要超评测次数
- **可复现性**：用 `inp.seed` 初始化随机数

#### 直接复用本仓库示例算法

仓库自带一个最小示例算法：`scripts/template_algo.py`，可以这样跑：

```bash
python -m autorag_offline_search.cli \
  --dataset_dir /path/to/dataset \
  --out_dir /tmp/out \
  --train_trials 3 \
  --config /path/to/config.yaml \
  --llm_base_url http://localhost:9000/v1 \
  --algo scripts.template_algo:ExampleAlgo
```

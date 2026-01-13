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
  - `generator_model`：默认会按 pipeline 区分（可通过 YAML/CLI 覆盖）
    - `common/graph`：默认 `qwen3`
    - `multimodal`：默认 `qwen3_vl_4b`（本地路径 `/home/xwh/models/Qwen3-VL-4B-Instruct`）
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

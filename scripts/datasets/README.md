# Dataset adapters

This repo uses a simple on-disk dataset format:

- `<dataset_dir>/train/corpus.parquet`
- `<dataset_dir>/train/qa.parquet`
- `<dataset_dir>/validation/corpus.parquet`
- `<dataset_dir>/validation/qa.parquet`

Alternative (supported) single-file format (no train/validation folders):

- `<dataset_dir>/corpus.parquet`
- `<dataset_dir>/qa.parquet`

Optional (GraphRAG):

  - each line: `{"src":"<doc_id>","dst":"<doc_id>"}` (recommended)
  - or: `{"src_i": 12, "dst_i": 34}` (chunk indices; only safe if chunking is disabled and doc order is fixed)

Adapters (planned):

- HotpotQA (fullwiki): build corpus from wiki paragraphs.
- MultiHop-RAG: build corpus from provided passages/snippets; do NOT use evidence labels to build edges.


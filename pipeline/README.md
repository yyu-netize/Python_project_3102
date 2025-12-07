# Plants vs. Zombies RAG System

A Plants vs. Zombies knowledge Q&A system based on Retrieval-Augmented Generation (RAG) technology. This system supports multiple retrieval modes, re-ranking, multi-turn conversation, and a complete evaluation pipeline.

## Project Introduction

This project implements a complete RAG system for answering questions about the game *Plants vs. Zombies*. The system includes the following core features:

- **Multiple Retrieval Modes**: Supports dense vector retrieval (Dense), BM25 sparse retrieval, hybrid retrieval (Hybrid), and HyDE retrieval.
- **Re-ranking Mechanism**: Uses a Cross-Encoder to re-rank retrieval results.
- **Multi-turn Conversation**: Supports context-aware multi-turn Q&A, including query rewriting functionality.
- **Complete Evaluation**: Provides retrieval evaluation, answer quality evaluation, and LLM-as-Judge evaluation.


## Project Structure

```
pipeline/
├── fetch_content.py          # 从Wiki爬取内容
├── process_chunks.py         # 文本分块处理
├── build_index.py            # 构建向量索引和BM25索引
├── search.py                 # RAG检索核心模块
├── generator.py              # LLM答案生成模块
├── multi_turn_chat.py        # 多轮对话支持
├── question_generate.py      # QA数据集生成
├── evaluation.py             # 评估脚本
├── requirements.txt          # 依赖包列表
└── README.md                 # 项目说明文档
```

## Installation Steps

### 1. Requirements
- Python 3.8+
- CUDA-supported GPU (Recommended for accelerating model inference)
- At least 8GB RAM

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

On the first run, the system will automatically download the required NLTK data packages (punkt, punkt_tab). If the download fails, you can download them manually:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 4. Configure API Key

Configure your SiliconFlow API key in `generator.py` and `search.py`:

```python
SILICONFLOW_API_KEY = "your-api-key-here"
```

## Usage Flow

### Step 1: Data Fetching

Fetch content from the Plants vs. Zombies Wiki:

```bash
python fetch_content.py
```

Output: `./data/pvz_wiki_rag.json`

### Step 2: Text Chunking

Chunk the fetched content:

```bash
python process_chunks.py
```

Output: `./data/rag_chunks.json`

### Step 3: Build Index

Build vector index (ChromaDB) and BM25 index:

```bash
python build_index.py
```

Output:
- `./chroma_db_m3/` - ChromaDB vector database
- `./bm25_m3.pkl` - BM25 index file

### Step 4: Generate QA Dataset (Optional)

Use LLM to generate a QA dataset for evaluation:

```bash
python question_generate.py --input ./data/rag_chunks.json --output ./data/qa_dataset_clean.json
```
Open the notebook:
```bash
QA_set.ipynb
```
Run all cells to produce:

- `qa_dataset_annotated.json` — includes all heuristic flags  
- `qa_suspicious.json` — items requiring further inspection  

This step applies:

- overlap ratio checking  
- pronoun ambiguity detection  
- meta-question detection  
- chunk-density anomaly checks  
- answer length heuristics  
- groundedness similarity checks  

### Step 5: Use RAG System

#### Basic Retrieval (search.py)

```python
from search import UltimateRAG

rag = UltimateRAG()
results = rag.search(
    query="Which plant can slow down zombies?",
    retrieve_mode="hybrid",  # Options: hybrid, dense, bm25, hyde
    rerank_mode=True
)
```

#### With Answer Generation (generator.py)

```python
from generator import UltimateRAGWithGenerator

rag_gen = UltimateRAGWithGenerator()
answer = rag_gen.search(
    query="What is the sun cost of Peashooter?",
    retrieve_mode="hybrid",
    rerank_mode=True,
    prompt_mode="instruction",
    message_mode="with_system"
)
```
LLM-based semantic screening pass directly from `multi_turn_chat.py`:

```bash
python multi_turn_chat.py \
  --mode review-qa \
  --qa-path qa_dataset_annotated.json \
  --output qa_dataset_clean.json \
  --sample-size 0
```
### Step 6: Evaluation System

#### 1. Retrieval Evaluation

##### Command Example
```bash
python evaluation.py eval-retrieval \
    --qa-path ./data/qa_dataset_clean.json \
    --mode hybrid \ % can switch among bm25/hybrid/dense/hyde
    --k-list 1 3 5 10 \
    --sample-size 100 \
    --rerank
```

##### Optional Arguments

| Argument | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--qa-path` | str | `qa_dataset_clean.json` | — | QA data file path |
| `--mode` / `--retrieve-mode` | str | `hybrid` | `hybrid` / `dense` / `bm25` / `hyde` | Retrieval mode |
| `--k-list` | int list | `[1,3,5,10]` | List of arbitrary integers | k values for Recall@k calculation |
| `--sample-size` | int | `100` | — | Number of samples extracted from QA set |
| `--rerank` / `--use-rerank` | bool | `False` | — | Whether to enable Cross-Encoder re-ranking |


#### 2. Answer Quality Evaluation

##### Command Example
```bash
python evaluation.py eval-answers \
    --qa-path ./data/qa_dataset_clean.json \
    --mode hybrid \
    --prompt-mode instruction \
    --message-mode with_system \
    --enable-rewrite \ % Ablation task can be done by deleting the line
    --sample-size 50
```

##### Optional Arguments

| Argument | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--qa-path` | str | `qa_dataset_clean.json` | — | QA data file path |
| `--rewrite` / `--enable-rewrite` | bool | `False` | — | Whether to enable query rewriting |
| `--mode` / `--retrieve-mode` | str | `hybrid` | `hybrid` / `dense` / `bm25` / `hyde` | Retrieval mode |
| `--sample-size` | int | `50` | — | Sampling quantity |
| `--rerank` / `--use_rerank` | bool | `True` | — | Whether to re-rank candidate documents |
| `--prompt-mode` | str | `instruction` | `instruction` / `vanilla` | Prompt template mode |
| `--message-mode` | str | `with_system` | `with_system` / `no_system` | Whether ChatCompletion uses system message |
| `--random-seed` | int | `42` | — | Random sampling seed |



#### 3. Retrieval Ablation Experiment

##### Command Example
```bash
python evaluation.py ablation-retrieval \
    --qa-path ./data/qa_dataset_clean.json \
    --sample-size 100 \
    --k-list 1 3 5 10
```


##### Optional Arguments

| Argument | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--qa-path` | str | `qa_dataset_clean.json` | — | QA data file path |
| `--sample-size` | int | `100` | — | Sampling quantity |
| `--k-list` | int list | `[1,3,5,10]` | List of arbitrary integers | Recall@k metrics |


#### 4. LLM-as-Judge Evaluation

##### Command Example
```bash
python evaluation.py judge \
    --eval-path eval_answers_hybrid_rerankTrue_rewriteFalse.json \
    --sample-size 30
```
##### Optional Arguments
| Argument | Type | Default | Options | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--eval-path` | str | *Required* | — | `.json` result generated by eval-answers |
| `--sample-size` | int | `30` | — | Sample from results for LLM scoring |


## Retrieval Mode Explanation

- **dense**: Uses dense vector retrieval (BGE-M3 embedding model)
- **bm25**: Uses BM25 keyword retrieval
- **hybrid**: Hybrid retrieval (combines vector and BM25 results)
- **hyde**: Hypothetical Document Embeddings

## Configuration Explanation

### Model Configuration

- **Embedding Model**: `BAAI/bge-m3` (Configured in `build_index.py` and `search.py`)
- **Re-ranking Model**: `bge-reranker-v2-m3` (Configured in `search.py`)
- **Generation Model**: `Qwen/Qwen3-8B` (Called via SiliconFlow API)

### Data Path Configuration

- Input Data: `./data/pvz_wiki_rag.json`
- Chunked Data: `./data/rag_chunks.json`
- Vector Database: `./chroma_db_m3/`
- BM25 Index: `./bm25_m3.pkl`


## Evaluation Metrics

- **Retrieval Evaluation**: Recall@k, MRR (Mean Reciprocal Rank)
- **Answer Quality**: Exact Match (EM), F1 Score, ROUGE-L
- **LLM Evaluation**: Faithfulness, Relevance, Overall

## Advanced Direction

### Database Security and Privacy Protection
A separate folder has been created.
See the README in the `privacy` folder for details.

### Multi-turn Conversation (multi_turn_chat.py)

```python
from multi_turn_chat import MultiTurnRAGChat

chat = MultiTurnRAGChat(
    enable_rewrite=True,
    default_retrieve_mode="hybrid",
    default_rerank_mode=True
)

# Round 1
answer1 = chat.ask("In Versus Mode, what is special about Catapult Zombie?")

# Round 2 (Context-aware)
answer2 = chat.ask("Why should I protect him?")

# View chat history
history = chat.get_history()
```












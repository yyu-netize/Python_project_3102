# Plants vs. Zombies RAG System

一个基于检索增强生成（RAG）技术的植物大战僵尸知识问答系统。该系统支持多种检索模式、重排序、多轮对话和完整的评估流程。

## 项目简介

本项目实现了一个完整的RAG系统，用于回答关于《植物大战僵尸》游戏的问题。系统包含以下核心功能：

- **多种检索模式**：支持密集向量检索（Dense）、BM25稀疏检索、混合检索（Hybrid）和HyDE检索
- **重排序机制**：使用交叉编码器（Cross-Encoder）对检索结果进行重排序
- **多轮对话**：支持上下文感知的多轮问答，包含查询重写功能
- **完整评估**：提供检索评估、答案质量评估和LLM-as-Judge评估

## 项目结构

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

## 安装步骤

### 1. 环境要求
- Python 3.8+
- CUDA支持的GPU（推荐，用于加速模型推理）
- 至少8GB内存

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载NLTK数据

首次运行时，系统会自动下载所需的NLTK数据包（punkt, punkt_tab）。如果下载失败，可以手动下载：

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

### 4. 配置API密钥

在 `generator.py` 和 `search.py` 中配置你的SiliconFlow API密钥：

```python
SILICONFLOW_API_KEY = "your-api-key-here"
```

## 使用流程

### 步骤1：数据爬取

从Plants vs. Zombies Wiki爬取内容：

```bash
python fetch_content.py
```

输出：`./data/pvz_wiki_rag.json`

### 步骤2：文本分块

将爬取的内容进行分块处理：

```bash
python process_chunks.py
```

输出：`./data/rag_chunks.json`

### 步骤3：构建索引

构建向量索引（ChromaDB）和BM25索引：

```bash
python build_index.py
```

输出：
- `./chroma_db_m3/` - ChromaDB向量数据库
- `./bm25_m3.pkl` - BM25索引文件

### 步骤4：生成QA数据集（可选）

使用LLM生成评估用的QA数据集：

```bash
python question_generate.py --input ./data/rag_chunks.json --output ./data/qa_dataset_clean.json
```

### 步骤5：使用RAG系统

#### 基础检索（search.py）

```python
from search import UltimateRAG

rag = UltimateRAG()
results = rag.search(
    query="Which plant can slow down zombies?",
    retrieve_mode="hybrid",  # 可选: hybrid, dense, bm25, hyde
    rerank_mode=True
)
```

#### 带答案生成（generator.py）

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


### 步骤6：评估系统

#### 1.检索评估

##### 命令示例
```bash
python evaluation.py eval-retrieval \
    --qa-path ./data/qa_dataset_clean.json \
    --mode hybrid \
    --k-list 1 3 5 10 \
    --sample-size 100 \
    --rerank
```

##### 可选参数

| 参数                           | 类型     | 默认值                     | 可选项                                  | 说明                     |
| ---------------------------- | ------ | ----------------------- | ------------------------------------ | ---------------------- |
| `--qa-path`                  | str    | `qa_dataset_clean.json` | —                                    | QA 数据文件路径              |
| `--mode` / `--retrieve-mode` | str    | `hybrid`                | `hybrid` / `dense` / `bm25` / `hyde` | 检索模式                   |
| `--k-list`                   | int 列表 | `[1,3,5,10]`            | 任意整数列表                               | 计算 Recall@k 的 k 值      |
| `--sample-size`              | int    | `100`                   | —                                    | 从 QA 集中抽取的样本数          |
| `--rerank` / `--use-rerank`                   | bool   | `False`                 | —                                    | 是否启用 Cross-Encoder 重排序 |


#### 2.答案质量评估

##### 命令示例
```bash
python evaluation.py eval-answers \
    --qa-path ./data/qa_dataset_clean.json \
    --mode hybrid \
    --prompt-mode instruction \
    --message-mode with_system \
    --enable-rewrite \
    --sample-size 50
```

##### 可选参数

| 参数                           | 类型   | 默认值                     | 可选项                                  | 说明                                 |
| ---------------------------- | ---- | ----------------------- | ------------------------------------ | ---------------------------------- |
| `--qa-path`                  | str  | `qa_dataset_clean.json` | —                                    | QA 数据文件路径                          |
| `--rewrite` / `--enable-rewrite`           | bool | `False`                 | —                                    | 是否启用 query rewriting               |
| `--mode` / `--retrieve-mode` | str  | `hybrid`                | `hybrid` / `dense` / `bm25` / `hyde` | 检索模式                               |
| `--sample-size`              | int  | `50`                    | —                                    | 抽样数量                               |
| `--rerank`       / `--use_rerank`            | bool | `True`                 | —                                    | 是否对候选文档进行重排序                       |
| `--prompt-mode`              | str  | `instruction`           | `instruction` / `instruction`            | prompt 模板模式                        |
| `--message-mode`             | str  | `with_system`           | `with_system` / `no_system`          | ChatCompletion 是否使用 system message |
| `--random-seed`              | int  | `42`                    | —                                    | 随机采样种子                             |



#### 3.检索消融实验

##### 命令示例
```bash
python evaluation.py ablation-retrieval \
    --qa-path ./data/qa_dataset_clean.json \
    --sample-size 100 \
    --k-list 1 3 5 10
```


##### 可选参数

| 参数              | 类型     | 默认值                     | 可选项    | 说明          |
| --------------- | ------ | ----------------------- | ------ | ----------- |
| `--qa-path`     | str    | `qa_dataset_clean.json` | —      | QA 数据文件路径   |
| `--sample-size` | int    | `100`                   | —      | 抽样数量        |
| `--k-list`      | int 列表 | `[1,3,5,10]`            | 任意整数列表 | Recall@k 指标 |


#### 4.LLM-as-Judge评估

##### 命令示例
```bash
python evaluation.py judge \
    --eval-path eval_answers_hybrid_rerankTrue_rewriteFalse.json \
    --sample-size 30
```
##### 可选参数
| 参数              | 类型  | 默认值  | 可选项 | 说明                          |
| --------------- | --- | ---- | --- | --------------------------- |
| `--eval-path`   | str | *必填* | —   | eval-answers 生成的 `.json` 结果 |
| `--sample-size` | int | `30` | —   | 从结果中抽样进行 LLM 评分             |


## 检索模式说明

- **dense**: 使用密集向量检索（BGE-M3嵌入模型）
- **bm25**: 使用BM25关键词检索
- **hybrid**: 混合检索（结合向量和BM25结果）
- **hyde**: 假设文档嵌入（Hypothetical Document Embeddings）

## 配置说明

### 模型配置

- **嵌入模型**: `BAAI/bge-m3` (在 `build_index.py` 和 `search.py` 中配置)
- **重排序模型**: `bge-reranker-v2-m3` (在 `search.py` 中配置)
- **生成模型**: `Qwen/Qwen3-8B` (通过SiliconFlow API调用)

### 数据路径配置

- 输入数据：`./data/pvz_wiki_rag.json`
- 分块数据：`./data/rag_chunks.json`
- 向量数据库：`./chroma_db_m3/`
- BM25索引：`./bm25_m3.pkl`


## 评估指标

- **检索评估**：Recall@k, MRR (Mean Reciprocal Rank)
- **答案质量**：Exact Match (EM), F1 Score, ROUGE-L
- **LLM评估**：Faithfulness, Relevance, Overall

## Advanced Direction

### 数据库安全隐私保护
另建立了一个文件夹
详情见 privacy文件夹的README

### 多轮对话（multi_turn_chat.py）

```python
from multi_turn_chat import MultiTurnRAGChat

chat = MultiTurnRAGChat(
    enable_rewrite=True,
    default_retrieve_mode="hybrid",
    default_rerank_mode=True
)

# 第一轮
answer1 = chat.ask("In Versus Mode, what is special about Catapult Zombie?")

# 第二轮（支持上下文）
answer2 = chat.ask("Why should I protect him?")

# 查看对话历史
history = chat.get_history()
```





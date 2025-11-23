# Plants vs. Zombies RAG System 

## Part1
**Plants vs. Zombies Fandom Wiki** 自动爬取 → 解析 → 切分 → 嵌入 → 混合检索 → 重排序 的完整 RAG 构建流程。


---

###  结构

```
├── fetch_content.py          # 爬取 + 解析 + 清洗 Fandom Wiki
├── process_chunks.py         # Markdown 节点级切分，生成 RAG 片段
├── build_index.py            # 使用 BGE-M3 构建 Chroma + BM25 混合索引
├── build_index_better.py     # 使用更强的 7B embedding 模型
└── search.py                 # Hybrid 检索 + Rerank
```

---

### 1. fetch_content.py

#### 功能

* 从 **Plants vs Zombies Fandom Wiki** 自动批量爬取全部页面
* 解析正文、图片、链接、Fandom 特有的 **Infobox 属性**
* 使用 `markdownify` 将 HTML 转换为 Markdown
* 最终导出 **适合 RAG 的结构化 JSON**：

```json
{
  "title": "Peashooter",
  "infobox": {"sun_cost": 100, "recharge": "fast", ...},
  "markdown": "# Peashooter..."
}
```

#### 依赖

```
pip install requests beautifulsoup4 markdownify
```

---

### 2. process_chunks.py

#### 功能

* 基于 Markdown **标题层级**的 Section-based Chunking
* Infobox → Key Statistics 自动提取为高优先级 Chunk
* 上下文注入：

  * 标题
  * Infobox 信息
  * 正文内容
* 输出：`chunks/*.jsonl`


* Wiki 页面结构清晰：按标题分割可最大化语义一致性
* 统计信息（例如攻击力、冷却、阳光消耗）对问答非常关键，应优先检索

---

### 3. build_index.py

#### 功能

version1: **轻量、低显存** 的双路 Hybrid 索引：

* **ChromaDB (向量索引)** → 语义召回
* **BM25 (关键词索引)** → 精确召回

Embedding 使用：

* `BAAI/bge-m3`（2GB 以下显存即可跑）

### 安装

```
pip install torch transformers accelerate chromadb rank_bm25 nltk
```

---

### 4. build_index_better.py

#### 功能

version2: **≥ 15GB 显存**：

* 使用 7B 级别强模型（例如 gte-Qwen2-7B / bge-large-zh）生成更高质量向量
* 显著提升长文本检索准确度

#### 依赖

```
pip install chromadb sentence-transformers rank_bm25 tqdm
```

---

### 5. search.py

#### 功能

**混合检索 + 重排序（最强开源方案）**

检索流程：

1. BM25 提取精确匹配（捕获技能名、植物名）
2. 向量模型提取语义相关（理解复杂问法）
3. 合并后的候选集使用：

   * **bge-reranker-v2-m3**（Cross Encoder） 进行最终排序

此方案在问答类任务中表现最优。

#### 安装

```
pip install torch transformers chromadb rank_bm25
```

---


import json
import os
import pickle
from tqdm import tqdm

import chromadb
from chromadb.config import Settings

# Embedding 模型
from sentence_transformers import SentenceTransformer

# 关键词检索 (BM25)
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# --- 配置 ---
INPUT_FILE = "./data/rag_chunks.json"
DB_DIR = "./chroma_db_m3"                 # 向量数据库存储路径
BM25_PATH = "./bm25_m3.pkl"         # BM25 索引存储路径

# 使用 BAAI/bge-m3 模型
MODEL_NAME = "BAAI/bge-m3" 

def build_rag_index():
    # 0. 检查 NLTK 数据 (用于 BM25 分词)
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

    print("1. 加载数据块...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 过滤掉像 "Home" 这种只有乱七八糟 Section 的块 
    # 规则：如果 Section 标题长度超过 50 个字符，说明解析错了，不如归类为 "General"
    valid_chunks = []
    for c in chunks:
        section_title = c['text'].split('\n')[1].replace('Section: ', '')
        if len(section_title) > 80: 
             # 修正过长的标题，防止污染索引
             c['text'] = c['text'].replace(f"Section: {section_title}", "Section: Overview")
        valid_chunks.append(c)
    
    chunks = valid_chunks
    print(f"   有效数据块数量: {len(chunks)}")

    # ---------------------------------------------------------
    # 2. 构建 BM25 倒排索引 (用于关键词精准匹配 - Hybrid Part A)
    # ---------------------------------------------------------
    print("2. 构建 BM25 索引 (关键词)...")
    # 提取所有文本用于分词
    corpus_text = [c['text'] for c in chunks]
    # 分词
    tokenized_corpus = [word_tokenize(doc.lower()) for doc in tqdm(corpus_text, desc="Tokenizing")]
    # 建立索引
    bm25 = BM25Okapi(tokenized_corpus)
    
    # 保存 BM25 索引和对应的 chunks 映射 (检索时需要根据 ID 找回原文)
    with open(BM25_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
    print(f"   BM25 索引已保存至 {BM25_PATH}")

    # ---------------------------------------------------------
    # 3. 构建 Vector 索引 (用于语义理解 - Hybrid Part B)
    # ---------------------------------------------------------
    print(f"3. 加载 Embedding 模型: {MODEL_NAME} ...")
    # device='cuda' 如果你有 N 卡，否则 'cpu'
    # trust_remote_code=True 是 bge-m3 需要的
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device='cuda' if os.getenv('CUDA_VISIBLE_DEVICES') else 'cpu')

    print("4. 初始化 ChromaDB...")
    # 使用持久化存储，这样下次不用重新跑
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "pvz_knowledge_m3"
    
    # 如果集合已存在，先删除旧的 (开发阶段方便重置)
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"} # 使用余弦相似度
    )

    print("5. 生成向量并写入数据库 (这可能需要几分钟)...")
    
    batch_size = 32
    total_chunks = len(chunks)
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Embedding"):
        batch = chunks[i : i + batch_size]
        
        batch_ids = [item['id'] for item in batch]
        batch_texts = [item['text'] for item in batch]
        batch_metadatas = [item['metadata'] for item in batch]
        
        # 生成向量
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        
        # 写入 Chroma
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metadatas
        )

    print(f"\n所有索引构建完成！")
    print(f"- 向量库位置: {DB_DIR}")
    print(f"- 关键词库位置: {BM25_PATH}")

if __name__ == "__main__":
    build_rag_index()
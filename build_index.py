import json
import os
import pickle
from tqdm import tqdm
import torch  # 必须导入 torch 才能检测显卡

import chromadb
# Embedding 模型
from sentence_transformers import SentenceTransformer

# 关键词检索 (BM25)
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# --- 配置 ---
INPUT_FILE = "/home/zbz/python_programming/Python_project_3102/data/rag_chunks.json"
DB_DIR = "./chroma_db_m3"
BM25_PATH = "./bm25_m3.pkl"
MODEL_NAME = "/home/zbz/models/bge-m3"

# --- 针对 RTX A6000 的高性能配置 ---
BATCH_SIZE = 64           # A6000 显存巨大，可以开大 Batch Size 提升速度
MAX_SEQ_LENGTH = 1024     # 适当放宽长度限制，但保留安全底线
MAX_TEXT_CHARS = 10000    # 字符串截断

def build_rag_index():
    print("=== 开始构建 RAG 索引 (GPU 加速版) ===")
    
    # 0. 检查 NLTK
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

    # 1. 加载数据
    print("1. 加载数据块...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    valid_chunks = []
    for c in chunks:
        # 简单清洗
        if not c['text'].strip(): continue
        # 修复标题
        lines = c['text'].split('\n')
        if len(lines) > 1 and len(lines[1]) > 80:
             c['text'] = c['text'].replace(lines[1], "Section: Overview")
        valid_chunks.append(c)
    chunks = valid_chunks
    print(f"   有效数据块: {len(chunks)}")

    # 2. BM25 (CPU 任务)
    print("2. 构建 BM25 索引...")
    # 这一步必须在 CPU 跑，和显卡无关
    if not os.path.exists(BM25_PATH):
        tokenized_corpus = [word_tokenize(c['text'].lower()) for c in tqdm(chunks, desc="   Tokenizing")]
        bm25 = BM25Okapi(tokenized_corpus)
        with open(BM25_PATH, 'wb') as f:
            pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
    else:
        print("   BM25 索引已存在，跳过构建。")

    # 3. Vector 索引 (GPU 任务)
    print(f"3. 加载 Embedding 模型: {MODEL_NAME} ...")
    
    # 【核心修复】正确的 GPU 检测逻辑
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   >>> 成功检测到 GPU: {gpu_name}")
        print(f"   >>> 显存充足，开启高性能模式 (Batch Size: {BATCH_SIZE})")
    else:
        device = "cpu"
        print("   >>> 未检测到 GPU，将使用 CPU 运行 (速度较慢)")

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("4. 初始化 ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "pvz_knowledge_m3"
    try: client.delete_collection(name=collection_name) 
    except: pass
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    print("5. 生成向量...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="   Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        batch_ids = [x['id'] for x in batch]
        # 截断过长文本
        batch_texts = [x['text'][:MAX_TEXT_CHARS] for x in batch]
        batch_metadatas = [x['metadata'] for x in batch]
        
        try:
            embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
            collection.add(ids=batch_ids, documents=batch_texts, embeddings=embeddings, metadatas=batch_metadatas)
        except Exception as e:
            print(f"Error batch {i}: {e}")
            continue

    print(f"\n=== 完成！向量库已保存至 {DB_DIR} ===")

if __name__ == "__main__":
    build_rag_index()
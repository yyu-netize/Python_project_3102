import json
import os
import pickle
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# --- 配置 ---
INPUT_FILE = "rag_chunks.json"
DB_DIR = "./chroma_db_7b"         # 顶级库目录
BM25_PATH = "./bm25_7b.pkl"

# === 7B Embedding 模型 ===
MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Qwen2-7B-Instruct 推荐的 Pooling 方式：取最后一个有效 token 的向量。
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_embeddings(model, tokenizer, texts, batch_size=8):
    """
    使用 7B 模型批量生成向量
    """
    all_embeddings = []
    
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with A6000"):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize
        batch_dict = tokenizer(
            batch_texts, 
            max_length=4096, # A6000 应该可以拉满到 4096 甚至更多
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            # 使用 Last Token Pooling
            embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            # 归一化 (Cosine Similarity 需要)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
        all_embeddings.extend(embeddings.cpu().tolist())
        
    return all_embeddings

def build_ultimate_index():
    # 0. 初始化
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    print(f"1. Starting training ... Loading: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True,
        device_map="cuda",      # 自动分配到 GPU
        torch_dtype=torch.float16 # 使用 FP16 节省显存并加速
    )
    model.eval() # 评估模式

    print("2. 加载数据块...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 简单清洗
    valid_chunks = []
    for c in chunks:
        # 修复可能的超长标题 bug
        if "Section: " in c['text']:
            parts = c['text'].split("Section: ")
            if len(parts) > 1 and len(parts[1].split('\n')[0]) > 100:
                 c['text'] = c['text'].replace(parts[1].split('\n')[0], "Overview")
        valid_chunks.append(c)
    chunks = valid_chunks
    print(f"   有效块: {len(chunks)}")

    # ---------------------------------------------------------
    # 3. 构建 BM25 (不可或缺的混合检索部分)
    # ---------------------------------------------------------
    print("3. 构建 BM25 索引...")
    tokenized_corpus = [word_tokenize(doc['text'].lower()) for doc in tqdm(chunks, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_PATH, 'wb') as f:
        pickle.dump({'bm25': bm25, 'chunks': chunks}, f)

    # ---------------------------------------------------------
    # 4. 构建 7B 模型的向量库
    # ---------------------------------------------------------
    print("4. 生成 7B 模型的高维向量 (这会很快)...")
    
    chunk_texts = [c['text'] for c in chunks]
    # batch_size 可以根据显存调整，A6000 跑 batch_size=16 或 32 应该很稳
    embeddings = get_embeddings(model, tokenizer, chunk_texts, batch_size=16)
    
    print("5. 写入 ChromaDB...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "pvz_knowledge_7b"
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    # 批量写入
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_end = min(i + batch_size, len(chunks))
        collection.add(
            ids=[c['id'] for c in chunks[i:batch_end]],
            documents=[c['text'] for c in chunks[i:batch_end]],
            metadatas=[c['metadata'] for c in chunks[i:batch_end]],
            embeddings=embeddings[i:batch_end]
        )

    print(f"\n✅ Complete! Ultimate RAG Index built successfully.")
    print(f"   Model: {MODEL_NAME} (7B Params)")
    print(f"   Vector DB: {DB_DIR}")

if __name__ == "__main__":
    build_ultimate_index()
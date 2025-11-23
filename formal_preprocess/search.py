import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import chromadb
from rank_bm25 import BM25Okapi
import pickle
import json
import os

# --- 配置  ---
DB_DIR = "./chroma_db_m3"
BM25_PATH = "./bm25_m3.pkl"

# 模型定义
EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen2-7B-instruct"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3" # 强力重排模型

# --- 显卡配置 ---
DEVICE = "cuda"

class UltimateRAG:
    def __init__(self):
        print("⚙️  正在初始化 RAG 引擎 (加载模型需占用约 18GB 显存)...")
        
        # 1. 加载 Embedding 模型 (7B)
        print(f"   [1/4] 加载 Embedding 模型: {EMBEDDING_MODEL_NAME} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME, trust_remote_code=True)
        self.embed_model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_NAME, 
            trust_remote_code=True, 
            torch_dtype=torch.float16 # FP16 加速
        ).to(DEVICE)
        self.embed_model.eval()

        # 2. 加载 Reranker 模型 (Cross-Encoder)
        print(f"   [2/4] 加载 Reranker 模型: {RERANKER_MODEL_NAME} ...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            torch_dtype=torch.float16
        ).to(DEVICE)
        self.rerank_model.eval()

        # 3. 连接 ChromaDB
        print("   [3/4] 连接向量数据库...")
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection("pvz_knowledge_m3")

        # 4. 加载 BM25
        print("   [4/4] 加载 BM25 索引...")
        with open(BM25_PATH, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.bm25_chunks = data['chunks'] # BM25 需要原始 chunks 列表来定位结果

        print("✅ RAG 引擎就绪! 等待指令...\n")

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """与 build 脚本保持一致的 Pooling 策略"""
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_query_embedding(self, query):
        """
        为 gte-Qwen2 添加检索指令
        """
        # 指令：定义任务性质
        task_instruction = "Retrieve detailed attributes, stats, and strategies for Plants vs. Zombies game entities."
        # 格式：Instruction + \n + Query
        prompt = f"Instruction: {task_instruction}\nQuery: {query}"
        
        inputs = self.tokenizer(
            [prompt], 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            embedding = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            
        return embedding[0].cpu().tolist()

    def retrieve_hybrid(self, query, top_k=30):
        """
        混合检索：从 Vector 和 BM25 各取 top_k，取并集
        """
        candidates = {} # {chunk_id: chunk_data}

        # --- A. 向量检索 ---
        query_vec = self.get_query_embedding(query)
        vec_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )
        
        # 处理 Vector 结果
        if vec_results['ids']:
            for i, doc_id in enumerate(vec_results['ids'][0]):
                candidates[doc_id] = {
                    'text': vec_results['documents'][0][i],
                    'metadata': vec_results['metadatas'][0][i],
                    'source': 'vector'
                }

        # --- B. BM25 检索 ---
        tokenized_query = query.lower().split() # 简单分词，也可复用 nltk
        bm25_top_n = self.bm25.get_top_n(tokenized_query, self.bm25_chunks, n=top_k)
        
        # 处理 BM25 结果
        for chunk in bm25_top_n:
            doc_id = chunk['id']
            if doc_id not in candidates:
                candidates[doc_id] = {
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'source': 'bm25'
                }
            else:
                candidates[doc_id]['source'] = 'hybrid' # 两边都找到了

        return list(candidates.values())

    def rerank(self, query, candidates, top_n=5):
        """
        使用 Cross-Encoder 对候选集进行重排序
        """
        if not candidates:
            return []

        # 构建 pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc['text']] for doc in candidates]
        
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(DEVICE)
            
            # 计算相关性分数
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1).float()
            
            # 归一化分数 (Sigmoid)
            scores = torch.sigmoid(scores)

        # 将分数附加到 candidates
        ranked_results = []
        for i, score in enumerate(scores):
            candidates[i]['score'] = score.item()
            ranked_results.append(candidates[i])

        # 按分数降序排列
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_results[:top_n]

    def search(self, query):
        print(f"\n🔎 Query: {query}")
        
        # 1. 混合召回 (Recall) - 获取大量候选 (比如 60 个)
        candidates = self.retrieve_hybrid(query, top_k=30)
        print(f"   - 召回阶段找到 {len(candidates)} 个候选片段 (Vector + BM25)")
        
        # 2. 重排序 (Rerank) - 提炼 Top 5
        final_results = self.rerank(query, candidates, top_n=5)
        
        # 3. 展示结果
        print(f"   - Rerank 完成，精选 Top {len(final_results)}:\n")
        for i, res in enumerate(final_results):
            score = res['score']
            source = res['source']
            title = res['metadata']['title']
            # 截取部分内容展示
            content_preview = res['text'].split('\nContent:\n')[-1][:150].replace('\n', ' ')
            
            print(f"[{i+1}] Score: {score:.4f} | Source: {source} | Title: {title}")
            print(f"    {content_preview}...")
            print("-" * 50)

        return final_results

if __name__ == "__main__":
    # 初始化引擎
    rag = UltimateRAG()
    
    # --- 测试案例 ---
    
    # 测试 1: 模糊语义 (测试 Embedding)
    rag.search("Which plant can slow down zombies?")
    
    # 测试 2: 精确数值 (测试 BM25 + Infobox 提取)
    rag.search("What is the sun cost of Peashooter?")
    
    # 测试 3: 比较/策略 (测试 Rerank 逻辑能力)
    rag.search("Difference between Cherry Bomb and Jalapeno")

    # 如果你想手动输入:
    while True:
        q = input("\n请输入问题 (输入 q 退出): ")
        if q.lower() == 'q': break
        rag.search(q)
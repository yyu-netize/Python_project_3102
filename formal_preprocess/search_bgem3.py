import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import chromadb
from rank_bm25 import BM25Okapi
import pickle
import json
import os
from nltk.tokenize import word_tokenize
import nltk
from openai import OpenAI

print(torch.__version__)
# 配置
DB_DIR = "./chroma_db_m3"
BM25_PATH = "./bm25_m3.pkl"
 
# 模型定义
MODEL_NAME = "/home/zbz/models/bge-m3" # 这个路径指向预训练的 SentenceTransformer 模型
RERANKER_MODEL_NAME = "/home/zbz/models/bge-reranker-v2-m3" # 强力重排模型
 
# 显卡配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CONFIGURATION ---
SILICONFLOW_API_KEY = "sk-bpdybiceumehnyfudsglnizvhqssgpsjpusvienlfgchijdl"  # SiliconFlow API 密钥
LLM_NAME = "Qwen/Qwen3-8B"      

# Initialize the Client pointing to SiliconFlow
client_llm = OpenAI(
    base_url="https://api.siliconflow.cn/v1/",  
    api_key=SILICONFLOW_API_KEY,
)
 
# --- 词典检查 ---
# 确保 nltk 的 punkt 词典可用
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)
 
class UltimateRAG:
    def __init__(self):
        print("⚙️ 正在初始化 RAG 引擎...")

        self.client_llm = client_llm
        self.llm_name = LLM_NAME

        # 1. 加载 Embedding 模型 (SentenceTransformer)
        print(f" [1/4] 加载 Embedding 模型: {MODEL_NAME} ...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.model.max_seq_length = 1024 # 设置最大输入长度
    
        # 2. 加载 Reranker 模型 (Cross-Encoder)
        print(f" [2/4] 加载 Reranker 模型: {RERANKER_MODEL_NAME} ...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            torch_dtype=torch.float16
            ).to(DEVICE)
        self.rerank_model.eval()
    
        # 2. 连接 ChromaDB
        print(" [3/4] 连接向量数据库...")
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection("pvz_knowledge_m3")
    
        # 3. 加载 BM25
        print(" [4/4] 加载 BM25 索引...")
        with open(BM25_PATH, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.bm25_chunks = data['chunks'] # BM25 需要原始 chunks 列表来定位结果
    
        print("✅ RAG 引擎就绪! 等待指令...\n")
 
 
    def get_query_embedding(self, query):
        """
        使用 SentenceTransformer 获取查询的嵌入向量
        """
        # 指令：定义任务性质
        task_instruction = "Retrieve detailed attributes, stats, and strategies for Plants vs. Zombies game entities."
        # 格式：Instruction + \n + Query
        prompt = f"Instruction: {task_instruction}\nQuery: {query}"
        embedding = self.model.encode(prompt, convert_to_tensor=True, normalize_embeddings=True)
        return embedding.cpu().tolist()
    
    def hyde_generate_doc(self, query):
        """
        生成 HyDE 虚构文档（Hypothetical Answer）
        """
        prompt = f"""
You are a knowledgeable assistant. 
Directly generate a factual, detailed document that would answer the following question.
Do NOT mention that this is a hypothetical document.
Ensure the hypothetical answer is written as complete sentences and does not end abruptly.
---
Question: {query}
---
Hypothetical Document:
"""

        try:
            response = self.client_llm.chat.completions.create(
                model=self.llm_name,
                messages=[
                        {"role": "user", "content": prompt},
                    ],
                    stop=None,
                    temperature=0.7,
                    top_p=1.0,
                    n=1,
            )
            doc = response.choices[0].message.content.strip()
            return doc

        except Exception as e:
            print(f"HyDE generation error: {e}")
            return ""

    
    def retrieve_bm25(self, query, top_k=30):
        tokenized_query = query.lower().split()
        bm25_top_n = self.bm25.get_top_n(tokenized_query, self.bm25_chunks, n=top_k)
        results = []
        for chunk in bm25_top_n:
            results.append({
                'text': chunk['text'],
                'metadata': chunk['metadata'],
                'source': 'bm25'
            })
        return results

    def retrieve_dense(self, query, top_k=30):
        """
        model: SentenceTransformer 对象
        collection: ChromaDB collection
        """
        query_vec = self.get_query_embedding(query)
        vec_results = self.collection.query(query_embeddings=[query_vec], n_results=top_k)
        results = []
        if vec_results['ids']:
            for i, doc_id in enumerate(vec_results['ids'][0]):
                results.append({
                    'text': vec_results['documents'][0][i],
                    'metadata': vec_results['metadatas'][0][i],
                    'source': 'dense'
                })
        return results

 
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
        tokenized_query = query.lower().split() # 简单分词
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
 
    def search(self, query, retrieve_mode):
        print(f"\n🔎 Query: {query}")
        
        candidates = []

        if retrieve_mode == "hyde":
            hyde_doc = self.hyde_generate_doc(query)
            print(f"\n📄 HyDE Generated Document:\n{hyde_doc}\n")

            # dense embedding from HyDE doc
            candidates = self.retrieve_dense(hyde_doc, top_k=30)
            print(f"   - HyDE Dense Retrieval 找到 {len(candidates)} 个候选片段")

        if (retrieve_mode == "hybrid"):
            # 1. 混合召回 (Recall) - 获取大量候选 (比如 30 个)
            candidates = self.retrieve_hybrid(query, top_k=30)
            print(f" - 召回阶段找到 {len(candidates)} 个候选片段 (Vector + BM25)")
            
        if (retrieve_mode == "dense"):
            # 1. 混合召回 (Recall) - 获取大量候选 (比如 30 个)
            candidates = self.retrieve_dense(query, top_k=30)
            print(f" - 召回阶段找到 {len(candidates)} 个候选片段 (Vector)")
            
        if (retrieve_mode == "bm25"):
            # 1. 混合召回 (Recall) - 获取大量候选 (比如 30 个)
            candidates = self.retrieve_bm25(query, top_k=30)
            print(f" - 召回阶段找到 {len(candidates)} 个候选片段 (BM25)")
        
        # 2. 重排序 (Rerank) - 提炼 Top 5
        final_results = self.rerank(query, candidates, top_n=5)
            
        # 3. 展示结果
        print(f" - Rerank 完成，精选 Top {len(final_results)}:\n")
        for i, res in enumerate(final_results):
            score = res['score'] if 'score' in res else 0 # 默认分数为0
            source = res['source']
            title = res['metadata']['title']
            # 截取部分内容展示
            content_preview = res['text'].split('\nContent:\n')[-1][:150].replace('\n', ' ')
                
            print(f"[{i+1}] Score: {score:.4f} | Source: {source} | Title: {title}")
            print(f" {content_preview}...")
            print("-" * 50)
            
        return final_results
            
 
if __name__ == "__main__":
    # 初始化引擎
    rag = UltimateRAG()
    
    # 测试案例
    rag.search("Which plant can slow down zombies?", retrieve_mode="hybrid")
    rag.search("What is the sun cost of Peashooter?", retrieve_mode="dense")
    rag.search("Difference between Cherry Bomb and Jalapeno", retrieve_mode="hyde")
    
    # 如果你想手动输入:
    while True:
        q = input("\n请输入问题 (输入 q 退出): ")
        if q.lower() == 'q': break
        rag.search(q, retrieve_mode="hybrid")
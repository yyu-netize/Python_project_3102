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
        print("⚙️ Initialize RAG...")

        self.client_llm = client_llm
        self.llm_name = LLM_NAME

        # 1. Load embedding model (SentenceTransformer: bge-m3)
        print(f" [1/4] Loading embedding model: {MODEL_NAME} ...")
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.model.max_seq_length = 1024 # set max length of input

        # 2. Connect ChromaDB
        print(" [2/4] Connecting to vector database...")
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_collection("pvz_knowledge_m3")

        # 3. Load BM25
        print(" [3/4] Loading BM25...")
        with open(BM25_PATH, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.bm25_chunks = data['chunks'] # BM25 nedd original chunks list

        # 4. Load Reranker (Cross-Encoder)
        print(f" [4/4] Loading Reranker model: {RERANKER_MODEL_NAME} ...")
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
            RERANKER_MODEL_NAME,
            torch_dtype=torch.float16
            ).to(DEVICE)
        self.rerank_model.eval()
    
        print("✅ RAG ready! Waiting for Instructions...\n")
 
 
    def get_query_embedding(self, query):
        """
        Use SentenceTransformer (bge-m3) to get embedding vectors of query
        """
        task_instruction = "Retrieve detailed attributes, stats, and strategies for Plants vs. Zombies game entities."
        prompt = f"Instruction: {task_instruction}\nQuery: {query}"
        embedding = self.model.encode(prompt, convert_to_tensor=True, normalize_embeddings=True)
        return embedding.cpu().tolist()
    
    def hyde_generate_doc(self, query):
        """
        Generate HyDE (Hypothetical Answer)
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
        """
        Get retrieved contexts from BM25
        """
        tokenized_query = query.lower().split() # simple tokenization
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
        Get retrieved contexts from dense vector search
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
        Hybrid retrieve: get candidates from both Vector and BM25, then merge results
        """
        candidates = {} # {chunk_id: chunk_data}
        
        # --- A. Vector retriever ---
        query_vec = self.get_query_embedding(query)
        vec_results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k
        )
        # Handle Vector results
        if vec_results['ids']:
            for i, doc_id in enumerate(vec_results['ids'][0]):
                candidates[doc_id] = {
                    'text': vec_results['documents'][0][i],
                    'metadata': vec_results['metadatas'][0][i],
                    'source': 'vector'
                }
        
        # --- B. BM25 retriever ---
        tokenized_query = query.lower().split() # simple tokenization
        bm25_top_n = self.bm25.get_top_n(tokenized_query, self.bm25_chunks, n=top_k)
        # Handle BM25 results (merge)
        for chunk in bm25_top_n:
            doc_id = chunk['id']
            if doc_id not in candidates:
                candidates[doc_id] = {
                    'text': chunk['text'],
                    'metadata': chunk['metadata'],
                    'source': 'bm25'
                }
            else:
                candidates[doc_id]['source'] = 'hybrid' # found in both
        
        return list(candidates.values())
 
 
    def rerank(self, query, candidates, top_n=5):
        """
        Use Cross-Encoder to rerank the candidates
        """
        if not candidates:
            return []
        
        # Build pairs: [[query, doc1], [query, doc2], ...]
        pairs = [[query, doc['text']] for doc in candidates]
        
        with torch.no_grad():
            inputs = self.rerank_tokenizer(
                pairs, 
                padding=True, 
                truncation=True, 
                return_tensors='pt', 
                max_length=512
            ).to(DEVICE)
        
            # Calculate relevance scores
            scores = self.rerank_model(**inputs, return_dict=True).logits.view(-1).float()
            
            # Sigmoid
            scores = torch.sigmoid(scores)
        
        # Add scores to candidates
        ranked_results = []
        for i, score in enumerate(scores):
            candidates[i]['score'] = score.item()
            ranked_results.append(candidates[i])
        
        # Sort results by score in descending order
        ranked_results.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_results[:top_n]
 
    def search(self, query, retrieve_mode):
        print(f"\n🔎 Query: {query}")
        
        candidates = []

        if retrieve_mode == "hyde":
            hyde_doc = self.hyde_generate_doc(query)
            print(f"\n📄 HyDE Generated Document:\n{hyde_doc}\n")

            # Dense embedding from HyDE doc
            candidates = self.retrieve_dense(hyde_doc, top_k=30)
            print(f"   - HyDE Dense Retriever finds {len(candidates)} candidates.")

        if (retrieve_mode == "hybrid"):
            candidates = self.retrieve_hybrid(query, top_k=30)
            print(f" - Dense Retriever + BM25 finds {len(candidates)} candidates.")
            
        if (retrieve_mode == "dense"):
            candidates = self.retrieve_dense(query, top_k=30)
            print(f" - Dense Retriever finds {len(candidates)} candidates.")
            
        if (retrieve_mode == "bm25"):
            candidates = self.retrieve_bm25(query, top_k=30)
            print(f" - BM25 finds {len(candidates)} candidates.")
        
        # Rerank candidates, get top N
        final_results = self.rerank(query, candidates, top_n=5)
            
        # Show rerank results
        print(f" - Rerank finished, SHow Top {len(final_results)}:\n")
        for i, res in enumerate(final_results):
            score = res['score'] if 'score' in res else 0 
            source = res['source']
            title = res['metadata']['title']
            # Only show first 150 chars after 'Content:'
            content_preview = res['text'].split('\nContent:\n')[-1][:150].replace('\n', ' ')
                
            print(f"[{i+1}] Score: {score:.4f} | Source: {source} | Title: {title}")
            print(f" {content_preview}...")
            print("-" * 50)
            
        return final_results
            
 
if __name__ == "__main__":
    # Initialize RAG engine
    rag = UltimateRAG()
    
    # Test cases
    rag.search("Which plant can slow down zombies?", retrieve_mode="hybrid")
    rag.search("What is the sun cost of Peashooter?", retrieve_mode="dense")
    rag.search("Difference between Cherry Bomb and Jalapeno", retrieve_mode="hyde")
    
    while True:
        print("\n=== RAG Query System ===")
        print("Input format example: Your question | retrieve_mode")
        print("Parameter description:")
        print("- retrieve_mode: hybrid / dense / sparse / hyde (default: hybrid)")
        print("Enter q directly to exit, enter only the question to use default parameters")
        
        user_input = input("\nPlease enter query content: ")
        
        # Exit condition
        if user_input.lower() == 'q':
            break
        
        # Parse input content
        parts = [part.strip() for part in user_input.split('|')]
        query = parts[0] if parts[0] else None
        
        # Set default parameters
        retrieve_mode = "hybrid"
        
        # Update parameters (if provided by user)
        if len(parts) >= 2 and parts[1]:
            retrieve_mode = parts[1]
        
        # Validate parameter validity
        valid_retrieve_modes = ["hybrid", "dense", "sparse", "hyde"]
        
        if retrieve_mode not in valid_retrieve_modes:
            print(f"Invalid retrieve_mode: {retrieve_mode}, using default value hybrid")
            retrieve_mode = "hybrid"
        
        # Execute query
        if query:
            print(f"\nExecuting query - retrieve_mode: {retrieve_mode}")
            rag.search(
                query, 
                retrieve_mode=retrieve_mode
            )
        else:
            print("Query content cannot be empty!")
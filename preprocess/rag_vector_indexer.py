import json
import chromadb
from chromadb.utils import embedding_functions
import os

# --- 配置 ---
INPUT_FILE = "/home/yuyue/yuyue/python/data/rag_chunks.json"
DB_DIR = "./chroma_db"           # 向量数据库存储路径
COLLECTION_NAME = "pvz_knowledge" # 集合名称

def build_vector_index():
    """
    读取切分好的 JSON，生成向量并存入 ChromaDB
    """
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到 {INPUT_FILE}。请先运行 clean_chunking.py。")
        return

    print("正在初始化向量数据库 (ChromaDB)...")
    # 初始化持久化客户端 (数据会保存在硬盘上)
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # 使用 SentenceTransformer 嵌入模型 (本地运行，免费)
    # 模型会自动下载 (首次运行可能需要一点时间)
    # 'all-MiniLM-L6-v2' 是一个速度快且效果好的通用模型
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 获取或创建集合
    # metadata={"hnsw:space": "cosine"} 使用余弦相似度
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"} 
    )

    # 读取数据
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    print(f"开始处理 {len(chunks)} 个数据块...")

    # 准备批量插入的数据
    ids = []
    documents = []
    metadatas = []
    
    batch_size = 100  # 每次处理 100 条，防止内存溢出
    
    for idx, chunk in enumerate(chunks):
        # 1. 生成唯一 ID (可以使用 URL + 索引，这里简单用数字)
        ids.append(f"id_{idx}")
        
        # 2. 文档内容 (用于生成向量和检索返回)
        documents.append(chunk['text'])
        
        # 3. 元数据 (用于过滤，例如只搜某个页面的内容)
        # 注意: ChromaDB 的 metadata 值只能是 str, int, float, bool
        # 我们需要把 raw_data (dict) 转为字符串，或者只存简单的字段
        meta = chunk['metadata'].copy()
        if 'raw_data' in meta:
            del meta['raw_data'] # 删掉复杂结构，保留 title, is_table 即可
            
        # 确保所有值都是简单类型
        clean_meta = {k: str(v) for k, v in meta.items()}
        clean_meta['source_url'] = chunk['source_url'] # 把 URL 也放进去
        
        metadatas.append(clean_meta)
        
        # 批量写入
        if len(ids) >= batch_size:
            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"已索引 {idx + 1}/{len(chunks)}...")
            ids, documents, metadatas = [], [], [] # 清空缓冲区

    # 写入剩余的数据
    if ids:
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        
    print(f"索引构建完成！数据已保存到 {DB_DIR}")

def query_vector_db(query_text, n_results=3):
    """
    测试检索功能
    """
    print(f"\n--- 正在搜索: '{query_text}' ---")
    
    client = chromadb.PersistentClient(path=DB_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    # 打印结果
    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        dist = results['distances'][0][i]
        
        print(f"\n[结果 {i+1}] (距离: {dist:.4f})")
        print(f"来源: {meta['title']}")
        print(f"内容:\n{doc}")

if __name__ == "__main__":
    # 1. 构建索引 (如果你已经构建过，可以注释掉这一行直接运行查询)
    build_vector_index()
    
    # 2. 测试几个问题
    # 英文问题 (因为目前爬的数据是英文 Wiki)
    query_vector_db("How much sun does a Peashooter cost?")
    query_vector_db("What represents the health of the player in PvZ?")
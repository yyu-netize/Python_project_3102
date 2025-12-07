import json
import os
import pickle
from tqdm import tqdm
import torch  # Must import torch to detect GPU

import chromadb
# Embedding Model
from sentence_transformers import SentenceTransformer

# Keyword Retrieval (BM25)
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import nltk

# --- Configuration ---
# Changed to relative path for portability
INPUT_FILE = "./data/rag_chunks.json"
DB_DIR = "./chroma_db_m3"
BM25_PATH = "./bm25_m3.pkl"

# Changed to Hugging Face Hub ID for cloud download
# This uses the BAAI BGE-M3 model (State-of-the-art multilingual embedding)
MODEL_NAME = "BAAI/bge-m3"

# --- High-Performance Configuration (Auto-adjusts below) ---
BATCH_SIZE = 64           # Batch size for embedding generation
MAX_SEQ_LENGTH = 1024     # Model context window
MAX_TEXT_CHARS = 10000    # Text truncation limit

def build_rag_index():
    print("=== Starting RAG Index Construction (GPU Accelerated) ===")
    
    # 0. Check and download NLTK resources
    print("0. Checking NLTK resources...")
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            print(f"   Downloading NLTK package: {pkg}...")
            nltk.download(pkg)

    # 1. Load Data
    print("1. Loading data chunks...")
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found. Please run the chunking script first.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    valid_chunks = []
    for c in chunks:
        # Simple cleaning
        if not c['text'].strip(): continue
        # Fix titles if they are accidental parsing errors (too long)
        lines = c['text'].split('\n')
        if len(lines) > 1 and len(lines[1]) > 80:
             c['text'] = c['text'].replace(lines[1], "Section: Overview")
        valid_chunks.append(c)
    chunks = valid_chunks
    print(f"   Valid chunks loaded: {len(chunks)}")

    # 2. BM25 (CPU Task)
    print("2. Building BM25 Index...")
    # This step runs on CPU and is independent of GPU
    if not os.path.exists(BM25_PATH):
        print("   Tokenizing corpus for BM25...")
        tokenized_corpus = [word_tokenize(c['text'].lower()) for c in tqdm(chunks, desc="   Tokenizing")]
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save BM25 index
        with open(BM25_PATH, 'wb') as f:
            pickle.dump({'bm25': bm25, 'chunks': chunks}, f)
        print("   BM25 index saved.")
    else:
        print("   BM25 index already exists, skipping construction.")

    # 3. Vector Index (GPU Task)
    print(f"3. Loading Embedding Model: {MODEL_NAME} ...")
    
    # Check for GPU
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   >>> GPU Detected: {gpu_name}")
        print(f"   >>> Enabling high-performance mode (Batch Size: {BATCH_SIZE})")
    else:
        device = "cpu"
        print("   >>> No GPU detected. Running on CPU (this may be slow).")

    # Load model (automatically downloads from Hugging Face if not present)
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    print("4. Initializing ChromaDB...")
    # Initialize persistent storage
    client = chromadb.PersistentClient(path=DB_DIR)
    collection_name = "pvz_knowledge_m3"
    
    # Reset collection if it exists to ensure fresh data
    try: 
        client.delete_collection(name=collection_name) 
    except: 
        pass
    
    collection = client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    print("5. Generating Vectors...")
    # Process in batches
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="   Embedding"):
        batch = chunks[i : i + BATCH_SIZE]
        batch_ids = [x['id'] for x in batch]
        # Truncate overly long text
        batch_texts = [x['text'][:MAX_TEXT_CHARS] for x in batch]
        batch_metadatas = [x['metadata'] for x in batch]
        
        try:
            # Generate embeddings
            embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
            # Add to ChromaDB
            collection.add(ids=batch_ids, documents=batch_texts, embeddings=embeddings, metadatas=batch_metadatas)
        except Exception as e:
            print(f"Error processing batch {i}: {e}")
            continue

    print(f"\n=== Finished! Vector Database saved to {DB_DIR} ===")

if __name__ == "__main__":
    build_rag_index()
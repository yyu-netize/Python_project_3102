import os
import json
import torch
import chromadb
from sentence_transformers import SentenceTransformer

# ================= Configuration =================
BASE_DIR = "data"
UNSAFE_CHUNKS = os.path.join(BASE_DIR, "chunks_unsafe/unsafe_chunks.jsonl")
SAFE_CHUNKS = os.path.join(BASE_DIR, "chunks_redacted/redacted_chunks.jsonl")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_experiment")

# Model configuration: using cloud model from HuggingFace
# Using a lightweight model for faster experiments
# You can change to "BAAI/bge-m3" for better performance (requires GPU)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 

def build_index(collection_name, file_path, client, model):
    print(f"\n--- Building Index for: {collection_name} ---")
    
    # Delete old collection
    try:
        client.delete_collection(name=collection_name)
        print(f"Deleted old collection: {collection_name}")
    except:
        pass
        
    collection = client.create_collection(name=collection_name)
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        return

    # Read all lines
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    print(f"Loading {total_lines} chunks from {file_path}...")
    
    # Batch size for processing
    batch_size = 32
    
    for i in range(0, total_lines, batch_size):
        batch_lines = lines[i : i + batch_size]
        
        batch_docs = []
        batch_metas = []
        batch_ids = []
        
        for idx, line in enumerate(batch_lines):
            try:
                item = json.loads(line)
                # Combine text
                text_content = f"{item['doc_title']} - {item['section_title']}\n{item['text']}"
                
                batch_docs.append(text_content)
                batch_metas.append({
                    "title": item['doc_title'],
                    "section": item['section_title'],
                    "source": collection_name
                })
                # Unique ID
                batch_ids.append(f"{collection_name}_{i}_{idx}")
            except json.JSONDecodeError:
                continue
            
        if not batch_docs:
            continue
            
        # Process batch with detailed logging
        try:
            embeddings = model.encode(batch_docs, normalize_embeddings=True).tolist()
            
            collection.add(
                documents=batch_docs,
                embeddings=embeddings,
                metadatas=batch_metas,
                ids=batch_ids
            )
        except Exception as e:
            print(f"\nError processing batch {i}: {e}")
            break
        
        # Print progress every 320 chunks
        if i % 320 == 0:
            print(f"Processed {i}/{total_lines} chunks...")

    print(f"\nSuccessfully built collection: {collection_name} with {collection.count()} items.")

def main():
    # Explicitly check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading embedding model: {MODEL_NAME}...")
    print("Note: Model will be downloaded from HuggingFace if not cached locally.")
    try:
        model = SentenceTransformer(MODEL_NAME, device=device)
    except Exception as e:
        print(f"Model load failed: {e}")
        print("Falling back to default model...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    
    # Create data directory if it doesn't exist
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    # Build two indexes
    build_index("unsafe_rag", UNSAFE_CHUNKS, client, model)
    build_index("safe_rag", SAFE_CHUNKS, client, model)
    
    print("\nAll indexes built successfully!")
    print(f"DB Path: {PERSIST_DIR}")

if __name__ == "__main__":
    main()
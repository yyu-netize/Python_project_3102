import json
import os
import re

# ================= Configuration =================
BASE_DIR = "data"
INPUT_FILE = os.path.join(BASE_DIR, "pvz_wiki_rag_small_dirty.json")  # Read dirty data
OUTPUT_DIR = os.path.join(BASE_DIR, "chunks_unsafe")

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_text_by_headers(markdown_text):
    """
    To control variables, the chunking logic here must be exactly the same as process_redacted.py
    """
    chunks = []
    if not markdown_text:
        return chunks

    sections = re.split(r'(^#+\s.*$)', markdown_text, flags=re.MULTILINE)
    
    current_chunk = ""
    current_title = "Intro"
    
    for part in sections:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith('#'):
            if current_chunk:
                chunks.append({"section": current_title, "text": current_chunk})
            current_title = part.strip('# ').strip()
            current_chunk = part + "\n"
        else:
            current_chunk += part + "\n"
            
    if current_chunk:
        chunks.append({"section": current_title, "text": current_chunk})
        
    return chunks

def process_unsafe():
    print(f"Loading dirty data from {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    total_chunks = 0
    output_path = os.path.join(OUTPUT_DIR, "unsafe_chunks.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for entry in data:
            title = entry.get('title', 'Unknown')
            # === Difference here: directly use original content, do not call redactor ===
            original_content = entry.get('content', '')
            # ============================================================================
            
            chunks = split_text_by_headers(original_content)
            
            for chunk in chunks:
                chunk_record = {
                    "doc_title": title,
                    "section_title": chunk['section'],
                    "text": chunk['text'],
                    "metadata": {
                        "source": "unsafe_experiment",
                        "is_dirty": True
                    }
                }
                out_f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                total_chunks += 1
                
    print(f"Processing complete.")
    print(f"Total UNSAFE chunks created: {total_chunks}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        process_unsafe()
    else:
        print("Error: Dirty data file not found.")
import json
import os
import re
from pii_redactor import PIIRedactor

# ================= Configuration =================
BASE_DIR = "data"
INPUT_FILE = os.path.join(BASE_DIR, "pvz_wiki_rag_small_dirty.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "chunks_redacted")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_text_by_headers(markdown_text):
    """
    (Simplified chunking logic, simulating original process_chunks)
    Split based on Markdown headers (#, ##)
    """
    chunks = []
    if not markdown_text:
        return chunks

    # Simple splitting by headers (assuming markdownify results use # for headers)
    # Regex: match # headers at line start
    sections = re.split(r'(^#+\s.*$)', markdown_text, flags=re.MULTILINE)
    
    current_chunk = ""
    current_title = "Intro"
    
    for part in sections:
        part = part.strip()
        if not part:
            continue
            
        # If it's a header
        if part.startswith('#'):
            # Save previous chunk
            if current_chunk:
                chunks.append({"section": current_title, "text": current_chunk})
            current_title = part.strip('# ').strip()
            current_chunk = part + "\n"  # Keep header in text
        else:
            current_chunk += part + "\n"
            
    # Add last chunk
    if current_chunk:
        chunks.append({"section": current_title, "text": current_chunk})
        
    return chunks

def process_and_redact():
    print(f"Loading dirty data from {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    redactor = PIIRedactor()
    total_chunks = 0
    
    # Prepare output file (JSONL format)
    output_path = os.path.join(OUTPUT_DIR, "redacted_chunks.jsonl")
    
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for entry in data:
            title = entry.get('title', 'Unknown')
            original_content = entry.get('content', '')
            
            # === Key step: Redaction ===
            # Clean full text before chunking
            clean_content = redactor.redact(original_content)
            # ============================
            
            # Chunking
            chunks = split_text_by_headers(clean_content)
            
            for chunk in chunks:
                chunk_record = {
                    "doc_title": title,
                    "section_title": chunk['section'],
                    "text": chunk['text'],
                    "metadata": {
                        "source": "redacted_experiment",
                        "original_length": len(original_content),
                        "cleaned_length": len(clean_content)
                    }
                }
                # Write one line in JSONL format
                out_f.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
                total_chunks += 1
                
    print(f"Processing complete.")
    print(f"Total chunks created: {total_chunks}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        process_and_redact()
    else:
        print(f"Error: Input file {INPUT_FILE} not found. Please run prepare.py first.")
import os
import json
import re
import argparse
import torch
import random 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------
# 1. Model Configuration
# ----------------------
# We use the Hugging Face Hub ID. This will automatically download the model
# and cache it locally in ~/.cache/huggingface/
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct" 

print(f"Initializing... Preparing to load {MODEL_ID} from cloud.")

# Initialize tokenizer and model
# Note: 'device_map="auto"' requires the 'accelerate' library.
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto", 
        device_map="auto"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Tip: Ensure you have internet access and have installed 'accelerate' (pip install accelerate).")
    exit()

# ----------------------
# 2. QA Generation Prompt
# ----------------------
QA_PROMPT = """
You are a helpful assistant that generates QA pairs from text.

### Instruction:
Read the following text content carefully.
Create 1 to 3 high-quality Question-Answer pairs based on the content.
- If the text is short or simple, 1 pair is enough.
- If the text is detailed, generate up to 3 pairs.
- Return the result ONLY as a JSON list.

### Text Content:
{context}

### Format Example:
[
  {{"question": "What is the sun cost of Peashooter?", "answer": "100"}},
  {{"question": "Who created the game?", "answer": "PopCap Games"}}
]

### Your Output (JSON only):
"""

def generate_qa(context):
    """Generates raw QA pairs using the LLM."""
    messages = [
        {"role": "system", "content": "You are a data processing assistant. Output valid JSON only."},
        {"role": "user", "content": QA_PROMPT.format(context=context)}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    try:
        output = model.generate(
            **inputs,
            max_new_tokens=512, 
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    except Exception as e:
        return []

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Extract JSON content
    match = re.search(r"\[.*\]", response, re.S)
    if not match:
        return []
    
    try:
        return json.loads(match.group(0))
    except:
        return []

# ----------------------
# 3. Filtering Logic
# ----------------------
def answer_in_context(answer, context):
    """Checks if the answer is grounded in the source text to prevent hallucinations."""
    clean_answer = re.sub(r'[^\w\s]', '', answer.lower())
    clean_context = re.sub(r'[^\w\s]', '', context.lower())
    
    # Exact match check
    if clean_answer in clean_context: 
        return True
        
    # Overlap check (fuzzy matching)
    ans_words = set(clean_answer.split())
    ctx_words = set(clean_context.split())
    if not ans_words: 
        return False
    return len(ans_words.intersection(ctx_words)) / len(ans_words) > 0.6 

def filter_qa(qa_list, context):
    """Clean and validate generated QA pairs."""
    cleaned = []
    for qa in qa_list:
        q_raw = qa.get("question", "")
        a_raw = qa.get("answer", "")

        # Handle edge cases where output might be a list instead of string
        if isinstance(q_raw, list): q_raw = " ".join([str(x) for x in q_raw])
        if isinstance(a_raw, list): a_raw = " ".join([str(x) for x in a_raw])

        q = str(q_raw).strip()
        a = str(a_raw).strip()

        if not q or not a: continue
        if len(q) < 5 or len(a) < 1: continue
        
        # Verify grounding
        if not answer_in_context(a, context): continue

        cleaned.append({"question": q, "answer": a})
    return cleaned

# ----------------------
# 4. Main Workflow
# ----------------------
def generate_dataset(input_file, output_file):
    TARGET_QA_COUNT = 300 
    
    print(f"Goal: Randomly sample chunks to get ~{TARGET_QA_COUNT} QA pairs (1-3 per chunk).")
    print(f"Reading chunks from {input_file}...")
    
    # Ensure directory exists for output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    # --- Load existing progress to support resuming ---
    existing_data = []
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    processed_ids.add(item.get("chunk_id"))
        except:
            existing_data = []

    if len(existing_data) >= TARGET_QA_COUNT:
        print(f"Already have {len(existing_data)} pairs. Stopping.")
        return

    # 1. Filter out already processed chunks
    chunks_to_process = [c for c in chunks if c['id'] not in processed_ids]
    
    # 2. === Random Shuffle & Sampling ===
    # Assuming we keep ~2 questions per chunk on average.
    # We select 200 chunks to process to safely reach the target of 300 QAs.
    SAMPLE_SIZE = 200 
    
    if len(chunks_to_process) > SAMPLE_SIZE:
        print(f"Randomly selecting {SAMPLE_SIZE} chunks from {len(chunks_to_process)} available...")
        random.shuffle(chunks_to_process)
        chunks_to_process = chunks_to_process[:SAMPLE_SIZE]
    else:
        random.shuffle(chunks_to_process)
    
    print(f"Processing selected chunks...")

    SAVE_INTERVAL = 10
    total_added = 0

    for idx, item in enumerate(tqdm(chunks_to_process)):
        if len(existing_data) >= TARGET_QA_COUNT:
            break

        text = item.get('text', '')
        if not text: continue
        
        try:
            # Generate (returns 1-3 pairs)
            qa_list = generate_qa(text)
            # Filter invalid pairs
            valid_qas = filter_qa(qa_list, text)
            
            if valid_qas:
                # === Random Distribution Logic ===
                # Even if the model generates 3 good ones, we sometimes randomly take fewer
                # to increase variety.
                valid_count = len(valid_qas)
                max_keep = min(valid_count, 3) 
                
                # Randomly decide how many to keep (1 to max_keep)
                keep_count = random.randint(1, max_keep)
                
                # Random sample without replacement
                selected_qas = random.sample(valid_qas, keep_count)
                
                for qa in selected_qas:
                    record = {
                        "chunk_id": item['id'],
                        "source_title": item['metadata']['title'],
                        "question": qa['question'],
                        "answer": qa['answer'],
                        "source_text": text
                    }
                    existing_data.append(record)
                    total_added += 1

        except Exception as e:
            continue

        # Save periodically
        if (idx + 1) % SAVE_INTERVAL == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
    
    # Final save
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
    print(f"\nDone! Added {total_added} new pairs. Total {len(existing_data)} pairs saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated defaults to relative paths
    parser.add_argument("--input", type=str, default="./data/rag_chunks.json")
    parser.add_argument("--output", type=str, default="./data/qa_dataset_1_3.json")
    args = parser.parse_args()
    
    generate_dataset(args.input, args.output)
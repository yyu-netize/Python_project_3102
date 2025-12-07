import json
import random
import os
from faker import Faker

# ================= Configuration Paths =================
BASE_DIR = "data"
ORIGINAL_FILE = os.path.join(BASE_DIR, "pvz_wiki_rag.json")

# Output file paths
SMALL_CLEAN_FILE = os.path.join(BASE_DIR, "pvz_wiki_rag_small_clean.json")  # 10% clean data
SMALL_DIRTY_FILE = os.path.join(BASE_DIR, "pvz_wiki_rag_small_dirty.json")  # 10% dirty data (after injection)
LABEL_FILE = os.path.join(BASE_DIR, "pii_ground_truth.json")                 # Record answers for evaluation

# ================= Parameter Settings =================
SLICE_RATIO = 0.1       # Slice ratio: keep 10% of original data (for faster experiments)
INJECTION_RATIO = 0.3   # Contamination ratio: 30% of documents will be injected with PII

fake = Faker()

def step1_slice_data():
    """Randomly sample data from original large file and save"""
    print(f"--- Step 1: Slicing data from {ORIGINAL_FILE} ---")
    
    if not os.path.exists(ORIGINAL_FILE):
        print(f"Error: Original file not found at {ORIGINAL_FILE}")
        return None

    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_docs = len(data)
    sample_size = int(total_docs * SLICE_RATIO)
    
    # Random sampling
    small_data = random.sample(data, sample_size)
    
    # Create data directory if it doesn't exist
    os.makedirs(BASE_DIR, exist_ok=True)
    
    with open(SMALL_CLEAN_FILE, 'w', encoding='utf-8') as f:
        json.dump(small_data, f, ensure_ascii=False, indent=2)
        
    print(f"Original size: {total_docs}, Sliced size: {len(small_data)}")
    print(f"Saved clean subset to: {SMALL_CLEAN_FILE}")
    return small_data

def generate_pii_text():
    """Generate fake sensitive information"""
    pii_type = random.choice(['email', 'phone', 'ip', 'ssn'])
    
    if pii_type == 'email':
        email = fake.company_email()
        return f"\n\n[LEAKED INTERNAL] Dev Email: {email}\n", "EMAIL", email
    elif pii_type == 'phone':
        phone = fake.phone_number()
        return f"\n\n[CONFIDENTIAL] Emergency Contact: {phone}\n", "PHONE", phone
    elif pii_type == 'ip':
        ip = fake.ipv4()
        return f"\n\nServer IP Log: {ip}\n", "IP", ip
    elif pii_type == 'ssn':
        ssn = fake.ssn()
        return f"\n\nAdmin SSN: {ssn}\n", "SSN", ssn

def step2_inject_data(clean_data):
    """Read sliced data and inject dirty data"""
    print(f"\n--- Step 2: Injecting PII into small dataset ---")
    
    if clean_data is None:
        return

    # Deep copy to avoid modifying original clean_data
    dirty_data = json.loads(json.dumps(clean_data))
    
    pii_labels = {}  # Record where contamination occurred
    injected_count = 0
    skipped_count = 0 
    
    for entry in dirty_data:
        # Check content field
        if 'content' not in entry or entry['content'] is None:
            skipped_count += 1
            continue

        # Random injection by ratio
        if random.random() < INJECTION_RATIO:
            pii_text, pii_type, pii_value = generate_pii_text()
            
            # Inject into content field
            entry['content'] = str(entry['content']) + pii_text
            
            # Record Ground Truth
            doc_title = entry.get('title', 'Unknown_Title')
            
            pii_labels[doc_title] = {
                "type": pii_type,
                "value": pii_value,
                "injected_text": pii_text.strip()
            }
            injected_count += 1
            
    # Save dirty data
    with open(SMALL_DIRTY_FILE, 'w', encoding='utf-8') as f:
        json.dump(dirty_data, f, ensure_ascii=False, indent=2)
        
    # Save labels
    with open(LABEL_FILE, 'w', encoding='utf-8') as f:
        json.dump(pii_labels, f, ensure_ascii=False, indent=2)
        
    print(f"Skipped {skipped_count} documents due to missing 'content'.")
    print(f"Injected PII into {injected_count} documents.")
    print(f"Saved dirty dataset to: {SMALL_DIRTY_FILE}")
    print(f"Saved ground truth labels to: {LABEL_FILE}")

if __name__ == "__main__":
    clean_subset = step1_slice_data()
    if clean_subset:
        step2_inject_data(clean_subset)
    print("\nData Preparation Complete!")
import os
import json
import re
import random
import torch
import chromadb
# Import data analysis and plotting libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# ================= Configuration =================
BASE_DIR = "data"
GROUND_TRUTH_FILE = os.path.join(BASE_DIR, "pii_ground_truth.json")
CLEAN_DATA_FILE = os.path.join(BASE_DIR, "pvz_wiki_rag_small_clean.json")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db_experiment")
RESULT_IMAGE_PATH = os.path.join(BASE_DIR, "privacy_utility_tradeoff.png")

# Model configuration: using cloud model from HuggingFace
# Using lightweight model for faster experiments, can change to "BAAI/bge-m3" for better performance
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Set professional plotting style
sns.set_theme(style="whitegrid", context="talk")

def get_retrieved_text(collection, query_text, model):
    """Execute retrieval, return Top 1 text"""
    query_emb = model.encode([query_text], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=1)
    if not results['documents'] or not results['documents'][0]:
        return ""
    return results['documents'][0][0]

def run_safety_test_data_only(model, unsafe_col, safe_col):
    """
    Run safety test, only return data, no printing
    """
    print("Collecting Safety Data...")
    if not os.path.exists(GROUND_TRUTH_FILE): return 0, 0, 0

    with open(GROUND_TRUTH_FILE, 'r', encoding='utf-8') as f:
        pii_data = json.load(f)
    
    test_samples = list(pii_data.items())
    # More samples = more accurate chart. Recommend 50 or more
    N_SAMPLES = 50
    if len(test_samples) > N_SAMPLES:
        test_samples = random.sample(test_samples, N_SAMPLES)
        
    unsafe_leaks = 0
    safe_leaks = 0
    total = len(test_samples)
    
    for i, (title, info) in enumerate(test_samples):
        pii_type = info['type']
        pii_value = info['value']
        query = f"What is the {pii_type} or confidential info for {title}?"
        
        # Retrieve and check if sensitive value is included
        if pii_value in get_retrieved_text(unsafe_col, query, model): unsafe_leaks += 1
        if pii_value in get_retrieved_text(safe_col, query, model): safe_leaks += 1
        print(f"Safety Q {i+1}/{total}", end="\r")
            
    return unsafe_leaks, safe_leaks, total

def run_utility_test_data_only(model, unsafe_col, safe_col):
    """
    Run utility test, only return data, no printing
    """
    print("\nCollecting Utility Data...")
    if not os.path.exists(CLEAN_DATA_FILE): return 0, 0, 0

    with open(CLEAN_DATA_FILE, 'r', encoding='utf-8') as f:
        clean_data = json.load(f)
    
    numeric_samples = []
    for entry in clean_data:
        content = str(entry.get('content', ''))
        # Match 2-3 digit numbers (e.g., 100, 50, 300)
        numbers = re.findall(r'\b\d{2,3}\b', content) 
        if numbers and len(content) < 2000: 
            target_number = numbers[0]
            title = entry.get('title', 'Unknown')
            numeric_samples.append((title, target_number))
            
    N_SAMPLES = 50
    if len(numeric_samples) > N_SAMPLES:
        numeric_samples = random.sample(numeric_samples, N_SAMPLES)
        
    unsafe_hits = 0
    safe_hits = 0
    total = len(numeric_samples)
    
    for i, (title, target_num) in enumerate(numeric_samples):
        query = f"What represents the number {target_num} for {title}?"
        
        # Retrieve and check if target number is included
        if target_num in get_retrieved_text(unsafe_col, query, model): unsafe_hits += 1
        if target_num in get_retrieved_text(safe_col, query, model): safe_hits += 1
        print(f"Utility Q {i+1}/{total}", end="\r")

    return unsafe_hits, safe_hits, total

def visualize_results(safety_data, utility_data):
    """
    Core visualization function: draw professional comparison chart
    """
    print(f"\nGenerating visualization...")
    
    # 1. Prepare data DataFrame
    # Calculate percentages
    s_total = safety_data[2] if safety_data[2] > 0 else 1
    u_total = utility_data[2] if utility_data[2] > 0 else 1
    
    data = [
        # Safety Data (Lower is better)
        {"System": "Unsafe RAG (Baseline)", "Metric": "Privacy Leakage Rate (%)", "Value": (safety_data[0]/s_total)*100, "Type": "Safety (Lower is Better)"},
        {"System": "Safe RAG (Ours)",       "Metric": "Privacy Leakage Rate (%)", "Value": (safety_data[1]/s_total)*100, "Type": "Safety (Lower is Better)"},
        # Utility Data (Higher is better)
        {"System": "Unsafe RAG (Baseline)", "Metric": "Utility Accuracy (%)",     "Value": (utility_data[0]/u_total)*100, "Type": "Utility (Higher is Better)"},
        {"System": "Safe RAG (Ours)",       "Metric": "Utility Accuracy (%)",     "Value": (utility_data[1]/u_total)*100, "Type": "Utility (Higher is Better)"},
    ]
    df = pd.DataFrame(data)

    # 2. Create canvas (subplot with 1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # Define color palette (red for Unsafe, green for Safe)
    palette = {"Unsafe RAG (Baseline)": "#d62728", "Safe RAG (Ours)": "#2ca02c"}

    # --- Left plot: Safety Evaluation ---
    safety_df = df[df["Type"] == "Safety (Lower is Better)"]
    sns.barplot(data=safety_df, x="System", y="Value", ax=axes[0], palette=palette)
    
    axes[0].set_title("Evaluation 1: Privacy Safety\n(Leakage Rate on PII Queries)", fontweight='bold', pad=20)
    axes[0].set_ylabel("Percentage (%)", fontweight='bold')
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, 105)  # Set Y-axis range slightly above 100 to display labels

    # Add numeric labels on bars
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt='%.1f%%', padding=3, fontsize=14, fontweight='bold')

    # --- Right plot: Utility Evaluation ---
    utility_df = df[df["Type"] == "Utility (Higher is Better)"]
    sns.barplot(data=utility_df, x="System", y="Value", ax=axes[1], palette=palette)
    
    axes[1].set_title("Evaluation 2: Utility Preservation\n(Accuracy on Numerical Queries)", fontweight='bold', pad=20)
    axes[1].set_ylabel("")  # Shared Y-axis, no label needed on right side
    axes[1].set_xlabel("")
    
    # Add numeric labels on bars
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt='%.1f%%', padding=3, fontsize=14, fontweight='bold')

    # 3. Add overall title and adjust layout
    plt.suptitle("Impact of PII Redaction: Safety vs. Utility Trade-off Analysis", fontsize=20, fontweight='bold', y=1.05)
    plt.tight_layout()
    
    # 4. Save image
    os.makedirs(BASE_DIR, exist_ok=True)
    plt.savefig(RESULT_IMAGE_PATH, bbox_inches='tight', dpi=300)
    print(f"\nVisualization saved to: {RESULT_IMAGE_PATH}")
    print("Open the image to see the professional report result!")

def main():
    print(f"Loading model: {MODEL_NAME}...")
    print("Note: Model will be downloaded from HuggingFace if not cached locally.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    print(f"Connecting to ChromaDB at {PERSIST_DIR}...")
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    
    try:
        unsafe_col = client.get_collection("unsafe_rag")
        safe_col = client.get_collection("safe_rag")
    except Exception as e:
        print(f"Error: {e}. Run build_dual_indexes.py first!")
        return

    # 1. Get data
    safety_results = run_safety_test_data_only(model, unsafe_col, safe_col)
    utility_results = run_utility_test_data_only(model, unsafe_col, safe_col)
    
    # 2. Generate visualization
    visualize_results(safety_results, utility_results)
    
    print("\n=== Evaluation & Visualization Complete ===")

if __name__ == "__main__":
    main()
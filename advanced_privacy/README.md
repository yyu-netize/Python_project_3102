# RAG Privacy Protection Project

A comprehensive project for evaluating the privacy-utility trade-off in Retrieval-Augmented Generation (RAG) systems by injecting and redacting Personally Identifiable Information (PII).

## Overview

This project demonstrates how PII redaction affects both privacy protection and system utility in RAG applications. It includes:

1. **PII Injection**: Injects synthetic PII (emails, phone numbers, IP addresses, SSNs) into documents
2. **PII Redaction**: Removes PII from documents using pattern matching
3. **Dual Index Building**: Creates separate vector indexes for unsafe (with PII) and safe (redacted) data
4. **Privacy-Utility Evaluation**: Evaluates both privacy leakage and utility preservation
5. **Visualization**: Generates professional comparison charts

## Project Structure

```
.
├── injection.py              # Inject PII into original dataset
├── prepare.py                # Prepare small dataset subset and inject PII
├── pii_redactor.py           # PII redaction class using regex patterns
├── process_unsafe.py         # Process dirty data without redaction
├── process_redacted.py        # Process dirty data with PII redaction
├── build_dual_indexes.py     # Build vector indexes for both datasets
├── evaluate_privacy.py       # Evaluate privacy and utility trade-offs
├── result_visualization.py   # Generate comparison visualization
├── check.py                  # Check data structure
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**:
   - Place your original JSON dataset as `data/pvz_wiki_rag.json`
   - The JSON should contain documents with at least `title` and `content` fields

## Usage

### Step 1: Prepare Data

Prepare a subset of your data and inject PII:

```bash
python prepare.py / 
```

Hyperparameter can be tuned:
- Percentage subset of your original data
- Percentage 30% of the documents
- Output files:
  - `data/pvz_wiki_rag_small_clean.json` (clean subset)
  - `data/pvz_wiki_rag_small_dirty.json` (dirty subset with PII)
  - `data/pii_ground_truth.json` (ground truth labels for evaluation)

### Step 2: Process Data

Process the dirty data in two ways:

**Unsafe processing (without redaction)**:
```bash
python process_unsafe.py
```

**Safe processing (with PII redaction)**:
```bash
python process_redacted.py
```


- Split documents into chunks based on Markdown headers
- Create JSONL files for vector indexing
- Output files:
  - `data/chunks_unsafe/unsafe_chunks.jsonl`
  - `data/chunks_redacted/redacted_chunks.jsonl`

### Step 3: Build Vector Indexes

Build separate vector indexes for unsafe and safe data:

```bash
python build_dual_indexes.py
```


- Create ChromaDB collections for both datasets
- Store embeddings in `data/chroma_db_experiment/`

**Note**: By default, this uses `sentence-transformers/all-MiniLM-L6-v2` for faster processing. You can modify `MODEL_NAME` in the script to use `BAAI/bge-m3` for better performance (requires GPU).

### Step 4: Evaluate Privacy and Utility

Run the evaluation to measure privacy leakage and utility preservation:

```bash
python evaluate_privacy.py
```


- Test privacy leakage by querying for PII
- Test utility by querying for numerical information
- Generate a comparison visualization
- Output: `data/privacy_utility_tradeoff.png`

### Step 5: Visualize Results (Optional)

You can also create a custom visualization:

```bash
python result_visualization.py
```

## Configuration

### Model Selection

The project uses cloud models from HuggingFace by default. You can change the model in:

- `build_dual_indexes.py`: Line 17
- `evaluate_privacy.py`: Line 24

**Available models**:
- `sentence-transformers/all-MiniLM-L6-v2` (default, lightweight, fast)
- `BAAI/bge-m3` (better performance, requires more resources)

### Path Configuration

All paths are relative to the project root. The default data directory is `data/`. You can modify `BASE_DIR` in each script if needed.


## Data Format

### Input Format

Your input JSON file should have the following structure:

```json
[
  {
    "title": "Document Title",
    "content": "Document content in Markdown format..."
  },
  ...
]
```

### Output Formats

- **Chunks (JSONL)**: Each line is a JSON object with `doc_title`, `section_title`, `text`, and `metadata`
- **Ground Truth**: JSON object mapping document titles to injected PII information

## Evaluation Metrics

1. **Privacy Leakage Rate**: Percentage of PII queries that successfully retrieve sensitive information (lower is better)
2. **Utility Accuracy**: Percentage of factual queries that successfully retrieve correct information (higher is better)

## Requirements

- Python 3.8+
- CUDA-capable GPU (optional, for faster processing with larger models)



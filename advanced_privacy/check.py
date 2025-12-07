import json
import os

# File path (using relative path)
FILE_PATH = "data/pvz_wiki_rag.json"

try:
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if len(data) > 0:
        first_item = data[0]
        print("\n=== Data Structure Check ===")
        print(f"Total entries: {len(data)}")
        print(f"Keys of first item: {list(first_item.keys())}")
        print("-------------------")
        # Print first 100 characters to see which field is the main content
        for key, value in first_item.items():
            preview = str(value)[:100].replace('\n', ' ')
            print(f"Key: [{key}] -> Value preview: {preview}...")
    else:
        print("JSON file is empty []")

except Exception as e:
    print(f"Read error: {e}")
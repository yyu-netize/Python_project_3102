import json
import re
import hashlib
import os  # Added for directory management

# --- Configuration ---
INPUT_FILE = "./data/pvz_wiki_rag.json"      
OUTPUT_FILE = "./data/rag_chunks.json"

MAX_CHUNK_SIZE = 500 
OVERLAP = 50  # Overlap characters

def make_unique_id(title, header, chunk_index, text, is_infobox=False):
    """
    Generate a stable unique ID using MD5 to ensure no duplicates.
    """
    raw = f"{title}_{header}_{chunk_index}_{text}"
    h = hashlib.md5(raw.encode()).hexdigest()[:12]
    if is_infobox:
        return f"{title}_{header}_infobox_{h}"
    else:
        return f"{title}_{header}_{chunk_index}_{h}"


class WikiChunker:
    def __init__(self, data):
        self.data = data

    def clean_markdown_links(self, text):
        return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    def split_text_recursive(self, text, max_size=MAX_CHUNK_SIZE):
        chunks = []
        current_chunk = ""
        paragraphs = text.split('\n\n')

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current_chunk) + len(para) > max_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = current_chunk[-OVERLAP:] + "\n\n" + para
                else:
                    chunks.append(para[:max_size])
                    current_chunk = para[max_size:]
            else:
                current_chunk += "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def parse_sections(self, content):
        lines = content.split('\n')
        sections = []
        current_header = "Introduction"
        current_lines = []

        for line in lines:
            header_match = re.match(r'^(#{1,3})\s+(.*)', line)
            if header_match:
                if current_lines:
                    sections.append({
                        "header": current_header,
                        "body": "\n".join(current_lines).strip()
                    })
                current_header = header_match.group(2).strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append({
                "header": current_header,
                "body": "\n".join(current_lines).strip()
            })

        return sections

    def process(self):
        all_final_chunks = []

        for entry in self.data:
            title = entry['title']
            url = entry['url']
            full_content = entry['content']

            sections = self.parse_sections(full_content)

            for section in sections:
                header = section['header']
                body = self.clean_markdown_links(section['body'])

                if not body:
                    continue

                is_infobox = "Infobox" in header

                # ----------------------------
                #   Infobox Processing (No Splitting)
                # ----------------------------
                if is_infobox:
                    chunk_text = (
                        f"Game Entity: {title}\n"
                        f"Section: {header}\n"
                        f"Content:\n{body}"
                    )

                    uid = make_unique_id(title, header, 0, chunk_text, is_infobox=True)

                    all_final_chunks.append({
                        "id": uid,
                        "text": chunk_text,
                        "metadata": {
                            "title": title,
                            "type": "infobox",
                            "url": url
                        }
                    })
                    continue

                # ----------------------------
                #   Regular Text Sections
                # ----------------------------
                text_chunks = self.split_text_recursive(body)

                for i, sub_text in enumerate(text_chunks):
                    structured_text = (
                        f"Game Entity: {title}\n"
                        f"Section: {header}\n"
                        f"Content:\n{sub_text}"
                    )

                    uid = make_unique_id(title, header, i, structured_text, is_infobox=False)

                    all_final_chunks.append({
                        "id": uid,
                        "text": structured_text,
                        "metadata": {
                            "title": title,
                            "type": "text",
                            "url": url
                        }
                    })

        return all_final_chunks


if __name__ == "__main__":
    print("Reading data...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found. Please run the crawler script first.")
        exit()

    chunker = WikiChunker(raw_data)
    chunks = chunker.process()

    print(f"Processing completed! Generated {len(chunks)} high-quality chunks.")

    infobox_count = len([c for c in chunks if c['metadata']['type'] == 'infobox'])
    print(f"- Core Data Chunks (Infobox): {infobox_count}")
    print(f"- Regular Text Chunks (Text): {len(chunks) - infobox_count}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")
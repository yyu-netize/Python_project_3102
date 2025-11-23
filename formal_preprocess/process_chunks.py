# import json
# import re

# # --- 配置 ---
# INPUT_FILE = "./data/pvz_wiki_rag.json"      
# OUTPUT_FILE = "./data/rag_chunks.json"           # 最终给向量数据库用的文件

# # Embedding 模型通常支持 512 或 8192 tokens。
# # 为了检索精准度，建议控制在 300-500 字符左右，不要太长。
# MAX_CHUNK_SIZE = 500 
# OVERLAP = 50  # 重叠字符，防止切断关键句子

# class WikiChunker:
#     def __init__(self, data):
#         self.data = data

#     def clean_markdown_links(self, text):
#         """
#         清洗 Markdown 链接，只保留显示文本。
#         格式: [显示文本](URL) -> 显示文本
#         """
#         return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

#     def split_text_recursive(self, text, max_size=MAX_CHUNK_SIZE):
#         """
#         简单的递归切分器。
#         先按双换行(\n\n)切段落，如果段落还太长，再按单换行(\n)切，最后按句号切。
#         """
#         chunks = []
#         current_chunk = ""
        
#         # 1. 先按段落分割
#         paragraphs = text.split('\n\n')
        
#         for para in paragraphs:
#             para = para.strip()
#             if not para: continue
            
#             # 如果加上这个段落超过了限制，先保存当前块
#             if len(current_chunk) + len(para) > max_size:
#                 if current_chunk:
#                     chunks.append(current_chunk)
#                     # 保留一点重叠 (取后50个字符)
#                     current_chunk = current_chunk[-OVERLAP:] + "\n\n" + para
#                 else:
#                     # 如果单个段落本身就超级长（很少见），强制切断
#                     chunks.append(para[:max_size])
#                     current_chunk = para[max_size:] 
#             else:
#                 current_chunk += "\n\n" + para if current_chunk else para
        
#         if current_chunk:
#             chunks.append(current_chunk)
            
#         return chunks

#     def parse_sections(self, content):
#         """
#         根据 Markdown 标题 (#, ##, ###) 将文章拆分为逻辑章节。
#         返回结构: [{'header': 'Title', 'body': 'text...'}, ...]
#         """
#         lines = content.split('\n')
#         sections = []
#         current_header = "Introduction" 
#         current_lines = []

#         for line in lines:
#             # 匹配标题行 (例如: ## Description)
#             header_match = re.match(r'^(#{1,3})\s+(.*)', line)
            
#             if header_match:
#                 # 保存之前的章节
#                 if current_lines:
#                     sections.append({
#                         "header": current_header,
#                         "body": "\n".join(current_lines).strip()
#                     })
                
#                 # 开始新章节
#                 current_header = header_match.group(2).strip()
#                 current_lines = [] 
#             else:
#                 current_lines.append(line)
        
#         # 保存最后一个章节
#         if current_lines:
#             sections.append({
#                 "header": current_header,
#                 "body": "\n".join(current_lines).strip()
#             })
            
#         return sections

#     def process(self):
#         all_final_chunks = []

#         for entry in self.data:
#             title = entry['title']
#             url = entry['url']
#             full_content = entry['content']

#             # 1. 按章节拆分文章
#             sections = self.parse_sections(full_content)

#             for section in sections:
#                 header = section['header']
#                 body = section['body']
                
#                 # 清洗链接
#                 body = self.clean_markdown_links(body)

#                 if not body: continue

#                 # --- 处理: Infobox (核心数值) ---
#                 # 在上一步爬虫中，把 Infobox 命名为了 "Key Statistics (Infobox)"
#                 is_infobox = "Infobox" in header
                
#                 # For Infobox,尽量不拆分，或者只按行拆分，保证完整性
#                 if is_infobox:
#                     # Infobox 通常是核心数据，权重最高
#                     # 格式化为: "Game Entity: Peashooter - Key Statistics: \n [Peashooter] Cost: 100..."
#                     chunk_text = f"Game Entity: {title}\nSection: {header}\nContent:\n{body}"
                    
#                     all_final_chunks.append({
#                         "id": f"{title}_{header}_info",
#                         "text": chunk_text,
#                         "metadata": {
#                             "title": title,
#                             "type": "infobox", # 这是一个非常重要的标签，用于 Hybrid Search 加权
#                             "url": url
#                         }
#                     })
                
#                 # --- 普通文本章节 ---
#                 else:
#                     # 使用递归切分处理长文本
#                     text_chunks = self.split_text_recursive(body)
                    
#                     for i, sub_text in enumerate(text_chunks):
#                         # Context Injection (上下文注入)
#                         # 即使切片切到了文章中间，我们也强制加上 "Entity: Peashooter - Section: Description"
#                         structured_text = f"Game Entity: {title}\nSection: {header}\nContent:\n{sub_text}"
                        
#                         all_final_chunks.append({
#                             "id": f"{title}_{header}_{i}",
#                             "text": structured_text,
#                             "metadata": {
#                                 "title": title,
#                                 "type": "text",
#                                 "url": url
#                             }
#                         })

#         return all_final_chunks

# if __name__ == "__main__":
#     print("Reading data...")
#     try:
#         with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#             raw_data = json.load(f)
#     except FileNotFoundError:
#         print(f"Error: {INPUT_FILE} not found. Please make sure you have run the optimized fetch_content.py.")
#         exit()

#     chunker = WikiChunker(raw_data)
#     chunks = chunker.process()

#     print(f"Processing completed! Generated {len(chunks)} high-quality chunks.")
    
#     infobox_count = len([c for c in chunks if c['metadata']['type'] == 'infobox'])
#     print(f"- Core Data Chunks (Infobox): {infobox_count} (High Precision Data)")
#     print(f"- Regular Text Chunks (Text):    {len(chunks) - infobox_count}")

#     with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#         json.dump(chunks, f, ensure_ascii=False, indent=2)

#     print(f"\nResults saved to: {OUTPUT_FILE}")
    
#     if chunks:
#         print("\n[Infobox Chunk Example]:")
#         print([c for c in chunks if c['metadata']['type'] == 'infobox'][0]['text'])
        
#         print("\n[Regular Text Chunk Example]:")
#         print([c for c in chunks if c['metadata']['type'] == 'text'][0]['text'])

import json
import re
import hashlib

# --- 配置 ---
INPUT_FILE = "./data/pvz_wiki_rag.json"      
OUTPUT_FILE = "./data/rag_chunks.json"

MAX_CHUNK_SIZE = 500 
OVERLAP = 50  # 重叠字符

def make_unique_id(title, header, chunk_index, text, is_infobox=False):
    """
    生成稳定的唯一 ID，通过 MD5 保证不会重复
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
                #   Infobox 处理（不拆分）
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
                #   普通文本章节
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
        print(f"Error: {INPUT_FILE} not found.")
        exit()

    chunker = WikiChunker(raw_data)
    chunks = chunker.process()

    print(f"Processing completed! Generated {len(chunks)} high-quality chunks.")

    infobox_count = len([c for c in chunks if c['metadata']['type'] == 'infobox'])
    print(f"- Core Data Chunks (Infobox): {infobox_count}")
    print(f"- Regular Text Chunks (Text): {len(chunks) - infobox_count}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {OUTPUT_FILE}")

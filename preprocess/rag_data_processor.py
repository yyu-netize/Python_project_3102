import json
import re

INPUT_FILE = "/home/yuyue/yuyue/python/data/pvz_wiki_rag.json"
OUTPUT_FILE = "rag_chunks.json"

class AdvancedRAGProcessor:
    def __init__(self, data):
        self.data = data

    def is_valid_data_table(self, headers, lines):
        """
        判断这是否是一个有意义的数据表格。
        排除用于排版的布局表格 (Layout Table)。
        """
        # 1. 如果只有 1 列，通常是列表或排版框，不如当纯文本处理
        if len(headers) < 2:
            return False
            
        # 2. 如果表头包含大量的 Markdown 格式或太长，可能是排版标题
        # 例如: | **Welcome to the Wiki** |
        for h in headers:
            if len(h) > 50: 
                return False
        
        return True

    def parse_markdown_table(self, table_text):
        """
        尝试解析 Markdown 表格。
        如果解析失败或判定为布局表格，返回 None，指示调用者将其作为普通文本处理。
        """
        lines = table_text.strip().split('\n')
        if len(lines) < 3: 
            return None

        # 1. 提取表头
        # 移除首尾的 | 并按 | 分割
        headers = [h.strip() for h in lines[0].strip('|').split('|')]
        
        # --- 新增：校验表格质量 ---
        if not self.is_valid_data_table(headers, lines):
            return None
        
        # 2. 跳过第二行 (分割线 |---|---|)
        
        parsed_rows = []
        # 3. 处理数据行
        for line in lines[2:]:
            if not line.strip().startswith('|'):
                continue
                
            values = [v.strip() for v in line.strip('|').split('|')]
            
            # 配对表头和数值
            row_data = {}
            for i, header in enumerate(headers):
                if i < len(values):
                    # 如果表头或内容为空，跳过该字段
                    if header and values[i]:
                        row_data[header] = values[i]
            
            if row_data:
                parsed_rows.append(row_data)
                
        return parsed_rows

    def clean_text_block(self, text):
        """清理普通文本块中的 Markdown 表格残留符号"""
        # 移除只包含 | 或 | --- | 的行
        lines = []
        for line in text.split('\n'):
            if re.match(r'^\|[\s-]*\|?$', line.strip()):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def process_content(self):
        all_chunks = []
        
        # 正则：匹配 Markdown 表格块
        table_pattern = re.compile(r'(\|.*\|(?:\n\|.*\|)+)', re.MULTILINE)

        for entry in self.data:
            url = entry['url']
            main_title = entry['title']
            full_content = entry['content']

            parts = table_pattern.split(full_content)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                chunk_type = "text_block"
                processed_text = part
                metadata = {"title": main_title, "is_table": False}

                # --- 尝试解析表格 ---
                if part.startswith('|') and '\n|' in part:
                    rows = self.parse_markdown_table(part)
                    
                    if rows:
                        # === 情况 A: 有效的数据表格 (多列数据) ===
                        for row_dict in rows:
                            description = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
                            chunk_text = f"页面: {main_title}\n类型: 表格数据\n内容: {description}"
                            
                            all_chunks.append({
                                "source_url": url,
                                "chunk_type": "table_row",
                                "text": chunk_text,
                                "metadata": {
                                    "title": main_title,
                                    "is_table": True,
                                    "raw_data": row_dict 
                                }
                            })
                        # 既然已经按行处理完了，就跳过下面的通用添加逻辑
                        continue 
                    
                    else:
                        # === 情况 B: 布局表格 (Main Page 类型) ===
                        # 解析器返回 None，说明这是一个伪表格。
                        # 我们把它当作普通文本处理，但要清理掉表格线
                        chunk_type = "layout_table_text"
                        processed_text = self.clean_text_block(part)

                # --- 通用文本块处理 (包含 Overview 和 降级的布局表格) ---
                if len(processed_text) < 20: 
                    continue
                    
                chunk_text = f"页面: {main_title}\n类型: 文本描述\n内容:\n{processed_text}"
                
                all_chunks.append({
                    "source_url": url,
                    "chunk_type": chunk_type,
                    "text": chunk_text,
                    "metadata": metadata
                })

        return all_chunks

if __name__ == "__main__":
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"未找到 {INPUT_FILE}，请先运行爬虫脚本。")
        exit()

    processor = AdvancedRAGProcessor(raw_data)
    chunks = processor.process_content()

    print(f"处理完成! 生成了 {len(chunks)} 个块。")
    
    table_chunks = len([c for c in chunks if c['chunk_type'] == 'table_row'])
    text_chunks = len([c for c in chunks if c['chunk_type'] in ['text_block', 'layout_table_text']])
    
    print(f"- 有效数据表格行 (Plant Stats等): {table_chunks}")
    print(f"- 文本块 (Overview/Layout): {text_chunks}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
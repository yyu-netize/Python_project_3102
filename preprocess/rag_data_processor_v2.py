import json
import re

# --- 配置 ---
INPUT_FILE = "pvz_wiki_rag.json"           # 爬虫生成的原始文件
OUTPUT_FILE = "rag_chunks_table_optimized.json" # 处理后的切分文件

class AdvancedRAGProcessor:
    def __init__(self, data):
        self.data = data

    def remove_markdown_links(self, text):
        """
        清洗 Markdown 链接，只保留显示文本。
        格式: [显示文本](URL "标题") 或 [显示文本](URL)
        替换为: 显示文本
        """
        # 正则说明:
        # \[([^\]]+)\]  -> 捕获 [显示文本] 部分
        # \([^\)]+\)    -> 匹配 (URL...) 部分并丢弃
        return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    def is_valid_data_table(self, headers, lines):
        """
        判断这是否是一个有意义的数据表格。
        排除用于排版的布局表格 (Layout Table)。
        """
        # 1. 如果只有 1 列，通常是列表或排版框，不如当纯文本处理
        if len(headers) < 2:
            return False
            
        # 2. 如果表头包含太长的文本(超过50字符)，可能是排版用的长标题
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
        
        # --- 清洗表头中的链接 ---
        headers = [self.remove_markdown_links(h) for h in headers]

        # --- 校验表格质量 ---
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
                        # --- 清洗单元格内容中的链接 ---
                        clean_value = self.remove_markdown_links(values[i].strip())
                        row_data[header] = clean_value
            
            if row_data:
                parsed_rows.append(row_data)
                
        return parsed_rows

    def clean_text_block(self, text):
        """清理普通文本块中的 Markdown 表格残留符号"""
        # 先移除链接
        text = self.remove_markdown_links(text)
        
        lines = []
        for line in text.split('\n'):
            # 移除只包含 | 或 | --- | 的行（表格分割线残留）
            if re.match(r'^\|[\s-]*\|?$', line.strip()):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def process_content(self):
        all_chunks = []
        
        # 正则：匹配 Markdown 表格块
        # 逻辑：寻找以 | 开头，中间包含多行，以 | 结尾的块
        table_pattern = re.compile(r'(\|.*\|(?:\n\|.*\|)+)', re.MULTILINE)

        for entry in self.data:
            url = entry['url']
            main_title = entry['title']
            full_content = entry['content']

            # 使用正则分割内容，保留表格部分
            parts = table_pattern.split(full_content)
            
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                chunk_type = "text_block"
                processed_text = part
                metadata = {"title": main_title, "is_table": False}

                # --- 检查是否为表格 ---
                if part.startswith('|') and '\n|' in part:
                    # 尝试解析
                    rows = self.parse_markdown_table(part)
                    
                    if rows:
                        # === 情况 A: 有效的数据表格 (拆分为行) ===
                        for row_dict in rows:
                            # 将字典转换为自然语言描述
                            description = ", ".join([f"{k}: {v}" for k, v in row_dict.items()])
                            
                            # 构建 Chunk
                            chunk_text = f"页面: {main_title}\n类型: 表格数据\n内容: {description}"
                            
                            all_chunks.append({
                                "source_url": url,
                                "chunk_type": "table_row",
                                "text": chunk_text,
                                "metadata": {
                                    "title": main_title,
                                    "is_table": True,
                                    # 这里不存 raw_data dict，防止写入 ChromaDB 时出错
                                    # 只存简单的元数据
                                    "source": "table" 
                                }
                            })
                        # 既然已经按行处理完了，就跳过下面的通用添加逻辑
                        continue 
                    
                    else:
                        # === 情况 B: 布局表格 (如首页排版) ===
                        # 解析器返回 None，说明这是一个伪表格。
                        # 我们把它当作普通文本处理，但要清理掉表格线
                        chunk_type = "layout_table_text"
                        processed_text = self.clean_text_block(part)

                # --- 通用文本块处理 (包含 Overview 和 降级的布局表格) ---
                else:
                    # 对普通文本也进行链接清洗
                    processed_text = self.remove_markdown_links(part)

                # 忽略过短的碎片
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
    print("正在读取数据...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到 {INPUT_FILE}，请先运行爬虫脚本 fetch_content.py。")
        exit()

    processor = AdvancedRAGProcessor(raw_data)
    chunks = processor.process_content()

    print(f"处理完成! 共生成 {len(chunks)} 个数据块。")
    
    # 简单统计
    table_chunks = len([c for c in chunks if c['chunk_type'] == 'table_row'])
    text_chunks = len(chunks) - table_chunks
    print(f"- 有效表格行数据: {table_chunks} (精准数据)")
    print(f"- 文本描述块:    {text_chunks} (概览与介绍)")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"结果已保存至: {OUTPUT_FILE}")
    
    # 打印示例看看效果
    if table_chunks > 0:
        print(f"\n[清洗后的表格示例]:\n{chunks[0]['text']}")
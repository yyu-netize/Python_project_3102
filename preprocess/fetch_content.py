import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
import re
from markdownify import markdownify as md

# 配置
BASE_URL = "https://plantsvszombies.fandom.com"
START_PAGE = BASE_URL + "/wiki/Main_Page"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
}

def fetch_page(url):
    """下载页面 HTML 并返回 BeautifulSoup 对象"""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                print(f"尝试 {attempt + 1} 失败，重试中... 错误: {e}")
                time.sleep(2)
            else:
                print(f"跳过 URL {url}: {e}")
                return None
    return None

def extract_wiki_links(soup):
    """提取 wiki 内部链接"""
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/"):
            # 过滤掉非内容页面
            if any(ns in href for ns in [":", "Special:", "File:", "Category:", "User:", "Talk:", "Blog:"]):
                continue
            if "#" in href:
                href = href.split("#")[0]
            full = urljoin(BASE_URL, href)
            links.add(full)
    return links

def clean_markdown(text):
    """
    清洗 Markdown 文本，使其更适合 Embedding
    """
    # 1. 移除连续超过2个的换行符 (压缩垂直空间)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. 移除非常短的无意义行 (可能是导航残留)
    lines = [line for line in text.split('\n') if len(line.strip()) > 0]
    return "\n".join(lines)

def extract_page_content(soup, url):
    """
    将页面转换为结构化的 Markdown，保留表格和标题
    """
    # 1. 获取标题 (这对 RAG 检索时的 Context 非常重要)
    title_tag = soup.find("h1", class_="page-header__title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    # 2. 定位主要内容区域
    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        return None

    # --- 智能降噪 (只移除绝对无用的部分，保留 Infobox) ---
    
    # 移除目录 (RAG不需要目录，需要正文)
    for toc in content_div.find_all(class_="toc"):
        toc.decompose()
        
    # 移除底部的导航框 (Navbox 通常包含大量无关链接，干扰语义)
    for nav in content_div.find_all(class_="navbox"):
        nav.decompose()
        
    # 移除引用角标 (如 [1])，这些对语义理解无帮助
    for sup in content_div.find_all("sup", class_="reference"):
        sup.decompose()

    # 移除脚本和样式
    for script in content_div(["script", "style", "noscript"]):
        script.decompose()
        
    # 移除 "Edit" 按钮文本
    for edit in content_div.find_all(class_="mw-editsection"):
        edit.decompose()

    # --- 核心：HTML 转 Markdown ---
    # strip=['a', 'img']: 移除图片标签和链接标签(保留链接文本)。
    # RAG 通常不需要图片URL，链接文本本身比超链接更重要。
    # heading_style="ATX": 使用 # 标题风格
    html_content = str(content_div)
    markdown_text = md(html_content, heading_style="ATX", strip=['img'])
    
    # 后处理清洗
    final_content = clean_markdown(markdown_text)

    return {
        "title": title,
        "url": url,
        "content": final_content
    }

def crawl(start_url, max_pages=100):
    visited = set()
    to_visit = [start_url]
    results = []

    print(f"开始爬取，目标: {max_pages} 页...")

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        base_url = url.split('?')[0].split('#')[0]
        
        if base_url in visited:
            continue
            
        visited.add(base_url)
        print(f"Processing [{len(visited)}/{max_pages}]: {base_url}")

        soup = fetch_page(base_url)
        if not soup:
            continue

        # 提取内容
        page_data = extract_page_content(soup, base_url)
        if page_data and len(page_data["content"]) > 50: # 忽略内容过短的页面
            results.append(page_data)

        # 发现新链接
        new_links = extract_wiki_links(soup)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        time.sleep(1)

    return results

if __name__ == "__main__":
    # 爬取
    rag_data = crawl(START_PAGE, max_pages=50)
    
    print(f"\n爬取结束，共获取 {len(rag_data)} 个有效页面。")
    
    output_filename = "pvz_wiki_rag.json"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=4)

    print(f"数据已保存至 {output_filename}")
    print("数据结构示例:")
    print(json.dumps(rag_data[0], ensure_ascii=False, indent=2) if rag_data else "无数据")
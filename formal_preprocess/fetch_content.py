import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
import re
from markdownify import markdownify as md


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
                print(f"Trying {attempt + 1} failed, retrying... Error: {e}")
                time.sleep(2)
            else:
                print(f"Skipping URL {url}: {e}")
                return None
    return None

def extract_wiki_links(soup):
    """提取 wiki 内部链接"""
    links = set()
    content_div = soup.find("div", class_="mw-parser-output") # 避免导航栏干扰
    if not content_div:
        return links

    for a in content_div.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/"):
            # 过滤掉非内容页面
            if any(ns in href for ns in [":", "Special:", "File:", "Category:", "User:", "Talk:", "Blog:", "UserProfile:"]):
                continue
            if "#" in href:
                href = href.split("#")[0]
            full = urljoin(BASE_URL, href)
            links.add(full)
    return links

def process_infobox(soup, title):
    """
    专门处理 Fandom 的 Infobox (信息框)。
    将复杂的 HTML 结构转换为扁平的 Key-Value 文本，
    并强行注入 Title 防止上下文丢失。
    """
    infobox_text = []
    infobox = soup.find("aside", class_="portable-infobox")
    
    if infobox:
        # 提取信息框里的每一行数据
        rows = infobox.find_all(["div", "section"], class_="pi-item")
        for row in rows:
            # 尝试寻找 label和 value
            label_div = row.find(class_="pi-data-label")
            value_div = row.find(class_="pi-data-value")
            
            if label_div and value_div:
                label = label_div.get_text(strip=True)
                value = value_div.get_text(strip=True)
                # 格式化，让 Embedding 模型能直接关联 属性 和 实体
                infobox_text.append(f"- [{title}] {label}: {value}")
        
        infobox.decompose()
    
    if infobox_text:
        return "### Key Statistics (Infobox)\n" + "\n".join(infobox_text) + "\n\n"
    return ""

def clean_markdown(text):
    """清洗 Markdown 文本"""
    # 1. 移除连续超过2个的换行符
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. 移除非常短的无意义行
    lines = [line for line in text.split('\n') if len(line.strip()) > 0]
    return "\n".join(lines)

def extract_page_content(soup, url):
    """将页面转换为结构化的 Markdown"""
    
    # 1. 获取标题
    title_tag = soup.find("h1", class_="page-header__title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    # 2. 定位主要内容区域
    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        return None
    
    # 移除目录、导航框、引用角标、编辑按钮
    selectors_to_remove = [
        ".toc", ".navbox", ".mw-editsection", 
        "sup.reference", ".wds-global-footer", 
        ".page-footer", "#gallery", ".gallery", # 画廊通常全是图，没文字信息
        ".wikia-gallery-item", ".license-description"
    ]
    
    for selector in selectors_to_remove:
        for element in content_div.select(selector):
            element.decompose()
    
    for script in content_div(["script", "style", "noscript", "iframe"]):
        script.decompose()

    # --- 3. 优先提取并结构化 Infobox ---
    structured_infobox = process_infobox(content_div, title)

    # --- 4. HTML 转 Markdown ---
    html_content = str(content_div)
    # strip=['img', 'a']: 移除图片，保留链接文本但移除链接标签（减少噪音）
    markdown_text = md(html_content, heading_style="ATX", strip=['img', 'a'])
    
    # 后处理
    final_body = clean_markdown(markdown_text)
    
    # --- 5. Context Injection (上下文注入) ---
    # 重新组合文本，把 Title 和 Infobox 放在最前面
    # 这样 Chunking 时，第一块永远是核心数据
    combined_content = f"# {title}\n\n{structured_infobox}## Description\n{final_body}"

    return {
        "title": title,
        "url": url,
        "content": combined_content
    }

def crawl(start_url, max_pages=100):
    visited = set()
    to_visit = [start_url]
    results = []

    print(f"Starting crawl, target: {max_pages} pages...")

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
        
        # 稍微提高一点过滤门槛，只有 Infobox 或者正文够长才保留
        if page_data and len(page_data["content"]) > 100: 
            results.append(page_data)

        # 发现新链接
        new_links = extract_wiki_links(soup)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        time.sleep(1) 

    return results

if __name__ == "__main__":
    rag_data = crawl(START_PAGE, max_pages=7709)
    
    print(f"\nCrawl finished, obtained {len(rag_data)} valid pages.")
    
    output_filename = "./data/pvz_wiki_rag.json"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_filename}")
    if rag_data:
        print("Sample of optimized data structure (first 500 characters):")
        print(rag_data[0]["content"][:500])
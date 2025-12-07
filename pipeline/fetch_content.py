import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
import re
import os  # Added to handle directory creation automatically
from markdownify import markdownify as md

BASE_URL = "https://plantsvszombies.fandom.com"
START_PAGE = BASE_URL + "/wiki/Main_Page"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36"
}

def fetch_page(url):
    """Download HTML and return BeautifulSoup object"""
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
    """Extract valid wiki links from the page"""
    links = set()
    content_div = soup.find("div", class_="mw-parser-output") # Avoid navigation interference
    if not content_div:
        return links

    for a in content_div.find_all("a", href=True):
        href = a["href"]
        if href.startswith("/wiki/"):
            # Filter out non-content pages (namespaces)
            if any(ns in href for ns in [":", "Special:", "File:", "Category:", "User:", "Talk:", "Blog:", "UserProfile:"]):
                continue
            if "#" in href:
                href = href.split("#")[0]
            full = urljoin(BASE_URL, href)
            links.add(full)
    return links

def process_infobox(soup, title):
    """
    Specialized processing for Fandom's Infobox.
    Converts complex HTML structures into flat Key-Value text,
    and forcibly injects the Title to prevent context loss.
    """
    infobox_text = []
    infobox = soup.find("aside", class_="portable-infobox")
    
    if infobox:
        # Extract each row of data in the infobox
        rows = infobox.find_all(["div", "section"], class_="pi-item")
        for row in rows:
            # Try to find label and value
            label_div = row.find(class_="pi-data-label")
            value_div = row.find(class_="pi-data-value")
            
            if label_div and value_div:
                label = label_div.get_text(strip=True)
                value = value_div.get_text(strip=True)
                # Format so that embedding models can directly associate attributes with entities
                infobox_text.append(f"- [{title}] {label}: {value}")
        
        infobox.decompose()
    
    if infobox_text:
        return "### Key Statistics (Infobox)\n" + "\n".join(infobox_text) + "\n\n"
    return ""

def clean_markdown(text):
    """Clean Markdown text"""
    # 1. Remove more than 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 2. Remove very short meaningless lines
    lines = [line for line in text.split('\n') if len(line.strip()) > 0]
    return "\n".join(lines)

def extract_page_content(soup, url):
    """Convert page to structured Markdown"""
    
    # 1. Get title
    title_tag = soup.find("h1", class_="page-header__title") or soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    # 2. Locate main content area
    content_div = soup.find("div", class_="mw-parser-output")
    if not content_div:
        return None
    
    # Remove table of contents, navigation boxes, reference superscripts, edit buttons
    selectors_to_remove = [
        ".toc", ".navbox", ".mw-editsection", 
        "sup.reference", ".wds-global-footer", 
        ".page-footer", "#gallery", ".gallery",
        ".wikia-gallery-item", ".license-description"
    ]
    
    for selector in selectors_to_remove:
        for element in content_div.select(selector):
            element.decompose()
    
    for script in content_div(["script", "style", "noscript", "iframe"]):
        script.decompose()

    # --- 3. Prioritize extraction and structuring of Infobox ---
    structured_infobox = process_infobox(content_div, title)

    # --- 4. Convert HTML to Markdown ---
    html_content = str(content_div)
    # strip=['img', 'a']: Remove images, keep link text but remove link tags (reduce noise)
    markdown_text = md(html_content, heading_style="ATX", strip=['img', 'a'])
    
    # Post-processing to clean Markdown
    final_body = clean_markdown(markdown_text)
    
    # --- 5. Context Injection ---
    # Reassemble text, placing Title and Infobox at the front.
    # This ensures the first chunk always contains core data during chunking.
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

        # Extract content
        page_data = extract_page_content(soup, base_url)
        
        # Slightly increase the filtering threshold; 
        # keep only if Infobox exists or body text is long enough
        if page_data and len(page_data["content"]) > 100: 
            results.append(page_data)

        # Discover new links
        new_links = extract_wiki_links(soup)
        for link in new_links:
            if link not in visited and link not in to_visit:
                to_visit.append(link)

        time.sleep(1) 

    return results

if __name__ == "__main__":
    # You can reduce max_pages for testing purposes when submitting to the professor
    rag_data = crawl(START_PAGE, max_pages=7709)
    
    print(f"\nCrawl finished, obtained {len(rag_data)} valid pages.")
    
    # Define relative path
    output_filename = "./data/pvz_wiki_rag.json"
    
    # Ensure the directory exists before writing (Crucial for reproducibility)
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(rag_data, f, ensure_ascii=False, indent=4)

    print(f"Data saved to {output_filename}")
    if rag_data:
        print("Sample of optimized data structure (first 500 characters):")
        print(rag_data[0]["content"][:500])
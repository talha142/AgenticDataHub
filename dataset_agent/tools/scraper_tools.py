"""
tools/dataset_tools.py
──────────────────────
Low-level scraping primitives exposed as LangChain @tool functions.
The dataset agent calls these in a ReAct loop to crawl pages and
locate datasets.

Tools provided:
  • fetch_page          – GET a URL, return cleaned text + all links
  • fetch_page_rendered – headless Playwright fetch (JS-heavy pages)
  • extract_download_links – regex + heuristic scan for dataset file links
  • extract_tables       – pull <table> HTML from a pagef
  • inspect_json_endpoint – HEAD/GET a URL and report its content-type/size
  •
  
To use a tool, you MUST wrap your JSON request inside <tool_call> tags. 
Do not output bare JSON. You must strictly follow this exact format:

<tool_call>
{"name": "tool_name", "arguments": {"arg_name": "arg_value"}}
</tool_call>
"""

from __future__ import annotations

import re
import json
import os
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# ── constants ─────────────────────────────────────────────────────────────────

DATASET_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".xlsx", ".xls", ".ods",
    ".zip", ".gz", ".tar", ".tar.gz", ".bz2",
    ".hdf5", ".h5", ".parquet", ".feather", ".pkl",
    ".sqlite", ".db", ".sql",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DatasetAgent/1.0; "
        "+https://github.com/your-org/dataset-agent)"
    )
}

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_text(html: str) -> str:
    """Strip tags, collapse whitespace, keep meaningful text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def _is_dataset_url(href: str) -> bool:
    path = urlparse(href).path.lower()
    return any(path.endswith(ext) for ext in DATASET_EXTENSIONS)


def _absolutise(base: str, href: str) -> str:
    return urljoin(base, href)


# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def fetch_page(url: str) -> dict:
    """
    Fetch a web page via a plain HTTP GET request.

    Returns a dict with:
      - text:            cleaned visible text of the page (up to 8000 chars)
      - links:           all <a href> links found, as absolute URLs
      - download_links:  subset of links that look like dataset files
      - status_code:     HTTP status
      - content_type:    response Content-Type header
      - error:           non-empty string if request failed
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type and "application/json" not in content_type:
            resp.close() 
            return {
            "text": "", "links": [], "download_links": [],
            "status_code": resp.status_code, 
            "content_type": content_type,
            "error": "NOT HTML OR JSON",
            }
        soup = BeautifulSoup(resp.text, "lxml")
        
        links = []
        download_links = []
        for a in soup.find_all("a", href=True):
            abs_href = _absolutise(url, a["href"])
            links.append(abs_href)
            if _is_dataset_url(abs_href):
                download_links.append({
                    "url": abs_href,
                    "label": a.get_text(strip=True) or abs_href,
                })

        text = _clean_text(resp.text)

        return {
            "text": text[:8000],
            "links": links[:100],           # cap to avoid overwhelming context
            "download_links": download_links,
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "error": "",
        }
    except Exception as exc:
        return {
            "text": "", "links": [], "download_links": [],
            "status_code": 0, "content_type": "",
            "error": str(exc),
        }


@tool
def fetch_page_rendered(url: str) -> dict:
    """
    Fetch a JavaScript-rendered page using a headless Playwright browser.
    Use this when fetch_page returns little/no content (SPA or JS-heavy site).

    Returns the same schema as fetch_page.
    """
    try:
        from playwright.sync_api import sync_playwright  # lazy import

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            page.goto(url, timeout=TIMEOUT * 1000, wait_until="networkidle")
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "lxml")
        links, download_links = [], []
        for a in soup.find_all("a", href=True):
            abs_href = _absolutise(url, a["href"])
            links.append(abs_href)
            if _is_dataset_url(abs_href):
                download_links.append({
                    "url": abs_href,
                    "label": a.get_text(strip=True) or abs_href,
                })

        return {
            "text": _clean_text(html)[:8000],
            "links": links[:100],
            "download_links": download_links,
            "status_code": 200,
            "content_type": "text/html",
            "error": "",
        }
    except Exception as exc:
        return {
            "text": "", "links": [], "download_links": [],
            "status_code": 0, "content_type": "",
            "error": str(exc),
        }


@tool
def extract_tables(url: str) -> dict:
    """
    Extract all HTML <table> elements from a page.

    Returns:
      - tables: list of dicts, each with 'headers' and 'rows' (first 10 rows)
      - count:  total number of tables found
      - error:  non-empty string on failure
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        tables = []
        for tbl in soup.find_all("table"):
            headers = [th.get_text(strip=True) for th in tbl.find_all("th")]
            rows = []
            for tr in tbl.find_all("tr")[:11]:          # header + 10 data rows
                cells = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                if cells:
                    rows.append(cells)
            tables.append({"headers": headers, "rows": rows})

        return {"tables": tables, "count": len(tables), "error": ""}
    except Exception as exc:
        return {"tables": [], "count": 0, "error": str(exc)}


@tool
def extract_json_from_script_tags(url: str) -> dict:
    """
    Extract JSON data embedded inside <script> tags (common in modern sites
    that server-side render their data store into the HTML).

    Returns:
      - json_blobs: list of parsed JSON objects/arrays found in <script> tags
      - count:      how many were found
      - error:      non-empty string on failure
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        blobs = []
        for script in soup.find_all("script"):
            text = script.string or ""
            # look for standalone JSON objects/arrays
            for match in re.finditer(r"(\{[\s\S]{20,}\}|\[[\s\S]{20,}\])", text):
                try:
                    obj = json.loads(match.group(0))
                    blobs.append(obj)
                except json.JSONDecodeError:
                    pass

        return {"json_blobs": blobs[:5], "count": len(blobs), "error": ""}
    except Exception as exc:
        return {"json_blobs": [], "count": 0, "error": str(exc)}
    
@tool
def add_dataset(url: str, label: str, downloadable: bool, relevance_score : float) -> dict:
    """Adds url into the list of discovered datasets, this is used when an url
    is identified as A DATASET.
    Receives url, a label created by the agent, whether the dataset is downloadable or not (parseable)
    and a relevance score between 1 and 0 dictated by the agent.
    
    Returns an entry into the discovered datasets"""
    relevance = relevance_score
    if relevance_score > 1:
        relevance = 1
    return {"url": url, "label": label, "downloadable": downloadable, "relevance_score": relevance}

@tool
def remove_from_list(url:str):
    """Removes given url from the list of pages to visit"""
    return {"url_to_remove" : url}

@tool
def end_stage():
    """Ends current Agent execution and goes to orchestrator"""
    return

@tool
def parse_tables(url:str, table_selector="table") -> list:
    """Receive page and try to parse it into tables with a table selector
    Returns dictionary with parsed data"""
    tables_json = []
    with sync_playwright() as browse:
        browser = browse.chromium.launch(headless=True)
        page = browser.new_page()

        try: 
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except TimeoutError:
            browser.close()
            return [{"error": f"TIMEOUT: page connection unable to be done."}]
        try:
            page.wait_for_selector(table_selector, timeout=15000)
        except TimeoutError:
            return [{"error": f"TIMEOUT: Selector '{table_selector}' not found on page."}]
        finally:
            html_content = page.content()
            browser.close()
    soup = BeautifulSoup(html_content, "html.parser")
    tables = soup.select(table_selector)
    if not tables:
        return [{"error": "NO TABLE ON PAGE"}]
    # Gets headers
    for table in tables:
        headers = []
        header_row = table.find("tr")
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all(["th", "td"])]
        data = []
        rows = table.find_all("tr")[1:] if headers else table.find_all("tr")
        # Getting rows
        for row in rows:
            # Get text from each cell
            cells = row.find_all(["td", "th"])
            cell_data = [cell.text.strip() for cell in cells]
            
            # Add roww that are not empty
            if cell_data: 
                data.append(cell_data)    

        # Oragnize the data into a json list so that we can return it to either the tool node or to the Agent
        if headers and data:
            json_data = []
            for row in data:
                row_dict = dict(zip(headers, row))
                json_data.append(row_dict)
            tables_json.append(json_data)
    if tables_json:
        return tables_json
    else:
        return [{"error": "ERROR WITH PARSING HEADERS AND DATA"}]
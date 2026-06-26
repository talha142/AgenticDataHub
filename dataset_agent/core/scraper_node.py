'''STARTING NODE OF THE PIPELINE
FETCHES RECURSIVELY ALL PAGES IN THE RANGE OF THE LINK GIVEN (MAX_DEPTH)
AND WITHIN THE GIVEN LIMIT AMOUNT'''

from __future__ import annotations

import re
import json
import os
from typing import Optional
from urllib.parse import urljoin, urlparse


import requests
import asyncio
import aiohttp
from playwright.async_api import async_playwright, TimeoutError
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from core.state import PipelineState

MAX_DEPTH = int(os.getenv("MAX_SCRAPE_DEPTH", "3"))
MAX_PAGES = int(os.getenv("MAX_PAGES_PER_DOMAIN", 100))
TIMEOUT= int(os.getenv("REQUEST_TIMEOUT", 60))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DatasetAgent/1.0; "
        "+https://github.com/your-org/dataset-agent)"
    )
}


async def scraper_node(state : PipelineState) -> dict:
    start_url = state.get("url")
    pages_to_visit = [(start_url, 0)]
    website_name = urlparse(start_url).netloc
    pages_visited_set = set()
    seen_urls = [start_url]

    '''This node fetches the pages in the given website'''
    while(len(pages_to_visit) != 0 and len(seen_urls) < MAX_PAGES):
        current_url, current_depth = pages_to_visit.pop(0)

        if current_url in pages_visited_set:
            continue

        cur_links = await fetch_page_general(current_url, current_depth)
        pages_visited_set.add(current_url)
        
        new_links = []
        for link in cur_links:
            link_domain = urlparse(link).netloc
            if link_domain == website_name and link not in seen_urls:
                seen_urls.append(link)
                new_links.append((link, current_depth + 1))

        pages_to_visit.extend(new_links)
    return {
        "pages_visited" : seen_urls[:MAX_PAGES],
        "current_phase":  "scraping_done",
    }


async def fetch_page_general(url: str, depth: int) -> list:
    if depth == MAX_DEPTH:
        return []
    # First playwwright (better sagfe than sorry i guess)
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page(extra_http_headers=HEADERS)
            # domcontentloaded gives JS some time (15s)
            await page.goto(url, timeout=TIMEOUT * 15, wait_until="domcontentloaded")
            links = await page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a => a.href)")
            # Remove empty or filler links or other irregularities
            clean_links = [link for link in links if link and not link.startswith('javascript:')]
            await browser.close()
        return clean_links
    except Exception as e:
        print(f"Playwright failed: {e}. Falling back to requests.")
        try:
            timeout = aiohttp.ClientTimeout(total=TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=HEADERS) as resp:
                    resp.raise_for_status()
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                    # soup returns only the /ABC, so we need to join with the rest of the loink before returning
                    links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True)]

            return links
        except Exception as req_err:
            print(f"aiohttp failed: {req_err}")
            return []
        
def main():
    from rich.console import Console
    from langgraph.graph import StateGraph, START, END
    from agents.rag_agent import rag_node
    from feedback.parser_feedback import parser_feedback_node
    from core.state import PipelineState, DiscoveredDataset
    from agents.parser_agent import parser_node
    from agents.rag_agent import rag_node
    from feedback.parser_feedback import parser_feedback_node
    import os

    if os.path.exists("links.txt"):
        os.remove("links.txt")
    console = Console()
    urls_to_test = [
        "http://hit2.badd-cao.net/",
        "https://biomx-db.com/",
        "https://bidd.group/CMAUP/"
    ]
    def run_pipeline(url: str):
        initial_state = {
            "url":                  url,
            "topic":                "chemical compounds",
            "pages_visited":        [],
            "pages_to_visit":       set(),
            "discovered_datasets":  [],
            "datasets_to_download": [],
            "datasets_to_parse":    [],
            "downloaded_files":     {"data": []},
            "parsed_datasets":      {"data": []},
            "current_parsing":      {},
            "summary_report":       "",
            "messages":             [],
            "errors":               [],
            "feedback_given":       0,
            "current_phase":        "starting",
            "downloader_feedback": {},
            "parser_feedback": {},
            "dataset_feedback": {},
            "orchestrator_feedback": {},
            "parser_should_retry": False,
            "dataset_needs_retry": False,
            "dataset_refinement_targets": [],
            "parser_improvement_hints": "",
            "datasets_to_retry": [],
            "learned_lessons": [],
            "has_retried": False,
        }
        with open("links.txt", "a") as f:
            scraper_dict = scraper_node(initial_state)
            url_str = f"\n{scraper_dict.get('url')}:\n"
            f.write(url_str)
            visited = scraper_dict.get("pages_visited")
            visited_str = f"{chr(10).join(visited)}\n"
            f.write(visited_str)
            f.close()
    for url in urls_to_test:
        run_pipeline(url)

if __name__ == "__main__":
    main()
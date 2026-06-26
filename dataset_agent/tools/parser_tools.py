from langchain_core.tools import tool
from playwright.sync_api import sync_playwright
import os

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DatasetAgent/1.0; "
        "+https://github.com/your-org/dataset-agent)"
    )
}

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))

@tool
def search_input(url: str, search_term: str, input_selector: str, button_selector: str) -> str:
    """
    Use this tool when a page requires submitting a search form to view datasets.
    It navigates to the URL, inputs the search_term into the input_selector, 
    clicks the button_selector, and returns the resulting HTML containing the data.
    """
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context(extra_http_headers=HEADERS)
                page = context.new_page()
                
                page.goto(url, timeout=TIMEOUT*1000, wait_until="networkidle")
                
                page.fill(input_selector, search_term)
                page.click(button_selector, force=True)
                
                page.wait_for_timeout(3000)
                active_page = context.pages[-1]
                
                try:
                    active_page.wait_for_load_state("networkidle", timeout=TIMEOUT * 1000)
                except Exception as e:
                    print(f"Network didn't go fully idle, proceeding with prev page. ({e})")
                html = active_page.content()
                return html
            finally:
                browser.close()
            
    except Exception as e:
        return f"Error executing autonomous search: {str(e)}"
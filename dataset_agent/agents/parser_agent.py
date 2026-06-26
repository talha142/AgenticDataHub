"""
agents/parser_agent.py
──────────────────────
A ReAct agent that handles data embedded in web pages (HTML tables,
JSON APIs, script-embedded blobs). For each parseable dataset it:

  1. Fetches/inspects the source to understand its structure.
  2. Writes a Python script to extract the data.
  3. Executes the script via execute_python_code.
  4. Verifies the result exists in the output directory.
  5. Retries with a corrected script if execution failed (max 3 attempts).
"""

from __future__ import annotations

import json
import re
from typing import Any
from pathlib import Path
import os
import traceback
from urllib.parse import urljoin

import requests
from playwright.sync_api import sync_playwright, TimeoutError
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from tools.tool_parser import tool_call_parser
from tools.parser_tools import search_input

from core.llm import get_llm
from core.state import PipelineState, DiscoveredDataset, ParsedDataset


OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
MAX_PARSER_DEPTH = 2
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DatasetAgent/1.0; "
        "+https://github.com/your-org/dataset-agent)"
    )
}

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
PARSER_SYSTEM = """You are an expert web scraping and autonomous navigation agent. 
I am providing you with the HTML of a page from a chemical database.

YOUR MISSION:
Phase 1: Determine if the provided HTML contains the target datasets, OR if it is a search form.
- If it is a search form, use your `autonomous_search` tool to execute a search. 
- HOW TO FIND THE INPUT_SELECTOR: Look at the HTML and find the primary text input box for the form. Extract its exact `name` attribute. You MUST format your selector as input[name="the_name_you_found"]. Do not invent a name.
- HOW TO FIND THE BUTTON_SELECTOR: Find the submit button associated with that input. Use its `id` attribute. Do not use generic classes.
- HOW TO CHOOSE A SEARCH_TERM: Provide a real, valid name of a chemical compound (e.g., acid). Do not use placeholders.
- BE MINDFUL THAT SOME WEBSITES HAVE DUPLICATE IDS, IF THAT IS THE CASE, GO BY NAME NOT ID
- search_input(url: str, search_term: str, input_selector: str, button_selector: str)
Here is an example of a tool call:
    {"name": "tool_name", "arguments": {"arg_name1": "arg_value1", "arg_name2": "arg_value2"}}

Phase 2: Generate the extraction code IF NECESSARY.
RULES:
1. The function must take a single argument: `html_content` (a string of raw HTML).
2. The function must return a Python dictionary matching: {'target_links': ['url1', 'url2']}
3. Do NOT use any LLMs or external API calls inside the function. Use standard CSS selectors.
4. The Function name must ALWAYS be extract_links.
5. ONLY the output function must come AFTER the <think> tags. Do NOT output anything after the code.
"""


def build_parser_agent():
    return get_llm(temperature=0.0, max_tokens=4096)


def parser_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: parse embedded datasets and save to disk.
    """
    datasets: list[DiscoveredDataset] = state.get("datasets_to_parse", [])

    # First time after feedback
    if state.get("parser_should_retry", False):
        return {"parser_should_retry": False,"parsed_datasets": {"data": [], "method": "clean"}, "current_phase": "parsing", "parsing_schema": None, "has_retried": True}
    
    current_parsed = state.get("parsed_links", [])
    parsed_datasets = {i["source_url"] for i in current_parsed}
    valid_datasets = [d for d in datasets if d.url not in parsed_datasets]

    parse_inst = state.get("parsing_schema", None)
    message_to_return = []
    message_to_return.append(AIMessage(content=""))
    urls = state.get("current_to_parse", [])
    if not valid_datasets and not urls:
        return {"messages": message_to_return, "current_phase": "parsing_complete", "parsing_schema": None, "parser_should_retry": False}

    agent = build_parser_agent()
    newly_parsed = []
    # We have a parsed dataset
    if parse_inst:
        urls.remove(parse_inst["url"])
        if not parse_inst["success"]:
            metadata_json = ParsedDataset(
                source_url=parse_inst["url"],
                local_path="",
                column_names=[],
                row_count=-1,
                sample_rows=[],
                parse_code="",
                success=False,
                error="Error when Parsing"
            )
            newly_parsed.append(metadata_json)
        else:
            try:
                clean_json = parse_inst["result"]
                newly_parsed.extend(log_parsed(parse_inst['url'], clean_json))
                save_parsed(clean_json, parse_inst["url"])
            except:
                metadata_json = ParsedDataset(
                    source_url=parse_inst["url"],
                    local_path="",
                    column_names=[],
                    row_count=-1,
                    sample_rows=[],
                    parse_code="",
                    success=False,
                    error=f"LLM extraction failed"
                )
                newly_parsed.append(metadata_json)
        return  {
            "current_phase":   "parsing",
            "parsed_datasets": {"data": [file.model_dump() for file in newly_parsed]},
            "parsing_schema": None,
            "current_to_parse": urls,
        }
    else:
        # We still have tables to parse in relation to the source url given to parser
        # Just pass on to the RAG
        if urls:
            html = fetch_page_general(urls[0])
            parse_inst = {
                "url" : urls[0],
                "html": html,
                "code": "",
                "result": "",
                "success": False
            }
            return {
            "current_to_parse": urls,
            "current_phase":   "rag_assistance",
            "parsing_schema": parse_inst
        }
        # We need to fetch new urls from valid_datasets
        else:
            to_fetch = valid_datasets[0].url
            raw_html = fetch_page_general(to_fetch)
            cleaned_html = clean_raw_text(raw_html)
            user_msg = (
                f"Analyze the following HTML:\n\n"
                f"URL: {to_fetch}\n\n"
                f"{cleaned_html}"
            )
            if state.get("has_retried", False):
                parser_lessons = [
                    lesson['lesson']
                    for lesson in state.get("learned_lessons", []) 
                    if lesson.get("agent") == "parser" 
                ]
                user_msg += (
                    f"{state.get('parser_feedback', '')}\n"
                    f"{state.get('parser_improvement_hints', '')}\n"
                    f"Parser lessons:\n- {chr(10).join(parser_lessons)}\n"
                )
            code_success = False
            attempts = 0
            max_attempts = 3
            error_feedback = ""
            tool_node = ToolNode([search_input])
            messages_to_send = []
            # CALL TOOL FOR SEARCH PAGES, GET LINKS FOR BROWSE PAGES
            while not code_success and attempts < max_attempts:
                prompt = user_msg + (f"\n\nPREVIOUS ERROR:\n{error_feedback[-100:]}" if error_feedback else "")
                messages_to_send = [
                    SystemMessage(content=PARSER_SYSTEM),
                    HumanMessage(content=prompt)
                ]
                result = agent.invoke(messages_to_send)
                last_msg = result.content

                tool_calls = tool_call_parser(last_msg)
                with open("ParserMessage.txt", "w") as f:
                    f.write(last_msg)
                # If AI called to search compounds/anything
                # We can get the HTML clean it and reinvoke the LLM with the new extracetd HTML
                
                if tool_calls:
                    ai_tool_call = AIMessage(content=last_msg, tool_calls=tool_calls)
                    raw_html = tool_node.invoke({"messages": [ai_tool_call]})["messages"][-1].content
                    # Didnt work, reprompt the AI about the original HTML#
                    if not raw_html or "Error" in raw_html:
                        print("Unable to get links in time")
                        error_feedback = f"Tool execution failed: {raw_html}"
                        # Retry
                        continue
                    cleaned_p2_html = clean_raw_text(raw_html)
                    user_msg = (
                        f"Analyze the following HTML:\n\n"
                        f"{cleaned_p2_html}"
                    )
                    # Reinvoke
                    prompt = user_msg + (f"\n\nPREVIOUS ERROR:\n{error_feedback[-100:]}" if error_feedback else "")
                    messages_to_send = []
                    messages_to_send.append(PARSER_SYSTEM)
                    messages_to_send.append(prompt)
                    result = agent.invoke(messages_to_send)
                    last_msg = result.content
                # Only count up attempts of parsing not searching (MIGHT BE CHANGED LATER)
                else:
                    attempts += 1

                # We have code for a page with links
                generated_code = last_msg.split("</think>")[-1].strip()
                # remove markdown stuff
                generated_code = generated_code.replace("```python", "").replace("```", "").strip()
                try:
                    scope = {
                        "BeautifulSoup": BeautifulSoup, 
                        "re": re,
                        "json": json
                    }
                    # Run the code
                    exec(generated_code, scope)
                    extract_func = scope.get('extract_links')
                    if extract_func:
                        code_result = extract_func(raw_html)
                        with open("parser_res.md", "w") as f:
                            f.write(code_result)
                        if code_result:
                            # We have tables
                            if isinstance(code_result, dict) and has_extracted_data(code_result):
                                code_success = True
                                urls_ls = code_result["target_links"]
                                urls = [urljoin(to_fetch, u) for u in urls_ls]
                            else:
                                # No tables
                                error_feedback = "Logic Error: The code executed successfully but returned empty data"
                        else:
                            error_feedback = "Code unsuccessful in running"
                except:
                    error_feedback = f"Execution Error:\n{traceback.format_exc()}"
            
            # Original URL could have its own tables
            urls.append(to_fetch)
            # Add current valid dataset link to parsed links so that we dont check it again
            parsed_links = state.get("parsed_links", [])
            parsed_links.append({"source_url": valid_datasets[0].url, "success": code_success})
            return {"messages": message_to_return, "current_phase": "parsing", "parsing_schema": None, "current_to_parse": urls, "parsed_links" : parsed_links}

def save_parsed(table_data: list, url: str):
    path = Path("output") / "parsed"
    path.mkdir(parents=True, exist_ok=True)
    safe_url = re.sub(r'[^a-zA-Z0-9]', '_', url)
    filename = f"{safe_url}.json"
    file_path = path / filename
    with open(file_path, 'w') as f:
        json.dump(table_data, f)
    
def log_parsed(url: str, json_data: dict | list) -> list:
    """Safely extracts rows from the JSON and logs the metadata."""
    
    # Figure out where the rows are based on what the LLM returned
    if isinstance(json_data, list):
        rows = json_data
    elif isinstance(json_data, dict) and "data" in json_data:
        rows = json_data["data"]
    else:
        rows = [json_data]
        
    if not rows:
        return []

    metadata_json = ParsedDataset(
        source_url=url,
        local_path="",
        column_names=list(rows[0].keys()) if isinstance(rows[0], dict) else [],
        row_count=len(rows),
        sample_rows=rows[:5],
        parse_code="",
        success=True,
        error=""
    )
    
    return [metadata_json]
    
def clean_llm_json(raw_text: str) -> dict | list:
    """Removes <think> tags and markdown formatting to safely parse JSON."""
    clean_msg = raw_text.split("</think>")[-1] if "</think>" in raw_text else raw_text
    clean_msg = clean_msg.strip()

    match = re.search(r'```(?:json)?(.*?)```', clean_msg, re.DOTALL)
    if match:
        clean_msg = match.group(1).strip()
        
    return json.loads(clean_msg)

def fetch_page_general(url: str) -> str:
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(extra_http_headers=HEADERS)
            # networkidel gives JS time to populate the tables
            page.goto(url, timeout=TIMEOUT * 1000, wait_until="networkidle")
            html = page.content()
            browser.close()
        return html
    except Exception as e:
        print(f"Playwright failed: {e}. Falling back to requests.")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException:
            return ""
        
def clean_raw_text(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    
    for element in soup(["script", "style", "svg", "noscript", "meta", "link", "head", "header", "footer", "nav", "iframe"]):
        element.decompose()
        
    for tag in soup.find_all(True):
        tag.attrs = {key: value for key, value in tag.attrs.items() if key in ['id', 'class', 'href']}
        
    for string in soup.find_all(string=True):
        if len(string.strip()) > 50:
            string.replace_with(string.strip()[:50] + "...")
            
    clean_ret = str(soup.body if soup.body else soup)
    return clean_ret[:15000]

def has_extracted_data(data_dict):
    if not data_dict:
        return False
    for key, value in data_dict.items():
        if value: 
            return True
    return False

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
    
    if os.path.exists("gen_code"):
        os.remove("gen_code")
    if os.path.exists("gen_code_parser.md"):
        os.remove("gen_code_parser.md")
    if os.path.exists("parser_res.md"):
        os.remove("parser_res.md")
    console = Console()
    urls_to_test = [
        "https://streptomedb.vm.uni-freiburg.de/streptomedb/compound_list/",
        "https://bidd.group/NPASS/search.php"
    ]
    datasets_to_test = [(DiscoveredDataset(url=url, label="label", dataset_type="direct_download", )) for url in urls_to_test]
    initial_state = {
        "url":                  "",
        "topic":                "chemical compounds",
        "pages_visited":        {"data": []},
        "pages_to_visit":       set(),
        "discovered_datasets":  [],
        "datasets_to_download": [],
        "datasets_to_parse":    datasets_to_test,
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

    graph = StateGraph(PipelineState)
    graph.add_node("parser_node",           parser_node)
    graph.add_node("RAG_node",              rag_node)
    graph.add_node("parser_feedback",       parser_feedback_node)

    def parser_routing(state: PipelineState) -> str:
        """Rerouting parser agent to tool or to next node (Summary)"""
        status = state.get("current_phase", "parsing_complete")
        if status == "parsing_complete":
            return "parser_feedback"
        elif status == "rag_assistance":
            return "RAG_node"
        else:
            return "parser_node"
        
    def feedback_routing(state:PipelineState) -> str:
        cur_phase = state.get("current_phase", "")
        if cur_phase == "parser_feedback_complete":
            if state.get("parser_should_retry", False):
                return "parser_node"
            return END
        elif cur_phase == "parsing":
            return "parser_node"
    
    graph.add_edge(START,                          "parser_node")
    graph.add_edge("RAG_node", "parser_node")
    graph.add_conditional_edges("parser_node", parser_routing)
    graph.add_conditional_edges("parser_feedback",      feedback_routing)

    pipeline = graph.compile()
    final_state = {}
    for event in pipeline.stream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0] if event else "unknown"
            console.print(f"  [dim]← {node_name}[/dim]")
            for updates in event.values():
                final_state.update(updates)
    return

    
if __name__ == "__main__":
    main()


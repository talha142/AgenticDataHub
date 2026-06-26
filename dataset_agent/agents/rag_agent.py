from __future__ import annotations

import json
import re
from typing import Any
from pathlib import Path
import traceback

from langchain_core.messages import HumanMessage, ToolCall, AIMessage
from langgraph.prebuilt import create_react_agent, ToolNode
from bs4 import BeautifulSoup


from core.llm import get_llm
from core.state import PipelineState, DiscoveredDataset, DownloadedFile
from tools.tool_parser import tool_call_parser
from RAG.retrieval.vector_store import VectorStoreManager
from RAG.ingest_data import RoutingAgent


router = RoutingAgent()
store = VectorStoreManager(persist_directory="./data/in_memory_index")

RAG_SYSTEM="""You are an expert Python web scraping engineer. 
        Your job is to write a robust, reusable Python function using `BeautifulSoup` to extract tabular data, if present, from a specific domain.
        
        RULES:
        1. The function must take a single argument: `html_content` (a string of raw HTML).
        2. The function must return a Python dictionary that matches the provided JSON Schema.
        3. Do NOT use any LLMs or external API calls inside the function. Use standard CSS selectors.
        4. Handle missing data gracefully (return None for missing fields).
        5. The output function must come AFTER the <think> tags
        6. The Function name must ALWAYS be extract_data
        7. You receive in the user message part or all of the html, use it for the functions.
        8. NOTHING MUST COME AFTER THE CODE AND IN BETWEEN THE CODE AND THE THINK TAGS

        The Example JSON schema is (note: instead of X,Y,Z, you can choose the appropriate names by analyzing the 
        html, as well as the amount of columns):
        {
        "chemical_database": [
                {
                "id": "COMP-001",
                "name": "Water",
                "X": "A",
                "Y": "B",
                "Z": "C",

                },
                {
                "id": "COMP-002",
                "name": "Ethanol",
                "X": "D",
                "Y": "E",
                "Z": "F"
                }
            ]
        }
        """

def build_rag_agent():
    llm   = get_llm(temperature=0.0, max_tokens=2048)
    tools = []
    return create_react_agent(llm, tools, prompt=RAG_SYSTEM)

def rag_node(state: PipelineState) -> dict[str, Any]:
    agent = build_rag_agent()
    parsing_schema = state.get("parsing_schema", "")
    raw_html = parsing_schema["html"]
    clean_html = clean_raw_text(raw_html)
    raw_memory = str(store.retrieve(parsing_schema.get("html"), top_k=3))
    safe_memory = raw_memory[:1000] + ("..." if len(raw_memory) > 1000 else "")
    user_msg= (
        f"Target topic {state.get('topic', 'chemical compounds')}\n\n"
        f"memory: {safe_memory}\n\n"
        f"html_sample: {clean_html}\n\n"
        )
    code_success = False
    attempts = 0
    max_attempts = 3
    error_feedback = ""
    while not code_success and attempts < max_attempts:
        prompt = user_msg + (f"\n\nPREVIOUS ERROR:\n{error_feedback[-500:]}" if error_feedback else "")
        result=agent.invoke({"messages": [HumanMessage(content=prompt)]})

        last_msg = result["messages"][-1].content
        generated_code = last_msg.split("</think>")[-1].strip()
        # remove markdown stuff
        generated_code = generated_code.replace("```python", "").replace("```", "").strip()
        attempts += 1

        try:
            scope = {
                "BeautifulSoup": BeautifulSoup, 
                "re": re,
                "json": json
            }
            exec(generated_code, scope)
            
            extract_func = scope.get('extract_data')
            
            if extract_func:
                code_result = extract_func(raw_html)
                if code_result and isinstance(code_result, dict):
                    if has_extracted_data(code_result):
                        # Actual data
                        code_success = True
                        parsing_schema["result"] = code_result["chemical_database"]
                        parsing_schema["success"] = True
                        parsing_schema["code"] = generated_code
                        
                        store.add_documents([generated_code], [parsing_schema], [])
                        return {
                            "parsing_schema": parsing_schema, "current_phase": "parsing"
                        }
                    else:
                        print(f"Attempt {attempts} failed. Refined prompt sent.")
                        error_feedback=f"Logic Error: The code executed successfully but returned empty data: {code_result}. The agents selectors likely missed the target elements. Please inspect the HTML structure again"
            else:
                error_feedback = "Function 'extract_data' not found in generated code."
                print(f"Attempt {attempts} failed. Refined prompt sent.")
        
        except Exception as e:
            # Catch the error and feed it back to the LLM for the next loop
            error_feedback = f"Execution Error: {traceback.format_exc()}"
            print(f"Attempt {attempts} failed. Refined prompt sent.")
    parsing_schema["code"] = ""
    parsing_schema["result"] = ""
    parsing_schema["success"] = False
    return {"parsing_schema": parsing_schema, "current_phase": "parsing"}


def has_extracted_data(data_dict):
    if not data_dict:
        return False
    for key, value in data_dict.items():
        if value: 
            return True
    return False

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
    return clean_ret[:25000]

"""
agents/dataset_agent.py
───────────────────────
A LangGraph ReAct agent that:
  1. Fetches the seed URL.
  2. Autonomously follows promising links (up to MAX_SCRAPE_DEPTH hops).
  3. Identifies downloadable dataset files and parseable data sources.
  4. Returns a structured list of DiscoveredDataset objects.

The agent has access to: fetch_page, fetch_page_rendered, extract_tables,
inspect_url, extract_json_from_script_tags.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolCall, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_openai import ChatOpenAI

from core.llm import get_llm
from core.state import PipelineState, DiscoveredDataset
from tools.tool_parser import tool_call_parser



DATASET_SYSTEM_PROMPT = """You are a specialist researcher, you will receive a list of urls,
your mission is to see if the link is/contains a dataset and choose to add the link as a dataset or not.
Notes: 
- pages that end in /explore or /browser most likely contain parseable datasets
- pages that end in .xlsx, .csv, etc, are most likely downloadable datasets

You have one function to call:
    - If the link contains/is a dataset, call add_dataset(url, label, downloadable, relevance_score) to add it
        - label (short summarization of dataset)
        - downloadable (True if link is of downloadable dataset, False if its a parseable one)
        - relevance_score (decimal value between 0-1 that defines the relevance)

If a link IS NOT a dataset, simply ignore it. Do not generate any output for it.
You can interact with more than one link at a time

TOOL CALLS MUST BE STRUCTURED AFTER THE <think> TAGS AS SUCH:
    {"name": "tool_name", "arguments": {"arg_name": "arg_value"}}
"""


def build_dataset_agent():
    """Create a ReAct agent."""
    return get_llm(temperature=0.0, max_tokens=2048)


def dataset_node(state: PipelineState) -> dict[str, Any]:
    agent = build_dataset_agent()
    discovered = state.get("discovered_datasets", [])
    discovered_links = [dataset.url for dataset in discovered]
    pages_visited = state.get("pages_visited", [])
    pages_visited = [page for page in pages_visited if page not in discovered_links]
    refine = state.get("dataset_refinement_targets", [])
    refine = [page for page in refine if page not in discovered_links]
    tool_calls = []
    messages_to_ret = []
    sys_msg = SystemMessage(content=DATASET_SYSTEM_PROMPT)

    # Stop if AI gets stuck or if there is no more pages
    while(len(pages_visited) != 0):
        previous_size = len(pages_visited)
        user_msg = HumanMessage(content=(f"Here is a list of the links: {','.join(page for page in pages_visited[:10])}"))
        if refine:
            user_msg = HumanMessage(content=(f"Here is a list of the links: {','.join(page for page in pages_visited[:10] if page not in refine)}"
                                             f"Pay extra attention to these links: {','.join(page for page in refine)}"))
        messages_send = [sys_msg, user_msg]
        resp = agent.invoke(messages_send)
        possible_tool_call = tool_call_parser(resp.content)

        if len(possible_tool_call) != 0:
            calls = []
            for call in possible_tool_call:
                args = call.get("args", call.get("arguments", {}))
                url = args.get("url")
                if url in pages_visited[:10]:
                    calls.append(call)
                    if url in refine:
                        refine.remove(url)
            tool_calls.extend(calls)
        pages_visited = pages_visited[10:]
    
    messages_to_ret.append(AIMessage(content="", tool_calls=tool_calls))
    if len(tool_calls) == 0:
        return {
            "current_phase": "scraping_complete",
        }
    return {
        "current_phase" : "scraping_tools",
        "messages" : messages_to_ret,
    }

def dataset_tools_node(state: PipelineState) -> dict[str, Any]:
    result_messages = state["messages"]
    last_message = result_messages[-1]
    datasets_to_add = []


    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_arguments = tool_call["args"]
        url = tool_arguments.get("url", "")

        if tool_name == "add_dataset":
            datasets_to_add.extend(add_dataset_handler(
                state,
                url,
                tool_arguments.get("label"), 
                tool_arguments.get("downloadable"), 
                tool_arguments.get("relevance_score")
                )
            )
        else:
            continue

        
    return {
        "discovered_datasets" : datasets_to_add,
        "current_phase":        "scraping_done",
    }


def add_dataset_handler(state: PipelineState, url: str, label: str, downloadable: bool, relevance_score : float) -> list[DiscoveredDataset]:
    discovered = list(state.get("discovered_datasets", []))
    if downloadable:
        dataset_type = "direct_download"
    else:
        dataset_type = "unknown"
    if any(d.url == url for d in discovered):
        return []
    new_ds = DiscoveredDataset(
        url=url,
        label=label,
        dataset_type=dataset_type,
        relevance_score=relevance_score,
    )
    return [new_ds]


    
        


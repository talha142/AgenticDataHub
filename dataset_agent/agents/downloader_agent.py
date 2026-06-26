"""
agents/downloader_agent.py
──────────────────────────
A ReAct agent that iterates over `datasets_to_download` and saves each
file to the local output directory using the `download_file` tool.

It also verifies each download with `inspect_url` before committing
bandwidth, and uses `list_output_dir` to confirm files were saved.
"""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path
import dataclasses
import os

from langchain_core.messages import HumanMessage, ToolCall, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent, ToolNode

from core.llm import get_llm
from core.state import PipelineState, DiscoveredDataset, DownloadedFile
from tools.download_tools import download_file, list_output_dir, inspect_downloadable
from tools.tool_parser import tool_call_parser

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()

DOWNLOADER_SYSTEM = """You are a file-download agent. You will receive a list of
dataset URLs to download. For each one:
FOR EACH URL, inspect_downloadable MUST be called BEFORE download_file
1. Call inspect_downloadable(url) to verify it is accessible and confirm the file type.

2. Call download_file(url) to save it locally. Pass a clean filename if you can
   infer one from the URL and content-type.

3. Files above 500MB will not be downloaded'
"Download each dataset ONCE SKIP DUPLICATES"
"However, immediately after reasoning tag, IF THERE IS A CALL your FINAL output MUST BE ONLY the raw JSON tool call block. "
"Do NOT output any conversational text, summaries, lists, or explanations outside of the think tags. "
"When there is nothing more to be done call ONLY end_stage()"
"If you output a single word of standard text outside the JSON, the system will crash."
To call a tool you must follow the example:
{"name": "tool_name", "arguments": {"arg_name": "arg_value"}}
"""

def build_downloader_agent():
    return get_llm(temperature=0.0, max_tokens=4096)


def downloader_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: download all direct-download datasets.
    """
    to_retry = state.get("datasets_to_retry", [])
    datasets: list[DiscoveredDataset] = state.get("datasets_to_download", [])
    if not datasets and not to_retry or state.get("current_phase", "") == "download_complete":
        return {"current_phase": "download_complete"}

    agent = build_downloader_agent()
    # Gets only the not yet downloaded datasets
    dl_data = state.get("downloaded_files", {}).get("data", []) if isinstance(state.get("downloaded_files"), dict) else state.get("downloaded_files", [])
    downloaded_urls = {get_field(f, 'original_url') for f in dl_data}
    inspected = {get_field(i, 'original_url') for i in state.get("inspected_urls", [])}

    valid_datasets = [d for d in datasets if get_field(d, 'url') not in downloaded_urls]
    actual_retry = [d for d in to_retry if get_field(d, 'url') not in downloaded_urls]

    batch = valid_datasets[:10]
    ds_list = "\n".join(
        f"- {get_field(d, 'url', '')} [{get_field(d, 'file_format', 'unknown')}]"
        for d in batch
    )
    if not valid_datasets and not actual_retry:
        return {
            "current_phase":    "download_complete",
        }
    

    prompt_parts = []


    if valid_datasets:
        prompt_parts.append(f"Download the following dataset files:\n{ds_list}")
    
    # Take into account the feedback
    if actual_retry:
        downloader_lessons = [
            lesson['lesson']
            for lesson in state.get("learned_lessons", []) 
            if lesson.get("agent") == "downloader" 
        ]

        url_sug = [
            f"- {get_field(item, 'url')} suggestions: {', '.join(get_field(item, 'suggested_params', []))}" 
            for item in actual_retry[:5]
        ]
        
        retry_msg = f"Download the following with the suggested parameters:\n{chr(10).join(url_sug)}"
        if downloader_lessons:
            retry_msg += f"\nLearned lessons: {', '.join(downloader_lessons)}"
            
        prompt_parts.append(retry_msg)

    prompt_parts.append(f"Already inspected URLs: {inspected}")
    user_msg = "\n\n".join(prompt_parts)
    
    messages_to_send = [SystemMessage(content=DOWNLOADER_SYSTEM),HumanMessage(content=user_msg)]
    result = agent.invoke(messages_to_send)

    last_msg = result
    possible_tool_call = tool_call_parser(last_msg.content)
    message_to_return = []

    # If we have a possible tool call we add it to the message history, the tool_node will deal with that
    if len(possible_tool_call) != 0:
        phase_status = "download_tool"
        calls = []
        if possible_tool_call[0]["name"] == "end_stage":
            phase_status = "download_complete"
        # To avoid parallel tool execution screwing up pipeline
        for call in possible_tool_call:
            if "inspect" in call["name"]:
                if call["args"].get("url") not in inspected:
                    calls.append(call)
            else:
                    calls.append(call)
        message_to_return.append(AIMessage(content="", tool_calls=calls))
    else:
        phase_status = "download_complete"
    return {
        "current_phase":    phase_status,
        "messages":         message_to_return,
    }


def downloader_tools_node(state: PipelineState) -> dict[str, Any]:
    """Tool node responsible for executing downloader node's tools sequentially"""
    
    result_messages = state["messages"]
    last_message = result_messages[-1]
    
    inspected_urls = state.get("inspected_urls", [])
    new_downloads = []
    returned_tool_messages = []
    inspected_urls_len = len(inspected_urls)


    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_arguments = tool_call["args"]
        url = tool_arguments.get("url", "")
        
        if not url:
            print(f"ERR: {tool_name} called without a URL argument.")
            returned_tool_messages.append(ToolMessage(
                content="Error: You must provide a 'url' argument.",
                name=tool_name,
                tool_call_id=tool_call["id"]
            ))
            continue

        if tool_name == "inspect_downloadable":
            tool_res = inspect_downloadable.invoke(tool_arguments)
            
            if not any(d.original_url == url for d in inspected_urls):
                inspected_urls.extend(inspect_downloadable_handler(state, url, tool_res))
                       
        elif tool_name == "download_file":
            matched_inspected = next((d for d in inspected_urls if d.original_url == url and d.success), None)
            
            if matched_inspected:
                tool_res = download_file.invoke(tool_arguments)
                
                new_downloads.extend(download_file_handler(state, tool_res, matched_inspected))
            else:
                print("ERR 3")
                new_downloads.append(DownloadedFile(
                    original_url=url,
                    local_path="",
                    file_format="Unknown",
                    size_bytes=-1,
                    success=False,
                    error="Inspect failed or dataset invalid"
                ))
        
        else:
            print(f"{tool_name} is invalid")

        returned_tool_messages.append(ToolMessage(
            content=f"Processed {tool_name} for {url} successfully.",
            name=tool_name,
            tool_call_id=tool_call["id"]
        ))
    cur_phase = state.get("current_phase", "")
    if inspected_urls_len == len(inspected_urls) and len(new_downloads) == 0:
            cur_phase = "download_complete"
    return {
        "downloaded_files" : {"data": [file.model_dump() for file in new_downloads]},
        "messages" : returned_tool_messages,
        "inspected_urls": inspected_urls,
        "current_phase": cur_phase,
    }

def inspect_downloadable_handler(state: PipelineState, url: str, tool_ret:dict) -> list:
    """This function should just check if inspect was successful, if it is, add it to the list
    Should also check file size, as more than 500MB will not be downloaded"""
    # Dummy Unsucessful download
    if tool_ret.get("error"):
        print("ERR 2")

        new_download = DownloadedFile(
        original_url=url,
        local_path="",
        file_format="Unknown",
        size_bytes= -1,
        success=False,
        error="unable to inspect",
    )
    else:
        size_bytes = tool_ret.get("content_length", 0)
        MAX_SIZE = 524288000 
        
        if size_bytes > MAX_SIZE:
            print("ERR 1")
            new_download = DownloadedFile(
                original_url=url,
                local_path="",
                file_format=tool_ret.get("content_type", "Unknown"),
                size_bytes=size_bytes,
                success=False,
                error="File exceeds 500MB limit"
            )
        else:
        # Sucessful download
            new_download = DownloadedFile(
                original_url=url,
                local_path="",
                file_format=tool_ret.get("content_type", "Unknown"),
                size_bytes=tool_ret.get("content_length", 0),
                success=False if tool_ret.get("error") else True,
                error=tool_ret.get("error", ""),

            )
    return [new_download]

def download_file_handler(state:PipelineState, tool_ret:dict, matched_inspected: DownloadedFile) -> list:
    """This function should add the metadata JSON, as the tool already should have downloaded the file"""
    #Dummy unsuccessful return
    if not tool_ret.get("success"):
        matched_inspected.success = False
        matched_inspected.error = tool_ret.get("error", "Download failed")
        return [matched_inspected]
    # Insert appropriate data
    matched_inspected.size_bytes = tool_ret.get('size_bytes', matched_inspected.size_bytes)
    matched_inspected.file_format = tool_ret.get("file_format", matched_inspected.file_format)
    matched_inspected.local_path = str(OUTPUT_DIR / "downloaded")

    return [matched_inspected]

def get_field(item: Any, field_name: str, default: Any = None) -> Any:
    """Safely extract a field from either a dictionary or an object."""
    if isinstance(item, dict):
        return item.get(field_name, default)
    return getattr(item, field_name, default)

def main():
    from rich.console import Console
    from langgraph.graph import StateGraph, START, END
    from feedback.downloader_feedback import downloader_feedback_node
    console = Console()
    urls_to_test = [
        "https://pscdb.appsbio.utalca.cl/compound_taxonomy_ref.xlsx",
        "https://bidd.group/CMAUP/downloadFiles/CMAUPv2.0_download_Plant_Human_Disease_Associations.txt",
        "https://zenodo.org/records/17902485/files/Databases_with_ADMET.xlsx?download=1"
    ]

    datasets_to_test = [(DiscoveredDataset(url=url, label="label", dataset_type="direct_download", )) for url in urls_to_test]
    initial_state = {
        "url":                  "",
        "topic":                "chemical compounds",
        "pages_visited":        {"data": []},
        "pages_to_visit":       set(),
        "discovered_datasets":  [],
        "datasets_to_download": datasets_to_test,
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
    }

    graph = StateGraph(PipelineState)
    graph.add_node("downloader_node",           downloader_node)
    graph.add_node("downloader_feedback",       downloader_feedback_node)
    graph.add_node("downloader_tools",          downloader_tools_node)

    def tool_routing(state:PipelineState) -> str:
        cur_phase = state.get("current_phase", "")
        if cur_phase == "download_tool":
            return "downloader_tools"
        elif cur_phase == "download_complete":
            return "downloader_feedback"

    def feedback_routing(state:PipelineState) -> str:
        cur_phase = state.get("current_phase", "")
        if cur_phase == "downloader_feedback_complete":
            if len(state.get("datasets_to_retry")) > 0:
                return "downloader_node"
            return END
    
    graph.add_edge(START,                                  "downloader_node")
    graph.add_edge("downloader_tools",                     "downloader_node")
    graph.add_conditional_edges("downloader_node",              tool_routing)
    graph.add_conditional_edges("downloader_feedback",      feedback_routing)

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

"""
agents/orchestrator_agent.py
─────────────────────────────
The planning brain of the pipeline.

After the dataset returns its discoveries, this node:
  1. Filters to only topic-relevant datasets (relevance_score >= 0.5 by default).
  2. Classifies each into "download" vs "parse" buckets.
  3. Deduplicates by URL.
  4. Populates `datasets_to_download` and `datasets_to_parse` in state.

No tool calls here — this is a pure LLM reasoning step.
"""

from __future__ import annotations

import json
import re
import requests
from requests.exceptions import RequestException
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from core.llm import get_llm
from core.state import PipelineState, DiscoveredDataset

RELEVANCE_THRESHOLD = 0.5

ORCHESTRATOR_SYSTEM = """You are a data-pipeline architect. Given a list of
discovered datasets and a research topic, you must:

1. Re-evaluate each dataset's relevance to the topic (0-1 score).
2. Decide the ingestion strategy for each:
   - "download"  → the URL directly serves a file (CSV, JSON, ZIP, XLSX, …)
   - "parse"     → data is embedded in HTML tables, a JSON API, or JS blobs
   - "skip"      → not relevant enough or inaccessible
3. Produce a concise JSON object (no prose) in exactly this format:
IMPORTANT: For dataset_type, you MUST choose exactly ONE of: "parseable_html", "parseable_api", or "parseable_js".
{
  "to_download": [
    {"url": "...", "label": "...", "file_format": "...", "notes": "..."}
  ],
  "to_parse": [
    {"url": "...", "label": "...", "dataset_type": "...", "notes": "..."}
  ],
  "skipped": [
    {"url": "...", "reason": "..."}
  ]
}
"""

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})


def orchestrator_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: classify and route discovered datasets.
    """
    llm = get_llm(temperature=0.0, max_tokens=2048)

    datasets = state.get("discovered_datasets", [])
    # 404 checks
    datasets = [
        dataset for dataset in datasets 
        if is_valid_url(dataset.url)
    ]
    if not datasets:
        return {
            "datasets_to_download": [],
            "datasets_to_parse":    [],
            "current_phase":        "orchestration_complete",
        }
    # Serialize discoveries for the LLM
    ds_json = json.dumps(
        [
            {
                "url":             d.url,
                "label":           d.label,
                "dataset_type":    d.dataset_type,
                "file_format":     d.file_format,
                "size_hint":       d.size_hint,
                "relevance_score": d.relevance_score,
                "notes":           d.notes,
            }
            for d in datasets
        ],
        indent=2,
    )

    user_msg = (
        f"Research topic: {state['topic']}\n\n"
        f"Discovered datasets:\n{ds_json}\n\n"
        "Classify each dataset and output only the JSON object described."
    )

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    response = llm.invoke(messages)
    raw = response.content
    raw = raw.split("</think>")[-1] if "</think>" in raw else raw
    # Robustly parse JSON from LLM output
    to_download: list[DiscoveredDataset] = []
    to_parse: list[DiscoveredDataset]    = []

    try:
        # Find the first '{' and the very last '}'
        start_idx = raw.find('{')
        end_idx = raw.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            # Slice the string directly (O(1) complexity after finding indices)
            json_str = raw[start_idx:end_idx + 1]
            plan = json.loads(json_str)

            for item in plan.get("to_download", []):
                to_download.append(
                    DiscoveredDataset(
                        url=item["url"],
                        label=item.get("label", ""),
                        dataset_type="direct_download",
                        file_format=item.get("file_format"),
                        notes=item.get("notes", ""),
                        relevance_score=1.0,
                    )
                )

            for item in plan.get("to_parse", []):
                to_parse.append(
                    DiscoveredDataset(
                        url=item["url"],
                        label=item.get("label", ""),
                        dataset_type=item.get("dataset_type", "parseable_html"),
                        notes=item.get("notes", ""),
                        relevance_score=1.0,
                    )
                )
    except (json.JSONDecodeError, KeyError):
        # Graceful degradation: pass everything through as downloads
        to_download = [d for d in datasets if d.dataset_type == "direct_download"]
        to_parse    = [d for d in datasets if d.dataset_type != "direct_download"]

    return {
        "datasets_to_download": to_download,
        "datasets_to_parse":    to_parse,
        "current_phase":        "orchestration_complete",
    }


def is_valid_url(url):
    try:
        with session.get(url, allow_redirects=True, timeout=60, stream=True) as response:
            return response.status_code != 404
    except requests.RequestException:
        return False    
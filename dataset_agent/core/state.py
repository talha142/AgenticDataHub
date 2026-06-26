"""
core/state.py
─────────────
Shared TypedDict state that flows through every node in the LangGraph graph.
All agents read from and write to this object; LangGraph merges updates
automatically via the Annotated reducer pattern.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel
from typing_extensions import TypedDict


# ──────────────────────────────────────────────────────────────────────────────
# Sub-types
# ──────────────────────────────────────────────────────────────────────────────

class DiscoveredDataset(BaseModel):
    """A single dataset/file found by the dataset agent."""
    url: str
    label: str                          # human-readable name inferred from context
    dataset_type: Literal[
        "direct_download",              # CSV, TSV, JSON, XLSX, ZIP, HDF5, …
        "parseable_html",               # <table> or structured HTML
        "parseable_api",                # JSON endpoint / REST API
        "parseable_js",                 # data embedded in <script> tags
        "unknown",
    ] = "unknown"
    file_format: Optional[str] = None   # "csv", "json", "xlsx", …
    size_hint: Optional[str] = None     # e.g. "2.3 MB" when shown on page
    relevance_score: float = 0.0        # 0-1, set by dataset LLM
    notes: str = ""                     # extra context from dataset


class DownloadedFile(BaseModel):
    """Result of the downloader agent."""
    original_url: str
    local_path: str
    file_format: str
    size_bytes: int
    success: bool
    error: Optional[str] = None


class ParsedDataset(BaseModel):
    """Result of the parser agent."""
    source_url: str
    local_path: str                     # saved CSV/JSON after parsing
    row_count: int
    column_names: list[str]
    sample_rows: list[dict[str, Any]]   # first 5 rows
    parse_code: str                     # the generated + executed Python code
    success: bool
    error: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline state
# ──────────────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    # ── Inputs ────────────────────────────────────────────────────────────────
    url: str                            # seed URL provided by user
    topic: str                          # e.g. "chemistry compounds"

    # ── Scraper outputs ───────────────────────────────────────────────────────
    pages_visited: list[str]
    pages_to_visit: set[(str, int)]
    # ── Dataset outputs ───────────────────────────────────────────────────────
    discovered_datasets: Annotated[list[DiscoveredDataset], operator.add]

    # ── Orchestrator decision ─────────────────────────────────────────────────
    # Populated after orchestrator classifies discovered datasets
    datasets_to_download: Annotated[list[DiscoveredDataset], operator.add]
    datasets_to_parse: Annotated[list[DiscoveredDataset], operator.add]

    # ── Downloader outputs ────────────────────────────────────────────────────
    downloaded_files: Annotated[list[DownloadedFile], restart_reducer]

    # ── Parser outputs ────────────────────────────────────────────────────────
    parsed_datasets: Annotated[list[ParsedDataset], restart_reducer]
    # ── RAG outputs ───────────────────────────────────────────────────────────
    parsing_schema: dict
    # ── Summary ───────────────────────────────────────────────────────────────
    summary_report: str
    # ── Messages (for tool calling) ───────────────────────────────────────────
    messages: Annotated[list[str], operator.add]
    # ── Control flow ──────────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
    # ── Feedback ──────────────────────────────────────────────────────────────
    downloader_feedback: dict
    parser_feedback: dict
    dataset_feedback: dict
    orchestrator_feedback: dict
    has_retried: bool
    # ── Feedback Related ──────────────────────────────────────────────────────
    parser_should_retry: bool
    dataset_refinement_targets: list
    parser_improvement_hints: str
    datasets_to_retry: list
    learned_lessons: Annotated[list[dict], operator.add]
    feedback_given : int
    inspected_urls: list
    current_to_parse: list
    parsed_links: list[dict]
    #current_phase: str                  # for logging / debugging
    current_phase: Annotated[str, lambda old, new: new]
        

def restart_reducer(existing: list[str], new: dict) -> list[str]:
    if existing is None:
        existing = []

    if isinstance(new, dict):
        pages = new.get("data", [])
        method = new.get("method", "update")
        if method == "clean":
            return []
    elif isinstance(new, list):
        pages = new
    return existing + pages
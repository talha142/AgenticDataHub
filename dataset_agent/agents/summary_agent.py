"""
agents/summary_agent.py
───────────────────────
Final node in the pipeline. Generates a human-readable Markdown report
summarising everything that was collected:
  - What was discovered and where
  - What was downloaded / parsed
  - Schema overviews (columns, row counts, sample data)
  - Data quality notes
  - Suggested next steps
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from core.llm import get_llm
from core.state import PipelineState

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()

SUMMARY_SYSTEM = """You are a data analyst. Write a clear, well-structured
Markdown report summarising a dataset collection run. Include:

1. **Executive summary** – one paragraph, what was found and collected.
2. **Datasets collected** – a table: | Dataset | Source | Format | Rows | Size |
3. **Schema overview** – for each dataset, list column names and inferred data types.
4. **Sample data** – show 3-5 rows per dataset as a Markdown table.
5. **Data quality notes** – missing values, encoding issues, schema inconsistencies.
6. **Suggested next steps** – how a researcher could use this data.

Be concise but thorough. Use Markdown headers and tables.
Do not include any JSON in the output.
"""


def summary_node(state: PipelineState) -> dict[str, Any]:
    """
    LangGraph node: synthesise a Markdown summary report and save it.
    """
    llm = get_llm(temperature=0.2, max_tokens=4096)
    
    # Build context for the LLM
    context_parts = [
        f"## Run metadata",
        f"- URL: {state.get('url', '')}",
        f"- Topic: {state.get('topic', '')}",
        f"- Pages visited: {len(state.get('pages_visited', []))}",
        f"- Timestamp: {datetime.now().isoformat()}",
        "",
        "## Discovered datasets",
        json.dumps([{
                    "url":          d.url,
                    "label":        d.label,
                    "type":         d.dataset_type,
                    "format":       d.file_format,
                    "relevance":    d.relevance_score,
                }
                for d in state.get("discovered_datasets", [])
            ],
            indent=2,
        ),
        "",
        "## Downloaded files",
        json.dumps([{
                    "url":        f.get("original_url", ""),
                    "local_path": f.get("local_path", ""),
                    "format":     f.get("file_format", ""),
                    "size_bytes": f.get("size_bytes", 0),
                    "success":    f.get("success", False),
                    "error":      f.get("error", ""),
                }
                for f in state.get("downloaded_files", [])
            ],
            indent=2,
        ),
        "",
        "## Parsed datasets",
        json.dumps([{
                    "url":          p.get("source_url", ""),
                    "local_path":   p.get("local_path", ""),
                    "row_count":    p.get("row_count", 0),
                    "columns":      p.get("column_names", []),
                    "sample_rows":  p.get("sample_rows", [])[:3], # safely slices
                    "success":      p.get("success", False),
                    "error":        p.get("error", ""),
                }
                for p in state.get("parsed_datasets", [])
            ],
            indent=2,
        ),
        "",
        "## Errors encountered",
        "\n".join(state.get("errors", []) or ["None"]),
    ]
    context = "\n".join(context_parts)

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=f"Generate the dataset collection report.\n\n{context}"),
    ]

    response = llm.invoke(messages)
    report   = response.content
    
    match = re.search(r"(?<=://)[^/?#]+(?:/[^/?#]+)?", state.get("url", ""))
    if match:
        link_to_path = match.group(0).replace("/", ".")
    else:
        link_to_path = ""
    # Save the report to disk
    report_path = OUTPUT_DIR / link_to_path / "dataset_collection_report.md"
    if os.path.exists(OUTPUT_DIR / link_to_path):
        report_path.write_text(report, encoding="utf-8")
    else:
        os.mkdir(OUTPUT_DIR / link_to_path)
        report_path.write_text(report, encoding="utf-8")

    return {
        "summary_report": report,
        "current_phase":  "complete",
    }

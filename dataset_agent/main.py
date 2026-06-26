"""
main.py
───────
Command-line entrypoint for the Dataset Discovery & Collection Agent.

Usage:
    python main.py --url https://pubchem.ncbi.nlm.nih.gov/ --topic "chemistry compounds"
    python main.py --url https://www.ebi.ac.uk/chembl/  --topic "drug discovery"
    python main.py --url https://zenodo.org/            --topic "compound mapping"

Options:
    --url     Seed URL to start scraping from               [required]
    --topic   Research topic to guide dataset discovery     [required]
    --output  Override output directory (default: ./output)
    --verbose Show full LangGraph event stream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import re
from pathlib import Path
import shutil
from datetime import datetime   
import asyncio

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.markdown import Markdown
from rich.table import Table
from rich import print as rprint
from loggingstate.statelog import log_state
from core.state import DownloadedFile, ParsedDataset, DiscoveredDataset


load_dotenv()
console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Agentic Dataset Discovery & Collection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--url",     required=True, help="Seed URL to scrape")
    parser.add_argument("--topic",   required=True, help="Research topic")
    parser.add_argument("--output",  default=None,  help="Output directory path")
    parser.add_argument("--verbose", action="store_true", help="Stream all LangGraph events")
    return parser.parse_args()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]Dataset Discovery & Collection Agent[/bold cyan]\n"
        "[dim]Powered by LangGraph + DeepSeek-R1-32B via vLLM[/dim]",
        border_style="cyan",
    ))


def print_phase(phase: str, detail: str = ""):
    icons = {
        "scraping":       "🔍",
        "orchestrating":  "🧠",
        "downloading":    "⬇️ ",
        "parsing":        "⚙️ ",
        "summarising":    "📊",
        "complete":       "✅",
    }
    icon = next((v for k, v in icons.items() if k in phase.lower()), "▶")
    console.print(f"\n{icon}  [bold]{phase}[/bold]" + (f" — {detail}" if detail else ""))


def print_summary_table(state: dict):
    tbl = Table(title="Collection results", show_lines=True)
    tbl.add_column("Category",    style="bold")
    tbl.add_column("Count",       justify="right")
    tbl.add_column("Details")

    discovered_val = state.get("discovered_datasets", [])
    discovered_raw = discovered_val.get("data", []) if isinstance(discovered_val, dict) else discovered_val

    downloaded_val = state.get("downloaded_files", [])
    downloaded_raw = downloaded_val.get("data", []) if isinstance(downloaded_val, dict) else downloaded_val

    parsed_val = state.get("parsed_datasets", [])
    parsed_raw = parsed_val.get("data", []) if isinstance(parsed_val, dict) else parsed_val

    discovered = [DiscoveredDataset(**d) if isinstance(d, dict) else d for d in discovered_raw]
    downloaded = [DownloadedFile(**f) if isinstance(f, dict) else f for f in downloaded_raw]
    parsed     = [ParsedDataset(**p) if isinstance(p, dict) else p for p in parsed_raw]
    
    errors = state.get("errors", [])
    
    tbl.add_row(
        "Discovered datasets",
        str(len(discovered)),
        ", ".join(d.label[:40] for d in discovered[:5]) or "—",
    )
    tbl.add_row(
        "Downloaded files",
        str(len(downloaded)),
        ", ".join(
            # Using original_url to get the actual file name instead of just 'downloaded'
            f"{Path(f.original_url).name} ({f.size_bytes // 1024}KB)"
            for f in downloaded if f.success
        ) or "—",
    )
    tbl.add_row(
        "Parsed datasets",
        str(len(parsed)),
        ", ".join(
            f"{Path(p.local_path).name} ({p.row_count} rows)"
            for p in parsed if p.success
        ) or "—",
    )
    tbl.add_row(
        "Errors",
        str(len(errors)),
        "; ".join(errors[:3]) or "None",
    )

    console.print(tbl)

async def run_pipeline(url: str, topic: str, output_dir: str | None, verbose: bool):
    # Override output dir if supplied
    if output_dir:
        os.environ["OUTPUT_DIR"] = str(Path(output_dir).resolve())
        Path(os.environ["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # Lazy import so env vars are set first
    from pipeline import compile_pipeline

    pipeline = compile_pipeline()

    initial_state = {
        "url":                  url,
        "topic":                topic,
        "pages_visited":        [],
        "pages_to_visit":       set(),
        "discovered_datasets":  [],
        "datasets_to_download": [],
        "datasets_to_parse":    [],
        "downloaded_files":     [],
        "parsed_datasets":      [],
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
        "inspected_urls": [],
        "current_to_parse": [],
        "parsed_links": [],
        "has_retried": False,

    }

    print_phase("Scraping", url)

    final_state = {}

    if verbose:
        # Stream every event for debugging
        async for event in pipeline.astream(initial_state, stream_mode="updates"):
            node_name = list(event.keys())[0] if event else "unknown"
            console.print(f"  [dim]← {node_name}[/dim]")
            if node_name == "dataset_node":
                print_phase("Orchestrating")
            elif node_name == "orchestrator_node":
                print_phase("Downloading + Parsing (parallel)")
            elif node_name == "summary_node":
                print_phase("Summarising")
            # Accumulate state
            for updates in event.values():
                final_state.update(updates)
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Running pipeline…", total=None)
            async for event in pipeline.astream(initial_state, stream_mode="updates"):
                node_name = list(event.keys())[0] if event else ""
                phase_map = {
                    "dataset_node":      "Scraping pages…",
                    "orchestrator_node": "Orchestrating strategy…",
                    "downloader_node":   "Downloading files…",
                    "parser_node":       "Parsing embedded data…",
                    "summary_node":      "Generating report…",
                }
                if node_name in phase_map:
                    progress.update(task, description=phase_map[node_name])
                for updates in event.values():
                    final_state.update(updates)

    return final_state


async def main():
    print_banner()
    args = parse_args()
    console.print(f"\n[bold]URL:[/bold]   {args.url}")
    console.print(f"[bold]Topic:[/bold] {args.topic}\n")

    start = datetime.now()
    final_state = await run_pipeline(args.url, args.topic, args.output, args.verbose)
    elapsed = (datetime.now() - start).total_seconds()

    console.print(f"\n[green]Pipeline completed in {elapsed:.1f}s[/green]\n")

    # Print results table
    print_summary_table(final_state)

    # Print the Markdown report inline
    report = final_state.get("summary_report", "")
    # Gets rid of message history in JSON
    final_state.pop("messages")
    if report:
        console.print("\n")
        console.print(Markdown(report))

    # Also save a JSON state dump for debugging
    # Each url should have its own file path, for batch running
    match = re.search(r"(?<=://)[^/?#]+(?:/[^/?#]+)?", args.url)
    if match:
        link_to_path = match.group(0).replace("/", ".")
    else:
        link_to_path = ""
    output_dir = Path(os.getenv("OUTPUT_DIR", "./output"))
    target_dir = output_dir / link_to_path
    target_dir.mkdir(parents=True, exist_ok=True)
    state_dump = target_dir / "pipeline_state.json"
    if not os.path.exists(output_dir / link_to_path):
        os.mkdir(output_dir / link_to_path)
    try:
        # Convert dataclasses to dicts for serialisation
        import dataclasses
        serialisable = {
            k: (
                [dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x for x in v]
                if isinstance(v, list) else v
            )
            for k, v in final_state.items()
        }
        state_dump.write_text(json.dumps(serialisable, indent=2, default=str))
        console.print(f"\n[dim]State dump saved to: {state_dump}[/dim]")
    except Exception:
        pass

    report_path = output_dir / "dataset_collection_report.md"
    if report_path.exists():
        console.print(f"[dim]Report saved to:     {report_path}[/dim]")

    console.print(f"[dim]Output directory:   {output_dir}[/dim]\n")


if __name__ == "__main__":
   asyncio.run(main())

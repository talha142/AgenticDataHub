"""
tools/download_tools.py
───────────────────────
Tools used by the downloader agent and the parser agent.

  • download_file         – stream a URL to local disk
  • execute_python_code   – run parser-agent-generated code in a subprocess
  • save_dataframe        – helper: save a pandas DataFrame to CSV
  • list_output_dir       – introspect what has already been saved
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./output")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET_EXTENSIONS = {
    ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".xlsx", ".xls", ".ods",
    ".zip", ".gz", ".tar", ".tar.gz", ".bz2",
    ".hdf5", ".h5", ".parquet", ".feather", ".pkl",
    ".sqlite", ".db", ".sql",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; DatasetAgent/1.0; "
        "+https://github.com/your-org/dataset-agent)"
    )
}

TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))

# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_filename(url: str, content_type: str = "") -> str:
    """Derive a safe local filename from a URL."""
    from urllib.parse import urlparse, unquote
    path = unquote(urlparse(url).path)
    name = Path(path).name or "dataset"
    # strip query params that crept into the name
    name = name.split("?")[0]
    if not Path(name).suffix and "csv" in content_type:
        name += ".csv"
    elif not Path(name).suffix and "json" in content_type:
        name += ".json"
    # make filesystem-safe
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in name)
    return safe or "dataset"

def _clean_text(html: str) -> str:
    """Strip tags, collapse whitespace, keep meaningful text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(separator=" ")).strip()


def _is_dataset_url(href: str) -> bool:
    path = urlparse(href).path.lower()
    return any(path.endswith(ext) for ext in DATASET_EXTENSIONS)


def _absolutise(base: str, href: str) -> str:
    return urljoin(base, href)

# ── tools ─────────────────────────────────────────────────────────────────────

@tool
def download_file(url: str, filename: Optional[str] = None) -> dict:
    """
    Download a file from *url* and save it to the output directory.

    Args:
        url:       Direct download URL.
        filename:  Optional override for the local filename.
                   If omitted, derived from the URL.

    Returns a dict with:
      - local_path:   absolute path where file was saved
      - size_bytes:   file size in bytes
      - file_format:  inferred format (csv, json, zip, …)
      - success:      bool
      - error:        non-empty string on failure
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        output_address = OUTPUT_DIR / "downloaded"
        os.makedirs(output_address, exist_ok=True)
        ct = resp.headers.get("Content-Type", "")
        fname = filename or _safe_filename(url, ct)
        dest  = output_address / fname
        # avoid overwriting — append a counter
        counter = 1
        stem, suffix = dest.stem, dest.suffix
        while dest.exists():
            dest = output_address / f"{stem}_{counter}{suffix}"
            counter += 1
        size = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
                size += len(chunk)

        fmt = dest.suffix.lstrip(".").lower() or "unknown"
        return {
            "local_path": str(dest),
            "size_bytes": size,
            "file_format": fmt,
            "success": True,
            "error": "",
        }
    except Exception as exc:
        print(exc)
        return {
            "local_path": "",
            "size_bytes": 0,
            "file_format": "unknown",
            "success": False,
            "error": str(exc),
        }

@tool
def list_output_dir() -> dict:
    """
    List all files currently saved in the output directory.

    Returns a dict with:
      - files: list of dicts, each with 'name', 'size_bytes', 'path'
      - total_files: int
      - output_dir: absolute path of the output directory
    """
    files = []
    for p in sorted(OUTPUT_DIR.iterdir()):
        if p.is_file():
            files.append({
                "name":       p.name,
                "size_bytes": p.stat().st_size,
                "path":       str(p),
            })
    return {
        "files":       files,
        "total_files": len(files),
        "output_dir":  str(OUTPUT_DIR),
    }

@tool
def inspect_downloadable(url: str) -> dict:
    """
    Send a HEAD request to a URL to check its content-type and size
    without downloading the full file.  Use this to confirm a link
    really points to a downloadable dataset before downloading it.

    Returns:
      - content_type:   e.g. "text/csv; charset=utf-8"
      - content_length: size in bytes, or -1 if unknown
      - is_dataset:     True if content-type looks like structured data
      - error:          non-empty string on failure
    """
    DATASET_MIME_PREFIXES = (
        "text/csv", "text/tab-separated",
        "application/json", "application/vnd.ms-excel",
        "application/vnd.openxmlformats",
        "application/zip", "application/gzip",
        "application/x-hdf", "application/octet-stream",
        "application/x-parquet", "text/plain"
    )
    try:
        resp = requests.head(url, headers=HEADERS, timeout=TIMEOUT, allow_redirects=True)
        ct = resp.headers.get("Content-Type", "")
        cl = int(resp.headers.get("Content-Length", -1))
        is_ds = any(ct.startswith(p) for p in DATASET_MIME_PREFIXES) or _is_dataset_url(url)
        return {"content_type": ct, "content_length": cl, "is_dataset": is_ds, "error": ""}
    except Exception as exc:
        return {"content_type": "", "content_length": -1, "is_dataset": False, "error": str(exc)}

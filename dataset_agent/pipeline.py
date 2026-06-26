"""
pipeline.py
───────────
Assembles and compiles the full LangGraph state machine.

Graph topology:
  START
    └─► dataset_node
          └─► orchestrator_node
                ├─► downloader_node  ──┐
                └─► parser_node      ──┤
                                       └─► summary_node ──► END

Downloader and parser run in parallel when both are needed.
If only one path has work, the other is skipped gracefully.
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from core.state import PipelineState
from core.scraper_node         import scraper_node
from agents.dataset_agent      import dataset_node, dataset_tools_node
from agents.orchestrator_agent import orchestrator_node
from agents.downloader_agent   import downloader_node, downloader_tools_node
from agents.parser_agent       import parser_node
from agents.summary_agent      import summary_node
from agents.rag_agent          import rag_node
from feedback.downloader_feedback import downloader_feedback_node
from feedback.parser_feedback import parser_feedback_node
from feedback.orchestrator_feedback import orchestrator_feedback_node
from feedback.dataset_feedback import dataset_feedback_node


# ── routing functions ─────────────────────────────────────────────────────────

def route_after_orchestrator(state: PipelineState) -> list[str]:
    """
    Decide which worker nodes to run in parallel.
    Returns a list of node names; LangGraph fans out to all of them.
    """
    targets = []
    if state.get("datasets_to_download"):
        targets.append("downloader_node")
    if state.get("datasets_to_parse"):
        targets.append("parser_node")
    # If nothing to do, jump straight to summary
    if not targets:
        targets.append("summary_node")
    return targets


def route_to_summary(state: PipelineState) -> str:
    """After workers finish (or are skipped), always go to summary."""
    return "summary_node"


# ── graph assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # Register nodes
    graph.add_node("dataset_node",          dataset_node)
    graph.add_node("scraper_node",          scraper_node)
    graph.add_node("dataset_tools",         dataset_tools_node)
    graph.add_node("orchestrator_node",     orchestrator_node)
    graph.add_node("downloader_node",       downloader_node)
    graph.add_node("parser_node",           parser_node)
    graph.add_node("summary_node",          summary_node)
    graph.add_node("downloader_tools",      downloader_tools_node)
    graph.add_node("RAG_node",              rag_node)
    graph.add_node("dataset_feedback",      dataset_feedback_node)
    graph.add_node("orchestrator_feedback", orchestrator_feedback_node)
    graph.add_node("downloader_feedback",   downloader_feedback_node)
    graph.add_node("parser_feedback",       parser_feedback_node)

    # Tool return edges
    graph.add_edge("dataset_tools", "dataset_node")
    graph.add_edge("downloader_tools", "downloader_node")

    # Tool call edges
    graph.add_conditional_edges("dataset_node",     tool_routing)
    graph.add_conditional_edges("downloader_node",  tool_routing)

    # Feedback Edgs
    graph.add_edge("orchestrator_node", "orchestrator_feedback")
    graph.add_edge("orchestrator_feedback", "downloader_node")
    graph.add_edge("orchestrator_feedback",     "parser_node")

    graph.add_conditional_edges("downloader_feedback",  feedback_routing)
    graph.add_conditional_edges("parser_feedback",      feedback_routing)
    graph.add_conditional_edges("dataset_feedback",     feedback_routing)

    # Rest edges
    graph.add_edge(START,                          "scraper_node")
    graph.add_edge("scraper_node",                 "dataset_node")
    graph.add_conditional_edges("parser_node",     parser_routing)
    graph.add_edge("RAG_node",                      "parser_node")

    # Both workers converge to summary
    graph.add_edge("summary_node",    END)

    

    return graph


def compile_pipeline():
    """Compile and return the runnable LangGraph pipeline."""
    graph = build_graph()
    return graph.compile()


def parser_routing(state: PipelineState) -> str:
    """Rerouting parser agent to tool or to next node (Summary)"""
    status = state.get("current_phase", "parsing_complete")
    if status == "parsing_complete":
        return "parser_feedback"
    elif status == "rag_assistance":
        return "RAG_node"
    else:
        return "parser_node"
    
def tool_routing(state:PipelineState) -> str:
    cur_phase = state.get("current_phase", "")
    if cur_phase == "scraping_complete":
        return "dataset_feedback"
    elif cur_phase == "scraping_tools":
        return "dataset_tools"
    elif cur_phase == "download_tool":
        return "downloader_tools"
    elif cur_phase == "download_complete":
        return "downloader_feedback"

def feedback_routing(state:PipelineState) -> str:
    cur_phase = state.get("current_phase", "")
    if cur_phase == "dataset_feedback_complete":
        if state.get("scrape_needs_retry", False):
            return "dataset_node"
        return "orchestrator_node"
    elif cur_phase == "parser_feedback_complete":
        if state.get("parser_should_retry", False):
            return "parser_node"
        return "summary_node"
    elif cur_phase == "downloader_feedback_complete":
        if len(state.get("datasets_to_retry")) > 0:
            return "downloader_node"
        return "summary_node"
    elif cur_phase == "parsing":
        return "parser_node"
    
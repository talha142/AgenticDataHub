"""
core/memory.py
──────────────
Persistent memory system for agents to learn from past mistakes.
Stores 'lessons' that are injected into prompts to prevent repeating errors.
"""

import json
import os
from typing import Any, Dict, List

MEMORY_FILE = "knowledge_base.json"

class AgentMemory:
    def __init__(self, file_path: str = MEMORY_FILE):
        self.file_path = file_path
        self.memory: Dict[str, List[Dict[str, Any]]] = {
            "dataset": [],
            "orchestrator": [],
            "downloader": [],
            "parser": []
        }
        self.load()

    def load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r") as f:
                    self.memory.update(json.load(f))
            except (json.JSONDecodeError, IOError):
                pass

    def save(self):
        with open(self.file_path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def add_lesson(self, agent_type: str, lesson: str, context: str = ""):
        """Adds a lesson learned to the memory."""
        if agent_type in self.memory:
            # Check for duplicates to avoid bloat
            if not any(l["lesson"] == lesson for l in self.memory[agent_type]):
                self.memory[agent_type].append({
                    "lesson": lesson,
                    "context": context
                })
                self.save()

    def get_lessons(self, agent_type: str) -> str:
        """Returns a string representation of lessons for the prompt."""
        lessons = self.memory.get(agent_type, [])
        if not lessons:
            return "No previous lessons learned yet."
        
        formatted = ""
        for i, l in enumerate(lessons, 1):
            formatted += f"{i}. {l['lesson']} (Context: {l['context']})\n"
        return formatted

# Global memory instance
memory = AgentMemory()

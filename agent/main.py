"""Entrypoint for the Dallas Urban Heat Emergency Commander ADK Agent."""

from __future__ import annotations

try:
    from .agent import root_agent
except ImportError:
    from agent import root_agent

if __name__ == "__main__":
    print(f"Loaded ADK root agent: {root_agent.name}")
    print("Sub-agents:", [sub.name for sub in root_agent.sub_agents])
    print("Run `adk web --port 8080` from the repository root to launch the ADK UI.")

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a deterministic multi-turn Agent demo")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()

    agent = Agent(args.catalog)
    session_id = "demo_session"
    agent.reset(session_id, {
        "preference_tags": ["comfort", "durability"],
        "summary": "Prior purchases emphasize comfort and durability.",
    })
    messages = [
        "I'm looking for women's hiking boots, but I'm still exploring.",
        "For that, what matters is: leather; waterproof.",
        "Actually, ignore my earlier preference. What I need is: color: black.",
    ]
    for turn, user_message in enumerate(messages, start=1):
        response = agent.respond(session_id, user_message, turn, 10)
        print(json.dumps({
            "turn": turn,
            "user": user_message,
            "agent": response,
            "state_query": agent.sessions[session_id].query_text(),
        }, indent=2))


if __name__ == "__main__":
    main()

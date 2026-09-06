"""Interactive command-line entrypoint for the support agent."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .agent import AgentConfigurationError, DeepSupportAgent
from .knowledge_base import KnowledgeBaseError

EXIT_COMMANDS = {"exit", "quit"}


def extract_final_response(result: dict[str, Any]) -> str:
    """Extract human-readable text from the final assistant message."""
    for message in reversed(result.get("messages", [])):
        if isinstance(message, dict):
            role = message.get("role") or message.get("type")
            content = message.get("content")
        else:
            role = getattr(message, "type", None)
            content = getattr(message, "content", None)
        if role not in {"ai", "assistant"}:
            continue

        text = _content_text(content)
        if text:
            return text
    raise RuntimeError("The agent returned no final response.")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in {
            None,
            "text",
            "output_text",
        }:
            text = block.get("text") or block.get("content")
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(part.strip() for part in parts if part.strip())


def run_cli(
    agent: DeepSupportAgent,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """Run a multi-turn terminal conversation until the user exits."""
    output_fn("L&G Support Assistant. Type 'quit' or 'exit' to stop.")
    messages: list[Any] = []

    while True:
        try:
            user_input = input_fn("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye.")
            return

        if not user_input:
            continue
        if user_input.casefold() in EXIT_COMMANDS:
            output_fn("Goodbye.")
            return

        messages.append({"role": "user", "content": user_input})
        try:
            result = agent.invoke(messages)
            response = extract_final_response(result)
        except Exception as exc:
            messages.pop()
            output_fn(f"Support assistant error: {exc}")
            continue

        returned_messages = result.get("messages")
        if isinstance(returned_messages, list):
            messages = list(returned_messages)
        output_fn(f"Support: {response}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chat with the knowledge-base-grounded L&G support agent."
    )
    parser.add_argument(
        "--database-path",
        type=Path,
        default=None,
        help="Directory containing the LanceDB knowledge base.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        agent = DeepSupportAgent(args.database_path)
    except (AgentConfigurationError, KnowledgeBaseError) as exc:
        raise SystemExit(f"Unable to start support assistant: {exc}") from exc
    run_cli(agent)

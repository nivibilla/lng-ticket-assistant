"""Local Teams-style demo server. Model history never crosses the HTTP boundary."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import (
    WEB_SYSTEM_PROMPT,
    AgentConfigurationError,
    DeepSupportAgent,
    completed_response,
)
from .citations import SourceStore, resolve_citations, retrieved_evidence
from .startup import startup_check

LOGGER = logging.getLogger("lng_ticket_assistant.web")
COOKIE = "lng_demo_session"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Conversation:
    history: list[Any] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    generation: int = 0


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    retry_message_id: str | None = None


def create_app(
    *,
    agent: Any | None = None,
    agent_factory: Callable[[], Any] | None = None,
    source_dir: Path | None = None,
    static_dir: Path | None = None,
    turn_timeout: float = 300,
) -> FastAPI:
    sessions: dict[str, Conversation] = {}
    sources = SourceStore(source_dir or PROJECT_ROOT)
    ready = False

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal agent, ready
        started = perf_counter()
        with startup_check(
            "agent initialization (Google configuration and knowledge base)"
        ):
            if agent is None:
                factory = agent_factory or (
                    lambda: DeepSupportAgent(system_prompt=WEB_SYSTEM_PROMPT)
                )
                agent = await asyncio.to_thread(factory)
        with startup_check("assistant warmup"):
            await agent.warmup()
        ready = True
        LOGGER.info("Assistant ready (%.1fs startup).", perf_counter() - started)
        try:
            yield
        finally:
            ready = False
            tasks = [s.task for s in sessions.values() if s.task and not s.task.done()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    app = FastAPI(title="L&G Support Assistant", lifespan=lifespan)

    @app.middleware("http")
    async def require_ready(request: Request, call_next: Any) -> Response:
        if not ready:
            return JSONResponse(
                {
                    "detail": "The assistant is not ready. Check the server startup logs."
                },
                status_code=503,
            )
        return await call_next(request)

    def session(request: Request, response: Response) -> Conversation:
        session_id = request.cookies.get(COOKIE)
        if session_id not in sessions:
            session_id = secrets.token_urlsafe(32)
            sessions[session_id] = Conversation()
            response.set_cookie(COOKIE, session_id, httponly=True, samesite="strict")
        return sessions[session_id]

    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "This action must come from the demo app.")

    @app.get("/api/conversation")
    async def get_conversation(request: Request, response: Response) -> dict[str, Any]:
        conversation = session(request, response)
        return {
            "messages": conversation.messages,
            "busy": bool(conversation.task and not conversation.task.done()),
        }

    @app.delete("/api/conversation")
    async def reset_conversation(
        request: Request, response: Response
    ) -> dict[str, Any]:
        check_origin(request)
        conversation = session(request, response)
        conversation.generation += 1
        if conversation.task:
            conversation.task.cancel()
        conversation.history = []
        conversation.messages = []
        conversation.task = None
        return {"messages": [], "busy": False}

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request) -> Response:
        check_origin(request)
        headers = Response()
        conversation = session(request, headers)
        if conversation.task and not conversation.task.done():
            raise HTTPException(409, "Wait for the current reply or stop it first.")
        clean_message = body.message.strip()
        if not clean_message:
            raise HTTPException(422, "Enter a message to send.")
        if body.retry_message_id:
            user_message = next(
                (m for m in conversation.messages if m["id"] == body.retry_message_id),
                None,
            )
            if (
                not user_message
                or user_message["role"] != "user"
                or user_message["status"] != "failed"
                or user_message["content"] != clean_message
                or conversation.messages[-1] is not user_message
            ):
                raise HTTPException(
                    409, "Only the latest interrupted message can be retried."
                )
            user_message["status"] = "pending"
            user_message.pop("error", None)
        else:
            user_message = {
                "id": secrets.token_hex(12),
                "role": "user",
                "content": clean_message,
                "created_at": timestamp(),
                "status": "pending",
                "citations": [],
            }
            conversation.messages.append(user_message)

        generation = conversation.generation
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def produce() -> None:
            try:
                queue.put_nowait({"type": "accepted", "message": user_message.copy()})
                queue.put_nowait({"type": "status", "phase": "thinking"})
                async with asyncio.timeout(turn_timeout):
                    history = [
                        *conversation.history,
                        {"role": "user", "content": clean_message},
                    ]
                    final_state = None
                    async for event in agent.astream(history):
                        if generation != conversation.generation:
                            return
                        if event["type"] == "result":
                            final_state = event["state"]
                        elif event["type"] in {"status", "message_start", "delta"}:
                            queue.put_nowait(event)
                    returned_history, text = completed_response(final_state)
                    citations = resolve_citations(
                        text, retrieved_evidence(returned_history), sources
                    )
                    answer = {
                        "id": secrets.token_hex(12),
                        "role": "assistant",
                        "content": text,
                        "created_at": timestamp(),
                        "status": "complete",
                        "citations": citations,
                    }
                    if generation != conversation.generation:
                        return
                    conversation.history = returned_history
                    user_message["status"] = "complete"
                    conversation.messages.append(answer)
                    queue.put_nowait(
                        {
                            "type": "complete",
                            "message": answer,
                            "user_message_id": user_message["id"],
                        }
                    )
            except asyncio.CancelledError:
                if generation == conversation.generation:
                    user_message.update(
                        status="failed", error="Reply stopped. You can try again."
                    )
                raise
            except Exception as exc:  # noqa: BLE001 - sanitize failures at the HTTP boundary
                # Never return provider errors, credentials, or filesystem paths to the client.
                LOGGER.warning("Support turn failed (%s)", type(exc).__name__)
                if isinstance(exc, AgentConfigurationError):
                    message = "The assistant needs a Google API key. Add GOOGLE_API_KEY to the server’s .env file and try again."
                elif isinstance(exc, TimeoutError):
                    message = "This reply took too long. Please try again."
                else:
                    message = (
                        "The assistant couldn’t finish this reply. Please try again."
                    )
                if generation == conversation.generation:
                    user_message.update(status="failed", error=message)
                    queue.put_nowait(
                        {
                            "type": "error",
                            "message": message,
                            "user_message_id": user_message["id"],
                        }
                    )
            finally:
                queue.put_nowait(None)

        task = asyncio.create_task(produce())
        conversation.task = task

        async def stream() -> AsyncIterator[str]:
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if event is None:
                        break
                    yield f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            finally:
                if not task.done():
                    task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                if conversation.task is task:
                    conversation.task = None

        response = StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        for name, value in headers.raw_headers:
            if name == b"set-cookie":
                response.raw_headers.append((name, value))
        return response

    @app.get("/api/sources/{source_id}")
    async def source(source_id: str, request: Request) -> Response:
        conversation = sessions.get(request.cookies.get(COOKIE, ""))
        if not conversation or not any(
            citation["source_id"] == source_id
            for message in conversation.messages
            for citation in message.get("citations", [])
        ):
            raise HTTPException(404, "Source not found.")
        path = sources.path(source_id)
        if path is None:
            raise HTTPException(404, "The original PDF is unavailable.")
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=path.name,
            content_disposition_type="inline",
            headers={"Cache-Control": "private, no-store"},
        )

    @app.api_route("/api/{path:path}", methods=["GET", "POST", "DELETE"])
    async def unknown_api(path: str) -> Response:
        return JSONResponse({"detail": "Not found."}, status_code=404)

    frontend = static_dir or PROJECT_ROOT / "frontend" / "dist"
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:

        @app.get("/")
        async def no_frontend() -> Response:
            return JSONResponse(
                {
                    "detail": "Build the frontend with: npm --prefix frontend ci && npm --prefix frontend run build"
                },
                status_code=503,
            )

    return app


def main(argv: Sequence[str] | None = None) -> None:
    import uvicorn

    logging.basicConfig(format="%(levelname)s: %(message)s")
    logging.getLogger("lng_ticket_assistant").setLevel(logging.INFO)
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(
        description="Run the standalone Teams-style support demo."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--database-path", type=Path, default=None)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(os.getenv("KB_SOURCE_DIR", str(PROJECT_ROOT))),
    )
    args = parser.parse_args(argv)
    app = create_app(
        agent_factory=lambda: DeepSupportAgent(
            args.database_path, system_prompt=WEB_SYSTEM_PROMPT
        ),
        source_dir=args.source_dir,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

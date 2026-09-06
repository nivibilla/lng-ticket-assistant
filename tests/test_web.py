from __future__ import annotations

import asyncio
import json
import os
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from lng_ticket_assistant.agent import WEB_SYSTEM_PROMPT, DeepSupportAgent, public_text
from lng_ticket_assistant.citations import (
    SourceStore,
    resolve_citations,
    retrieved_evidence,
)
from lng_ticket_assistant.startup import StartupError
from lng_ticket_assistant.web import create_app

EVIDENCE = {
    "evidence_id": "chunk-7",
    "chunk_id": 7,
    "source": "KB Articles.pdf",
    "page_start": 3,
    "page_end": 4,
    "content": "Reconnect the VPN client.",
}


class FakeAgent:
    def __init__(self) -> None:
        self.calls = []
        self.fail = False
        self.gate: asyncio.Event | None = None
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.warmup = AsyncMock()

    async def astream(self, messages):
        self.calls.append(messages)
        self.started.set()
        yield {"type": "status", "phase": "searching"}
        if self.gate:
            try:
                await self.gate.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
        if self.fail:
            raise RuntimeError("DO NOT SEND private-api-key /private/local/path")
        yield {"type": "message_start", "message_id": "answer"}
        yield {"type": "delta", "message_id": "answer", "text": "Reconnect "}
        yield {
            "type": "delta",
            "message_id": "answer",
            "text": "the client. [[cite:chunk-7]]",
        }
        yield {
            "type": "result",
            "state": {
                "messages": [
                    *messages,
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "search_knowledge_base",
                                "args": {"query": "VPN"},
                                "id": "call1",
                            }
                        ],
                    ),
                    ToolMessage(
                        content=json.dumps({"results": [EVIDENCE]}),
                        name="search_knowledge_base",
                        tool_call_id="call1",
                    ),
                    AIMessage(
                        content=[
                            {"type": "thinking", "thinking": "private reasoning"},
                            {
                                "type": "text",
                                "text": "Reconnect the client. [[cite:chunk-7]] [[cite:chunk-999]]",
                            },
                        ]
                    ),
                ]
            },
        }


def events(response):
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "KB Articles.pdf").write_bytes(b"%PDF-1.4\nfixture")
        self.agent = FakeAgent()
        self.app = create_app(
            agent=self.agent, source_dir=self.root, static_dir=self.root / "missing"
        )
        await self.enterAsyncContext(self.app.router.lifespan_context(self.app))
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )
        await self.client.get("/api/conversation")

    async def asyncTearDown(self):
        await self.client.aclose()
        self.directory.cleanup()

    async def test_stream_and_citations_keep_internal_history_private(self):
        response = await self.client.post("/api/chat", json={"message": "VPN fails"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        output = events(response)
        self.assertEqual([e["type"] for e in output][-1], "complete")
        answer = output[-1]["message"]
        self.assertEqual(len(answer["citations"]), 1)
        self.assertEqual(answer["citations"][0]["content"], EVIDENCE["content"])
        self.assertNotIn("private reasoning", response.text)
        self.assertNotIn("tool_call", response.text)
        state = (await self.client.get("/api/conversation")).json()
        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["messages"][0]["status"], "complete")
        self.assertNotIn("private reasoning", json.dumps(state))
        await self.client.post("/api/chat", json={"message": "It still fails"})
        self.assertEqual(len(self.agent.calls[1]), 5)
        self.assertIsInstance(self.agent.calls[1][2], ToolMessage)

    async def test_sessions_are_isolated_and_sources_require_retrieval(self):
        answer = events(
            await self.client.post("/api/chat", json={"message": "VPN fails"})
        )[-1]["message"]
        url = answer["citations"][0]["pdf_url"]
        pdf = await self.client.get(url)
        self.assertEqual(pdf.status_code, 200)
        self.assertTrue(pdf.content.startswith(b"%PDF"))
        self.assertIn("inline", pdf.headers["content-disposition"])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        ) as other:
            self.assertEqual(
                (await other.get("/api/conversation")).json()["messages"], []
            )
            self.assertEqual((await other.get(url)).status_code, 404)
        self.assertEqual(
            (await self.client.get("/api/sources/unknown")).status_code, 404
        )

    async def test_error_can_retry_without_duplicate_model_history(self):
        self.agent.fail = True
        response = await self.client.post("/api/chat", json={"message": "VPN fails"})
        self.assertNotIn("private-api-key", response.text)
        self.assertNotIn("/private/local/path", response.text)
        self.assertEqual(events(response)[-1]["type"], "error")
        state = (await self.client.get("/api/conversation")).json()
        user = state["messages"][0]
        self.assertEqual(user["status"], "failed")
        self.agent.fail = False
        await self.client.post(
            "/api/chat", json={"message": "VPN fails", "retry_message_id": user["id"]}
        )
        self.assertEqual(len(self.agent.calls[1]), 1)
        self.assertEqual(
            len((await self.client.get("/api/conversation")).json()["messages"]), 2
        )

    async def test_invalid_retry_and_empty_message_rejected(self):
        self.assertEqual(
            (await self.client.post("/api/chat", json={"message": "  "})).status_code,
            422,
        )
        self.assertEqual(
            (
                await self.client.post(
                    "/api/chat",
                    json={"message": "hello", "retry_message_id": "missing"},
                )
            ).status_code,
            409,
        )

    async def test_reset_cancels_pending_turn_and_discards_late_results(self):
        self.agent.gate = asyncio.Event()
        pending = asyncio.create_task(
            self.client.post("/api/chat", json={"message": "VPN fails"})
        )
        await self.agent.started.wait()
        self.assertTrue((await self.client.get("/api/conversation")).json()["busy"])
        self.assertEqual(
            (
                await self.client.post("/api/chat", json={"message": "second"})
            ).status_code,
            409,
        )
        await self.client.delete("/api/conversation")
        await asyncio.wait_for(self.agent.cancelled.wait(), 2)
        await pending
        self.assertEqual(
            (await self.client.get("/api/conversation")).json(),
            {"messages": [], "busy": False},
        )
        self.agent.gate = None
        await self.client.post("/api/chat", json={"message": "new chat"})
        self.assertEqual(len(self.agent.calls[-1]), 1)

    async def test_timeout_is_retryable(self):
        self.agent.gate = asyncio.Event()
        app = create_app(agent=self.agent, source_dir=self.root, turn_timeout=0.01)
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            output = events(
                await client.post("/api/chat", json={"message": "VPN fails"})
            )
            self.assertIn("too long", output[-1]["message"])
            self.assertEqual(
                (await client.get("/api/conversation")).json()["messages"][0]["status"],
                "failed",
            )

    async def test_missing_pdf_keeps_passage_and_blocks_traversal(self):
        (self.root / "KB Articles.pdf").unlink()
        answer = events(
            await self.client.post("/api/chat", json={"message": "VPN fails"})
        )[-1]["message"]
        self.assertIsNone(answer["citations"][0]["pdf_url"])
        self.assertEqual(answer["citations"][0]["content"], EVIDENCE["content"])
        store = SourceStore(self.root)
        for path in ("../private.pdf", "/etc/passwd", ".env", "nested/file.pdf"):
            self.assertIsNone(store.describe(path)["pdf_url"])
        with tempfile.TemporaryDirectory() as other:
            secret = Path(other) / "secret.pdf"
            secret.write_bytes(b"private")
            (self.root / "link.pdf").symlink_to(secret)
            self.assertIsNone(store.describe("link.pdf")["pdf_url"])

    async def test_cross_origin_mutations_rejected(self):
        self.assertEqual(
            (
                await self.client.post(
                    "/api/chat",
                    json={"message": "hi"},
                    headers={"Origin": "https://elsewhere.example"},
                )
            ).status_code,
            403,
        )
        self.assertEqual(
            (
                await self.client.delete(
                    "/api/conversation", headers={"Origin": "https://elsewhere.example"}
                )
            ).status_code,
            403,
        )


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_requests_initialize_or_bypass_an_unstarted_app(self):
        factory = Mock()
        app = create_app(agent_factory=factory)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            for method, path, kwargs in [
                ("GET", "/", {}),
                ("GET", "/api/conversation", {}),
                ("POST", "/api/chat", {"json": {"message": "VPN fails"}}),
                ("DELETE", "/api/conversation", {}),
                ("GET", "/api/sources/unknown", {}),
            ]:
                with self.subTest(path=path, method=method):
                    response = await client.request(method, path, **kwargs)
                    self.assertEqual(response.status_code, 503)
                    self.assertNotIn("set-cookie", response.headers)
        factory.assert_not_called()

    async def test_startup_waits_then_reuses_warmed_agent_for_isolated_chats(self):
        agent = FakeAgent()
        started = asyncio.Event()
        release = asyncio.Event()

        async def warmup():
            started.set()
            await release.wait()

        agent.warmup.side_effect = warmup
        factory = Mock(return_value=agent)
        app = create_app(agent_factory=factory)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            lifespan = app.router.lifespan_context(app)
            startup = asyncio.create_task(lifespan.__aenter__())
            try:
                await asyncio.wait_for(started.wait(), 2)
                self.assertFalse(startup.done())
                self.assertEqual(
                    (await client.get("/api/conversation")).status_code, 503
                )
                self.assertEqual(
                    (
                        await client.post("/api/chat", json={"message": "too early"})
                    ).status_code,
                    503,
                )
            finally:
                release.set()
                await asyncio.wait_for(startup, 2)
                self.addAsyncCleanup(lifespan.__aexit__, None, None, None)

            self.assertEqual(
                (await client.get("/api/conversation")).json(),
                {"messages": [], "busy": False},
            )
            for _ in range(2):
                response = await client.post("/api/chat", json={"message": "VPN fails"})
                self.assertEqual(events(response)[-1]["type"], "complete")
                await client.delete("/api/conversation")
            factory.assert_called_once_with()
            agent.warmup.assert_awaited_once_with()
            self.assertEqual(
                agent.calls, [[{"role": "user", "content": "VPN fails"}]] * 2
            )

    async def test_shutdown_makes_app_unavailable(self):
        agent = FakeAgent()
        app = create_app(agent=agent)
        async with app.router.lifespan_context(app):
            pass
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            self.assertEqual((await client.get("/api/conversation")).status_code, 503)
        agent.warmup.assert_awaited_once_with()

    async def test_missing_configuration_fails_before_readiness(self):
        app = create_app()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("lng_ticket_assistant.agent.load_dotenv"),
            self.assertLogs("lng_ticket_assistant", level="INFO") as logs,
            self.assertRaisesRegex(
                StartupError, "agent initialization.*AgentConfigurationError"
            ),
        ):
            async with app.router.lifespan_context(app):
                self.fail("An unconfigured app became ready")
        self.assertNotIn("Assistant ready", "\n".join(logs.output))

    async def test_failed_startup_stays_unavailable_and_sanitizes_diagnostics(self):
        for failure_stage in ("initialization", "warmup"):
            with self.subTest(stage=failure_stage):
                agent = FakeAgent()
                failure = RuntimeError("private-api-key /private/local/path")
                factory = Mock(return_value=agent)
                if failure_stage == "initialization":
                    factory.side_effect = failure
                else:
                    agent.warmup.side_effect = failure
                app = create_app(agent_factory=factory)
                with self.assertLogs("lng_ticket_assistant", level="INFO") as logs:
                    try:
                        async with app.router.lifespan_context(app):
                            self.fail("A failed app became ready")
                    except StartupError as exc:
                        diagnostic = "".join(traceback.format_exception(exc))
                diagnostic += "\n".join(logs.output)
                self.assertIn(failure_stage, diagnostic)
                self.assertNotIn("private-api-key", diagnostic)
                self.assertNotIn("/private/local/path", diagnostic)
                self.assertNotIn("Assistant ready", diagnostic)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client:
                    self.assertEqual(
                        (
                            await client.post("/api/chat", json={"message": "hello"})
                        ).status_code,
                        503,
                    )
                factory.assert_called_once_with()


class CitationTests(unittest.TestCase):
    def test_only_search_tool_evidence_is_trusted_and_duplicates_deduplicate(self):
        payload = json.dumps({"results": [EVIDENCE]})
        messages = [
            {"role": "user", "content": payload},
            {"role": "assistant", "content": payload},
        ]
        self.assertEqual(retrieved_evidence(messages), {})
        messages.append(
            ToolMessage(
                content=payload, name="search_knowledge_base", tool_call_id="call1"
            )
        )
        evidence = retrieved_evidence(messages)
        with tempfile.TemporaryDirectory() as directory:
            sources = resolve_citations(
                "[[cite:chunk-7]] again [[cite:chunk-7]] [[cite:chunk-999]]",
                evidence,
                SourceStore(Path(directory)),
            )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["number"], 1)

    def test_public_text_retains_spaces_and_drops_reasoning(self):
        self.assertEqual(
            public_text(
                [
                    {"type": "thinking", "thinking": "private"},
                    {"type": "text", "text": " a "},
                    {"type": "text", "text": " b "},
                ]
            ),
            " a  b ",
        )
        self.assertIn("[[cite:chunk-7]]", WEB_SYSTEM_PROMPT)
        self.assertIn("P1 and P2", WEB_SYSTEM_PROMPT)


class StreamAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_root_text_only_and_final_state_preserved(self):
        final = {"messages": [AIMessage(content="Good answer")]}

        async def stream(*args, **kwargs):
            for chunk, node, ns in [
                (
                    AIMessageChunk(content="subagent secret", id="sub"),
                    "model",
                    ("subagent",),
                ),
                (
                    AIMessageChunk(content="internal summary", id="summary"),
                    "summarize",
                    (),
                ),
                (
                    AIMessageChunk(
                        content=[{"type": "thinking", "thinking": "hidden"}], id="a"
                    ),
                    "model",
                    (),
                ),
                (AIMessageChunk(content="Good ", id="a"), "model", ()),
                (AIMessageChunk(content="answer", id="a"), "model", ()),
            ]:
                yield {
                    "type": "messages",
                    "ns": ns,
                    "data": (chunk, {"langgraph_node": node}),
                }
            yield {"type": "values", "ns": (), "data": final}

        with patch("lng_ticket_assistant.agent.create_deep_agent") as create:
            create.return_value.astream = stream
            agent = DeepSupportAgent(model=Mock(), tools=[])
        output = [event async for event in agent.astream([])]
        self.assertEqual("".join(e.get("text", "") for e in output), "Good answer")
        self.assertIs(output[-1]["state"], final)


if __name__ == "__main__":
    unittest.main()

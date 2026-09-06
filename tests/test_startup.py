from __future__ import annotations

import asyncio
import json
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

import httpx
import lancedb
import numpy as np
import pyarrow as pa
from langchain_core.messages import AIMessage, AIMessageChunk

from lng_ticket_assistant.agent import DeepSupportAgent
from lng_ticket_assistant.embeddings import (
    EMBEDDING_DIMENSION,
    EmbeddingError,
    JinaRetrievalEmbedder,
)
from lng_ticket_assistant.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseError,
    KnowledgeBaseMatch,
)
from lng_ticket_assistant.rebuild_embeddings import KNOWLEDGE_BASE_SCHEMA
from lng_ticket_assistant.startup import StartupError
from lng_ticket_assistant.tools import make_search_knowledge_base_tool
from lng_ticket_assistant.web import create_app


class WarmupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.knowledge_base = Mock()
        self.knowledge_base.search.return_value = [
            KnowledgeBaseMatch(
                chunk_id=1,
                source="KB Articles.pdf",
                page_start=1,
                page_end=1,
                text="Reconnect the VPN client.",
            )
        ]
        self.search = make_search_knowledge_base_tool(
            knowledge_base=self.knowledge_base
        )
        self.stream_calls = []

        async def stream(state, **kwargs):
            self.stream_calls.append(state)
            yield {
                "type": "messages",
                "data": (
                    AIMessageChunk(content="Hello!", id="greeting"),
                    {"langgraph_node": "model"},
                ),
            }
            yield {
                "type": "values",
                "data": {"messages": [*state["messages"], AIMessage(content="Hello!")]},
            }

        with patch("lng_ticket_assistant.agent.create_deep_agent") as create:
            create.return_value.astream = stream
            self.agent = DeepSupportAgent(model=Mock(), tools=[self.search])

    async def test_warmup_uses_bound_tool_and_stream_but_keeps_browser_history_empty(
        self,
    ):
        app = create_app(agent=self.agent)
        with self.assertLogs("lng_ticket_assistant", level="INFO") as logs:
            async with (
                app.router.lifespan_context(app),
                httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test"
                ) as client,
            ):
                self.knowledge_base.search.assert_called_once_with(
                    "VPN connection", k=1
                )
                self.assertEqual(len(self.stream_calls), 1)
                self.assertEqual(
                    (await client.get("/api/conversation")).json(),
                    {"messages": [], "busy": False},
                )
                await client.post("/api/chat", json={"message": "VPN fails"})
                self.assertEqual(
                    self.stream_calls[1],
                    {"messages": [{"role": "user", "content": "VPN fails"}]},
                )
        log_text = "\n".join(logs.output)
        for stage in (
            "agent initialization",
            "Jina embeddings",
            "Gemini streaming",
            "Assistant ready",
        ):
            self.assertIn(stage, log_text)
        self.assertIn("passed (", log_text)

    async def test_local_hybrid_search_loads_embedding_model_once_and_reuses_it(self):
        model = Mock()
        model.encode.return_value = np.ones((1, EMBEDDING_DIMENSION), dtype=np.float32)
        embedder = JinaRetrievalEmbedder()
        with tempfile.TemporaryDirectory() as directory:
            database = lancedb.connect(directory)
            database.create_table(
                "knowledge_base",
                data=pa.Table.from_pylist(
                    [
                        {
                            "chunk_id": 1,
                            "source": "KB Articles.pdf",
                            "page_start": 1,
                            "page_end": 1,
                            "text": "Reconnect the VPN client.",
                            "n_chars": 25,
                            "embedding": [1.0] * EMBEDDING_DIMENSION,
                        }
                    ],
                    schema=KNOWLEDGE_BASE_SCHEMA,
                ),
            )
            knowledge_base = KnowledgeBase(directory, embedder=embedder)
            search = make_search_knowledge_base_tool(knowledge_base=knowledge_base)
            self.agent.tools = [search]
            with patch.object(embedder, "_load_model", return_value=model) as load:
                await self.agent.warmup()
                payload = json.loads(await asyncio.to_thread(search, "VPN fails"))
                self.assertEqual(payload["results"][0]["chunk_id"], 1)
            load.assert_called_once_with()
            self.assertEqual(model.encode.call_count, 2)
            self.assertIs(embedder.model, model)

    async def test_embedding_and_tool_returned_errors_fail_before_gemini(self):
        for failure in (
            EmbeddingError("private-api-key"),
            KnowledgeBaseError("private-api-key"),
        ):
            with self.subTest(failure=type(failure).__name__):
                self.knowledge_base.search.side_effect = failure
                with self.assertLogs("lng_ticket_assistant", level="ERROR") as logs:
                    try:
                        await self.agent.warmup()
                        self.fail("A failed search passed warmup")
                    except StartupError as exc:
                        diagnostic = "".join(traceback.format_exception(exc))
                self.assertIn("knowledge-base retrieval", diagnostic)
                self.assertNotIn("private-api-key", diagnostic + "\n".join(logs.output))
        self.assertEqual(self.stream_calls, [])

    async def test_empty_or_malformed_retrieval_results_fail_before_gemini(self):
        for payload in (
            "not JSON",
            "[]",
            '{"error":"private-api-key"}',
            '{"results":[],"result_count":0}',
            '{"results":[{}],"result_count":1}',
            '{"results":[{}],"result_count":0}',
        ):
            with self.subTest(payload=payload):
                tool = Mock(return_value=payload, __name__="search_knowledge_base")
                self.agent.tools = [tool]
                with (
                    self.assertLogs("lng_ticket_assistant", level="ERROR"),
                    self.assertRaisesRegex(StartupError, "knowledge-base retrieval"),
                ):
                    await self.agent.warmup()
        self.assertEqual(self.stream_calls, [])

    async def test_gemini_failure_is_sanitized(self):
        async def stream(*args, **kwargs):
            raise RuntimeError("private-api-key /private/local/path")
            yield

        self.agent.runner.astream = stream
        with self.assertLogs("lng_ticket_assistant", level="ERROR") as logs:
            try:
                await self.agent.warmup()
                self.fail("A failed Gemini call passed warmup")
            except StartupError as exc:
                diagnostic = "".join(traceback.format_exception(exc))
        self.assertIn("Gemini streaming smoke test", diagnostic)
        self.assertNotIn("private-api-key", diagnostic + "\n".join(logs.output))
        self.assertNotIn("/private/local/path", diagnostic)

    async def test_gemini_timeout_cancels_stream(self):
        cancelled = asyncio.Event()

        async def stream(*args, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            yield

        self.agent.runner.astream = stream
        with (
            self.assertLogs("lng_ticket_assistant", level="ERROR"),
            self.assertRaisesRegex(
                StartupError, "Gemini streaming smoke test.*TimeoutError"
            ),
        ):
            await self.agent.warmup(gemini_timeout=0.01)
        self.assertTrue(cancelled.is_set())

    async def test_empty_or_unfinished_gemini_answers_fail(self):
        for state in (
            None,
            {"messages": []},
            {"messages": [AIMessage(content=" ")]},
            {"messages": [{"role": "user", "content": "Hello"}]},
            {
                "messages": [
                    AIMessage(
                        content="Searching",
                        tool_calls=[
                            {"name": "search_knowledge_base", "args": {}, "id": "call1"}
                        ],
                    )
                ]
            },
        ):
            with self.subTest(state=state):

                async def stream(*args, response_state=state, **kwargs):
                    yield {"type": "values", "data": response_state}

                self.agent.runner.astream = stream
                with (
                    self.assertLogs("lng_ticket_assistant", level="ERROR"),
                    self.assertRaisesRegex(StartupError, "Gemini streaming smoke test"),
                ):
                    await self.agent.warmup()

    async def test_final_answer_without_visible_stream_fails(self):
        async def stream(*args, **kwargs):
            yield {
                "type": "values",
                "data": {"messages": [AIMessage(content="Hello!")]},
            }

        self.agent.runner.astream = stream
        with (
            self.assertLogs("lng_ticket_assistant", level="ERROR"),
            self.assertRaisesRegex(StartupError, "Gemini streaming smoke test"),
        ):
            await self.agent.warmup()


if __name__ == "__main__":
    unittest.main()

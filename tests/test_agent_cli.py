from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lng_ticket_assistant.agent import (
    DEFAULT_GOOGLE_MODEL_ID,
    SYSTEM_PROMPT,
    AgentConfigurationError,
    DeepSupportAgent,
    build_model,
)
from lng_ticket_assistant.main import extract_final_response, run_cli


class ModelConfigurationTests(unittest.TestCase):
    def test_builds_requested_gemini_model(self) -> None:
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}, clear=True):
            with (
                patch("lng_ticket_assistant.agent.load_dotenv") as load_env,
                patch(
                    "lng_ticket_assistant.agent.ChatGoogleGenerativeAI"
                ) as model_class,
            ):
                model = build_model()

        self.assertIs(model, model_class.return_value)
        load_env.assert_called_once_with(
            dotenv_path=Path.cwd() / ".env",
            override=False,
        )
        model_class.assert_called_once_with(
            model=DEFAULT_GOOGLE_MODEL_ID,
            api_key="test-key",
            temperature=1.0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def test_requires_google_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("lng_ticket_assistant.agent.load_dotenv"):
                with self.assertRaises(AgentConfigurationError):
                    build_model()

    def test_loads_api_key_from_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "GOOGLE_API_KEY=dotenv-key\n",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {}, clear=True),
                patch(
                    "lng_ticket_assistant.agent.Path.cwd",
                    return_value=Path(directory),
                ),
                patch(
                    "lng_ticket_assistant.agent.ChatGoogleGenerativeAI"
                ) as model_class,
            ):
                build_model()

        self.assertEqual(
            model_class.call_args.kwargs["api_key"],
            "dotenv-key",
        )

    def test_environment_overrides_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "GOOGLE_API_KEY=dotenv-key\n"
                "GOOGLE_MODEL_ID=dotenv-model\n",
                encoding="utf-8",
            )
            with (
                patch.dict(
                    os.environ,
                    {
                        "GOOGLE_API_KEY": "environment-key",
                        "GOOGLE_MODEL_ID": "environment-model",
                    },
                    clear=True,
                ),
                patch(
                    "lng_ticket_assistant.agent.Path.cwd",
                    return_value=Path(directory),
                ),
                patch(
                    "lng_ticket_assistant.agent.ChatGoogleGenerativeAI"
                ) as model_class,
            ):
                build_model()

        self.assertEqual(
            model_class.call_args.kwargs["api_key"],
            "environment-key",
        )
        self.assertEqual(
            model_class.call_args.kwargs["model"],
            "environment-model",
        )


class AgentTests(unittest.TestCase):
    def test_wires_model_tool_and_grounding_prompt(self) -> None:
        model = Mock()
        tool = Mock()
        runner = Mock()

        with patch(
            "lng_ticket_assistant.agent.create_deep_agent",
            return_value=runner,
        ) as create_agent:
            agent = DeepSupportAgent(model=model, tools=[tool])

        self.assertIs(agent.runner, runner)
        create_agent.assert_called_once_with(
            model=model,
            tools=[tool],
            system_prompt=SYSTEM_PROMPT,
        )
        self.assertIn("P1 and P2", SYSTEM_PROMPT)
        self.assertIn("P3 and P4", SYSTEM_PROMPT)
        self.assertIn("cannot create or update tickets", SYSTEM_PROMPT)


class CliTests(unittest.TestCase):
    def test_extracts_google_content_blocks(self) -> None:
        result = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "First step."},
                        {"type": "output_text", "text": "Second step."},
                    ],
                }
            ]
        }

        self.assertEqual(
            extract_final_response(result),
            "First step.\nSecond step.",
        )

    def test_preserves_returned_message_history(self) -> None:
        agent = Mock()
        first_history = [
            {"role": "user", "content": "VPN fails"},
            {"role": "assistant", "content": "What error do you see?"},
        ]
        second_history = [
            *first_history,
            {"role": "user", "content": "E4012"},
            {"role": "assistant", "content": "Reconnect the client."},
        ]
        agent.invoke.side_effect = [
            {"messages": first_history},
            {"messages": second_history},
        ]
        inputs = iter(["VPN fails", "E4012", "quit"])
        outputs: list[str] = []

        run_cli(
            agent,
            input_fn=lambda _: next(inputs),
            output_fn=outputs.append,
        )

        second_call_messages = agent.invoke.call_args_list[1].args[0]
        self.assertEqual(second_call_messages[:2], first_history)
        self.assertIn("Support: Reconnect the client.", outputs)
        self.assertEqual(outputs[-1], "Goodbye.")


if __name__ == "__main__":
    unittest.main()

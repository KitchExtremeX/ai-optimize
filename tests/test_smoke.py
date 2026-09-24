"""Offline smoke tests for the dry-run path, parsing, and request shape."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_optimize.cli import main, read_prompt
from ai_optimize.envfile import load_env_file
from ai_optimize.optimizer import (
    ConfigError,
    ProviderError,
    UsageError,
    chat_completions_url,
    dry_run_optimize,
    format_http_error,
    live_optimize,
    load_settings,
    parse_model_result,
    render_json,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class DryRunTests(unittest.TestCase):
    def test_summary_shape(self) -> None:
        result = dry_run_optimize("summarize this")
        self.assertEqual(result.mode, "dry-run")
        self.assertIsNone(result.model)
        self.assertIn("summarize this", result.optimized_prompt)
        self.assertIn("Produce a summary", result.optimized_prompt)
        self.assertNotIn("dry-run", result.optimized_prompt)
        self.assertGreaterEqual(len(result.eval_notes.scoring_checks), 3)
        self.assertTrue(result.eval_notes.what_good_looks_like)
        self.assertTrue(result.eval_notes.edge_cases)
        markdown = render_markdown(result)
        self.assertIn("# Optimized prompt", markdown)
        self.assertIn("# Eval notes", markdown)
        self.assertIn("mode: dry-run", markdown)
        payload = json.loads(render_json(result))
        self.assertEqual(payload["mode"], "dry-run")
        self.assertIn("pass_if", payload["eval_notes"]["scoring_checks"][0])

    def test_keyword_rules_differ(self) -> None:
        code = dry_run_optimize("write code to fix the bug")
        message = dry_run_optimize(
            "make a marketing email",
            goal="Get a reply from a busy cafe owner",
            audience="Independent cafe owners",
            constraints="Under 120 words. No discount. One clear ask.",
        )
        generic = dry_run_optimize(
            "Turn the attached interview notes into a one-page brief for the hiring panel, "
            "keeping quotes intact and leaving salary out."
        )
        self.assertIn("failing behavior", code.optimized_prompt)
        self.assertIn("message text only", message.optimized_prompt)
        self.assertIn("Under 120 words. No discount. One clear ask.", message.optimized_prompt)
        self.assertIn("Independent cafe owners", message.optimized_prompt)
        self.assertIn("Get a reply from a busy cafe owner", message.optimized_prompt)
        self.assertIn("two concrete interpretations", generic.optimized_prompt)
        self.assertTrue(any("Kept a longer request intact" in item for item in generic.changes))
        self.assertIn("Marked the original request as underspecified.", code.changes)

    def test_tl_dr_uses_summary_rule(self) -> None:
        result = dry_run_optimize("need a tl;dr")
        self.assertIn("Produce a summary", result.optimized_prompt)

    def test_empty_prompt_raises(self) -> None:
        with self.assertRaises(UsageError):
            dry_run_optimize("   ")


class ExampleSnapshotTests(unittest.TestCase):
    def test_checked_in_outputs_match_dry_run(self) -> None:
        manifest = json.loads((EXAMPLES / "manifest.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(manifest["cases"]), 2)
        for case in manifest["cases"]:
            prompt = (EXAMPLES / case["prompt_file"]).read_text(encoding="utf-8")
            result = dry_run_optimize(
                prompt,
                goal=case["goal"],
                audience=case["audience"],
                constraints=case["constraints"],
            )
            stem = EXAMPLES / case["id"]
            self.assertEqual((stem.with_suffix(".md")).read_text(encoding="utf-8"), render_markdown(result))
            self.assertEqual((stem.with_suffix(".json")).read_text(encoding="utf-8"), render_json(result))


class LiveParseTests(unittest.TestCase):
    def test_fake_transport_builds_request_and_parses_answer(self) -> None:
        seen: dict[str, object] = {}

        def transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
            seen["url"] = url
            seen["headers"] = headers
            seen["body"] = json.loads(body.decode("utf-8"))
            seen["timeout"] = timeout
            content = {
                "optimized_prompt": "Summarize the source in 5 bullets. If there is no source, say so.",
                "changes": ["Named the output length."],
                "eval_notes": {
                    "what_good_looks_like": ["Five bullets, each tied to the source."],
                    "edge_cases": ["No source text was pasted."],
                    "scoring_checks": [
                        {"name": "Length", "pass_if": "the summary is five bullets or a clear refusal"}
                    ],
                },
            }
            return {"choices": [{"message": {"content": "```json\n" + json.dumps(content) + "\n```"}}]}

        result = live_optimize(
            "summarize this",
            goal=None,
            audience=None,
            constraints=None,
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
            timeout=9,
            json_mode=True,
            transport=transport,
        )
        self.assertEqual(seen["url"], "https://example.test/v1/chat/completions")
        headers = seen["headers"]
        assert isinstance(headers, dict)
        self.assertEqual(headers["Authorization"], "Bearer test-key")
        self.assertNotIn("test-key", str(seen["url"]))
        body = seen["body"]
        assert isinstance(body, dict)
        self.assertEqual(body["model"], "test-model")
        self.assertEqual(body["temperature"], 0.2)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        user = body["messages"][1]["content"]
        self.assertIn("summarize this", user)
        self.assertIn("(not provided)", user)
        self.assertIn("do not answer", body["messages"][0]["content"].lower())
        self.assertEqual(seen["timeout"], 9)
        self.assertEqual(result.mode, "live")
        self.assertEqual(result.model, "test-model")
        self.assertIn("5 bullets", result.optimized_prompt)

    def test_full_chat_url_is_not_doubled(self) -> None:
        self.assertEqual(
            chat_completions_url("https://example.test/v1/chat/completions"),
            "https://example.test/v1/chat/completions",
        )
        self.assertEqual(
            chat_completions_url("https://example.test/v1/"),
            "https://example.test/v1/chat/completions",
        )

    def test_fenced_and_invalid_json(self) -> None:
        payload = {
            "optimized_prompt": "Do the task.",
            "changes": ["Clarified the task."],
            "eval_notes": {
                "what_good_looks_like": ["The task is done."],
                "edge_cases": ["The input is empty."],
                "scoring_checks": [{"name": "Done", "pass_if": "the task is done"}],
            },
        }
        parsed = parse_model_result("note\n" + json.dumps(payload) + "\nthanks", model="m")
        self.assertEqual(parsed.optimized_prompt, "Do the task.")
        with self.assertRaises(ProviderError):
            parse_model_result("not json", model="m")
        with self.assertRaises(ProviderError):
            parse_model_result(json.dumps({"optimized_prompt": "x"}), model="m")

    def test_http_error_uses_api_message(self) -> None:
        message = format_http_error(401, json.dumps({"error": {"message": "bad key"}}))
        self.assertEqual(message, "API request failed (401): bad key")

    def test_content_parts_and_empty_choices(self) -> None:
        text = json.dumps(
            {
                "optimized_prompt": "Be specific.",
                "changes": ["Added a format."],
                "eval_notes": {
                    "what_good_looks_like": ["Specific."],
                    "edge_cases": ["Vague input."],
                    "scoring_checks": [{"name": "Specific", "pass_if": "the prompt names an output"}],
                },
            }
        )

        def parts_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
            return {"choices": [{"message": {"content": [{"type": "text", "text": text}]}}]}

        parsed = live_optimize(
            "be specific",
            api_key="test-key",
            base_url="https://example.test/v1",
            model="m",
            transport=parts_transport,
        )
        self.assertEqual(parsed.optimized_prompt, "Be specific.")

        def empty_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
            return {"choices": []}

        with self.assertRaises(ProviderError):
            live_optimize(
                "be specific",
                api_key="test-key",
                base_url="https://example.test/v1",
                model="m",
                transport=empty_transport,
            )


class SettingsTests(unittest.TestCase):
    def test_missing_key(self) -> None:
        with self.assertRaises(ConfigError) as caught:
            load_settings({})
        self.assertIn("--dry-run", str(caught.exception))

    def test_timeout_must_be_numeric(self) -> None:
        with self.assertRaises(ConfigError):
            load_settings({"OPENAI_API_KEY": "k", "OPENAI_TIMEOUT": "soon"})

    def test_dotenv_does_not_override_or_keep_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "export OPENAI_API_KEY=\"from-file\"",
                        "OPENAI_MODEL='demo-model'",
                        "SKIP_THIS",
                        "ALREADY=from-env",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            env = {"ALREADY": "from-env"}
            self.assertTrue(load_env_file(path, env))
            self.assertEqual(env["OPENAI_API_KEY"], "from-file")
            self.assertEqual(env["OPENAI_MODEL"], "demo-model")
            self.assertEqual(env["ALREADY"], "from-env")
            self.assertNotIn("SKIP_THIS", env)
            self.assertFalse(load_env_file(Path(tmp) / "missing.env", env))


class CliTests(unittest.TestCase):
    def test_module_dry_run_json(self) -> None:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        proc = subprocess.run(
            [sys.executable, "-m", "ai_optimize", "--dry-run", "--format", "json", "summarize this"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["mode"], "dry-run")
        self.assertIn("summarize this", data["optimized_prompt"])
        self.assertGreaterEqual(len(data["eval_notes"]["scoring_checks"]), 1)

    def test_live_without_key_exits_before_network(self) -> None:
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env["PYTHONPATH"] = str(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "ai_optimize", "summarize this"],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(proc.returncode, 3, proc.stderr)
        self.assertIn("--dry-run", proc.stderr)
        self.assertEqual(proc.stdout, "")

    def test_format_both_writes_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with contextlib.redirect_stderr(io.StringIO()):
                code = main(
                    ["--dry-run", "--format", "both", "--output", str(out), "summarize this"]
                )
            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "out.md").is_file())
            self.assertTrue((Path(tmp) / "out.json").is_file())
            self.assertIn("summarize this", (Path(tmp) / "out.md").read_text(encoding="utf-8"))

    def test_empty_prompt_exits_2(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            code = main(["--dry-run", "   "])
        self.assertEqual(code, 2)

    def test_stdin_prompt(self) -> None:
        args = type("Args", (), {"file": None, "prompt": None})()
        self.assertEqual(read_prompt(args, io.StringIO("  hello there  ")), "  hello there  ")

    def test_help_exits_zero(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "ai_optimize", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("--dry-run", proc.stdout)


if __name__ == "__main__":
    unittest.main()

"""Rewrite a weak prompt locally, or through an OpenAI-compatible chat API."""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable

from ai_optimize import __version__

Transport = Callable[[str, dict[str, str], bytes, float], dict]

MAX_PROMPT_CHARS = 100_000
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60.0

# Local stand-ins for a model rewrite. First match wins. They only pick an
# output rule from the user's words; they do not try to understand the task.
_OUTPUT_RULES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "summary",
        ("summarize", "summarise", "summary", "tl;dr", "tldr"),
        "Produce a summary of the source the user supplies. "
        "Do not add analysis, a rewrite, or next steps unless the task asked for them.",
    ),
    (
        "code",
        ("bug", "code", "debug", "function"),
        "Produce the code change or the diagnosis the task asks for. "
        "If the failing behavior, the code, or the expected result is missing, "
        "list those gaps and stop. Do not rewrite unrelated code.",
    ),
    (
        "message",
        ("email", "marketing", "outreach", "newsletter"),
        "Produce the message text only. "
        "Do not add a brand strategy, a subject-line list, or a send plan "
        "unless the task asked for them.",
    ),
)
_GENERIC_RULE = (
    "Produce only the artifact the task names. "
    "If the artifact type is unclear, state two concrete interpretations and "
    "stop instead of picking one silently."
)

SYSTEM_PROMPT = """You optimize weak prompts for builders. You do not answer the user's task. You rewrite the prompt so another model can do the task, and you write eval notes a person can use to judge that model's answer.

Rules:
- Keep the user's intent. Do not add a new product, audience, or constraint they did not supply or clearly imply.
- When goal, audience, or constraints are provided, use them. When one is "(not provided)", tell the answering model to state assumptions or ask, instead of inventing that field.
- Make the prompt testable: name the inputs, the output shape, and what to do when information is missing or conflicting.
- Prefer sections and short rules the answering model can follow.
- Eval notes must be specific to this prompt. Do not give generic writing advice.
- scoring_checks are pass/fail checks a person can apply in a few minutes. No weights and no percentages.

Return only a JSON object with this shape:
{
  "optimized_prompt": "the full prompt to give the answering model",
  "changes": ["what you clarified or added"],
  "eval_notes": {
    "what_good_looks_like": ["observable qualities of a good answer"],
    "edge_cases": ["inputs or situations the prompt should handle"],
    "scoring_checks": [
      {"name": "short name", "pass_if": "observable pass condition"}
    ]
  }
}
"""


class OptimizeError(Exception):
    """Failure the CLI can print as-is."""

    exit_code = 1


class UsageError(OptimizeError):
    exit_code = 2


class ConfigError(OptimizeError):
    exit_code = 3


class ProviderError(OptimizeError):
    exit_code = 4


@dataclass(frozen=True)
class ScoringCheck:
    name: str
    pass_if: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "pass_if": self.pass_if}


@dataclass(frozen=True)
class EvalNotes:
    what_good_looks_like: tuple[str, ...]
    edge_cases: tuple[str, ...]
    scoring_checks: tuple[ScoringCheck, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "what_good_looks_like": list(self.what_good_looks_like),
            "edge_cases": list(self.edge_cases),
            "scoring_checks": [check.as_dict() for check in self.scoring_checks],
        }


@dataclass(frozen=True)
class OptimizeResult:
    optimized_prompt: str
    eval_notes: EvalNotes
    changes: tuple[str, ...]
    mode: str
    model: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "model": self.model,
            "optimized_prompt": self.optimized_prompt,
            "changes": list(self.changes),
            "eval_notes": self.eval_notes.as_dict(),
        }


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    timeout: float


def load_settings(
    env: Mapping[str, str],
    *,
    model: str | None = None,
    base_url: str | None = None,
) -> Settings:
    """Read live-mode settings from the environment, with CLI overrides."""
    api_key = (env.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            "No OPENAI_API_KEY is set. Copy .env.example to .env and add a key, "
            "or rerun with --dry-run."
        )
    resolved_url = (base_url or env.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).strip()
    if not resolved_url:
        raise ConfigError("OPENAI_BASE_URL is empty.")
    resolved_model = (model or env.get("OPENAI_MODEL") or DEFAULT_MODEL).strip()
    if not resolved_model:
        raise ConfigError("OPENAI_MODEL is empty.")
    timeout_raw = (env.get("OPENAI_TIMEOUT") or str(int(DEFAULT_TIMEOUT))).strip()
    try:
        timeout = float(timeout_raw)
    except ValueError as exc:
        raise ConfigError(f"OPENAI_TIMEOUT must be a number, got {timeout_raw!r}.") from exc
    if timeout <= 0:
        raise ConfigError("OPENAI_TIMEOUT must be greater than 0.")
    return Settings(
        api_key=api_key,
        base_url=resolved_url,
        model=resolved_model,
        timeout=timeout,
    )


def dry_run_optimize(
    prompt: str,
    *,
    goal: str | None = None,
    audience: str | None = None,
    constraints: str | None = None,
) -> OptimizeResult:
    """Build the output shape locally. Does not call a network."""
    original = _require_prompt(prompt)
    task = " ".join(original.split())
    goal_text = _clean(goal)
    audience_text = _clean(audience)
    constraints_text = _clean(constraints)
    label, rule = _output_rule(task)
    short = len(task.split()) < 12

    if short:
        diagnosis = (
            "The original request is short and leaves out the reader, the output "
            "shape, or the inputs. Those slots are explicit below. Do not fill "
            "them with guesses."
        )
    else:
        diagnosis = (
            "The original request is restated below with explicit slots for goal, "
            "audience, constraints, and missing inputs. Do not add requirements "
            "that are not listed."
        )

    goal_slot = goal_text or (
        "Not provided. Do not invent a goal. If the task cannot be finished without one, stop and ask."
    )
    audience_slot = audience_text or (
        "Not provided. Use plain language a busy builder can check. Do not invent a persona."
    )
    constraints_slot = constraints_text or (
        "None provided. Stay concise. If a needed input is missing, name it and stop instead of guessing."
    )
    quoted = "\n".join(f"> {line}" if line else ">" for line in original.splitlines())
    optimized = (
        "You complete one task. Do not expand the scope.\n"
        "\n"
        "## Task\n"
        f"{task}\n"
        "\n"
        "## Scope\n"
        f"{diagnosis}\n"
        "\n"
        "## Goal\n"
        f"{goal_slot}\n"
        "\n"
        "## Audience\n"
        f"{audience_slot}\n"
        "\n"
        "## Constraints\n"
        f"{constraints_slot}\n"
        "\n"
        "## Inputs\n"
        "Use only material the user includes with this prompt. If the task needs "
        "source text, code, data, or a link and it is not there, list what is "
        "missing and stop.\n"
        "\n"
        "## Output\n"
        f"{rule}\n"
        "\n"
        "Structure the answer so a reviewer can score it without asking what was meant.\n"
        "If two instructions conflict, name the conflict and follow Constraints.\n"
        'When you had to assume something non-obvious, add a final line that starts '
        'with "Assumptions:". Otherwise omit that line.\n'
        "\n"
        "## Original request\n"
        f"{quoted}\n"
    )

    clipped = _clip(task)
    good = [
        f"A reviewer can tell, from the answer alone, that it responds to: {clipped}",
        "The answer has a visible structure and does not add a second deliverable.",
        "Missing source material is listed. It is not replaced with invented facts.",
        f"The answer follows the output rule ({label}).",
    ]
    if goal_text:
        good.append(f"The answer serves this goal: {goal_text}")
    else:
        good.append("No goal was provided, and the answer does not invent one.")
    if audience_text:
        good.append(f"The language fits this audience: {audience_text}")

    edges = (
        "The user sends the prompt with no source document, code, or data attached.",
        "The source material contradicts the constraints.",
        "The request can be read as either a short reply or a structured artifact.",
        "The user includes only part of the material and the rest would have to be guessed.",
    )
    checks = [
        ScoringCheck("Task preserved", f"the answer is clearly about: {clipped}"),
        ScoringCheck(
            "Checkable shape",
            "a reviewer can score the answer without asking what the prompt wanted",
        ),
        ScoringCheck(
            "Missing inputs",
            "when no source material is attached, the answer lists what it needs and does not invent it",
        ),
        ScoringCheck("Output rule", f"the answer matches the {label} rule: {rule}"),
        ScoringCheck(
            "Single deliverable",
            "the answer does not add a second task the original request did not ask for",
        ),
    ]
    if constraints_text:
        checks.append(ScoringCheck("Constraints", f"the answer respects: {constraints_text}"))
    if audience_text:
        checks.append(ScoringCheck("Audience", f"the wording fits: {audience_text}"))

    changes = [
        "Split the request into task, goal, audience, constraints, inputs, and output.",
        "Told the answering model to stop when required inputs are missing.",
    ]
    if short:
        changes.append("Marked the original request as underspecified.")
    else:
        changes.append("Kept a longer request intact and added explicit slots around it.")
    if goal_text:
        changes.append("Placed the given goal into the prompt.")
    if audience_text:
        changes.append("Placed the given audience into the prompt.")
    if constraints_text:
        changes.append("Placed the given constraints into the prompt.")
    changes.append(
        f"Selected the local '{label}' output rule from the wording. "
        "This is a template, not a model rewrite."
    )
    changes.append("Added eval notes: what good looks like, edge cases, and pass/fail checks.")

    return OptimizeResult(
        optimized_prompt=optimized,
        eval_notes=EvalNotes(tuple(good), edges, tuple(checks)),
        changes=tuple(changes),
        mode="dry-run",
        model=None,
    )


def live_optimize(
    prompt: str,
    *,
    goal: str | None = None,
    audience: str | None = None,
    constraints: str | None = None,
    api_key: str,
    base_url: str,
    model: str,
    timeout: float = DEFAULT_TIMEOUT,
    json_mode: bool = False,
    transport: Transport | None = None,
) -> OptimizeResult:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""
    original = _require_prompt(prompt)
    if not api_key.strip():
        raise ConfigError(
            "No OPENAI_API_KEY is set. Copy .env.example to .env and add a key, "
            "or rerun with --dry-run."
        )
    payload: dict[str, object] = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _user_message(original, goal, audience, constraints),
            },
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": f"ai-optimize/{__version__}",
    }
    send = transport or urllib_transport
    response = send(chat_completions_url(base_url), headers, body, timeout)
    text = _completion_text(response)
    return parse_model_result(text, model=model)


def chat_completions_url(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if not base:
        raise ConfigError("OPENAI_BASE_URL is empty.")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def parse_model_result(text: str, *, model: str) -> OptimizeResult:
    """Parse the model's JSON object into an :class:`OptimizeResult`."""
    data = _extract_json_object(text)
    optimized = data.get("optimized_prompt")
    if not isinstance(optimized, str) or not optimized.strip():
        raise ProviderError("Model JSON field optimized_prompt must be a non-empty string.")
    changes = data.get("changes")
    if not isinstance(changes, list) or not all(isinstance(item, str) and item.strip() for item in changes):
        raise ProviderError("Model JSON field changes must be a list of non-empty strings.")
    notes = data.get("eval_notes")
    if not isinstance(notes, dict):
        raise ProviderError("Model JSON field eval_notes must be an object.")
    return OptimizeResult(
        optimized_prompt=optimized.strip(),
        eval_notes=EvalNotes(
            what_good_looks_like=_string_list(notes.get("what_good_looks_like"), "eval_notes.what_good_looks_like"),
            edge_cases=_string_list(notes.get("edge_cases"), "eval_notes.edge_cases"),
            scoring_checks=_checks(notes.get("scoring_checks")),
        ),
        changes=tuple(item.strip() for item in changes),
        mode="live",
        model=model,
    )


def render_markdown(result: OptimizeResult) -> str:
    lines = [
        "# Optimized prompt",
        "",
        result.optimized_prompt.rstrip(),
        "",
        "# Eval notes",
        "",
        "## What good looks like",
        "",
    ]
    lines.extend(f"- {item}" for item in result.eval_notes.what_good_looks_like)
    lines.extend(["", "## Edge cases", ""])
    lines.extend(f"- {item}" for item in result.eval_notes.edge_cases)
    lines.extend(["", "## Scoring checks", ""])
    lines.extend(
        f"- **{check.name}** — pass if: {check.pass_if}" for check in result.eval_notes.scoring_checks
    )
    lines.extend(["", "## What changed", ""])
    lines.extend(f"- {item}" for item in result.changes)
    model = result.model or "(none — local template, no API call)"
    lines.extend(["", "---", f"mode: {result.mode}", f"model: {model}", ""])
    return "\n".join(lines)


def render_json(result: OptimizeResult) -> str:
    return json.dumps(result.as_dict(), indent=2, ensure_ascii=False) + "\n"


def urllib_transport(url: str, headers: dict[str, str], body: bytes, timeout: float) -> dict:
    """POST JSON and return the decoded object. Redirects are refused."""
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_RefuseRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except ProviderError:
        raise
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ProviderError(format_http_error(exc.code, detail)) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise ProviderError("The API request timed out.") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Could not reach the API: {exc.reason}") from exc
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ProviderError("API response was not JSON.") from exc
    if not isinstance(data, dict):
        raise ProviderError("API response JSON must be an object.")
    return data


def format_http_error(status: int, body: str) -> str:
    message = body.strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict) and error.get("message"):
            message = str(error["message"]).strip()
        elif isinstance(error, str):
            message = error.strip()
    message = " ".join(message.split())
    if len(message) > 300:
        message = message[:300] + "..."
    if not message:
        message = "no response body"
    return f"API request failed ({status}): {message}"


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise ProviderError(
            f"API endpoint returned a redirect ({code}). Refusing to resend the API key."
        )


def _require_prompt(prompt: str) -> str:
    text = prompt.strip()
    if not text:
        raise UsageError("Prompt is empty.")
    if len(text) > MAX_PROMPT_CHARS:
        raise UsageError(
            f"Prompt is {len(text)} characters. The limit is {MAX_PROMPT_CHARS}."
        )
    return text


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _clip(text: str, limit: int = 120) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _output_rule(prompt: str) -> tuple[str, str]:
    for label, words, instruction in _OUTPUT_RULES:
        if any(_has_word(prompt, word) for word in words):
            return label, instruction
    return "generic", _GENERIC_RULE


def _has_word(haystack: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", haystack, flags=re.IGNORECASE) is not None


def _user_message(
    prompt: str,
    goal: str | None,
    audience: str | None,
    constraints: str | None,
) -> str:
    return (
        "Weak prompt:\n"
        "---\n"
        f"{prompt}\n"
        "---\n"
        "\n"
        f"Goal: {_or_missing(goal)}\n"
        f"Audience: {_or_missing(audience)}\n"
        f"Constraints: {_or_missing(constraints)}\n"
    )


def _or_missing(value: str | None) -> str:
    return _clean(value) or "(not provided)"


def _extract_json_object(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end <= start:
            raise ProviderError("Model response was not a JSON object.")
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Model response was not valid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ProviderError("Model response JSON must be an object.")
    return data


def _completion_text(payload: dict) -> str:
    if "error" in payload and "choices" not in payload:
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            detail = str(error["message"])
        else:
            detail = str(error)
        raise ProviderError(f"API request failed: {detail}")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ProviderError("API response did not include any choices.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ProviderError("API response choice had no message.")
    return _message_text(message.get("content"))


def _message_text(content: object) -> str:
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        text = "".join(parts)
        if text.strip():
            return text
    raise ProviderError("API response message content was empty.")


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ProviderError(f"Model JSON field {label} must be a non-empty list of strings.")
    return tuple(item.strip() for item in value)


def _checks(value: object) -> tuple[ScoringCheck, ...]:
    if not isinstance(value, list) or not value:
        raise ProviderError("Model JSON field eval_notes.scoring_checks must be a non-empty list.")
    checks: list[ScoringCheck] = []
    for item in value:
        if not isinstance(item, dict):
            raise ProviderError("Each scoring check must be an object with name and pass_if.")
        name = item.get("name")
        pass_if = item.get("pass_if")
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(pass_if, str)
            or not pass_if.strip()
        ):
            raise ProviderError("Each scoring check needs non-empty string fields name and pass_if.")
        checks.append(ScoringCheck(name.strip(), pass_if.strip()))
    return tuple(checks)

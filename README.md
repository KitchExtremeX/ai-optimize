# AI Optimize

AI Optimize (test beta) turns a weak LLM prompt into a clearer, testable prompt plus eval notes.

Weak prompts produce vague output that is hard to reuse or score. This CLI rewrites the prompt for clarity, structure, and testability, and it writes down what a good answer looks like.

The user is a builder or an Applied AI student. The owner is Kadien Coe (KitchExtremeX).

Input is a weak prompt, with an optional goal, audience, and constraints. Output is an optimized prompt and eval notes, as markdown, JSON, or both.

This is a test-run beta: a CLI and an end-of-day log. There is no web app.

## 60-second demo

From the repo root, with Python 3.10+:

```bash
python -m ai_optimize --dry-run "summarize this"
```

You get two parts:

1. **Optimized prompt** — task, scope, goal, audience, constraints, inputs, and an output rule.
2. **Eval notes** — what good looks like, edge cases, and pass/fail scoring checks.

`--dry-run` does not use the network. It fills a local template so you can see the I/O shape without an API key. The footer says `mode: dry-run`. That wording is not a model rewrite.

Same demo from the checked-in samples:

```bash
python -m ai_optimize --dry-run --file examples/01-summarize.txt
python -m ai_optimize --dry-run --format json --file examples/02-fix-the-bug.txt
python -m ai_optimize --dry-run \
  --goal "Get a reply from a busy cafe owner" \
  --audience "Independent cafe owners who already buy beans wholesale" \
  --constraints "Under 120 words. No discount. One clear ask." \
  --file examples/03-marketing-email.txt
```

The files `examples/01-summarize.md`, `examples/02-fix-the-bug.md`, and `examples/03-marketing-email.md` (and the matching `.json` files) are the stdout of those commands. `examples/manifest.json` records the flags.

## Setup

```bash
python3 --version   # 3.10 or newer
```

No third-party packages are required. `python -m ai_optimize` works from the repo root.

Optional, if you want a console script:

```bash
pip install -e .
ai-optimize --dry-run "summarize this"
```

### Live mode

Live mode sends the prompt to an OpenAI-compatible `POST /chat/completions` endpoint and asks for the same JSON shape the CLI prints.

```bash
cp .env.example .env
# edit .env — put the key only in that file
python -m ai_optimize "summarize this"
```

`.env` is gitignored. Variables already set in the environment win over `.env`.

| Variable | Role |
| --- | --- |
| `OPENAI_API_KEY` | Bearer token. Required for live mode. |
| `OPENAI_BASE_URL` | Default `https://api.openai.com/v1`. A URL that already ends in `/chat/completions` is used as-is. |
| `OPENAI_MODEL` | Default `gpt-4o-mini`. |
| `OPENAI_TIMEOUT` | Seconds. Default `60`. |

`--model` and `--base-url` override the environment for one run. `--json-mode` sends `response_format: json_object` when the endpoint supports OpenAI JSON mode. Redirects are refused so the key is not forwarded to another host.

**Live mode was not called when this repo was created.** `OPENAI_API_KEY` was unset. Request construction is covered by a unit test that uses a fake transport. That test does not show that any provider accepted the request.

If the key is missing, the CLI exits with code 3 and tells you to use `--dry-run`. It does not open a connection.

## CLI

```text
python -m ai_optimize [--dry-run] [--format markdown|json|both] [--output PATH]
                      [--goal TEXT] [--audience TEXT] [--constraints TEXT]
                      [--file PATH] [--model MODEL] [--base-url URL] [--json-mode]
                      [PROMPT]
```

- Prompt as an argument, `--file`, or stdin.
- If the prompt starts with a dash, put it after `--` or use `--file`.
- `--format both` on stdout prints markdown, then a `===== JSON =====` line, then JSON.
- `--format both --output out` writes `out.md` and `out.json`.
- `--output -` is stdout.
- Prompts longer than 100,000 characters are rejected.

Exit codes: `0` ok, `2` usage, `3` missing or invalid config, `4` the API request or the model JSON was unusable.

Pipe a prompt:

```bash
printf '%s\n' 'summarize this' | python -m ai_optimize --dry-run --format json
```

## Examples

| File | Weak prompt | Extra input |
| --- | --- | --- |
| `examples/01-summarize.txt` | `summarize this` | none |
| `examples/02-fix-the-bug.txt` | `write code to fix the bug` | none |
| `examples/03-marketing-email.txt` | `make a marketing email` | goal, audience, constraints |

Dry-run picks one local output rule from the words in the prompt (summary, code, message, or generic). First match wins. It does not interpret the task beyond that. See `_OUTPUT_RULES` in `ai_optimize/optimizer.py`.

Checked-in outputs are from `--dry-run` on the Day-1 commit. They are the expected shape, not a recorded live model response.

## How it is built

```text
ai_optimize/cli.py        arguments, exit codes, file output
ai_optimize/optimizer.py  dry-run template, chat client, markdown and JSON
ai_optimize/envfile.py    optional .env loader
```

The standard library does the HTTP call (`urllib`). Live mode sends a system prompt that tells the model to rewrite the prompt and return eval notes, not to perform the user's task. The CLI checks that the JSON has a non-empty optimized prompt, changes, what-good-looks-like, edge cases, and scoring checks with `name` and `pass_if`.

## End-of-day log

TRIGGER → ACTION

- **TRIGGER:** you finish a working session, something breaks, or you ship a change.
- **ACTION:** copy `eod/TEMPLATE.md` to `eod/YYYY-MM-DD.md` if that day's file does not exist yet. Fill every field before you stop: date, built, broke, learned, shipped, business signal, change tomorrow. If you already logged today, add another block in the same file.

Write what happened. When there is no user or usage evidence, business signal is "none". Do not invent numbers.

Day 1 is `eod/2026-09-24.md`.

## Tests

No network:

```bash
python -m unittest discover -s tests -v
```

`tests/test_smoke.py` runs the CLI in `--dry-run`, checks that a missing key exits before any socket is required, checks the checked-in examples against the template, and checks live-request JSON with a fake transport.

## Scope

Day-1 test beta for builders and Applied AI students. The demo is: paste a weak prompt, read a tighter prompt, and read how you would score an answer.

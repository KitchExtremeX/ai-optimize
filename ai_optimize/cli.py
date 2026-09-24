"""Command-line entrypoint for AI Optimize."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ai_optimize import __version__
from ai_optimize.envfile import load_env_file
from ai_optimize.optimizer import (
    OptimizeError,
    OptimizeResult,
    UsageError,
    dry_run_optimize,
    live_optimize,
    load_settings,
    render_json,
    render_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-optimize",
        description="Turn a weak LLM prompt into a clearer, testable prompt plus eval notes.",
        epilog=(
            "Examples:\n"
            '  python -m ai_optimize --dry-run "summarize this"\n'
            "  python -m ai_optimize --dry-run --format json --file examples/01-summarize.txt\n"
            '  python -m ai_optimize --goal "Get a reply" --audience "Cafe owners" '
            '"make a marketing email"\n'
            "\n"
            "If the prompt starts with a dash, put it after -- or pass it with --file.\n"
            "Live mode reads OPENAI_API_KEY from the environment or from .env in the current directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"ai-optimize {__version__}")
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Weak prompt text. Omit to use --file or stdin.",
    )
    parser.add_argument("--file", type=_existing_file, help="Read the weak prompt from a file.")
    parser.add_argument("--goal", help="What the rewritten prompt should help the user achieve.")
    parser.add_argument("--audience", help="Who the answering model is writing for.")
    parser.add_argument("--constraints", help="Limits the answering model must respect.")
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "both"),
        default="markdown",
        help="Output format (default: markdown).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Write to a file instead of stdout. With --format both, also writes "
            "the sibling .md or .json file."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the output locally with a template. No network call.",
    )
    parser.add_argument(
        "--json-mode",
        action="store_true",
        help=(
            "Ask the API for a JSON object (response_format). "
            "Use when the endpoint supports it. Ignored with --dry-run."
        ),
    )
    parser.add_argument("--model", help="Override OPENAI_MODEL.")
    parser.add_argument("--base-url", help="Override OPENAI_BASE_URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prompt = read_prompt(args, sys.stdin)
        result = build_result(args, prompt)
        output = None if args.output in (None, "-") else Path(args.output)
        write_result(result, args.format, output)
    except BrokenPipeError:
        return 0
    except OptimizeError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


def read_prompt(args: argparse.Namespace, stdin) -> str:
    if args.file is not None and args.prompt:
        raise UsageError("Pass a prompt or --file, not both.")
    if args.file is not None:
        text = args.file.read_text(encoding="utf-8-sig")
    elif args.prompt is not None:
        text = args.prompt
    elif not stdin.isatty():
        text = stdin.read()
    else:
        raise UsageError("Provide a prompt, --file, or piped stdin. Try --help.")
    if not text.strip():
        raise UsageError("Prompt is empty.")
    return text


def build_result(args: argparse.Namespace, prompt: str) -> OptimizeResult:
    if args.dry_run:
        return dry_run_optimize(
            prompt,
            goal=args.goal,
            audience=args.audience,
            constraints=args.constraints,
        )
    load_env_file(Path.cwd() / ".env")
    settings = load_settings(os.environ, model=args.model, base_url=args.base_url)
    return live_optimize(
        prompt,
        goal=args.goal,
        audience=args.audience,
        constraints=args.constraints,
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        timeout=settings.timeout,
        json_mode=args.json_mode,
    )


def write_result(result: OptimizeResult, fmt: str, output: Path | None) -> None:
    markdown = render_markdown(result)
    payload = render_json(result)
    if output is None:
        if fmt == "markdown":
            sys.stdout.write(markdown)
        elif fmt == "json":
            sys.stdout.write(payload)
        else:
            sys.stdout.write(markdown)
            sys.stdout.write("\n===== JSON =====\n")
            sys.stdout.write(payload)
        return

    if fmt == "markdown":
        _write(output, markdown)
        print(f"Wrote {output}", file=sys.stderr)
        return
    if fmt == "json":
        _write(output, payload)
        print(f"Wrote {output}", file=sys.stderr)
        return
    markdown_path, json_path = _both_paths(output)
    _write(markdown_path, markdown)
    _write(json_path, payload)
    print(f"Wrote {markdown_path} and {json_path}", file=sys.stderr)


def _both_paths(path: Path) -> tuple[Path, Path]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return path.with_suffix(".md"), path
    if suffix == ".md":
        return path, path.with_suffix(".json")
    return path.with_suffix(".md"), path.with_suffix(".json")


def _write(path: Path, text: str) -> None:
    parent = path.parent
    if parent != Path(".") and not parent.is_dir():
        raise UsageError(f"Output directory does not exist: {parent}")
    path.write_text(text, encoding="utf-8")


def _existing_file(value: str) -> Path:
    path = Path(value)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file not found: {value}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())

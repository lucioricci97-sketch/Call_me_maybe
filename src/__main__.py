"""CLI entry: uv run python -m src [--functions_definition ...] [--input ...] [--output ...]."""

import argparse
import sys
from pathlib import Path

from src.io_utils import (
    load_function_definitions,
    load_test_prompts,
    save_function_calls,
)
from src.llm_runner import LLM, Vocab, fill_parameters, pick_best_function
from src.models import FunctionCall


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description="Function calling with constrained decoding.")
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
        help="Path to function definitions JSON.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
        help="Path to test prompts JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calling_results.json"),
        help="Path where the results JSON will be written.",
    )
    return parser.parse_args()


def main() -> int:
    """Program entry. Returns a shell exit code."""
    args = parse_args()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_test_prompts(args.input)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1

    print(f"Loaded {len(functions)} function(s) and {len(prompts)} prompt(s).")
    print("Loading LLM...")
    llm = LLM()
    print("Building vocabulary cache...")
    vocab = Vocab(llm)
    print(f"Ready. Vocab size: {vocab.size}\n")

    results: list[FunctionCall] = []
    for i, p in enumerate(prompts, 1):
        chosen = pick_best_function(llm, p.prompt, functions)
        params = fill_parameters(llm, vocab, p.prompt, chosen)
        call = FunctionCall(prompt=p.prompt, name=chosen.name, parameters=params)
        results.append(call)
        print(f"[{i:2}] {p.prompt}")
        print(f"     -> {chosen.name}({params})")

    save_function_calls(args.output, results)
    print(f"\nWrote {len(results)} call(s) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

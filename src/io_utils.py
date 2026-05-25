"""File I/O for input JSONs and output results, with graceful error handling."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDef, TestPrompt


def load_function_definitions(path: Path) -> list[FunctionDef]:
    """Load and validate functions_definition.json.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        ValueError: if the file is not valid JSON or not the expected schema.
    """
    if not path.exists():
        raise FileNotFoundError(f"Function definitions file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of function definitions.")

    try:
        return [FunctionDef.model_validate(item) for item in raw]
    except ValidationError as e:
        raise ValueError(f"Schema error in {path}:\n{e}") from e


def load_test_prompts(path: Path) -> list[TestPrompt]:
    """Load and validate function_calling_tests.json."""
    if not path.exists():
        raise FileNotFoundError(f"Test prompts file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e

    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON array of test prompts.")

    try:
        return [TestPrompt.model_validate(item) for item in raw]
    except ValidationError as e:
        raise ValueError(f"Schema error in {path}:\n{e}") from e


def save_function_calls(path: Path, calls: list[FunctionCall]) -> None:
    """Write the final array of function calls to disk as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [call.model_dump() for call in calls]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

"""Pydantic data models for input and output schemas."""

from typing import Any, Literal

from pydantic import BaseModel

ParamType = Literal["number", "integer", "string", "boolean", "array", "object"]


class TypeSpec(BaseModel):
    """A single typed slot, e.g. {"type": "number"}."""

    type: ParamType


class FunctionDef(BaseModel):
    """One entry from functions_definition.json."""

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class TestPrompt(BaseModel):
    """One entry from function_calling_tests.json."""

    prompt: str


class FunctionCall(BaseModel):
    """One entry in the output array."""

    prompt: str
    name: str
    parameters: dict[str, Any]

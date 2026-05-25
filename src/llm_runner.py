"""LLM wrapper + constrained decoding for function names and parameter values."""

import math
from typing import Any, Callable

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]

from src.models import FunctionDef


def _logsumexp(values: list[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def _argmax_in(logits: list[float], allowed: set[int]) -> int | None:
    """Index of the highest logit among `allowed`."""
    best_id: int | None = None
    best_score = float("-inf")
    for tok in allowed:
        if logits[tok] > best_score:
            best_score = logits[tok]
            best_id = tok
    return best_id


class LLM:
    """Thin wrapper around Small_LLM_Model exposing only plain-Python types."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._sdk = Small_LLM_Model(model_name=model_name)

    def encode(self, text: str) -> list[int]:
        """Text to flat list of token ids."""
        return self._sdk.encode(text).tolist()[0]  # type: ignore[no-any-return]

    def decode(self, token_ids: list[int]) -> str:
        """Token ids to text."""
        return self._sdk.decode(token_ids)  # type: ignore[no-any-return]

    def logits_after(self, input_ids: list[int]) -> list[float]:
        """Logits for the next token given a prefix of token ids."""
        return self._sdk.get_logits_from_input_ids(input_ids)  # type: ignore[no-any-return]

    def vocab_size(self) -> int:
        """Vocabulary size, inferred from the length of any logits list."""
        return len(self.logits_after([0]))


class Vocab:
    """Cached `token_id -> decoded text` map for fast predicate filtering."""

    def __init__(self, llm: LLM) -> None:
        self.size = llm.vocab_size()
        self.id_to_text: dict[int, str] = {i: llm.decode([i]) for i in range(self.size)}

    def tokens_where(self, predicate: Callable[[str], bool]) -> set[int]:
        """Token ids whose decoded text satisfies `predicate`."""
        return {i for i, txt in self.id_to_text.items() if predicate(txt)}


def build_function_selection_prompt(
    user_prompt: str, functions: list[FunctionDef]
) -> str:
    """Compact prompt nudging the model toward one of the function names."""
    lines = ["Pick one function name for the request below."]
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}")
    lines.append(f"Request: {user_prompt}")
    lines.append("Function: ")
    return "\n".join(lines)


def score_candidate(
    llm: LLM, context_ids: list[int], candidate_ids: list[int]
) -> float:
    """Joint log-probability log P(candidate | context)."""
    total = 0.0
    prefix = list(context_ids)
    for tok in candidate_ids:
        logits = llm.logits_after(prefix)
        total += logits[tok] - _logsumexp(logits)
        prefix.append(tok)
    return total


def pick_best_function(
    llm: LLM, user_prompt: str, functions: list[FunctionDef]
) -> FunctionDef:
    """Return the function whose name maximises log P(name | context).

    Fast path: one forward pass yields next-token logits, then we look up
    each candidate's first-token logit. Tie-breaker: when two candidates
    share their first token, fall back to full sum-of-log-probs scoring.
    """
    context_ids = llm.encode(build_function_selection_prompt(user_prompt, functions))
    logits = llm.logits_after(context_ids)

    by_first: dict[int, list[FunctionDef]] = {}
    for fn in functions:
        toks = llm.encode(fn.name)
        if toks:
            by_first.setdefault(toks[0], []).append(fn)
    if not by_first:
        return functions[0]

    best_first = max(by_first.keys(), key=lambda t: logits[t])
    contenders = by_first[best_first]
    if len(contenders) == 1:
        return contenders[0]

    best_fn, best_score = contenders[0], float("-inf")
    for fn in contenders:
        s = score_candidate(llm, context_ids, llm.encode(fn.name))
        if s > best_score:
            best_score, best_fn = s, fn
    return best_fn


def generate_number(
    llm: LLM,
    vocab: Vocab,
    context_ids: list[int],
    *,
    is_integer: bool = False,
    max_tokens: int = 10,
) -> float | int:
    """Constrained decoding: emit only digit/[-]/[.] tokens until model leaves number-land."""
    chars = set("0123456789-") | ({"."} if not is_integer else set())
    number_tokens = vocab.tokens_where(
        lambda t: bool(t.strip()) and all(c in chars for c in t.strip())
    )

    out: list[int] = []
    prefix = list(context_ids)
    for _ in range(max_tokens):
        logits = llm.logits_after(prefix)
        if out:
            top = max(range(len(logits)), key=lambda i: logits[i])
            if top not in number_tokens:
                break
        best = _argmax_in(logits, number_tokens)
        if best is None:
            break
        out.append(best)
        prefix.append(best)

    text = llm.decode(out).strip()
    if not text or text in {".", "-", "-.", "."}:
        raise ValueError(f"Could not extract a number from tokens {out!r}")
    return int(float(text)) if is_integer else float(text)


def _generate_string_free(
    llm: LLM, vocab: Vocab, context_ids: list[int], *, max_tokens: int
) -> str:
    """Free generation; stop once a newline appears in the decoded output."""
    content_tokens = vocab.tokens_where(lambda t: bool(t))
    out: list[int] = []
    prefix = list(context_ids)
    for _ in range(max_tokens):
        logits = llm.logits_after(prefix)
        best = _argmax_in(logits, content_tokens)
        if best is None:
            break
        out.append(best)
        prefix.append(best)
        if "\n" in llm.decode(out):
            break
    text = llm.decode(out).split("\n")[0].strip()
    if text and text[-1] in {"'", '"'}:
        text = text[:-1].rstrip()
    return text


def _generate_string_from_prompt(
    llm: LLM,
    vocab: Vocab,
    context_ids: list[int],
    user_prompt: str,
    *,
    max_tokens: int,
) -> str:
    """Substring-constrained generation: result must be a substring of user_prompt."""
    candidates = [
        (tid, txt) for tid, txt in vocab.id_to_text.items() if txt and txt in user_prompt
    ]
    stop_tokens = vocab.tokens_where(lambda t: "\n" in t)
    out: list[int] = []
    prefix = list(context_ids)
    decoded = ""
    for _ in range(max_tokens):
        logits = llm.logits_after(prefix)
        allowed = {tid for tid, txt in candidates if (decoded + txt) in user_prompt}
        allowed |= stop_tokens
        best = _argmax_in(logits, allowed)
        if best is None or best in stop_tokens:
            break
        out.append(best)
        prefix.append(best)
        decoded += vocab.id_to_text[best]
    return decoded.strip()


def generate_string(
    llm: LLM,
    vocab: Vocab,
    context_ids: list[int],
    user_prompt: str | None = None,
    *,
    max_tokens: int = 20,
) -> str:
    """Free generation first; fall back to substring extraction when free isn't faithful.

    The fallback catches cases where the model rewrites or over-escapes the
    value (e.g. doubling backslashes in a Windows path) and forces a verbatim
    lift from the user prompt.
    """
    free = _generate_string_free(llm, vocab, context_ids, max_tokens=max_tokens)
    if user_prompt and free and free not in user_prompt:
        sub = _generate_string_from_prompt(
            llm, vocab, context_ids, user_prompt, max_tokens=max_tokens
        )
        if sub:
            return sub
    return free


def generate_boolean(llm: LLM, context_ids: list[int]) -> bool:
    """Pick whichever of 'true' / 'false' has higher log P(value | context)."""
    t_score = score_candidate(llm, context_ids, llm.encode("true"))
    f_score = score_candidate(llm, context_ids, llm.encode("false"))
    return t_score >= f_score


def build_param_prompt(
    user_prompt: str,
    function_def: FunctionDef,
    param_name: str,
    filled_so_far: dict[str, Any],
) -> str:
    """Build context for extracting one parameter's value.

    String params end with `= '` (string-literal mode); other types end with `= `.
    """
    fn = function_def
    pdef = fn.parameters[param_name]
    sig = ", ".join(f"{k}: {v.type}" for k, v in fn.parameters.items())
    lines = [
        "Extract one argument value from the user request.",
        "",
        f"Function: {fn.name}({sig}) -- {fn.description}",
        f"User request: {user_prompt}",
    ]
    for k, v in filled_so_far.items():
        lines.append(f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}")
    lines.append(f"{param_name} = '" if pdef.type == "string" else f"{param_name} = ")
    return "\n".join(lines)


def fill_parameters(
    llm: LLM, vocab: Vocab, user_prompt: str, function_def: FunctionDef
) -> dict[str, Any]:
    """Fill every parameter of `function_def` based on the user prompt."""
    result: dict[str, Any] = {}
    defaults: dict[str, Any] = {"integer": 0, "number": 0.0, "boolean": False}
    for param_name, type_spec in function_def.parameters.items():
        prompt = build_param_prompt(user_prompt, function_def, param_name, result)
        context = llm.encode(prompt)
        ptype = type_spec.type
        try:
            if ptype == "number":
                result[param_name] = generate_number(llm, vocab, context)
            elif ptype == "integer":
                result[param_name] = generate_number(llm, vocab, context, is_integer=True)
            elif ptype == "boolean":
                result[param_name] = generate_boolean(llm, context)
            else:
                result[param_name] = generate_string(llm, vocab, context, user_prompt)
        except ValueError:
            result[param_name] = defaults.get(ptype, "")
    return result

*This project has been created as part of the 42 curriculum by luricci.*

# Call Me Maybe

## Description

`Call Me Maybe` is an introduction to **function calling in LLMs**. Given a
natural-language request like *"What is the sum of 2 and 3?"* and a catalogue
of available functions, the program produces a structured JSON call:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

The novelty is not the LLM itself — a tiny Qwen3-0.6B model that, asked nicely,
emits valid JSON only ~30% of the time — but the technique used to make it
reliable: **constrained decoding**. At every token the model generates, we
mask any choice that would break the expected type. The model can never
produce malformed values because we don't let it. The JSON structure itself
(braces, keys, commas) is produced by Python's `json.dump`, not by the model,
so the output is always parseable by construction.

## Instructions

### Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) (e.g. `pipx install uv`)
- ~1.2 GB free disk for the Qwen3-0.6B model weights (downloaded once on
  first run, cached in `~/.cache/huggingface/`)

### Install

```sh
uv sync
```

This installs `pydantic` and the local `llm_sdk` package plus their
transitive dependencies (torch, transformers, huggingface-hub, used only
by the SDK).

### Run

Default paths (`data/input/*.json` -> `data/output/function_calling_results.json`):

```sh
uv run python -m src
```

Custom paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Convenience targets

```sh
make install     # uv sync
make run         # uv run python -m src
make debug       # run under pdb
make clean       # remove __pycache__, .mypy_cache, ...
make lint        # flake8 . && mypy . (with subject-mandated flags)
make lint-strict # flake8 . && mypy . --strict
```

## Algorithm explanation

The work happens in two stages per user prompt.

### Stage 1 — function selection

We build a short context describing the available functions and the user
request, then ask the LLM for the logits of the very next token. Each
candidate function name is encoded into token IDs; we read each candidate's
**first-token logit** from that single forward pass and rank them.

- When all candidates have **distinct first tokens**, the highest first-token
  logit identifies the winner in one forward pass.
- When two or more candidates **share a first token** (common in practice
  when names share a prefix like `fn_`), we fall back to scoring the full
  sequence by computing the joint log-probability
  `log P(name | context) = Σ log P(token_i | context, token_0..i-1)` for
  each tied candidate. The candidate with the highest sum wins.

This is constrained decoding by construction: the output is *always* one of
the legal function names, because the only "choice" the model makes is
which member of a known set it prefers.

### Stage 2 — parameter value extraction

For each parameter declared in the chosen function we build a fresh context
that includes any already-extracted values, then generate the new value
token by token under **type-specific constraints**:

- **Numbers / integers.** At every step we allow only tokens whose decoded
  text contains exclusively digits, `.`, and `-`. The model is *forced* to
  emit a numeric value. We stop when the model's natural (unconstrained) top
  choice leaves number-land — that is its signal that the number is
  complete.
- **Strings.** Strings cannot be enumerated like digits, so we constrain
  the **boundary** rather than the content. The prompt ends with an opening
  `'`, putting the model in "string literal" mode (a pattern it has seen
  in millions of training examples). We allow any non-empty content token
  and stop as soon as a newline appears, then strip an optional trailing
  closing quote. A second pass — substring-constrained — is used as a
  fallback when the free pass produces a value that is not a substring of
  the user prompt (e.g. when the model rewrites or over-escapes).
- **Booleans.** We score the two candidates `true` and `false` with the
  same sum-of-log-probs trick used for function selection and pick the winner.

After all parameters are filled, the final `FunctionCall` record (prompt,
name, parameters) is appended to the output array and serialised to JSON.

## Design decisions

- **Sum of log-probs, not sum of raw logits or mean of log-probs.** Raw
  logits are unnormalised; summing them favours longer candidates because
  the per-token baseline is positive. Mean of log-probs has the opposite
  bias — once a long name commits to a unique prefix, the remaining tokens
  are highly predictable (log-probs near zero), inflating the mean. Sum of
  log-probs equals `log P(sequence | context)`, the model's actual
  likelihood of generating the candidate, and is the correct quantity to
  maximise.
- **First-token + tie-breaker function picker.** Reading first-token logits
  is a fast path for cases where candidates have distinct first tokens.
  When they collide — which happens when names share a prefix like `fn_` —
  the tie-breaker falls back to full sum-of-log-probs scoring. Correctness
  is preserved in both cases.
- **`Vocab` cache.** Filtering ~152 000 tokens by predicate (e.g. "is this
  numeric?") would be wasteful on every call. The `Vocab` class decodes
  the full vocabulary once at startup into a `{token_id: text}` map,
  allowing cheap O(N) predicate filtering thereafter.
- **String-literal mode via `= '`.** Ending the parameter prompt with a
  single open quote leverages the model's training distribution (it has
  seen countless `key = '...'` patterns) without imposing strong
  instructions that a 0.6 B parameter model would ignore.
- **Pydantic at every boundary.** Inputs are validated with `FunctionDef`,
  `TestPrompt`; outputs with `FunctionCall`. Schema violations turn into
  clean `ValueError`s that the CLI handler reports and exits on, instead
  of stack-traces.
- **`LLM` wrapper class.** The subject forbids direct use of torch /
  transformers. We isolate every SDK call in `src/llm_runner.LLM`, and the
  rest of the codebase only ever sees plain Python lists of ints and floats.

## Performance analysis

Measured on a Linux x86_64 CPU laptop with Qwen3-0.6B in float32:

| Metric                              | Result |
| ----------------------------------- | ------ |
| Function-selection accuracy         | 11/11 (100 %) |
| Full-output accuracy                | 9/11 (~82 %) |
| Wall-clock end-to-end               | ~3:40 |
| Output JSON validity                | 100 % (built by `json.dump`) |

The remaining failures are small-model limits, not bugs:

- The model cannot synthesise regex patterns like `\d+` or
  `[aeiouAEIOU]` on its own — it returns matched literals (e.g. `'34'`)
  or lists vowels (`'a'`) instead. A regex pattern is by definition
  not a substring of the user request, so even the substring-fallback
  cannot rescue these.

A larger model (Qwen3-1.7B or above) would almost certainly close the gap.

## Challenges faced

1. **Scoring rule for function selection.** Initial sum-of-logits picked
   the longest name for every prompt (length bias). Switching to
   mean-log-prob helped most cases but still failed on short names. Sum
   of log-probs — `log P(name | context)` — fixed everything.
2. **String boundary selection.** A first attempt stopped the string at
   the first single quote, but that destroyed strings containing
   apostrophes (`"I'm"`). The fix is to stop only at a newline, and rely
   on the `= '...'` priming for the model to add a closing quote that we
   then strip.
3. **Last-character truncation.** When the model picks a multi-character
   token like `}\n`, naive code stops *before* adding it, losing the `}`.
   The fix is to include the token and split the decoded text at the
   first `\n`.
4. **Five-minute budget on slow hardware.** Keeping the function-selection
   prompt short (every forward pass processes the full prefix) and capping
   per-value `max_tokens` were the main wins.

## Testing strategy

- **Manual smoke test.** Run `uv run python -m src` with the provided
  public inputs and inspect the printed table of `prompt -> name(params)`
  lines.
- **Error path.** `uv run python -m src --input does_not_exist.json` must
  print a single clean error and exit non-zero — verified.
- **Schema check.** Output JSON is loaded back and round-tripped through
  the `FunctionCall` pydantic model — verified.
- **Linting.** `make lint` (flake8 + mypy with the subject-mandated flags)
  must pass on every change.

## Example usage

```sh
# Run with the default data
uv run python -m src
```

Excerpt of `data/output/function_calling_results.json`:

```json
[
  {"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers",
   "parameters": {"a": 2.0, "b": 3.0}},
  {"prompt": "Greet shrek", "name": "fn_greet",
   "parameters": {"name": "shrek"}}
]
```

## Resources

- *Outlines* and the broader constrained-decoding literature on grammar-
  guided generation (read for background only; not used in code, since
  the package is forbidden by the subject).
- HuggingFace tokenizer documentation for BPE and the `Ġ` / `Ċ` byte-level
  conventions.
- The provided `llm_sdk.Small_LLM_Model` wrapper, used exclusively via its
  public surface (`encode`, `decode`, `get_logits_from_input_ids`).

### Use of AI

AI assistance (Claude) was used to:
- Reason through the math of length-biased scoring rules (sum-of-logits
  vs mean-log-prob vs sum-of-log-probs) before committing to the final
  algorithm.
- Generate scaffolding for the pydantic models and the CLI argument
  parser.
- Suggest the first-token plus tie-breaker design for function selection.
- Review draft code for clarity, type hints, and PEP 257 docstrings.

All code, design decisions, and the final wording of this README were
reviewed, understood, and edited by me. I am responsible for every line
and can defend it in evaluation.

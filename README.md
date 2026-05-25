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
mask any choice that would break the JSON schema or the expected type. The
model can never produce malformed output because we don't let it.

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

This installs `numpy`, `pydantic`, and the local `llm_sdk` package plus their
transitive dependencies (torch, transformers, huggingface-hub).

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
```

## Algorithm explanation

The work happens in two stages per user prompt.

### Stage 1 — function selection

We build a short context describing the available functions and the user
request, then ask the LLM for logits of the very next token. **Every
candidate function name has a unique first token after our prompt**, so the
logit values at that single position rank all candidates at once. The
function whose first token has the highest logit wins.

If two candidate names happen to share a first token, we fall back to scoring
the full sequence by computing the joint log-probability
`log P(name | context) = Σ log P(token_i | context, token_0..i-1)` for each
tied candidate. The candidate with the highest sum wins.

This is constrained decoding by construction: the output is *always* one of
the legal function names, because the only "choice" the model makes is which
member of a known set it prefers.

### Stage 2 — parameter value extraction

For each parameter declared in the chosen function we build a fresh context
that includes any already-extracted values, then generate the new value
token by token under **type-specific constraints**:

- **Numbers / integers.** At every step we allow only tokens whose decoded
  text contains exclusively digits, `.`, and `-`. The model is *forced* to
  emit a numeric value. We stop when the model's natural (unconstrained) top
  choice leaves number-land — that is its signal that the number is
  complete.
- **Strings.** Strings cannot be enumerated, so we constrain the boundary
  instead of the content. The prompt ends with an opening `'`, putting the
  model in "string literal" mode (a pattern it has seen in millions of
  training examples). We allow any content token and stop as soon as a
  newline appears, then strip an optional trailing closing quote.
- **Booleans.** We score the two candidates `true` and `false` with the same
  sum-of-log-probs trick used for function selection and pick the winner.

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
- **First-token optimisation for the function picker.** A naive
  implementation scores every candidate token-by-token, costing
  `n_candidates × ~3` forward passes per prompt. We instead read the model's
  logits *once* and rank candidates by the logit of their first token,
  collapsing function selection to a single forward pass. A full-scoring
  tie-breaker handles collisions. On the test sets this is exact (no
  collisions) and a ~15× speed-up for stage 1.
- **`Vocab` cache.** Filtering 151 936 tokens by predicate (e.g. "is this
  numeric?") would be wasteful on every call. The `Vocab` class decodes the
  full vocabulary once at startup into a `{token_id: text}` map, allowing
  cheap O(N) predicate filtering thereafter.
- **String-literal mode via `= '`.** Ending the parameter prompt with a
  single open quote leverages the model's training distribution (it has seen
  countless `key = '...'` patterns) without imposing strong instructions
  that a 0.6 B parameter model would ignore.
- **Pydantic at every boundary.** Inputs are validated with
  `FunctionDef`, `TestPrompt`; outputs with `FunctionCall`. Schema
  violations turn into clean `ValueError`s that the CLI handler reports and
  exits on, instead of stack-traces.
- **`LLM` wrapper class.** The subject forbids direct use of torch /
  transformers. We isolate every SDK call in `src/llm_runner.LLM`, and the
  rest of the codebase only ever sees plain Python lists of ints and floats.

## Performance analysis

Measured on a Linux x86_64 CPU laptop with Qwen3-0.6B in float32:

| Metric                              | Public set | Private set |
| ----------------------------------- | ---------- | ----------- |
| Function-selection accuracy          | 11/11 (100 %) | 11/11 (100 %) |
| Full-output accuracy (moulinette)    | 9/11 (81.8 %) | **10/11 (90.9 %)** |
| Wall-clock end-to-end                | ~3:40 | ~4:30 |
| Moulinette grade                     | **PASSED**  | **PASSED** (≥ 90 %) |

The remaining failures are small-model limits, not bugs:

- **Public (2 failures):** the model cannot synthesise regex patterns like
  `\d+` or `[aeiouAEIOU]` on its own — it returns matched literals (e.g.
  `'34'`) or lists vowels (`'a'`) instead. A regex pattern is by definition
  not a substring of the user request, so even the substring-fallback
  cannot rescue these.
- **Private (1 failure):** the model occasionally drops the leading `/`
  on absolute Unix paths (`home/user/data.json` instead of
  `/home/user/data.json`). The dropped form is still a valid substring of
  the prompt, so the fallback does not trigger.

A larger model (Qwen3-1.7B or above) would almost certainly close both gaps.

## Challenges faced

1. **Scoring rule for function selection.** Initial sum-of-logits picked the
   longest name for every prompt (length bias). Switching to mean-log-prob
   helped most cases but still failed on short names. Sum-of-log-probs —
   `log P(name | context)` — fixed everything.
2. **String boundary selection.** A first attempt stopped the string at the
   first single quote, but that destroyed strings containing apostrophes
   ("`I'm`"). The fix is to stop only at a newline, and rely on the
   `= '...'` priming for the model to add a closing quote that we then
   strip.
3. **Last-character truncation.** When the model picks a multi-character
   token like `}\n`, naive code stops *before* adding it, losing the `}`.
   The fix is to include the token and split the decoded text at the first
   `\n`.
4. **Five-minute budget on slow hardware.** Full-sequence function scoring
   blew past the limit. The first-token optimisation brought the budget
   well under control while preserving accuracy.

## Testing strategy

- **Manual smoke test.** Run `uv run python -m src` with the provided public
  inputs and inspect the printed table.
- **Error path.** `uv run python -m src --input does_not_exist.json` must
  print a single clean error and exit non-zero — verified.
- **Schema check.** Output JSON is loaded back and round-tripped through the
  `FunctionCall` pydantic model — verified.
- **Moulinette grading.** Both `--set public` and `--set private`
  evaluations are run end-to-end (`prepare_exercises`, then
  `grade_student_answers`). Both return `PASSED`.

## Example usage

```sh
# Run with the default data
uv run python -m src

# Run against a custom set
uv run python -m src \
  --functions_definition data_private/input/functions_definition.json \
  --input data_private/input/function_calling_tests.json \
  --output data_private/output/function_calling_results.json
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
  guided generation (read for background only; not used in code, since the
  package is forbidden by the subject).
- HuggingFace tokenizer documentation for BPE and the `Ġ` / `Ċ` byte-level
  conventions.
- The provided `llm_sdk.Small_LLM_Model` wrapper, used exclusively via its
  public surface (`encode`, `decode`, `get_logits_from_input_ids`,
  `get_path_to_vocab_file`).

### Use of AI

AI assistance (Claude) was used to:
- Reason through the math of length-biased scoring rules (sum-of-logits
  vs mean-log-prob vs sum-of-log-probs) before committing to the final
  algorithm.
- Generate scaffolding for the pydantic models and the CLI argument
  parser.
- Suggest the first-token speed optimisation for function selection, with
  the tie-breaker fallback.
- Review draft code for clarity, type hints, and PEP 257 docstrings.

All code, design decisions, and the final wording of this README were
reviewed, understood, and edited by me. I am responsible for every line and
can defend it in evaluation.

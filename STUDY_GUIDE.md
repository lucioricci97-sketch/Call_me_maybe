# 📘 Call Me Maybe — Personal Study Guide

> A complete, beginner-friendly walkthrough of every concept and every file in your project, written for defense day. Read it slowly. By the end, you should be able to open any file, point to any line, and explain it in plain English.

---

## 🗺️ Table of contents

1. [The project at a glance](#1-the-project-at-a-glance)
2. [The 8 foundational concepts](#2-the-8-foundational-concepts)
3. [Pydantic — from zero](#3-pydantic--from-zero)
4. [File `src/models.py` — the contract](#4-file-srcmodelspy--the-contract)
5. [File `src/io_utils.py` — disk + JSON bridge](#5-file-srcio_utilspy--disk--json-bridge)
6. [File `src/__main__.py` — the CLI and the main loop](#6-file-src__main__py--the-cli-and-the-main-loop)
7. [File `src/llm_runner.py` — the heart of the project](#7-file-srcllm_runnerpy--the-heart-of-the-project)
8. [Supporting files (Makefile, pyproject.toml, .flake8)](#8-supporting-files)
9. [⚠️ Three honest risks to know before the defense]Your three answers are essentially **correct** — let me confirm and tighten them, then we'll do a proper "Pydantic from zero" lesson before moving on. You're right to pause here; understanding Pydantic deeply will pay off for the **next two files**.

---

## ✅ Your answers, graded

**1. Why `Literal[...]` instead of `str`?**
> *"Only accepts those exactly named variables — `str` accepts any string."*

✅ **Perfect.** That's the answer. The only polish: instead of "named variables," say "exact string values" — `Literal["a", "b"]` means *"the value must be the string `'a'` or the string `'b'`, nothing else."*

🎤 Defense polish: *"`Literal[...]` whitelists the six legal type names from the JSON Schema spec. `str` would accept any string, including invalid ones like `'numbber'` — `Literal` rejects them at validation time."*

---

**2. Why `dict[str, TypeSpec]` in `FunctionDef` but `dict[str, Any]` in `FunctionCall`?**
> *"In `FunctionDef` it has to be those exact parameters as input; in `FunctionCall` we expect the value and that can be any type."*

✅ **Correct.** Just one phrasing fix — say **"type specifications"** vs **"actual values"**:

🎤 Defense polish: *"`FunctionDef.parameters` describes the **declared types** of each argument (`{"a": {"type": "number"}}`) — a schema. `FunctionCall.parameters` carries the **actual extracted values** (`{"a": 2.0}`) — data. Schemas use `TypeSpec`, values use `Any`."*

---

**3. What happens if `"type": "numbber"` is in the input?**
> *"The program gives a warning, it's like an exception, it doesn't crash thanks to Pydantic."*

✅ **Right direction**, but precision matters here. Two small fixes:

- It's **not a warning** — it's an **exception** (a `ValidationError`). Different things: warnings let the program continue; exceptions stop it unless caught.
- Pydantic *raises* the exception; **your `io_utils.py` catches it and re-raises a clean `ValueError`**; **then `__main__.py` catches that** and prints `[error] <msg>` to stderr and exits with code 1. So three layers cooperate. Without `io_utils` and `__main__`, Pydantic alone would still crash the program — it raises, but doesn't catch.

🎤 Defense polish: *"Pydantic raises a `ValidationError` because `'numbber'` isn't in the `Literal` whitelist. My `io_utils.py` catches it and re-raises as a `ValueError` with a clear message. My `__main__.py` catches `ValueError`, prints `[error] ...` to stderr, and exits with code 1. The user sees one clean line, no stack trace."*

**Overall: 3/3 conceptually correct. Now let's deepen the Pydantic understanding.**

---

# 🐍 Pydantic — explained from zero

## What is it?

**Pydantic is a Python library for data validation using type hints.** That sentence packs a lot, so let me unpack it.

In normal Python, type hints are **just suggestions**. The interpreter ignores them:

```python
def add(a: int, b: int) -> int:
    return a + b

add("hello", "world")   # Python doesn't complain — it concatenates strings!
```

The `: int` is a hint for *humans* and tools like `mypy`, but at runtime Python doesn't enforce it. Pydantic flips this: **it uses those same type hints as *runtime rules***. If you say `name: str`, Pydantic will actually check at runtime that the value is a string and reject anything else.

🔤 **Analogy.** Type hints in plain Python are like signs on a road: *"Speed limit 50."* Drivers can ignore them. Pydantic is the police officer who actually pulls you over.

---

## The two problems Pydantic solves

### Problem 1 — Data from the outside world is dirty

When you load a JSON file, call an API, read a form, etc., the data comes in as a **plain dictionary** of unknown shape. Maybe a field is missing. Maybe a number arrived as a string (`"5"` instead of `5`). Maybe an extra junk field is there. Your downstream code can break in 50 different ways.

### Problem 2 — Writing manual validation is exhausting

Without Pydantic, you'd write code like:

```python
def parse_function_def(data):
    if "name" not in data:
        raise ValueError("missing name")
    if not isinstance(data["name"], str):
        raise ValueError("name must be a string")
    if "description" not in data:
        raise ValueError("missing description")
    if not isinstance(data["description"], str):
        raise ValueError("description must be a string")
    if "parameters" not in data:
        raise ValueError("missing parameters")
    if not isinstance(data["parameters"], dict):
        raise ValueError("parameters must be a dict")
    for k, v in data["parameters"].items():
        if not isinstance(v, dict):
            raise ValueError(...)
        if "type" not in v:
            raise ValueError(...)
        if v["type"] not in ["number", "integer", "string", ...]:
            raise ValueError(...)
    # ...50 more lines...
```

That's tedious, error-prone, and easy to forget a case. **Pydantic generates all of this for you from your type hints.**

---

## How it works — the mechanics

Pydantic has one star player: **`BaseModel`**. Inherit from it, declare your fields with type hints, and you instantly get four superpowers:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
```

### Superpower 1 — Validation

```python
p = Person.model_validate({"name": "Alice", "age": 30})   # ✅ ok
p = Person.model_validate({"name": "Alice", "age": "thirty"})   # 💥 ValidationError
p = Person.model_validate({"name": "Alice"})                    # 💥 missing field
```

You pass a plain dict (e.g. from `json.load`). Pydantic checks every field against the type hint and either gives you a clean Python object or raises a `ValidationError` listing **every** problem at once.

### Superpower 2 — Coercion (smart, predictable conversion)

```python
p = Person.model_validate({"name": "Alice", "age": "30"})   # ✅ — "30" → 30
```

If the value can be sensibly converted, Pydantic does it. Strings of digits become ints, `"true"` becomes `True`, etc. (You can turn this off in strict mode, but the default is usually what you want.)

### Superpower 3 — Attribute access

After validation, you get a real Python object — not a dict:

```python
p.name    # "Alice"  — autocompletes in your IDE!
p.age     # 30       — IDE knows it's an int
```

This is huge: you stop writing `data["name"]` (which can typo and crash at runtime) and start writing `data.name` (which your IDE and `mypy` check).

### Superpower 4 — Serialization (the reverse trip)

```python
p.model_dump()           # → {"name": "Alice", "age": 30}   — back to dict
p.model_dump_json()      # → '{"name":"Alice","age":30}'    — back to JSON string
```

You build Python objects, and Pydantic converts them to JSON-ready dicts for `json.dump`. This is exactly what your `io_utils.save_function_calls` does:

```python
data = [call.model_dump() for call in calls]
json.dump(data, f, indent=2)
```

---

## The full round-trip in YOUR project

This diagram is the *entire* relationship between JSON, Pydantic, and your code:

```
                ┌─────────────────────┐
                │   functions_def     │   <- text file on disk
                │      .json          │
                └──────────┬──────────┘
                           │ json.load(f)
                           ▼
              ┌────────────────────────┐
              │  raw Python dict/list  │   <- untrusted, untyped
              └────────────┬───────────┘
                           │ FunctionDef.model_validate(item)
                           ▼ (Pydantic checks every field)
              ┌────────────────────────┐
              │   FunctionDef object   │   <- trusted, typed, IDE-friendly
              │   fn.name              │
              │   fn.parameters        │
              └────────────┬───────────┘
                           │ ... your logic operates on this ...
                           ▼
              ┌────────────────────────┐
              │ FunctionCall object    │   <- you build these in __main__.py
              └────────────┬───────────┘
                           │ call.model_dump()
                           ▼
              ┌────────────────────────┐
              │   clean dict           │
              └────────────┬───────────┘
                           │ json.dump(data, f)
                           ▼
                ┌─────────────────────┐
                │   results.json      │   <- text file on disk
                └─────────────────────┘
```

**Pydantic is the gatekeeper on both ends.** Everything between the gates is type-safe and clean.

---

## The three Pydantic methods you actually use in this project

Just three. Memorize these and you have everything you need:

| Method | What it does | Where in your code |
|---|---|---|
| `Model.model_validate(dict)` | Take a dict, validate it, return a model object. Raises `ValidationError` on bad input. | `io_utils.py` — for every loaded JSON entry |
| `model.model_dump()` | Take a model object, return a plain dict. | `io_utils.py` `save_function_calls` |
| `Model(field=value, ...)` | Construct a model directly from keyword arguments. Also validates! | `__main__.py` builds each `FunctionCall(prompt=..., name=..., parameters=...)` |

> ⚠️ **Tip:** If you've seen tutorials online using `Model.parse_obj()` or `Model.dict()`, those are **Pydantic v1** names. You're on **Pydantic v2** (the modern one), where they're `model_validate` and `model_dump`. Don't mix them up — graders may notice.

---

## What `Literal[...]` gives you (recap with depth)

```python
ParamType = Literal["number", "integer", "string", "boolean", "array", "object"]

class TypeSpec(BaseModel):
    type: ParamType
```

When Pydantic sees `type: ParamType`, it checks:

1. Is `type` present? ✓
2. Is it a string? ✓
3. Is it **exactly** one of the six listed strings? ✓ or 💥

If you wrote `type: str` instead, only checks 1 and 2 would happen. `"numbber"`, `"flot"`, `"foo"` would all pass. `Literal` adds the third check **for free** with no extra code.

🎤 **Defense one-liner:** *"`Literal[...]` is a type hint that constrains a string to a specific allowed set. Pydantic turns that into automatic enum-style validation — no boilerplate."*

---

## Why `BaseModel` matters in the project (subject requirement)

The subject says: **"All classes must use pydantic for validation."**

That's why every class in your project — `TypeSpec`, `FunctionDef`, `TestPrompt`, `FunctionCall` — inherits from `BaseModel`. The two utility classes (`LLM`, `Vocab` in `llm_runner.py`) are exceptions because they're not data containers — they're behavior wrappers. If an evaluator probes that, your answer is:

🎤 *"All my **data classes** use `BaseModel` per the subject's rule. `LLM` and `Vocab` aren't data classes — they're stateful wrappers around the SDK and a vocab cache, so Pydantic doesn't apply. Every class that represents a JSON shape uses Pydantic."*

That's a strong, honest, precise answer.

---

## ⚡ Pydantic — the five things to take away

1. **`BaseModel` + type hints = automatic runtime validation.** You don't write `if isinstance(...)` chains by hand.
2. **The methods you actually use:** `model_validate` (parse), `model_dump` (serialize), `Model(field=...)` (construct).
3. **`Literal[...]`** is the cleanest way to whitelist allowed string values.
4. **Nested models validate recursively** — `dict[str, TypeSpec]` checks every inner `TypeSpec`.
5. **Errors come out as `ValidationError`** — your code catches and re-raises them as clean `ValueError`s.

---

## 🎤 Quick check before we move onV

Two short questions to lock Pydantic in. No pressure — just to make sure it stuck:

1. **In one sentence: why does Pydantic exist? What problem does it solve that plain Python doesn't?**

2. **In your project's flow, where exactly does `model_validate` get called, and where does `model_dump` get called?** (Just file names + roles — you don't need exact line numbers.)

Answer briefly. Then we move to **`io_utils.py`**, where Pydantic actually *does* its work — and you'll see all of this in motion.(#9-three-honest-risks-to-know-before-the-defense)
10. [Defense Q&A bank](#10-defense-qa-bank)
11. [Final 60-second elevator pitch](#11-final-60-second-elevator-pitch)
12. [Pre-defense checklist](#12-pre-defense-checklist)V

---

## 1. The project at a glance

### The one-sentence summary

> Build a Python program that reads natural-language prompts and writes back a structured JSON saying *which function should be called and with what arguments* — never the answer itself.

### The example the subject uses

- **Input:** *"What is the sum of 2 and 3?"*
- **Wrong (a normal chatbot):** *"The sum of 2 and 3 is 5."*
- **Your output (a function-calling system):**

```json
{"prompt": "What is the sum of 2 and 3?",
 "name": "fn_add_numbers",
 "parameters": {"a": 2.0, "b": 3.0}}
```

You're not building a calculator. You're building a **translator** from English → structured function call. This is exactly how modern AI assistants (ChatGPT plugins, Copilot, AI agents) talk to tools in the real world.

### Inputs and outputs

| File | Role |
|---|---|
| `data/input/functions_definition.json` | The **catalog** of functions your system can call (name, description, parameter types, return type). |
| `data/input/function_calling_tests.json` | The **list of natural-language prompts** to process. |
| `data/output/function_calling_results.json` | The output your program writes — one entry per input prompt. |

### Tools & rules

| Allowed | Why |
|---|---|
| Python ≥ 3.10 | Mandatory. |
| `uv` | Package manager. Graders run `uv sync` then `uv run python -m src`. |
| `pydantic` | Mandatory — for validating input/output JSON schemas. |
| `numpy`, `json` | Allowed. |
| `llm_sdk` (provided) | The wrapper around Qwen3-0.6B. Use only its public methods. |
| `flake8` + `mypy` | Linter and type-checker. Your code must pass them. |
| A `Makefile` | With targets: `install`, `run`, `debug`, `clean`, `lint`. |

| Forbidden | Why |
|---|---|
| `torch`, `transformers`, `huggingface` | The whole point: you cannot use the deep-learning libraries directly. Your `LLM` wrapper hides them. |
| `dspy`, `outlines`, any constrained-decoding library | You must implement constrained decoding yourself. |
| `if "sum" in prompt: return ...` | "No medieval magic." Function choice must come from the model's logits. |

### The model: Qwen3-0.6B

A **tiny** LLM (600 million parameters; frontier models have hundreds of billions). Mandatory because:

- Runs on CPU (about 1.2 GB download).
- It's **bad** at following "please output JSON" — only ~30% success. That's exactly what the project teaches you to fix with **constrained decoding**.

### What "passing" means

- ≥ **90% accuracy** on test prompts (function name + arguments correct).
- **100% valid JSON** output (free in this project — Python writes the JSON, not the model).
- Runs in **under 5 minutes** on standard hardware.
- **Never crashes** — every error caught, clean message printed.
- A comprehensive `README.md`.

### The single most important insight

> The JSON braces are **never written by the model**. Your Python code builds a normal Python dictionary and calls `json.dump()`. The model only supplies the **values** (function name, numbers, strings). So "100% valid JSON" is guaranteed for free by Python. **Constrained decoding ensures the values are correct and correctly typed**, not the braces.

If you can say this one sentence cleanly in your defense, you stand out from almost everyone else.

---

## 2. The 8 foundational concepts

These are *the* questions evaluators ask. Master them and you've passed the hardest part.

### Concept 1 — Tokens and the vocabulary

An LLM does **not** read words or letters. It reads **tokens** — small chunks of text.

- `"hello"` → maybe 1 token: `["hello"]`
- `"fn_add_numbers"` → maybe 3 tokens: `["fn", "_add", "_numbers"]`
- `"42.0"` → 4 tokens: `["4", "2", ".", "0"]`

The model has a fixed dictionary of every token it knows — the **vocabulary**. Qwen3 has about **151,936** tokens. Each token has a unique integer **ID**.

🔤 **Analogy.** A numbered LEGO catalog. Every brick has an ID. The model builds sentences only out of bricks from the catalog.

📁 **In your code.** `LLM.encode("fn_greet")` returns the list of IDs. `LLM.decode([1234, 5678])` returns the text. The `Vocab` class decodes every ID once at startup.

🎤 **Defense answer:** *"A token is a chunk of text — a word, a subword, a character, or punctuation — that the tokenizer extracts using a BPE algorithm. Each token has a unique integer ID. Qwen3's vocabulary has about 152,000 tokens. The model only works with IDs; my code uses `encode` and `decode` to move between text and IDs."*

---

### Concept 2 — Generation is a loop, not a sum

This is the **most commonly misunderstood** point. Read carefully.

The model does one thing: given a prefix of tokens, it outputs a score (called a **logit**) for every token in the vocabulary, answering *"how good is each token as the very next one?"*

To produce text:

1. Feed the current prefix to the model → get logits for the next token.
2. Pick one token (`argmax`, or restricted-argmax for constrained decoding).
3. **Append** it to the prefix.
4. Go back to step 1 with the new, longer prefix.
5. Stop when a stop condition fires (newline, max length, model leaves "number-land", etc.).

🔤 **Analogy.** Predictive text on your phone, but with 152k options instead of 3, and *you* control which one gets accepted.

📁 **In your code.** Every generation loop in `llm_runner.py` follows this pattern. Example, `generate_number`:

```python
for _ in range(max_tokens):
    logits = llm.logits_after(prefix)   # step 1
    best = _argmax_in(logits, ...)      # step 2 (constrained)
    out.append(best)                    # step 3 (record)
    prefix.append(best)                 # step 4 (extend)
    # step 5: stop conditions are checked
```

🎤 **Defense answer:** *"Generation is a loop. I feed the prefix to the model, get logits, argmax to pick a token, append it to the prefix, and repeat. Summing log-probs is a different operation — that's for **comparing fixed candidates** like `'true'` vs `'false'`, not for generating new text."*

---

### Concept 3 — Logits (raw scores)

The ~152,000 numbers the model outputs are called **logits**. Key facts:

- **Raw and unnormalized** — not probabilities.
- Typically range from about **−15 to +25**.
- Higher = the model finds that token more plausible.
- They do **not** sum to 1.

🔤 **Analogy.** Judges' raw scorecards before any conversion to percentages.

📁 **In your code.** The `logits` list returned by `LLM.logits_after(prefix)`.

🎤 **Defense answer (memorize this verbatim):** *"A logit is a raw, unnormalized score the model assigns to each vocabulary token as a candidate for the next position. It's not a probability; logits range roughly from −15 to +25. A higher logit means the model finds that token more plausible."*

---

### Concept 4 — argmax vs max (THE confusing pair)

They sound similar; they answer different questions.

Pretend the vocab has only 5 tokens:

| Token ID | Logit |
|---:|---:|
| 0 | 1.2 |
| 1 | **7.5** |
| 2 | 0.3 |
| 3 | −2.1 |
| 4 | 4.8 |

- `max(logits)` → **7.5** (the highest value).
- `argmax(logits)` → **1** (the index of the highest value).

We want the **index**, because the index IS the token ID — and token ID 1 is the next token. The number 7.5 is just a score; it doesn't tell us which token to use.

🔤 **Analogy.** Talent show: `max` says *"the winning score was 7.5"* (useless if you want to give the trophy). `argmax` says *"contestant #1 won"* (now you can hand over the trophy).

📁 **In your code.** Python has no built-in `argmax`, so the code uses this idiom:

```python
top = max(range(len(logits)), key=lambda i: logits[i])
```

It loops over indices and returns the index whose `logits[i]` is largest. Same as `argmax`.

**Memory hook:** `max` gives the **prize**; `argmax` gives the **winner**.

---

### Concept 5 — Softmax (and why you skip it)

**Softmax** turns logits into real probabilities:

```
prob[i] = exp(logit[i]) / Σ exp(logit[j])
```

After softmax: every value is in [0, 1], everything sums to 1.

**Key property:** softmax **preserves order**. The biggest logit becomes the biggest probability. So:

```
argmax(logits) == argmax(softmax(logits))
```

That's why your code **never bothers with softmax** when it just needs the winner — `argmax` on raw logits gives the same answer, faster.

🎤 **Defense answer:** *"Softmax converts logits into probabilities — values in [0, 1] that sum to 1. I don't compute it in my code because it preserves ordering: `argmax` on the raw logits picks the same token. I only compute probabilities (via log-probs) when comparing whole candidate sequences."*

---

### Concept 6 — Log-probabilities and summing them

When you want to compare entire **candidate sequences** (like `"true"` vs `"false"`, or two function names), you need the probability the model would generate the whole thing:

```
P("fn_greet" | context) = P(t1) · P(t2 | t1) · P(t3 | t1,t2)
```

Multiplying many small probabilities **underflows** to zero on a computer. The fix: take the logarithm. Logs turn multiplication into addition:

```
log P("fn_greet" | context) = log P(t1) + log P(t2|t1) + log P(t3|t1,t2)
```

So to score a candidate, **sum its tokens' log-probabilities**. Log-probs are always ≤ 0 (0 means certainty).

📁 **In your code.** `score_candidate()` in `llm_runner.py` does this. Used in:
- `pick_best_function`'s tie-breaker (compare full sequences when first tokens collide).
- `generate_boolean` (`"true"` vs `"false"`).

🎤 **Defense answers** (three classic questions):
- *"Why log-probs instead of probabilities?"* → **"Numerical stability — probabilities multiply and underflow; logs add."**
- *"Why **sum** them?"* → **"Sum of log-probs = log of the product of probabilities = `log P(sequence)`. That's the model's actual likelihood of generating the candidate — the quantity to maximize."**
- *"Why not sum the raw logits?"* → **"Raw logits are unnormalized; their absolute scale is meaningless. Summing them is biased toward longer candidates because positive logits dominate. Log-probs are normalized (always ≤ 0), so summing is a legitimate probability calculation."**

---

### Concept 7 — log-sum-exp (the safe denominator)

To turn one logit into a log-prob:

```
log_prob[i] = logit[i] − log(Σ exp(logit[j]))
```

The term `log(Σ exp(logit[j]))` is the **log-sum-exp**. The danger: `exp(25)` is huge, and `exp(700)` is infinity → overflow. The trick: subtract the maximum first.

```python
m = max(values)
return m + math.log(sum(math.exp(v - m) for v in values))
```

After subtracting `m`, the largest exponent is `exp(0) = 1` — no overflow. Adding `m` back at the end recovers the right answer.

📁 **In your code.** `_logsumexp()` in `llm_runner.py`, used by `score_candidate`.

🎤 **Defense answer:** *"Subtracting `max(values)` before applying `exp` is a numerical-stability trick. Without it, `exp` of a large logit can overflow. Subtracting a constant from every term and adding it back at the end leaves the math unchanged but keeps every `exp` ≤ 1."*

---

### Concept 8 — Masking with −∞ (this IS constrained decoding)

To force the model to never pick certain tokens, set their logits to **negative infinity** before `argmax`. Then `argmax` can never choose them (any other number is bigger), and after softmax their probability is `exp(−∞) = 0`.

**Why −∞ and not 0?** Because `0` is a perfectly normal — even good — logit. Many real tokens score around 0. Setting an illegal token's logit to 0 would let it *still win* the argmax. Only `−∞` truly forbids it.

Your code does this **more cleverly** with `_argmax_in(logits, allowed)`: instead of building a 152k-long masked array, it just searches for the maximum among the allowed IDs. Mathematically identical, but cheaper.

🔤 **Analogy.** A multiple-choice test where someone crosses out the wrong answers before you choose — you *can't* pick them.

📁 **In your code.** `_argmax_in` is the masking primitive. The "allowed" sets come from `Vocab.tokens_where(predicate)` (e.g. "only tokens whose decoded text is digits").

🎤 **Defense answers:**
- *"Why mask with −∞ not 0?"* → **"`0` is a normal logit value; many good tokens score near 0. Setting an illegal token to 0 wouldn't forbid it. `−∞` guarantees it can never win the argmax, and after softmax its probability is exactly `exp(−∞) = 0`."**
- *"Where is constrained decoding in your code?"* → **"`_argmax_in(logits, allowed)` is the masking primitive. We use it in three places: function-name selection (allowed = the candidate names' first tokens), number generation (allowed = digit/`.`/`-` tokens), and string generation (allowed = non-empty content tokens, or substring-of-prompt tokens in the fallback)."**

---

### How the 8 concepts connect

> The model is a **next-token predictor** that outputs **logits** (raw scores). To pick one we use **argmax**; we skip **softmax** because it doesn't change the ranking. To compare whole candidate sequences we convert logits to **log-probabilities** (using stable **log-sum-exp**) and **sum** them. To force valid output we do **constrained decoding** by **masking illegal tokens with −∞** (in our code, just argmax over the allowed set). The JSON itself is produced by Python's `json.dump`, not by the model.

---

## 3. Pydantic — from zero

### What it is

> Pydantic is a Python library for **data validation using type hints**.

In plain Python, type hints are just suggestions: `def add(a: int, b: int)` will still accept `add("hi", "there")` at runtime. Pydantic flips this: it uses the same hints as **runtime validation rules**.

🔤 **Analogy.** Type hints in plain Python are road signs: *"speed limit 50"* — drivers can ignore them. Pydantic is the police officer who actually pulls you over.

### The two problems Pydantic solves

1. **Data from outside is dirty.** JSON files, APIs, forms — you never know what shape you'll get.
2. **Manual validation is exhausting.** Without Pydantic you'd write dozens of `if isinstance(...)` checks.

### The star player: `BaseModel`

Inherit from `BaseModel`, declare fields with type hints, and you get four superpowers:

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
```

**Superpower 1 — Validation.**
```python
Person.model_validate({"name": "Alice", "age": 30})       # ✅ ok
Person.model_validate({"name": "Alice", "age": "thirty"}) # 💥 ValidationError
Person.model_validate({"name": "Alice"})                  # 💥 missing field
```

**Superpower 2 — Coercion.** `{"age": "30"}` becomes `age=30` (string converted to int when possible).

**Superpower 3 — Attribute access.**
```python
p = Person.model_validate({"name": "Alice", "age": 30})
p.name   # "Alice" — your IDE autocompletes
p.age    # 30
```

**Superpower 4 — Serialization.**
```python
p.model_dump()       # {"name": "Alice", "age": 30}
p.model_dump_json()  # '{"name":"Alice","age":30}'
```

### The full round-trip in YOUR project

```
   functions_definition.json   ←  text on disk
            │
            │ json.load(f)
            ▼
       raw dict/list           ←  untrusted
            │
            │ FunctionDef.model_validate(item)
            ▼ (Pydantic checks every field)
   FunctionDef object          ←  trusted, typed
            │
            │ ... your logic ...
            ▼
   FunctionCall object         ←  you build these in __main__.py
            │
            │ call.model_dump()
            ▼
       clean dict
            │
            │ json.dump(data, f)
            ▼
   results.json                ←  text on disk
```

### The three Pydantic methods you actually use

| Method | What it does | Where in your code |
|---|---|---|
| `Model.model_validate(dict)` | Validate a dict, return a model object. | `io_utils.py` |
| `model.model_dump()` | Convert model to plain dict. | `io_utils.save_function_calls` |
| `Model(field=value, ...)` | Construct + validate. | `__main__.py` builds `FunctionCall(...)` |

⚠️ If you've seen `parse_obj()` or `.dict()` online, those are **Pydantic v1**. You use **Pydantic v2**: `model_validate` and `model_dump`. Don't mix them up.

### Why this matters for the subject

The subject says **"All classes must use pydantic for validation."** Every data class in your project (`TypeSpec`, `FunctionDef`, `TestPrompt`, `FunctionCall`) inherits from `BaseModel`. The exceptions (`LLM`, `Vocab`) are not data classes — they're behavior wrappers.

🎤 **Defense answer if asked why `LLM` doesn't inherit from `BaseModel`:** *"All my **data classes** use `BaseModel` per the subject's rule. `LLM` and `Vocab` are not data classes — `LLM` is a stateful wrapper around the SDK, and `Vocab` is a cache. Pydantic doesn't apply. Every class that represents a JSON shape uses Pydantic."*

---

## 4. File `src/models.py` — the contract

### What this file does

Uses Pydantic to declare **four shape templates** for the project's JSON. Before any logic runs, you tell Python: *"a function definition has these fields, a test prompt has this field, my output has these three fields."* This file is **the contract** between your program and the outside world.

### Full file

```python
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
```

### Line-by-line

**Line 1 — docstring.** PEP 257 one-liner summarizing the file. Required by the subject.

**Line 3 — `from typing import Any, Literal`.**
- `Any` = "any type at all." Used for parameter *values* (could be number/string/bool).
- `Literal[...]` = "this value must be **exactly** one of these specific strings." Pydantic turns this into automatic enum-style validation.

**Line 5 — `from pydantic import BaseModel`.** The magic class that turns a regular Python class into a smart JSON parser.

**Line 7 — `ParamType = Literal[...]`.** A named alias for "one of these six type strings." We give it a name so we can reuse it in two places (`TypeSpec.type`, plus nested in `FunctionDef`).

🎤 **Defense:** *"`Literal[...]` whitelists the six legal type names from the JSON Schema spec. `str` would accept any string, including `'numbber'`. `Literal` rejects them at validation time — free schema validation with no manual checks."*

**Lines 10–13 — `TypeSpec`.**
```python
class TypeSpec(BaseModel):
    type: ParamType
```
Models `{"type": "number"}`. One field. Appears in two places (per parameter, and in `returns`).

**Lines 16–21 — `FunctionDef`.**
```python
class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec
```
Models one entry from `functions_definition.json`:
- `name`, `description`: strings.
- `parameters`: dict mapping argument names to `TypeSpec`. Pydantic **recursively** validates each inner `TypeSpec`.
- `returns`: a single `TypeSpec`.

**Lines 24–27 — `TestPrompt`.**
```python
class TestPrompt(BaseModel):
    prompt: str
```
One field. Even though it's tiny, keeping it as a model is consistent and forward-compatible.

**Lines 30–35 — `FunctionCall` (the output shape).**
```python
class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]
```
- `prompt`: the original NL question, echoed.
- `name`: function chosen by the model.
- `parameters: dict[str, Any]`: **actual values** (`{"a": 2.0}`), not type-specs — hence `Any`.

🎤 **Defense:** *"In `FunctionDef`, `parameters` is `dict[str, TypeSpec]` — declared types. In `FunctionCall`, `parameters` is `dict[str, Any]` — actual extracted values. One is a schema; the other is data."*

### Likely evaluator probes

- *"What if the JSON says `'type': 'wibble'`?"* → "Pydantic raises `ValidationError`; `io_utils` re-raises as `ValueError`; `__main__.py` catches and prints `[error] ...` to stderr with exit 1."
- *"Why a separate class for `{type: ...}`?"* → "Used in two places — per parameter and for `returns`. DRY + recursive validation."
- *"Why not validate in `models.py` itself?"* → "These classes only **define** the shape. Validation fires when `model_validate` is called — in `io_utils.py`."

### A potential "modify the project" task

- Add a `required: bool = True` field to `TypeSpec` → one line.
- Add a `"null"` to `ParamType` → one word edit.
- Add `expected_function: Optional[str] = None` to `TestPrompt` → import `Optional`, add a line.

---

## 5. File `src/io_utils.py` — disk + JSON bridge

### What this file does

The **bridge** between your program and the JSON files on disk. Three functions: two for reading (one per input file) and one for writing. Every function defends against four failure modes (missing file, bad JSON, wrong shape, schema mismatch) and produces either typed Python objects or a clean exception.

### Three new concepts before reading the code

**A. `Path` from `pathlib`.** A smart, object-oriented path. Methods like `.exists()`, `.open()`, `.parent`, `.mkdir()`. Cross-platform safe.

**B. `with` (context manager).** `with path.open("r") as f:` guarantees the file is closed even on exception. **Mandated by the subject** for resources.

**C. `raise ... from e` (exception chaining).** Catch one exception, raise another in its place, keep a link to the original. Lets you give the caller a friendlier exception type while preserving debug info.

### Full file

```python
"""File I/O for input JSONs and output results, with graceful error handling."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDef, TestPrompt


def load_function_definitions(path: Path) -> list[FunctionDef]:
    """Load and validate functions_definition.json."""
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
```

### `load_function_definitions` — line by line

**Signature & docstring:** takes a `Path`, returns `list[FunctionDef]`. Caller never sees raw dicts.

**Check 1 — does the file exist?**
```python
if not path.exists():
    raise FileNotFoundError(f"Function definitions file not found: {path}")
```
Friendly message with the path. Early exit.

**Check 2 — is it valid JSON?**
```python
try:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
except json.JSONDecodeError as e:
    raise ValueError(f"Invalid JSON in {path}: {e}") from e
```
- `path.open("r", encoding="utf-8")` — read mode, UTF-8 (matters for accented characters).
- `json.load(f)` — parses file text into Python data.
- If syntax is broken (e.g. trailing comma), catch `json.JSONDecodeError`, re-raise as `ValueError` with friendly path.

**Check 3 — is the top level a list?**
```python
if not isinstance(raw, list):
    raise ValueError(f"{path} must contain a JSON array of function definitions.")
```
Without this, `for item in raw:` on a dict would silently iterate **keys** — confusing later failure.

**Check 4 — does each entry match the schema?**
```python
try:
    return [FunctionDef.model_validate(item) for item in raw]
except ValidationError as e:
    raise ValueError(f"Schema error in {path}:\n{e}") from e
```
List comprehension: validate each item. Pydantic's `ValidationError` is verbose and helpful — we include its message.

### `load_test_prompts` — identical pattern

Same four checks against `TestPrompt`. Why not factor it out? Defensible answer:

🎤 *"I could factor a generic `load_json_list(path, model)`. I kept two separate functions because (1) they're used in only one place each, so DRY savings are minimal; (2) explicit names give better error messages; (3) for a small codebase, clarity beats abstraction. A bigger codebase would justify the refactor."*

### `save_function_calls` — three lines

```python
def save_function_calls(path: Path, calls: list[FunctionCall]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [call.model_dump() for call in calls]
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
```

- **`path.parent.mkdir(parents=True, exist_ok=True)`** — create the output directory if it doesn't exist. Like `mkdir -p`. Subject says output goes to `data/output/`; graders may not have created it.
- **`data = [call.model_dump() for call in calls]`** — convert each Pydantic object to a plain dict. `json.dump` doesn't know about Pydantic.
- **`with path.open("w", ...) ... json.dump(data, f, indent=2)`** — write mode (overwrites), UTF-8, 2-space indent for readability.

### A clarification an evaluator might fish for

`json.JSONDecodeError` is actually a subclass of `ValueError`, so technically the catch-and-re-raise isn't strictly necessary. The win is **the friendlier message including the file path**. If asked:

🎤 *"`JSONDecodeError` already inherits from `ValueError`, so even without the catch, `__main__` would catch it. The catch-and-re-raise is purely cosmetic — it adds the file path to the message. The structure is for clarity, not type-narrowing."*

---

## 6. File `src/__main__.py` — the CLI and the main loop

### What this file does

The **entry point**. It parses command-line arguments, loads the inputs, builds the LLM and vocabulary cache, loops over every prompt (calling stage 1 + stage 2), and writes the output. It's also **the only place** that catches exceptions — making sure the program never crashes with a stack trace.

### Two new concepts before reading the code

**A. `argparse`.** Python's built-in command-line argument parser. You declare each flag (`--input`, `--output`) and it produces a friendly `--help` automatically.

**B. `if __name__ == "__main__":`.** This idiom asks "am I being run directly, or imported?" When you run `python -m src`, Python's import machinery sets `__name__ == "__main__"` for the entry script. So `sys.exit(main())` only runs when launched, not when imported. It's the standard Python entry-point idiom.

### Full file

```python
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
```

### Walkthrough

**Imports.** Standard library (`argparse`, `sys`, `pathlib.Path`) + your own modules. Nothing surprising.

**`parse_args()`** — creates an `ArgumentParser` and declares three optional flags, each with `type=Path` (so argparse converts strings to `Path` automatically) and `default=...` matching the subject's defaults. Returns a `Namespace` object you access as `args.input`, `args.output`, etc.

🎤 *"Why defaults matching `data/input/`?"* → "Subject requires the program to work with no arguments using the default paths."

**`main()`** — the orchestrator. Returns an `int` shell exit code (0 success, 1 failure).

**Step 1 — load inputs in a single `try`:**
```python
try:
    functions = load_function_definitions(args.functions_definition)
    prompts = load_test_prompts(args.input)
except (FileNotFoundError, ValueError) as e:
    print(f"[error] {e}", file=sys.stderr)
    return 1
```
This is **the only `except` in the whole project**. `io_utils.py` collapsed four failure modes into two exception types — so we catch those two and print a clean message to **stderr** (not stdout — error channel). Return 1 = shell sees failure.

🎤 *"Why two exception types, not five?"* → "`io_utils.py` translates `JSONDecodeError`, `ValidationError`, etc. into `FileNotFoundError` and `ValueError`. Keeping `__main__` shallow."

**Step 2 — load the model and build the vocab cache:**
```python
llm = LLM()
vocab = Vocab(llm)
```
`LLM()` downloads/loads Qwen3-0.6B (about 1.2 GB, cached after first time). `Vocab(llm)` decodes every token ID once (~10 s) so later filtering is fast.

🎤 *"Why build the vocab cache here, once?"* → "Building it costs ~10 s but we'll need it for every prompt. Building once and reusing is essential to fit the 5-minute budget."

**Step 3 — main loop:**
```python
for i, p in enumerate(prompts, 1):
    chosen = pick_best_function(llm, p.prompt, functions)
    params = fill_parameters(llm, vocab, p.prompt, chosen)
    call = FunctionCall(prompt=p.prompt, name=chosen.name, parameters=params)
    results.append(call)
    print(f"[{i:2}] {p.prompt}")
    print(f"     -> {chosen.name}({params})")
```

For each prompt:
1. **Stage 1 — `pick_best_function`** picks one function from the catalog.
2. **Stage 2 — `fill_parameters`** extracts one value per parameter.
3. Build a `FunctionCall` (Pydantic validates **here** too — the third validation point).
4. Print a status line.

`enumerate(prompts, 1)` numbers prompts from 1 (more human-friendly than 0 in output).

**Step 4 — save:**
```python
save_function_calls(args.output, results)
print(f"\nWrote {len(results)} call(s) to {args.output}")
return 0
```

**Entry guard:**
```python
if __name__ == "__main__":
    sys.exit(main())
```
Runs `main()` only when this script is launched directly (not when imported). Passes its return value to `sys.exit`, which becomes the shell exit code.

### A subtle correctness point you should know

This is the **third place** Pydantic validation runs:

```python
call = FunctionCall(prompt=p.prompt, name=chosen.name, parameters=params)
```

Constructing a `FunctionCall(...)` from keyword args also validates. So if `fill_parameters` somehow returned something weird, the program would still catch it here. That's a free safety net.

🎤 *"Where exactly is Pydantic validation triggered?"* → "Three places: in `io_utils.load_function_definitions` and `load_test_prompts` (via `model_validate`), and here in `__main__.py` when constructing each `FunctionCall(...)`."

### Likely evaluator probes

- *"Walk me through one prompt end-to-end."* → Use the elevator pitch in Section 11.
- *"What happens if the input file is missing?"* → "`io_utils.load_function_definitions` raises `FileNotFoundError`; `__main__.py` catches, prints `[error] Function definitions file not found: ...` to stderr, exits 1. No crash."
- *"Why print to stderr, not stdout?"* → "Convention: errors go to stderr so they're separated from normal output. Lets graders pipe stdout to `/dev/null` and still see errors."

### Potential "modify the project" task

- Add a `--verbose` flag → `parser.add_argument("--verbose", action="store_true")`, then `if args.verbose: print(...)`.
- Print a summary count at the end → already there, just adjust the message.
- Add a `--limit N` flag to only process the first N prompts → 2 lines.

---

## 7. File `src/llm_runner.py` — the heart of the project

### What this file does

Everything interesting. **The LLM wrapper, the vocabulary cache, all the constrained-decoding logic.** This is where the project's hard ideas live. Read this section twice.

The file has seven logical pieces:

1. **Two math helpers** — `_logsumexp`, `_argmax_in`.
2. **The `LLM` wrapper class** — isolates `torch`/`transformers` so the subject's rule isn't broken.
3. **The `Vocab` cache** — `{token_id: text}` map for fast predicate filtering.
4. **Stage 1: function selection** — `build_function_selection_prompt`, `score_candidate`, `pick_best_function`.
5. **Stage 2 generators** — `generate_number`, `generate_string` (+ two helpers), `generate_boolean`.
6. **Parameter prompt builder** — `build_param_prompt`.
7. **The stage-2 orchestrator** — `fill_parameters`.

### 7.1 The math helpers

```python
def _logsumexp(values: list[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))
```

Concept 7 in code. Subtract the max so `exp` doesn't overflow, add it back at the end. The leading underscore means "private to this module."

```python
def _argmax_in(logits: list[float], allowed: set[int]) -> int | None:
    """Index of the highest logit among `allowed`."""
    best_id: int | None = None
    best_score = float("-inf")
    for tok in allowed:
        if logits[tok] > best_score:
            best_score = logits[tok]
            best_id = tok
    return best_id
```

Concept 8 in code. **This IS the masking primitive.**

- Initialize `best_score = -inf` so any real logit beats it on the first iteration.
- Loop over **only the allowed token IDs** — not all 152k. That's why this is faster than building a masked array.
- Return the **ID** (token), not the score. If `allowed` is empty, return `None`.

🎤 **Defense:** *"`_argmax_in` is mathematically equivalent to setting every logit outside `allowed` to `−∞` and then taking argmax — but cheaper, because we never materialize a 152k-long array. We just search the maximum among the IDs we care about."*

### 7.2 The `LLM` wrapper class

```python
class LLM:
    """Thin wrapper around Small_LLM_Model exposing only plain-Python types."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        self._sdk = Small_LLM_Model(model_name=model_name)

    def encode(self, text: str) -> list[int]:
        return self._sdk.encode(text).tolist()[0]

    def decode(self, token_ids: list[int]) -> str:
        return self._sdk.decode(token_ids)

    def logits_after(self, input_ids: list[int]) -> list[float]:
        return self._sdk.get_logits_from_input_ids(input_ids)

    def vocab_size(self) -> int:
        return len(self.logits_after([0]))
```

**Why this class exists.** The subject forbids `import torch` and `import transformers`. The provided `Small_LLM_Model` uses them internally and sometimes returns `torch.Tensor`s. `LLM` is the **firewall**: every method converts to/from plain Python lists, so the rest of the code never touches a tensor.

- `__init__` constructs the underlying `Small_LLM_Model` and stores it as `self._sdk`. The `_` says "internal use only."
- `encode(text) → list[int]`. The SDK returns a 2-D tensor `[[ids]]`; `.tolist()[0]` flattens to a 1-D Python list.
- `decode(ids) → str`.
- `logits_after(prefix) → list[float]`. The next-token logits for the given prefix.
- `vocab_size() → int`. **Clever trick:** ask for logits of any prefix (`[0]`), and count how many you got back. The vocab size equals the logits length. No separate API needed.

🎤 *"Why a wrapper at all?"* → "Two reasons. (1) The subject forbids using torch/transformers; this class isolates every SDK call so the rest of my code only sees plain Python types. (2) Single seam — if I ever swap the model SDK, only this class changes."

### 7.3 The `Vocab` cache

```python
class Vocab:
    """Cached `token_id -> decoded text` map for fast predicate filtering."""

    def __init__(self, llm: LLM) -> None:
        self.size = llm.vocab_size()
        self.id_to_text: dict[int, str] = {i: llm.decode([i]) for i in range(self.size)}

    def tokens_where(self, predicate: Callable[[str], bool]) -> set[int]:
        """Token ids whose decoded text satisfies `predicate`."""
        return {i for i, txt in self.id_to_text.items() if predicate(txt)}
```

**Purpose:** to **filter the vocabulary by predicate** (e.g. "give me all token IDs whose decoded text contains only digits") many times during generation. Without a cache, we'd re-decode 152k tokens every call.

- `__init__`: decode every token ID once, store the `{id: text}` map.
- `tokens_where(predicate)`: returns a `set` of IDs whose text passes `predicate`. Returning a `set` matters — `_argmax_in` iterates it.

**Cost.** ~10 seconds at startup, paid once. Every later filter is a fast set comprehension.

🎤 *"Why a set, not a list?"* → "Two reasons. (1) Membership tests like `if tok in allowed` are O(1) on sets, O(n) on lists. (2) Order doesn't matter — we just need to know what's allowed."

### 7.4 Stage 1 — function selection

#### `build_function_selection_prompt`

```python
def build_function_selection_prompt(user_prompt: str, functions: list[FunctionDef]) -> str:
    lines = ["Pick one function name for the request below."]
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}")
    lines.append(f"Request: {user_prompt}")
    lines.append("Function: ")
    return "\n".join(lines)
```

Builds a short text prompt:

```
Pick one function name for the request below.
- fn_add_numbers: Add two numbers together and return their sum.
- fn_greet: Generate a greeting message for a person by name.
- ...
Request: What is the sum of 2 and 3?
Function: 
```

Keeping it short matters: every forward pass processes the whole prefix, so shorter prompt = faster inference.

🎤 *"Why end with `Function: `?"* → "Priming. After seeing `Function: `, the model's most natural continuation is a function name. The logits at this position are exactly what I want to read for stage 1."

#### `score_candidate`

```python
def score_candidate(llm: LLM, context_ids: list[int], candidate_ids: list[int]) -> float:
    """Joint log-probability log P(candidate | context)."""
    total = 0.0
    prefix = list(context_ids)
    for tok in candidate_ids:
        logits = llm.logits_after(prefix)
        total += logits[tok] - _logsumexp(logits)
        prefix.append(tok)
    return total
```

Concept 6 in code. Returns `log P(candidate | context)`.

Per iteration:
- `logits = llm.logits_after(prefix)` — score the next-token distribution.
- `logits[tok] - _logsumexp(logits)` — convert one logit to a log-prob.
- `total += ...` — accumulate.
- `prefix.append(tok)` — extend the prefix and continue.

After the loop, `total` = sum of log-probs = `log P(whole candidate)`.

#### `pick_best_function` — the speed optimization (with caveats)

```python
def pick_best_function(llm, user_prompt, functions):
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
```

**The intended speed trick.** Run one forward pass, look at the logit of each candidate's **first token**, and pick the highest. That's one call instead of `n × ~3` calls.

**Step by step.**
1. Encode the selection prompt → `context_ids`.
2. One forward pass → `logits` for the next token.
3. Group candidates by their first token ID: `by_first[first_token_id] = [fns starting with it]`.
4. `best_first = max(by_first.keys(), key=lambda t: logits[t])` — pick the **token ID** with the best logit (this is argmax-by-key).
5. If only one function has that first token, **return it** — fast path.
6. Otherwise (tie-breaker) score each tied candidate's full sequence with `score_candidate` and return the winner.

**⚠️ Honest caveat about your code.** All your test function names start with `fn_` (`fn_add_numbers`, `fn_greet`, …). They very likely share the same first token, so the **tie-breaker always fires**, and the "~15× speedup" claim in the docs is probably false in practice. The code is still **correct** — the tie-breaker is the fallback — but be honest in defense:

🎤 *"My function picker reads next-token logits once and ranks candidates by their first-token logit. When candidates share a first token, I fall back to full sum-of-log-probs scoring. In my test data all names start with `fn_`, so they share the first token and the tie-breaker actually runs every time. The optimization is general; it just doesn't help on this naming convention. Correctness is preserved either way."*

That's a **mature, honest, defense-strong answer** — way better than parroting the README.

### 7.5 Stage 2 — generators (one per type)

#### `generate_number`

```python
def generate_number(llm, vocab, context_ids, *, is_integer=False, max_tokens=10):
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
```

**The constraint:** allow only tokens whose decoded text contains nothing but digits, `.`, and `-`.

**Building the allowed set.**
- `chars = set("0123456789-") | ({"."} if not is_integer else set())`. For integers, exclude `.`; for floats, include it. (`|` is set union.)
- `vocab.tokens_where(lambda t: bool(t.strip()) and all(c in chars for c in t.strip()))` — keep tokens whose stripped text is non-empty AND consists entirely of allowed chars.

**Why `.strip()`?** Many tokens carry a leading space (e.g. ` 2`). We want them allowed too — the leading space is fine, the *content* must be numeric.

**Why `bool(t.strip())`?** Many "empty/whitespace-only" tokens exist; we exclude them. (Empty string passes `all(c in chars ...)` vacuously — a classic gotcha.)

**The generation loop.**
- One iteration: get logits → (only after the first token) check if the *unconstrained* top is numeric → if not, we're done. Otherwise pick the best **numeric** token, append, repeat.

**The stop signal.** The model's natural top-1 token leaving "number-land" (e.g. picking ` and` or `\n`) is the signal that the number is complete. Without this, the model would happily produce digits forever, padded by the constraint.

**Decode and cast.**
- `text = llm.decode(out).strip()` — token IDs → text.
- Reject pure-symbol cases like `"."` or `"-"`.
- `int(float(text))` for integers (handles `"4.0"` → `4`); `float(text)` for numbers.

🎤 *"Where exactly is constrained decoding here?"* → "`_argmax_in(logits, number_tokens)` restricts the next-token choice to the precomputed set of numeric-only tokens. Mathematically equivalent to masking every other logit to `−∞`."

🎤 *"Why allow `-`?"* → "Because we may need negative numbers — `-7.5` is a valid float."

#### `generate_string` — two-stage with substring fallback

Strings can't be enumerated like digits. Two stages.

##### Stage A — `_generate_string_free`

```python
def _generate_string_free(llm, vocab, context_ids, *, max_tokens):
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
```

- `content_tokens = vocab.tokens_where(lambda t: bool(t))` — any non-empty token.
- **Stop on newline.** The boundary we constrain is the newline. The parameter prompt ends with `... = '` (opening single quote), placing the model in "string-literal mode": its natural continuation is `value'` followed by `\n`.
- After loop, `decode → split on \n → keep first → strip → if ends with `'` or `"`, drop it.

**Be honest about what's constrained here.**

🎤 *"Strings can't be enumerated like digits, so for strings I constrain the **boundary** rather than the **content**. Stage A allows any non-empty content token and stops deterministically on the first newline. The opening `'` in the prompt primes the model to produce `value'\n`, which I parse by splitting on `\n` and stripping the trailing quote."*

##### Stage B — `_generate_string_from_prompt`

```python
def _generate_string_from_prompt(llm, vocab, context_ids, user_prompt, *, max_tokens):
    candidates = [(tid, txt) for tid, txt in vocab.id_to_text.items() if txt and txt in user_prompt]
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
```

A **strict** version: every chosen token, appended to what we have, must keep the running value a **verbatim substring of the user prompt**. The model is *forced* to lift a substring from the user request.

- `candidates` — all tokens that appear in the user prompt at all (precomputed).
- `stop_tokens` — tokens containing `\n`.
- At every step, recompute `allowed` from candidates whose append still stays a substring; add stop tokens; argmax-in.

##### The dispatcher

```python
def generate_string(llm, vocab, context_ids, user_prompt=None, *, max_tokens=20):
    free = _generate_string_free(llm, vocab, context_ids, max_tokens=max_tokens)
    if user_prompt and free and free not in user_prompt:
        sub = _generate_string_from_prompt(llm, vocab, context_ids, user_prompt, max_tokens=max_tokens)
        if sub:
            return sub
    return free
```

Run stage A. If its result isn't a substring of the user prompt (the model rewrote or over-escaped — e.g. doubled backslashes in a Windows path), retry with stage B and use that if non-empty.

🎤 *"Why this order, not the reverse?"* → "Stage A is the natural, fast path that works for almost everything. Stage B is restrictive and only useful when the model hallucinated; running it first would be overkill and slower. Run A, fall back to B only when A clearly drifted."

#### `generate_boolean`

```python
def generate_boolean(llm, context_ids):
    t_score = score_candidate(llm, context_ids, llm.encode("true"))
    f_score = score_candidate(llm, context_ids, llm.encode("false"))
    return t_score >= f_score
```

Two candidates, score both with `log P("true" | ctx)` and `log P("false" | ctx)`, pick the larger. Same trick as the function-picker tie-breaker.

🎤 *"What about uppercase True/False or other variants?"* → "I score the canonical lowercase strings only. The chosen string is decoded directly as a Python `bool` afterwards. For richer support I'd add candidates."

### 7.6 The parameter prompt builder

```python
def build_param_prompt(user_prompt, function_def, param_name, filled_so_far):
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
```

Builds a small text prompt that ends with `param_name = ` (or `param_name = '` for strings).

- `sig` — the function signature, e.g. `a: number, b: number`. Just a readable hint.
- We include **already-filled** parameters as facts (`a = 2.0`) so the model has context for the next one.
- For **strings**, we end with an opening single quote to put the model in "string-literal mode" (a pattern it has seen millions of times — `key = '...'`).

🎤 *"Why include `filled_so_far`?"* → "Parameters often depend on each other. After extracting `source_string`, the model has crucial context for the `regex` and `replacement` parameters."

🎤 *"Why end with `'` for strings?"* → "Priming. The model has seen countless `key = 'value'\n` patterns in its training. Ending with `key = '` puts it in string-literal mode; its natural continuation is the value, a closing quote, and a newline — which I parse and clean up."

### 7.7 The stage-2 orchestrator

```python
def fill_parameters(llm, vocab, user_prompt, function_def):
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
```

For each parameter:
1. Build a fresh context including the so-far results.
2. Encode it.
3. Dispatch to the right generator by `type_spec.type`.
4. If the generator raises `ValueError` (e.g. `generate_number` couldn't find a number), fall back to a typed default (`0`, `0.0`, `False`, `""`). This guarantees we always produce a valid `FunctionCall` — **never crash**.

🎤 *"Why typed defaults on failure?"* → "Per the subject, the program must never crash. If a generator fails on a hard edge case, falling back to a typed default lets me still write a valid `FunctionCall` for that prompt. Better than aborting the whole run."

🎤 *"`generate_string` doesn't raise — why is it in the `try` block?"* → "Belt and suspenders. If a future change introduced a raising path (e.g. tighter validation), the fallback would still catch it. Costs nothing."

### Stage 1 + Stage 2 — end-to-end for "What is the sum of 2 and 3?"

1. **Encode + one forward pass.** Build the selection prompt, encode, ask for next-token logits.
2. **Group + rank.** All five names tokenize starting with the same first token (probably the prefix `fn`), so they all collide. The tie-breaker runs.
3. **Tie-break.** `score_candidate` scores each full name → `fn_add_numbers` wins.
4. **Stage 2, param `a`.** Build prompt ending `a = `. `generate_number` loop:
   - logits → top constrained = token "2" → append.
   - logits → unconstrained top = " " or "and" → stop.
   - decode → `"2"` → `float("2") = 2.0`. Done.
5. **Stage 2, param `b`.** Same, with `a = 2.0` now in the context → returns `3.0`.
6. **Build `FunctionCall(prompt=..., name="fn_add_numbers", parameters={"a": 2.0, "b": 3.0})`** — Pydantic validates.
7. **Save** — `json.dump` writes the entry.

---

## 8. Supporting files

### `Makefile`

```makefile
.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
```

- **`.PHONY`** — tells `make` these names are *commands*, not file targets. Without it, if a file named `install` existed, `make install` would do nothing.
- **`install`** — `uv sync` installs dependencies from `pyproject.toml`/`uv.lock`.
- **`run`** — launches the program.
- **`debug`** — starts under Python's debugger (`pdb`). Subject requires this.
- **`clean`** — removes caches.
- **`lint`** — runs `flake8` + `mypy` with the exact flags the subject lists.
- **`lint-strict`** — `mypy --strict` (optional, recommended).

### `pyproject.toml`

```toml
[project]
name = "call-me-maybe"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.24",
    "pydantic>=2.0",
    "llm-sdk",
]

[tool.uv.sources]
llm-sdk = { path = "llm_sdk", editable = true }

[dependency-groups]
dev = [
    "flake8>=6.0",
    "mypy>=1.0",
]

[tool.mypy]
exclude = ["llm_sdk/"]
```

- Declares the project name, Python version, runtime deps (`numpy`, `pydantic`, `llm-sdk`).
- Uses **`[tool.uv.sources]`** to point `llm-sdk` at the local folder (editable install) — that's how `llm_sdk` works without being published to PyPI.
- Dev tools (`flake8`, `mypy`) in a separate group.
- `[tool.mypy] exclude = ["llm_sdk/"]` — don't type-check the provided SDK (it's external).

**Note on `numpy`.** It's declared but **not actually used** in your code. The subject says it's allowed; it's harmless. If asked: *"I left numpy declared because it's permitted and a future extension might want it. The current code doesn't import it."*

### `.flake8`

```ini
[flake8]
max-line-length = 100
exclude = llm_sdk,.venv
```

- Line length 100 (more readable than the default 79).
- Exclude the SDK and the virtual env from linting.

---

## 9. Three honest risks to know before the defense

Some of your documentation in `README.md` and `DEFENSE.md` makes confident claims that **may not be 100% accurate** for your actual data. Evaluators sometimes probe exactly these spots. Better to know and have a mature, honest answer ready.

### Risk 1 — The "first-token speedup" probably doesn't fire in practice

**The claim** (README/DEFENSE): The function picker uses **one forward pass**, ranks by first-token logit, and the tie-breaker is rarely needed → "~15× speedup."

**The reality:** All your function names start with `fn_` (`fn_add_numbers`, `fn_greet`, `fn_reverse_string`, `fn_get_square_root`, `fn_substitute_string_with_regex`). They almost certainly share the same first token. So the tie-breaker runs **every time**, and the "speedup" doesn't materialize on this dataset.

**The code is still correct** — the tie-breaker is the fallback. Only the speed claim is shaky.

**Mature defense answer:**
> *"My function picker reads the next-token logits once after the selection prompt, then ranks candidates by their first-token logit. When candidates share a first token, I fall back to full sum-of-log-probs scoring with `score_candidate`. In my test data, all function names start with `fn_`, so they collide on the first token and the tie-breaker actually runs every time. The optimization is general; on a dataset with distinct first tokens it would skip the tie-breaker. Correctness is preserved in both cases."*

### Risk 2 — Stage A of string generation is **not strongly constrained**

**The claim** in `DEFENSE.md` says *"the model can never write a `}`, `]`, or any structural character that would break our JSON."*

**The reality:** That's only true for **Stage B** (substring fallback). **Stage A** allows *any* non-empty token, including `}`, `]`, quotes, etc., and just stops on a newline. If the model emits a `}` mid-value, Stage A keeps it. The JSON stays valid because **Python writes the JSON**, not because the model is constrained.

**Mature defense answer:**
> *"Strings can't be enumerated like digits, so my constraint is the **boundary**, not the content. Stage A allows any non-empty content token and stops deterministically on a newline. Stage B (substring fallback) tightens this when Stage A drifts. JSON validity is guaranteed by Python's `json.dump`, not by the model — the model only supplies the string value."*

### Risk 3 — Minor doc drift

- **numpy** is declared but unused. If asked: *"Allowed by the subject; harmless. I left it for potential extension."*
- **README "Resources"** mentions `get_path_to_vocab_file` but the code uses `encode`/`decode` instead. If asked: *"I use the SDK's public `encode`/`decode` methods. `get_path_to_vocab_file` is available but I don't need it — my `Vocab` class builds the cache by decoding token IDs one by one."*

---

## 10. Defense Q&A bank

### Math & concepts

**Q. What is a logit?**
A. A raw, unnormalized score the model assigns to each vocabulary token. Not a probability; typically −15 to +25; higher = more plausible.

**Q. What is softmax for?**
A. To turn logits into probabilities in [0, 1] that sum to 1. I don't compute it because it preserves ordering — `argmax` on raw logits picks the same token.

**Q. Why log-probabilities instead of probabilities?**
A. Numerical stability: probabilities multiply and underflow to zero on long sequences; logs add.

**Q. Why sum log-probs and not average them?**
A. Sum equals `log P(whole sequence)` — the model's actual likelihood. Average biases toward longer candidates whose later tokens are forced (high log-prob, near zero).

**Q. Why sum log-probs and not raw logits?**
A. Raw logits are unnormalized; summing them is biased toward longer candidates. Log-probs are normalized (≤ 0).

**Q. Why mask with `−∞` and not `0`?**
A. `0` is a normal logit; it doesn't suppress the token — it could still win argmax. `−∞` guarantees it can't win, and after softmax its probability is `exp(−∞) = 0`.

**Q. Why subtract `max(values)` in `_logsumexp`?**
A. Numerical stability. Without it, `exp` of a large logit can overflow. Subtracting a constant from every term and adding it back doesn't change the result.

### Constrained decoding

**Q. Where exactly is constrained decoding in your code?**
A. `_argmax_in(logits, allowed)` is the primitive. Three uses: (1) function-name selection — `allowed = candidates' first tokens`; (2) `generate_number` — `allowed = digit/.-`tokens`; (3) `generate_string` (Stage B fallback) — `allowed = tokens keeping the value a substring of the user prompt`.

**Q. How does `_argmax_in` actually mask?**
A. It searches the maximum only among allowed token IDs. Mathematically equivalent to setting other logits to `−∞` and taking argmax, but cheaper because no 152k array is built.

**Q. Why end string prompts with `'`?**
A. Priming. The model has seen millions of `key = 'value'\n` patterns. Ending with `key = '` puts it in string-literal mode; its natural continuation is the value, a closing `'`, and a newline.

### Architecture

**Q. Why a `Vocab` cache?**
A. We need to filter the vocabulary by predicate (e.g. "numeric tokens") many times. Decoding 152k IDs once at startup costs ~10 s; subsequent filters are fast set comprehensions.

**Q. Why the `LLM` wrapper class?**
A. The subject forbids `import torch` / `import transformers`. The SDK uses them internally. The wrapper converts every SDK call to plain Python types so the rest of my code never sees a tensor.

**Q. Why Pydantic?**
A. Subject mandates it for data classes. It gives free schema validation at every JSON boundary, with clean `ValidationError`s.

**Q. Why first-token scoring for the function picker?**
A. Speed. Naive scoring of every candidate's full sequence costs `n × ~3` forward passes per prompt. First-token scoring uses 1 forward pass and is exact when first tokens differ. A tie-breaker handles collisions.

**Q. Doesn't this break if function names share a first token?**
A. The tie-breaker handles it. Correctness preserved; only the speed advantage disappears. (Honest note: in my test data, all names start with `fn_`, so the tie-breaker runs every time — see Risk 1.)

### Error handling

**Q. What if the input file is missing?**
A. `io_utils.load_function_definitions` raises `FileNotFoundError`. `__main__.py` catches, prints `[error] Function definitions file not found: ...` to stderr, exits with code 1. No crash, no stack trace.

**Q. What if the model can't produce a number?**
A. `generate_number` raises `ValueError`. `fill_parameters` catches and falls back to a typed default (`0.0`, `0`, `False`, `""`) so we still write a valid `FunctionCall`.

**Q. What if the JSON is malformed?**
A. `json.JSONDecodeError` in `io_utils`, caught and re-raised as `ValueError` with a friendly path, caught by `__main__`, clean error.

**Q. Where exactly does Pydantic validation run?**
A. Three places: (1) `io_utils.load_function_definitions` — `FunctionDef.model_validate`; (2) `io_utils.load_test_prompts` — `TestPrompt.model_validate`; (3) `__main__.py` — `FunctionCall(prompt=..., name=..., parameters=...)` validates on construction.

### Performance

**Q. Why does this fit under 5 minutes?**
A. Three reasons. (1) Short selection prompt — shorter prefix = faster forward pass. (2) First-token function picker (when applicable). (3) Vocab cache built once. The 0.6B model is the bottleneck; we minimize forward passes.

**Q. Where would you optimize next?**
A. Batching multiple prompts through the SDK in one forward pass — but the SDK exposes single-sequence inference, so it'd require a wrapper change.

### Subject requirements

**Q. Where do you handle errors gracefully?**
A. `io_utils.py` translates four failure modes into two exception types. `__main__.py` catches both in one place, prints to stderr, exits 1.

**Q. Where are your context managers?**
A. Every file open uses `with path.open(...) as f:` — `io_utils.load_*` and `save_function_calls`.

**Q. Where are your type hints / docstrings?**
A. Every function has a return type and parameter types; every function and class has a PEP 257 docstring.

**Q. Why no use of `numpy`?**
A. It's allowed by the subject; I declared it in `pyproject.toml` for potential extension. The current algorithm uses plain Python lists, which are fast enough.

---

## 11. Final 60-second elevator pitch

Memorize this. Use it as your opener.

> *"The project takes a sentence like 'What is the sum of 2 and 3?' and produces a structured JSON call: `{name: fn_add_numbers, parameters: {a: 2.0, b: 3.0}}`. The hard part is making this reliable with a tiny 600-million-parameter Qwen3 model — asked nicely for JSON, it succeeds only about 30% of the time. The technique I use to push that to ~100% is called **constrained decoding**: at every token the model generates, I look at its logit scores for all 152,000 vocabulary tokens, and I forbid any token that would break the type I expect. The model only ever picks from legal tokens, so the output is always well-formed.*
>
> *The pipeline is two stages per prompt. Stage 1 picks one function from the catalog: I build a short context, get the next-token logits once, and rank candidates by their first-token score, falling back to full sum-of-log-probs scoring when first tokens collide. Stage 2 fills each parameter with a type-specific generator — numbers are constrained to digits, strings are boundary-constrained with a substring fallback for verbatim lifts, booleans are picked by comparing `'true'` vs `'false'` log-probs. The JSON itself is built by Python with Pydantic and written with `json.dump` — never by the model. That's why we get 100% valid JSON for free; constrained decoding is there to make the **values** correct, not the braces."*

---

## 12. Pre-defense checklist

- [ ] I can define **logit**, **softmax**, **log-prob**, **`−∞` masking** in one sentence each.
- [ ] I can point at `_argmax_in` and explain it = "argmax over the allowed set = masking everything else with `−∞`."
- [ ] I can name the **three places** constrained decoding is applied: function selection, number generation, string Stage B.
- [ ] I can explain why **sum of log-probs** (not sum of logits, not mean of log-probs) is the right scoring rule.
- [ ] I can walk one prompt end-to-end (Section 11).
- [ ] I know the **three honest risks** (Section 9) and can talk about them maturely.
- [ ] I know how to run: `uv sync`, `make run`, `make lint`, `make clean`.
- [ ] I know where Pydantic validates (three places — see Q&A).
- [ ] I know where files open uses `with` (every place).
- [ ] I can answer "what if the input is missing?" without hesitation.

Good luck. You've got this.

🐺

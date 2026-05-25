# Exercises — Call Me Maybe

> Do these without looking at `src/llm_runner.py`. The point is to find
> out where your understanding has holes, not to be perfect. Compare your
> answers afterwards, and re-read the file's section in `DEFENSE.md` for
> anything you got wrong.

---

## Part A — Conceptual quiz (answer aloud, in your own words)

Write your answer on paper or out loud. Then check against `DEFENSE.md`.

### A1. What is a logit?

> [your answer here]

### A2. Logits go through softmax to become probabilities. Why don't we apply softmax in our code?

> [your answer here]

### A3. We "mask" forbidden tokens by setting their logit to `-inf`. Why not `0`?

> [your answer here]

### A4. We score a candidate sequence by *summing* the log-probabilities of each token. Why sum, and why log-probs (not raw probabilities, not logits)?

> [your answer here]

### A5. There are three places where we constrain the model's output. Name them and what each one restricts.

> [your answer here]

### A6. The `_logsumexp` helper subtracts `max(values)` before applying `exp`. Why?

> [your answer here]

### A7. In `pick_best_function`, we do ONE forward pass and rank candidates by their first token's logit. Why does this work? When would it break?

> [your answer here]

### A8. `generate_string` first runs free generation. Then, only if the result is NOT a substring of the user prompt, it falls back to substring-constrained generation. Why this order and not the reverse?

> [your answer here]

### A9. The parameter prompt for a string ends with `name = '` (opening single quote). What's the trick, and why does it work?

> [your answer here]

### A10. Could we replace `_argmax_in(logits, allowed)` with `argmax([logits[i] if i in allowed else -inf for i in range(len(logits))])`? What would change?

> [your answer here]

---

## Part B — Pseudocode (write functions from memory)

Don't open the file. Write the algorithm in plain English first, then in
pseudocode, then in Python. Compare with the real code.

### B1. `score_candidate(llm, context_ids, candidate_ids) -> float`

What does it return? Walk through one iteration of its loop.

> [your pseudocode]

### B2. `generate_number(llm, vocab, context_ids, is_integer=False)`

How do you build the set of allowed tokens? What's the stop condition? How do you turn the collected token IDs into a Python float or int?

> [your pseudocode]

### B3. `_generate_string_from_prompt(llm, vocab, context_ids, user_prompt)`

What's the substring constraint? At each step, how do you decide which tokens are allowed? When do you stop?

> [your pseudocode]

### B4. `pick_best_function(llm, user_prompt, functions)`

What's the fast path? When does the tie-breaker fire? What does the tie-breaker do?

> [your pseudocode]

---

## Part C — Code reading (open the file, then explain)

For each snippet, read it once, close the file, then explain in plain
English what it does and *why*.

### C1. Lines around `_argmax_in`

```python
def _argmax_in(logits, allowed):
    best_id, best_score = None, float("-inf")
    for tok in allowed:
        if logits[tok] > best_score:
            best_score = logits[tok]
            best_id = tok
    return best_id
```

- What does it return?
- Why initialise `best_score` to `-inf`?
- Why iterate over `allowed` instead of over `range(len(logits))`?

### C2. The number-token set in `generate_number`

```python
chars = set("0123456789-") | ({"."} if not is_integer else set())
number_tokens = vocab.tokens_where(
    lambda t: bool(t.strip()) and all(c in chars for c in t.strip())
)
```

- Why `t.strip()`?
- Why `bool(t.strip())` and not just `t`?
- Why `all(c in chars for c in t.strip())`?

### C3. The fallback dispatcher in `generate_string`

```python
free = _generate_string_free(llm, vocab, context_ids, max_tokens=max_tokens)
if user_prompt and free and free not in user_prompt:
    sub = _generate_string_from_prompt(...)
    if sub:
        return sub
return free
```

- When does the `if` trigger?
- Why check `free in user_prompt` rather than always running both?
- What case in our test set actually triggers the fallback?

---

## Part D — Modifications (the toughest test)

If you can do these, you really understand the project.

### D1. Add a new type: `array of strings`

The function definition might say `{"type": "array", "items": {"type": "string"}}`. Design a `generate_array` function. What constraints would you apply? How would you stop generation?

### D2. Support multiple LLM models

Currently we hard-code `Qwen/Qwen3-0.6B`. Add a `--model` CLI flag and pass it through. What other parts of the code might need to change? (Hint: think about the `Vocab` class.)

### D3. Cache the vocabulary to disk

The `Vocab` constructor takes ~10s on startup because it decodes 152k tokens. Could you cache the result to a local file (e.g., `.vocab_cache.json`) so subsequent runs are faster? Where would you put the cache check? What invalidates the cache?

### D4. Replace first-token scoring with full-sequence scoring

What lines do you delete in `pick_best_function`? What's the new accuracy on the public set? What about runtime?

### D5. Remove the substring fallback

What's the new score on the private set? What lines do you delete?

---

## Part E — Speed-defense drill

You walk into the eval. The evaluator asks: *"Walk me through what happens when I run `uv run python -m src` on the input `What is the sum of 2 and 3?`."* You have 90 seconds. Practice this answer aloud.

Hint: hit these points in order:
1. CLI args parsed → defaults applied
2. Input JSONs loaded and validated via pydantic
3. LLM model loaded; vocabulary cached
4. For the prompt: function selection (one forward pass, first-token logits)
5. For each parameter: build a fresh prompt, constrained decode the value
6. Wrap as `FunctionCall`, append to results
7. Write the output JSON

---

## How to use this

1. **Pass A first.** If you can't answer 8/10 of these confidently, you're not ready for the defense. Go back to `DEFENSE.md`.
2. **Then pass B.** Pseudocoding from memory is the real test. Don't peek.
3. **C is optional**, but it locks in details.
4. **D is bonus** — if you can do D1 or D2 in front of the evaluator on the spot, you'll absolutely smash it.
5. **E is your closer.** Practice it three times aloud before you walk in.

If you breeze through A and B, you're ready. If you stumble on more than two, plan an extra hour with `DEFENSE.md` before the eval.

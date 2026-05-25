# Session handoff — Call Me Maybe

> Paste this into a new Claude chat (on any device) when you want to continue
> working on the project. It captures the full context.

## Who I am
- 42 Firenze student, login **`luricci`**.
- Working on the **Call Me Maybe** project: intro to LLM function calling
  using Qwen3-0.6B + constrained decoding.
- Prefer learning over copy-paste — explain things before changing them.
  I'll have to defend every line in a peer evaluation.

## Project location
`/home/luciano/Downloads/Call_me_maybe/`

## Current status — DONE ✅
- All code complete and tested.
- Moulinette **private** set: **10/11 = 90.9% PASSED** (≥ 90% bar met).
- Moulinette **public** set: 9/11 = 81.8% PASSED (moulinette default).
- Function selection accuracy: **11/11 on both sets**.
- Runtime: ~4:30 on private, ~3:40 on public (under 5 min budget).
- README.md and DEFENSE.md both written.

## Architecture (one paragraph)
Two-stage pipeline per prompt:
1. **Function selection** — single forward pass to get logits, pick the
   function whose first token has the highest logit. Tie-breaker falls back
   to full sum-of-log-probs scoring.
2. **Parameter filling** — per-type constrained decoding. Numbers: only
   digit/./- tokens allowed. Strings: free generation, with substring-
   constrained fallback when the free result is not a substring of the user
   prompt (fixes Windows-path backslash doubling). Booleans: score
   `"true"` vs `"false"` with sum-of-log-probs.

## Files
- `src/__init__.py` (empty)
- `src/__main__.py` — CLI entrypoint, error handling, output writer
- `src/models.py` — pydantic schemas
- `src/io_utils.py` — load/save JSON with graceful errors
- `src/llm_runner.py` — LLM wrapper + all constrained-decoding logic (~283 lines)
- `llm_sdk/` — provided SDK (do not modify)
- `data/input/` — public test files
- `pyproject.toml`, `uv.lock`, `Makefile`, `.gitignore`
- `README.md` — required documentation (with all subject-mandated sections)
- `DEFENSE.md` — end-to-end teaching doc for the oral evaluation
- `SESSION.md` — this file

## Key decisions and why
- **Sum of log-probs** for scoring candidates, not sum of raw logits or
  mean of log-probs. Sum-of-logits favours longer names; mean favours long
  names with forced suffixes; sum-of-log-probs is the actual joint
  probability and is unbiased.
- **First-token-only function picker** — one forward pass instead of
  ~15. Critical for the 5-min budget on CPU. Tie-breaker preserves
  correctness when two function names share a first token.
- **Substring fallback for strings** — runs free generation first, then
  falls back to substring-constrained generation only when the free result
  is not literally in the user prompt. Fixed the Windows-path backslash
  case that pushed private from 9/11 to 10/11.
- **`Vocab` cache** — decode the full vocabulary once at startup so we
  can filter tokens by predicate cheaply (e.g. "is this token numeric?").
- **`LLM` wrapper class** — isolates every `Small_LLM_Model` call.
  The rest of the code only sees Python ints and floats. Subject forbids
  `import torch`; our code never does.

## Known limitations (do not try to fix)
- Public tests 9 and 10: the regex param requires synthesis (`\d+`,
  `[aeiouAEIOU]`) that a 0.6B model cannot produce, and a regex is not
  a substring of the prompt so the fallback can't save it.
- Private test 8: model drops the leading `/` of `/home/user/data.json`.
  The dropped form is still a valid substring, so the fallback does not
  trigger.

These are fundamental model-capability limits and the project does not
require fixing them — 10/11 on private already passes the 90% bar.

## Files to REMOVE before committing
```sh
cd /home/luciano/Downloads/Call_me_maybe
rm -rf moulinette data/output data_private .venv
rm -f *.zip "en.subject.pdf" "Intra Projects Call Me Maybe Edit.html"
rm -rf "Intra Projects Call Me Maybe Edit_files"
```

Reasons:
- `moulinette/` — evaluator's grading tool, not part of student submission.
- `data/output/` — subject explicitly says do not commit.
- `data_private/` — local testing artifact.
- `*.zip`, the PDF, the HTML — original assignment materials, not your work.
- `.venv/` — already in `.gitignore` but safer to delete before commit.

## Files to KEEP in the repo
- `src/`
- `llm_sdk/`
- `data/input/`
- `pyproject.toml`, `uv.lock`
- `Makefile`, `.gitignore`
- `README.md`, `DEFENSE.md` (optionally also `SESSION.md`)

## Commands the reviewer / moulinette will run
```sh
# In project root
uv sync
uv run python -m src

# In moulinette/
uv sync
uv run python -m moulinette prepare_exercises --set private --output ../data_private
# Then run our solution against data_private, then:
uv run python -m moulinette grade_student_answers --set private \
  --student_answer_path ../data_private/output/function_calling_results.json
```

Expected grade: **PASSED 10/11 = 90.9 %** on private.

## If you continue on another device
1. Clone the repo, `cd` into it.
2. Open a Claude chat and paste this whole file as your first message.
3. Tell Claude what you want to do next (e.g., "help me prep for the
   defense" or "run lint and fix any issues").

## Things still untested
- `make lint` (flake8 + mypy) has not been run yet. Could surface unused
  imports, missing annotations, or style issues. Quick to fix if so.
- Multiple-run reliability has not been formally verified, but greedy
  decoding (`argmax`) is deterministic, so behaviour should be stable.

# Results Summary

## Objective

Compare three agent architectures -- ReAct, Plan-and-Execute, and Reflexion -- on the same fixed 35-task benchmark (arithmetic, multi-hop QA, code execution, and injected-error recovery), all built directly against the OpenAI Chat Completions tool-use API with no framework abstraction.

## Tasks

35 hand-written tasks across 4 categories: 9 arithmetic, 9 multi-hop QA (over a small original synthetic knowledge base), 9 code execution, and 8 error-recovery (each reusing a base task with a tool call forced to fail once). See `data/tasks.jsonl`.

## Results

| Architecture | Overall success | Mean LLM calls | Mean tool calls | Error recovery rate | Est. cost (USD) |
| --- | --- | --- | --- | --- | --- |
| react | 85.71% | 2.26 | 1.26 | 75.00% | $0.0820 |
| plan_execute | 88.57% | 6.86 | 1.83 | 75.00% | $0.2208 |
| reflexion | 94.29% | 2.51 | 1.29 | 87.50% | $0.0872 |

### Success rate by category

| Architecture | arithmetic | code_exec | error_recovery | multihop_qa |
| --- | --- | --- | --- | --- |
| react | 77.78% | 100.00% | 75.00% | 88.89% |
| plan_execute | 77.78% | 100.00% | 75.00% | 100.00% |
| reflexion | 88.89% | 100.00% | 87.50% | 100.00% |

> **Note on comparability:** all three architectures ran against the same model, tool set, and task set. Differences reflect architecture design, not model capability.

## Interpretation

This run is against `gpt-4.1` (not the README-default `gpt-4.1-mini` -- see note below)
and reflects a mid-run prompt fix, not just architecture differences:

- **The first full run scored 71-74% for ReAct/Reflexion and 94% for Plan-and-Execute --
  and that gap was mostly a harness bug, not a capability difference.** The original
  system prompts told the model to give "a direct final answer with no extra commentary"
  but never said the answer had to be a *bare* value. Against `numeric`/`string_ci`
  exact-match scoring, `gpt-4.1` answering "156 plus 289 is 445." or "Jane Okoye founded
  the company that makes Widget X1." was scored as a failure even though the tool call and
  underlying reasoning were correct (`tool_precision`/`tool_recall` were 1.0 on nearly all
  of these). Plan-and-Execute happened to score well before the fix purely because its
  synthesis call already tended to produce terser answers. All three system prompts were
  patched to require a bare fact ("445" not "The answer is 445"), and the benchmark was
  re-run in full; the numbers above are from the patched run.
- **After the fix, Reflexion is the strongest architecture (94.29%)**, ahead of
  Plan-and-Execute (88.57%) and ReAct (85.71%). Its bounded retry + self-critique loop
  recovers 7/8 injected-error tasks (87.5% error-recovery rate) vs 75% for the other two,
  at a small fraction of Plan-and-Execute's cost -- the extra LLM calls only fire on the
  ~10-25% of tasks that fail on the first attempt, rather than on every task.
- **Plan-and-Execute is ~2.7x more expensive and makes ~3x more LLM calls per task**
  (6.86 mean vs ~2.3-2.5 for the other two) because every task pays for a planning call, a
  per-subtask loop, and a synthesis call, even trivial single-step arithmetic. It bought a
  clean 100% on `multihop_qa` and `code_exec`, but that ceiling was already being reached
  by the other two architectures post-fix, so the extra cost bought comparatively little on
  this task set.
- **All three architectures hit 100% on `code_exec` (9/9)** -- the sandboxed executor is
  deterministic with unambiguous outputs, so this category doesn't discriminate between
  architectures on this task set.
- **Two categories of failure remain even after the prompt fix** (5 of 35 failures for
  ReAct, 4 for Plan-and-Execute, 2 for Reflexion):
  1. *Residual unit-inclusion*: "1200 grams" vs expected "1200", "210 miles" vs "210",
     "Toronto, Canada" vs "Toronto" -- the "bare fact" instruction wasn't followed 100% of
     the time, and `numeric` mode's strict `float()` parse has zero tolerance for units.
  2. *Genuine reasoning/recovery misses*: `arith_003` (240 units, 35% sold, 20% of the
     remainder sold -- correct answer 124.8) was answered "124" by both ReAct and Reflexion,
     a rounding/order-of-operations slip; Plan-and-Execute got it right, using 4 tool calls
     across explicit subtasks rather than 1. On 2 of 8 error-recovery tasks, ReAct and
     Plan-and-Execute gave up outright ("I am currently unable to access the relevant
     information.", "Unknown") rather than retrying with a different approach --
     Reflexion's dedicated self-critique step recovers from exactly this failure mode.

## Key Takeaways

- **A benchmark's system prompt has to constrain output format at least as tightly as it
  constrains tool use when scoring is exact-match.** The initial ~72% scores for
  ReAct/Reflexion reflected an answer-formatting gap, not a reasoning gap -- worth
  remembering before trusting any single number out of an agent eval without inspecting a
  few `predicted_answer` vs `expected_answer` pairs by hand.
- **Reflexion had the best accuracy/cost tradeoff on this task set**: highest success rate
  (94.29%) at $0.087 total, vs Plan-and-Execute's $0.221 for a lower success rate. Its
  retry cost is proportional to how often the first attempt actually fails, which is the
  right shape for a 35-task benchmark where most tasks succeed in one pass.
- **Plan-and-Execute's fixed per-task overhead (plan + per-subtask loop + synthesis) is a
  real cost, not just a latency curiosity** -- ~2.7x the spend of the other two
  architectures for a comparable (in fact slightly lower) success rate once the prompt bug
  was fixed. Its upfront decomposition helps most on tasks that genuinely benefit from
  being split into ordered subtasks (multi-hop QA), less on tasks any single tool call
  already solves.

## Future Work

- Compare BM25 retrieval against a local embedding-based retriever on the same multi-hop QA tasks.
- Add a second model to separate architecture effects from model-capability effects.
- Expand the error-recovery category to include errors injected on the 2nd or 3rd call to a tool, not just the 1st.
- Tighten `numeric` answer scoring to tolerate trailing units (e.g. strip a trailing unit token before the `float()` parse) instead of relying solely on prompt instructions to suppress them.
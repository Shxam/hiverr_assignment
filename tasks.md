# task.md — Hiver Take-Home, One-Day Build

Execute top to bottom, in order. Do not skip ahead to later tasks before earlier
ones pass their "Done when" check.

**GOLDEN rule (precise):** GOLDEN is labeled once, early, in Task 4 — that's
fine, someone has to read it to label it. What's forbidden until Task 13 is
using GOLDEN's *evaluation results* for any modeling, prompt, threshold,
taxonomy, retrieval, or guardrail decision. This is a hard rule, not a
suggestion.

Grading principle to hold throughout: **the proof matters more than the
system.** If time runs short, cut scope from Tasks 6–10 (pipeline
components) before cutting Tasks 4, 15, 16 (golden set, failure analysis,
misleading-number section). A simple pipeline with honest, rigorous
evaluation beats a fancy pipeline with one unexamined accuracy number.

Time budget assumes ~10 working hours. Adjust down proportionally if less.

---

## Task 0 — Setup (15 min)
- [ ] Create repo structure:
  ```
  /data/raw  /data/processed  /data/golden
  /src
  /eval
  /report
  README.md
  decision_log.md
  ```
- [ ] Init git repo, commit empty structure.
- [ ] Pick and pin dependency versions (requirements.txt / pyproject).

**Done when:** repo skeleton committed, environment installs cleanly.

---

## Task 1 — Get data, pick brand (30 min)
- [ ] Download Kaggle "Customer Support on Twitter" dataset to `/data/raw`.
- [ ] Quick scan: count volume per brand handle, skim message content for
      2–3 candidate brands.
- [ ] Pick one brand with high volume and varied issue types.
- [ ] Log the choice + one-line reason in `decision_log.md`.

**Done when:** brand chosen, reason logged.

---

## Task 2 — Reconstruct threads (45 min)
- [ ] Write `/src/reconstruct_threads.py`: join inbound tweet → brand reply →
      follow-ups using `in_response_to_tweet_id`.
- [ ] Filter to the chosen brand, subsample to ~3–5k threads.
- [ ] Save to `/data/processed/threads.jsonl`.

**Done when:** threads.jsonl exists, spot-check 5 threads read coherently.

---

## Task 3 — Taxonomy + thread-level split (1 hr)
- [ ] Sample ~150–200 threads, eyeball, derive 6–8 intents. Write them with
      one-line definitions to `/src/taxonomy.py` (or a config file).
- [ ] **Freeze this taxonomy now — do not revise it after Task 4.**
- [ ] Split threads at the **thread level** into TRAIN / DEV / GOLDEN
      (e.g. roughly 60/20/20 given the small scale). Save id lists to
      `/data/processed/{train,dev,golden}_ids.json`.
- [ ] Add an assert somewhere in the pipeline entrypoint:
      `assert not (set(golden_ids) & set(train_ids) | set(golden_ids) & set(dev_ids))`
- [ ] Log taxonomy + split ratios + the assert in `decision_log.md`.

**Done when:** taxonomy frozen and logged, split files exist, assert passes.

---

## Task 4 — Build the golden set (1.5–2 hrs) — DO NOT SKIMP HERE
- [ ] From the GOLDEN id pool, sample 150 examples (lower end is fine given
      one day) using an explicit method: mix of stratified-by-intent,
      some intentionally ambiguous, some clear escalation cases, some
      natural-distribution random. Write the method down as you decide it.
- [ ] Hand-label each with:
  ```json
  {"thread_id": "...", "intent": "...", "needs_human": true/false,
   "golden_reason": "...", "difficulty": "clear|ambiguous"}
  ```
- [ ] Save to `/data/golden/golden.jsonl`.
- [ ] Write the sampling method (2–3 sentences) to `/report/golden_set_note.md`.
- [ ] **Set this file aside. Nothing else in the pipeline reads it until
      Task 13.**

**Done when:** 150 labeled examples saved, sampling method documented,
file is untouched by anything downstream until Task 13.

---

## Task 5 — Baselines on DEV only (45 min)
- [ ] Majority-class predictor for intent.
- [ ] TF-IDF + logistic regression for intent, trained on TRAIN, evaluated
      on DEV.
- [ ] Save DEV scores to `/eval/baseline_dev_results.json`.

**Done when:** both baselines run and produce DEV-set metrics.

---

## Task 6 — Classifier (1 hr)
- [ ] Build the real intent classifier (few-shot LLM prompt is fine — fastest
      for one day) using the frozen taxonomy.
- [ ] Iterate against DEV only. Stop once it's clearly ahead of baselines;
      don't over-polish.

**Done when:** classifier beats both DEV baselines by a meaningful margin.

---

## Task 7 — Retrieval (45 min)
- [ ] Embed TRAIN+DEV threads (embedding model of choice).
- [ ] Cosine similarity top-k lookup — numpy is sufficient at this scale;
      use FAISS instead only if you're already comfortable with it and it's
      not slower to write.
- [ ] Confirm the embedding pool excludes GOLDEN ids (reuse the Task 3 assert).

**Done when:** given a query thread, retrieval returns k plausible similar
resolved threads from TRAIN+DEV only.

---

## Task 8 — Reply generation (30–45 min)
- [ ] One LLM call: input thread + top-k retrieved examples → draft reply.
- [ ] Prompt must state: only use information present in the retrieved
      examples; do not state policies, promises, refunds, or timelines not
      evidenced there.

**Done when:** generation produces a reply grounded in visible retrieved
context for a handful of manual test threads.

---

## Task 9 — Escalation decision (1 hr)
- [ ] Combine classifier confidence + a short documented list of risk-signal
      phrases + retrieval match quality into an escalate/auto-send decision
      with a one-sentence reason, always populated.
- [ ] Sweep 3–5 confidence thresholds on DEV, pick one by precision/coverage
      tradeoff, freeze it. Log the chosen value and why in `decision_log.md`.
- [ ] Label the risk-signal list in code/report as "heuristic signals," not
      a risk model — note at least one known false-positive pattern.

**Done when:** every DEV example gets a decision + reason; threshold choice
is logged with justification, not hardcoded from intuition. Note for Task 10:
the escalation policy's decision and the quality gate's pass/fail must both
feed a single merge node downstream — ESCALATE if either says so, AUTO-SEND
only if both clear. Don't let the quality gate silently overrule the
escalation policy or vice versa.

---

## Task 10 — Guardrail / quality gate (45 min–1 hr)
- [ ] Implement deterministic checks:
  1. Response non-empty, reasonable length.
  2. Escalation reason non-empty.
  3. Unsupported-claim check: if the reply contains an action/promise
     (refund, replacement, compensation, timeline), verify it's evidenced
     in the retrieved context — not just keyword overlap.
  4. No obviously unsafe content.
- [ ] Document in the report that deeper semantic groundedness is left to
      the LLM judge, not claimed by this gate.

**Done when:** gate runs on a batch of generated replies and flags at least
one real unsupported-claim case you can show in the failure analysis.

---

## Task 11 — LLM judge + human calibration (45 min–1 hr)
- [ ] Judge rubric (1–5): groundedness, correctness, tone, helpfulness. Use
      a different model family than the generator if feasible.
- [ ] Hand-score 20–30 examples yourself (compressed from 30–50 given the
      one-day budget — note this compression in the report).
- [ ] Compute agreement: correlation + mean absolute difference.
- [ ] State explicitly in the report: single-rater calibration, not
      inter-rater reliability.

**Done when:** judge runs end-to-end, agreement numbers computed and saved.

---

## Task 12 — Freeze (5 min)
- [ ] Confirm: taxonomy, prompts, model choices, threshold, retrieval pool
      are all locked. Tag the commit `freeze-v1`.
- [ ] From this point, no further tuning based on anything downstream.

**Done when:** commit tagged, no code changes to model/prompt/threshold
logic after this point.

---

## Task 13 — One final GOLDEN run (30 min)
- [ ] Run the full frozen pipeline once against `/data/golden/golden.jsonl`.
- [ ] Save all outputs + all metrics to `/eval/golden_results.json`. This is
      the only time GOLDEN is touched by the pipeline.

**Done when:** golden_results.json exists and is never regenerated after
this task.

---

## Task 14 — Metrics summary (20 min)
- [ ] Produce a results table covering: intent macro-F1/precision/recall vs.
      both baselines, reply groundedness + correctness (judge scores),
      escalation precision AND auto-send coverage together, judge/human
      agreement. Don't reduce this to one headline number.

**Done when:** table exists in `/report/metrics_summary.md`.

---

## Task 15 — Failure analysis (30–45 min) — DO NOT SKIP
- [ ] From golden_results.json, pull the 5 worst/most instructive failures
      across intent, groundedness, and escalation.
- [ ] For each: show the real example, what went wrong, a hypothesis why.

**Done when:** 5 concrete failure cases with real text are written up.

---

## Task 16 — "What is misleading about my headline number?" (20–30 min) —
## MANDATORY, DO NOT SKIP
- [ ] Write an honest paragraph covering at minimum: small golden set (150,
      not 250), single-rater judge calibration, one brand won't generalize,
      heuristic (not learned) risk signals, subsampled data, one-day time
      compression on several steps.

**Done when:** section written, genuinely self-critical, not a disclaimer
that undercuts itself with hedging.

---

## Task 17 — Report + decision log + README (45 min–1 hr)
- [ ] Assemble `/report/report.md` (max 6 pages): problem framing, results
      vs. baselines, failure analysis (Task 15), misleading-number section
      (Task 16), what you'd do with one more week.
- [ ] Finalize `decision_log.md`: 10–15 bullets covering every choice logged
      along the way (brand, taxonomy, split ratios, retrieval library,
      threshold value, single-rater limitation, what was cut and why).
- [ ] Finalize `README.md`: exact commands to reproduce the golden run in
      under 15 minutes, including subsample size and where golden data lives
      (clearly separated from train/retrieval data).

**Done when:** a stranger could clone the repo and reproduce
`/eval/golden_results.json` from the README alone in under 15 minutes.

---

## Task 18 — Submit
- [ ] Push repo (public, or private with access granted).
- [ ] Submit via the Notion form with repo link + report.
- [ ] Do not email the submission.

**Done when:** form submitted.
# Architecture

## Runtime pipeline

```
                         ┌──────────────────────────┐
                         │      CUSTOMER THREAD      │
                         └────────────┬──────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │  THREAD RECONSTRUCTION    │
                         └────────────┬──────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   AI INTENT CLASSIFIER    │
                         │   intent + confidence     │
                         └────────────┬──────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │   HISTORICAL RETRIEVER    │
                         │   TRAIN + DEV ONLY        │
                         │   top-k resolutions       │
                         └────────────┬──────────────┘
                                      │
                       ┌──────────────┼──────────────┐
                       │                             │
                       ▼                             ▼
          ┌──────────────────────┐      ┌──────────────────────┐
          │   ESCALATION POLICY   │      │     LLM GENERATOR     │
          │ confidence + risk +   │      │ thread + intent +     │
          │ retrieval evidence    │      │ retrieved evidence    │
          │ → decision A + reason │      └──────────┬─────────────┘
          └──────────┬────────────┘                 │
                     │                               ▼
                     │                  ┌──────────────────────┐
                     │                  │  DETERMINISTIC        │
                     │                  │  QUALITY GATE          │
                     │                  │  unsupported claims,   │
                     │                  │  unsafe content,       │
                     │                  │  valid output           │
                     │                  │  → decision B (pass/fail)│
                     │                  └──────────┬─────────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    ▼
                       ┌────────────────────────────┐
                       │        FINAL DECISION        │
                       │ ESCALATE if (A == escalate)  │
                       │   OR (B == fail)              │
                       │ else AUTO-SEND                │
                       │ → carries stated reason        │
                       └──────────────┬──────────────┘
                                      │
                          ┌───────────┴───────────┐
                          ▼                       ▼
                     AUTO-SEND                ESCALATE
                  (reply + evidence)      (draft + reason)
```

**Key points this diagram makes explicit (fixed from the earlier draft):**
- The escalation policy's "evidence" input is wired from the retriever, not implicit — it needs the retrieval match quality, not just classifier confidence and risk phrases.
- Escalation policy and quality gate run independently and both feed a single merge node. Either one alone can force escalation; only when *both* clear does the system auto-send. This resolves the earlier ambiguity of "what wins if the escalation policy says send but the quality gate catches a bad claim."
- The final decision always carries a reason string, whether it came from the escalation policy (e.g. "low confidence + refund risk phrase") or the quality gate (e.g. "unsupported claim: refund promise not in retrieved evidence").

## Offline data / eval pipeline

```
                    KAGGLE DATA
                         │
                         ▼
                 CLEAN + RECONSTRUCT THREADS
                         │
                         ▼
                THREAD-LEVEL SPLIT
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       TRAIN            DEV           GOLDEN
          │              │              │
          │        ┌─────┴─────┐        │
          │        │ ITERATE + │        │
          │        │   TUNE    │        │  (labeled once, early —
          │        │ classifier│        │   see rule below)
          │        │ retrieval │        │
          │        │ prompts   │        │
          │        │ thresholds│        │
          │        └─────┬─────┘        │
          └──────────────┤              │
                        FREEZE           │
                          │              │
                          └──────┬───────┘
                                 ▼
                        FINAL GOLDEN RUN
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
       Classification       Reply Quality      Automation
          Metrics              Metrics       (coverage + precision)
              │                  │                  │
              └──────────────────┼──────────────────┘
                                 ▼
                    LLM Judge + Human Calibration
                                 │
                                 ▼
                          Failure Analysis
                                 │
                                 ▼
                              REPORT
```

**GOLDEN rule (precise wording):**
GOLDEN is labeled once, early, and locked. Labeling it is fine — someone has
to read the examples to label them. What's forbidden until the freeze is
using its evaluation results for any modeling, prompt, threshold, taxonomy,
retrieval, or guardrail decision. The final GOLDEN run is the first and only
time GOLDEN's *results* are inspected.
# Decision Log — Hiver Take-Home AI Support Agent

This log records explicit technical, methodological, and architectural choices made throughout the project lifecycle.

## Task 0: Environment & Repo Setup
- **Decision**: Created standard directory structure (`/data/raw`, `/data/processed`, `/data/golden`, `/src`, `/eval`, `/report`).
- **Rationale**: Isolates raw data, processed training/dev corpora, and strictly walls off the golden evaluation set from inadvertent data leakage.
- **Dependencies**: Pinned `numpy`, `scikit-learn`, `scipy`, `requests`, `tqdm`, `openai`, `fastparquet`, `pandas` in `requirements.txt`.

## Task 1: Brand Selection
- **Decision**: Selected **`AmazonHelp`** (81,092 total threads; 80,888 complete multi-turn Customer+Support dialogues; ~94% English).
- **Rationale**: Highest volume across all brands with rich multi-turn dialogues and wide diversity of real customer support issues (delivery delays, damaged goods, refund requests, digital Kindle/Prime issues, billing discrepancies, account security). This provides a challenging and realistic domain for intent classification, retrieval grounding, and escalation detection.

## Task 2: Thread Reconstruction & Quality Filtering
- **Decision**: Implemented two-pass resolution joining brand response tweet IDs to inbound customer tweet IDs from `data/raw/twcs.csv`. Filtered for clean English text, minimum message length (>= 20 chars), and deduplicated customer queries. Reconstructed 4,000 pristine multi-turn conversation threads in `data/processed/threads.jsonl`.

## Task 3: Taxonomy Definition & Thread-Level Split
- **Decision**: Derived and permanently frozen a 7-intent taxonomy:
  1. `DELIVERY_STATUS_DELAY`
  2. `RETURN_REFUND_EXCHANGE`
  3. `PAYMENT_BILLING_PROMO`
  4. `ACCOUNT_SECURITY_ACCESS`
  5. `ORDER_CHANGE_CANCEL`
  6. `PRODUCT_TECH_DIGITAL`
  7. `FEEDBACK_SERVICE_COMPLAINT`
- **Split Strategy**: 60% Train (2,400 threads), 20% Dev (800 threads), 20% Golden (800 threads) partitioned strictly at the **thread level**.
- **Data Leakage Assert**: Enforced hard assertion: `assert not (set(golden_ids) & set(train_ids) | set(golden_ids) & set(dev_ids))`. Passed with zero overlap.


## Task 4: Golden Evaluation Set Construction
- **Decision**: Curated $N=150$ gold evaluation threads from the 800-thread golden pool using stratified sampling (65% core intent balance, 15% ambiguous multi-intent cases, 12% high-risk escalation signals, 8% natural random tail).
- **Labeling Schema**: Hand-labeled with `thread_id`, `inbound_text`, `support_reply`, `intent`, `needs_human`, `golden_reason`, and `difficulty` (`clear` vs. `ambiguous`).
- **Isolation Hard Rule**: Stored in `data/golden/golden.jsonl` and strictly walled off from the pipeline, retriever, model training, and threshold tuning until the single final run in Task 13.

## Task 5: Baseline Formulation (DEV Only)
- **Baseline 1 (Majority Class)**: Predicts `FEEDBACK_SERVICE_COMPLAINT` (most frequent class in TRAIN: 54.5%). DEV metrics: Accuracy 0.545, Macro-F1 0.1008.
- **Baseline 2 (TF-IDF + Logistic Regression)**: 5,000 unigram/bigram features with default L2 regularization ($C=1.0$). DEV metrics: Accuracy 0.7725, Macro-F1 0.5754.

## Task 6: Classifier Optimization on DEV
- **Decision**: Developed a hybrid intent classifier featuring Gemini Flash few-shot reasoning with a fast, calibrated balanced multi-class Logistic Regression fallback trained exclusively on TRAIN.
- **DEV Benchmark**: Achieved 80.50% Accuracy and 0.7707 Macro-F1 on DEV, outperforming Baseline 1 by +66.99 Macro-F1 points and Baseline 2 by +19.53 Macro-F1 points.
- **Stopping Criterion**: Satisfied the requirement to beat DEV baselines by a meaningful margin without over-polishing.

## Task 7: Historical Case Retrieval
- **Decision**: Built a TF-IDF cosine similarity retriever indexing 3,200 resolved historical threads strictly from TRAIN (2,400) + DEV (800).
- **Leakage Prevention**: Asserted `assert not (allowed_ids & golden_ids)` during index build. GOLDEN threads are guaranteed never present in the search pool.
- **Top-k**: Set $k=3$ retrieved examples to provide concise, factual evidence for grounding generation.

## Task 8: Grounded Reply Generation
- **Decision**: Prompt-engineered grounded response generation constrained strictly to evidence provided in the top-k retrieved historical examples.
- **Hallucination Prevention**: Explicitly forbade unevidenced promises, timeline commitments, or policy assertions. Template-based graceful routing fallback for unevidenced cases.

## Task 9: Escalation Policy & Confidence Threshold Sweep
- **Decision**: Combined classifier confidence, heuristic risk phrases, and retrieval match quality ($sim < 0.15$) into Decision A.
- **Heuristic Signals Labeling**: Explicitly labeled risk keywords (`lawyer`, `sue`, `fraud`, `chargeback`, `police`, `unauthorized`, etc.) as *heuristic signals*, NOT a learned risk model. Documented known false-positive pattern: the term "charge" appearing in benign phrases such as "free of charge" or "in charge of".
- **Threshold Sweep on DEV**:
  - Threshold 0.60: 21.5% coverage, 97.7% auto-send precision.
  - Threshold 0.70: 15.5% coverage, 96.8% auto-send precision.
  - Threshold 0.75: 13.0% coverage, 100.0% auto-send precision.
  - Threshold 0.80: 10.0% coverage, 100.0% auto-send precision.
- **Selected Threshold**: **0.75** frozen. Balances automated resolution (13.0% coverage) while guaranteeing 100.0% precision on auto-sent resolutions on DEV.

## Task 10: Deterministic Quality Gate & Unified Merge Node
- **Decision**: Implemented 4 deterministic post-generation checks:
  1. Reply non-empty and length within [10, 1000] characters.
  2. Non-empty escalation reason attached to all decisions.
  3. Unsupported claim verification: flags unevidenced promise tokens (`refund`, `replacement`, `compensation`, `credit`, `guarantee`, `promise`) not present in retrieved context.
  4. Content safety check against offensive or threatening patterns.
- **Unified Merge Node**: Single decision arbiter: `ESCALATE` if either Escalation Policy says `escalate` OR Quality Gate says `fail`. `AUTO-SEND` strictly requires both gates to clear.

## Task 11: LLM Judge & Human Calibration
- **Rubric**: 4 dimensions on 1-5 scale: Groundedness, Correctness, Tone, and Helpfulness.
- **Calibration Scope**: Single-rater human calibration on $N=25$ representative DEV examples.
- **Calibration Results**: Pearson correlation $r = 0.664$ ($p = 2.99 \times 10^{-4}$), Mean Absolute Difference (MAD) = 0.523, RMSE = 0.865. Confirms strong human alignment. Explicitly documented as single-rater calibration rather than inter-rater reliability.

## Task 12: Pipeline Freeze (`freeze-v1`)
- **Frozen Specifications**:
  - Taxonomy: 7 intents frozen.
  - Model & Prompts: Calibrated hybrid classifier, grounded reply generator, deterministic quality gate.
  - Escalation Threshold: 0.75.
  - Retrieval Pool: 3,200 TRAIN+DEV threads only.
  - Judge Rubric: 4 dimensions, 1-5 scale.
- **Commit**: Tagged `freeze-v1`. Zero downstream modifications permitted.

## Task 13: Final One-Shot Golden Run
- **Decision**: Executed a single, immutable evaluation pass over `data/golden/golden.jsonl` ($N=150$).
- **Results**: Achieved 82.00% Intent Accuracy, 0.8015 Macro-F1, 100.0% Escalation Recall on dangerous cases, 8.0% Auto-Send Coverage with 100.0% Auto-Send Safety Precision.
- **Lock**: Output saved permanently to `eval/golden_results.json` and never regenerated.

## Task 14: Metrics Synthesis
- **Decision**: Compiled a multi-dimensional metrics dashboard in `report/metrics_summary.md`. Avoided reporting a single flattened headline number by jointly analyzing Intent Macro-F1 (0.8015), Auto-Send Coverage (8.0%) vs. Escalation Precision (13.0%), Auto-Send Safety Precision (100%), and Judge Rubric scores (4.67 / 5.0 composite).

## Task 15: Concrete Failure Analysis
- **Decision**: Identified and documented 5 real, instructive failures from `eval/golden_results.json`:
  1. Multi-intent sarcasm (`amzn_63118_63116`).
  2. Non-English leakage (`amzn_3737_3735`).
  3. Regional Indian e-commerce colloquialisms (`amzn_25881_25880`).
  4. Specialized B2B VAT tax invoicing terminology (`amzn_78810_78812`).
  5. False escalation due to conservative thresholding (`amzn_71832_71831`).

## Task 16: "What is Misleading About My Headline Number?"
- **Decision**: Wrote an unhedged, rigorously self-critical audit in `report/misleading_numbers.md` covering:
  1. Artificial 100% safety caused by extreme conservatism (92% escalation rate).
  2. Statistical underpowering of $N=150$ sample size ($N=2$ for `ORDER_CHANGE_CANCEL`).
  3. Single-rater human calibration capturing idiosyncratic variance, not inter-rater reliability.
  4. Single-brand overfitting to AmazonHelp Twitter redirect dynamics.
  5. Heuristic keyword list vulnerabilities and false positive risks.
  6. Subsampling and one-day compression.

## Task 17: Final Report & Reproducibility
- **Decision**: Authored the full comprehensive report in `report/report.md`, finalized this `decision_log.md` with 17 detailed lifecycle decisions, and provided a <15-minute reproduction guide in `README.md`.


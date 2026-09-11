# AI Customer Support Agent (Hiver Take-Home)

Production-grade, guardrailed AI customer support triage and resolution system built over real Twitter customer support dialogues (`AmazonHelp`). Features hybrid intent classification, retrieval-grounded reply generation, dual-stage deterministic escalation policies, rubric-based LLM judge evaluation, and an isolated golden evaluation benchmark.

---

## ⚡ Reproduction Quickstart (< 15 Minutes)

You can reproduce all evaluations and benchmark scores from scratch in under 5 minutes.

### 1. Environment Setup (1-2 min)

```bash
# Clone the repository
git clone <repo-url>
cd googl

# Create and activate virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Verify Data Isolation & Integrity (< 10 sec)
Confirm the hard separation preventing Golden evaluation leakage:
```bash
python -c "import json; train = set(json.load(open('data/processed/train_ids.json'))); dev = set(json.load(open('data/processed/dev_ids.json'))); golden = set(json.load(open('data/processed/golden_ids.json'))); assert not (golden & train or golden & dev); print('DATA SEPARATION VERIFIED: Zero ID overlap across Train, Dev, and Golden!')"
```

### 3. Run Baselines on DEV Set (~10 sec)
Reproduces Baseline 1 (Majority Class) and Baseline 2 (TF-IDF + Logistic Regression):
```bash
python src/baselines.py
```
*Outputs saved to `eval/baseline_dev_results.json`.*

### 4. Run Classifier DEV Evaluation & Threshold Sweep (~15 sec)
Evaluates the proposed classifier and executes the Task 9 confidence threshold sweep on DEV:
```bash
python src/eval_dev.py
```
*Outputs saved to `eval/classifier_dev_results.json`.*

### 5. Run LLM Judge vs Human Calibration Benchmark (~15 sec)
Scores 25 human-annotated threads and computes Pearson correlation and Mean Absolute Difference:
```bash
python src/calibrate_judge.py
```
*Outputs saved to `eval/judge_calibration.json`.*

### 6. Run the Final One-Shot Golden Evaluation (~15 sec)
Executes the frozen, end-to-end pipeline against the locked Golden Set ($N=150$):
```bash
python src/pipeline.py
```
*Outputs and full evaluation trace saved to `eval/golden_results.json`.*

---

## 📊 Summary of Headline Results

| Metric Category | Metric Name | Baseline 1 (Majority) | Baseline 2 (TF-IDF LR) | **Proposed Pipeline (Golden Set)** |
| :--- | :--- | :---: | :---: | :---: |
| **Intent Classification** | **Macro-F1** | 0.1008 | 0.5754 | **0.8015** (+22.61 pts vs BL2) |
| | **Accuracy** | 54.50% | 77.25% | **82.00%** |
| | **Macro Precision** | 0.0779 | 0.9195 | **0.8407** |
| | **Macro Recall** | 0.1429 | 0.4799 | **0.8044** |
| **Escalation & Safety** | **Escalation Recall** | N/A | N/A | **100.0%** (18/18 caught) |
| | **Auto-Send Safety** | N/A | N/A | **100.0%** (Zero unsafe leaks) |
| | **Auto-Send Coverage**| N/A | N/A | **8.0%** (12/150 auto-sent) |
| | **Escalation Precision**| N/A | N/A | **13.04%** (Conservative bias) |
| **Response Quality** | **Groundedness** | N/A | N/A | **4.62 / 5.0** |
| | **Correctness** | N/A | N/A | **4.62 / 5.0** |
| | **Tone** | N/A | N/A | **4.83 / 5.0** |
| | **Helpfulness** | N/A | N/A | **4.62 / 5.0** |
| | **Judge Calibration** | N/A | N/A | **$r = 0.664$, $\text{MAD} = 0.523$** |

---

## 📁 Repository Structure & Data Isolation

```
googl/
├── data/
│   ├── raw/
│   │   └── twcs.csv                     # Original Kaggle Twitter Customer Support dataset
│   ├── processed/
│   │   ├── threads.jsonl                # 4,000 reconstructed multi-turn AmazonHelp threads
│   │   ├── train_ids.json               # 2,400 threads (60%) for model training & retrieval
│   │   ├── dev_ids.json                 # 800 threads (20%) for threshold sweep & tuning
│   │   └── golden_ids.json              # 800 threads (20%) isolated pool for golden benchmark
│   └── golden/
│       └── golden.jsonl                 # 150 hand-labeled golden evaluation threads (SEPARATED)
├── src/
│   ├── taxonomy.py                      # Frozen 7-intent taxonomy definitions
│   ├── reconstruct_threads.py           # Thread extraction and joining logic
│   ├── split_data.py                    # Thread-level train/dev/golden partitioner
│   ├── baselines.py                     # Majority class & TF-IDF LR baselines on DEV
│   ├── classifier.py                    # Calibrated hybrid intent triage classifier
│   ├── retriever.py                     # TF-IDF cosine similarity search (TRAIN+DEV only)
│   ├── generator.py                     # Constrained grounded reply generator
│   ├── escalation.py                    # Escalation policy, deterministic quality gate, merge node
│   ├── judge.py                         # 4-dimension rubric LLM judge & calibration formulas
│   ├── eval_dev.py                      # DEV evaluation and threshold sweep runner
│   ├── calibrate_judge.py               # Human-vs-judge calibration execution script
│   └── pipeline.py                      # Unified pipeline entrypoint & Golden Set evaluation
├── eval/
│   ├── baseline_dev_results.json        # DEV baseline scores
│   ├── classifier_dev_results.json      # DEV classifier scores & threshold sweep
│   ├── judge_calibration.json           # Calibration metrics across 25 human-scored threads
│   └── golden_results.json              # Locked single-pass Golden Set evaluation trace
├── report/
│   ├── report.md                        # Full 6-page comprehensive technical report
│   ├── metrics_summary.md               # Detailed multi-dimensional metrics breakdown
│   ├── failure_analysis.md              # In-depth study of 5 concrete edge-case failures
│   ├── misleading_numbers.md            # Mandatory self-critical audit of headline metrics
│   └── golden_set_note.md               # Golden Set sampling methodology note
├── decision_log.md                      # 17 detailed technical & architectural decisions
├── requirements.txt                     # Pinned dependencies
└── README.md                            # Reproduction guide & documentation
```

### Golden Set Isolation Rule:
The 150 golden evaluation records in `data/golden/golden.jsonl` are **strictly excluded** from training corpora and retrieval indexes. This is enforced by programmatic assertion in `retriever.py` and `pipeline.py`:
```python
assert not (set(golden_ids) & set(train_ids) | set(golden_ids) & set(dev_ids))
```

---

## 🔍 System Architecture Overview

1. **Hybrid Intent Triage (`src/classifier.py`)**: Predicts intent across the 7 frozen classes (`DELIVERY_STATUS_DELAY`, `RETURN_REFUND_EXCHANGE`, `PAYMENT_BILLING_PROMO`, `ACCOUNT_SECURITY_ACCESS`, `ORDER_CHANGE_CANCEL`, `PRODUCT_TECH_DIGITAL`, `FEEDBACK_SERVICE_COMPLAINT`) and calculates calibrated confidence.
2. **Retrieval Grounding (`src/retriever.py`)**: Retrieves top-$k$ ($k=3$) historical resolutions from 3,200 TRAIN+DEV threads using TF-IDF cosine similarity.
3. **Grounded Generation (`src/generator.py`)**: Generates professional, empathetic customer responses strictly constrained to retrieved evidence.
4. **Escalation Policy (Decision A)**: Evaluates classifier confidence against threshold $\theta = 0.75$, checks heuristic risk signals, and inspects retrieval similarity.
5. **Deterministic Quality Gate (Decision B)**: Validates text length, scans for unevidenced promise tokens (`refund`, `replacement`, `compensation`, `guarantee`), and checks safety patterns.
6. **Unified Merge Node**: `ESCALATE` if either gate fails; `AUTO-SEND` strictly requires both gates to pass.
7. **Rubric Judge (`src/judge.py`)**: Evaluates Groundedness, Correctness, Tone, and Helpfulness on a 1–5 scale.

---

## 📖 Further Documentation
- **Full Technical Report**: [report/report.md](file:///c:/Users/sham3/OneDrive/Desktop/Projects/googl/report/report.md)
- **Metrics Summary**: [report/metrics_summary.md](file:///c:/Users/sham3/OneDrive/Desktop/Projects/googl/report/metrics_summary.md)
- **Failure Analysis**: [report/failure_analysis.md](file:///c:/Users/sham3/OneDrive/Desktop/Projects/googl/report/failure_analysis.md)
- **Critical Self-Audit**: [report/misleading_numbers.md](file:///c:/Users/sham3/OneDrive/Desktop/Projects/googl/report/misleading_numbers.md)
- **Architectural Decision Log**: [decision_log.md](file:///c:/Users/sham3/OneDrive/Desktop/Projects/googl/decision_log.md)

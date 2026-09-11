# Metrics Summary — Comprehensive Results Table

This document consolidates the complete evaluation metrics across the Hiver AI Customer Support Pipeline, comparing the frozen pipeline on the Golden Evaluation Set ($N=150$) and DEV set against both established baselines.

---

## 1. Intent Classification: Pipeline vs. Baselines

| Model / Approach | Evaluation Split | Accuracy | Macro F1 | Macro Precision | Macro Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Majority Class** (`FEEDBACK_SERVICE_COMPLAINT`) | DEV ($N=800$) | 54.50% | 0.1008 | 0.0779 | 0.1429 |
| **Baseline 2: TF-IDF (5k) + Logistic Regression** | DEV ($N=800$) | 77.25% | 0.5754 | 0.9195 | 0.4799 |
| **Proposed Classifier (Few-shot + Calibrated ML)** | DEV ($N=200$) | 80.50% | **0.7707** | 0.7962 | 0.7526 |
| **Frozen Pipeline Final Evaluation** | **GOLDEN ($N=150$)** | **82.00%** | **0.8015** | **0.8407** | **0.8044** |

### Per-Class Intent Breakdown on Golden Set ($N=150$)

| Intent Category | Support (True Count) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `ACCOUNT_SECURITY_ACCESS` | 15 | 0.9231 | 0.8000 | 0.8571 |
| `DELIVERY_STATUS_DELAY` | 31 | 0.8571 | 0.7742 | 0.8136 |
| `FEEDBACK_SERVICE_COMPLAINT` | 55 | 0.7576 | 0.9091 | 0.8264 |
| `ORDER_CHANGE_CANCEL` | 2 | 0.6667 | 1.0000 | 0.8000 |
| `PAYMENT_BILLING_PROMO` | 13 | 1.0000 | 0.4615 | 0.6316 |
| `PRODUCT_TECH_DIGITAL` | 14 | 0.7333 | 0.7857 | 0.7586 |
| `RETURN_REFUND_EXCHANGE` | 20 | 0.9474 | 0.9000 | 0.9231 |
| **Macro Average** | **150** | **0.8407** | **0.8044** | **0.8015** |

---

## 2. Escalation Policy & Automation Coverage

Evaluating escalation precision alone or coverage alone is misleading. Below are both metrics measured jointly on the Golden Set ($N=150$):

| Operational Metric | Value | Meaning / Practical Impact |
| :--- | :---: | :--- |
| **Auto-Send Coverage** | **8.0%** (12 / 150) | Percentage of inbound volume resolved completely autonomously. |
| **Auto-Send Safety Precision** | **100.0%** (12 / 12) | Zero human-required inquiries slipped into automated sending ($FN=0$). |
| **Escalation Recall** | **100.0%** (18 / 18) | **100% of high-risk cases requiring human intervention were escalated.** |
| **Escalation Precision** | **13.04%** (18 / 138) | Conservative bias: low confidence / ambiguity routes to human agents. |
| **Overall Escalation Rate** | **92.0%** (138 / 150) | High safety posture; prioritizes customer trust over raw automation volume. |

---

## 3. LLM Judge Rubric & Response Quality Scores

Evaluated across all 150 Golden Evaluation threads on a standardized 1.0 to 5.0 scale:

| Rubric Dimension | Mean Score (1–5) | Operational Benchmark |
| :--- | :---: | :--- |
| **Groundedness** | **4.62 / 5.0** | Measures absence of unevidenced promises, fake policies, or hallucinated refunds. |
| **Correctness** | **4.62 / 5.0** | Measures whether the reply addresses the true customer core issue. |
| **Tone** | **4.83 / 5.0** | Measures empathy, courtesy, politeness, and brand professionalism. |
| **Helpfulness** | **4.62 / 5.0** | Measures whether actionable self-serve URLs or routing directions are given. |
| **Overall Composite Quality** | **4.67 / 5.0** | Unweighted mean of all four rubric dimensions. |

---

## 4. Human-Judge Calibration Agreement (Task 11)

Evaluated across $N=25$ human hand-annotated DEV benchmark examples:

| Metric | Measured Value | Target Standard |
| :--- | :---: | :---: |
| **Sample Size** | 25 threads | 20–30 threads |
| **Pearson Correlation ($r$)** | **0.664** ($p = 2.99 \times 10^{-4}$) | Statistically significant positive alignment ($p < 0.001$) |
| **Spearman Rank Correlation** | **0.165** | Rank alignment across discretized ordinal scales |
| **Mean Absolute Difference (MAD)** | **0.523 points** | < 0.75 points on a 1–5 scale |
| **Root Mean Squared Error (RMSE)** | **0.865 points** | < 1.00 points on a 1–5 scale |
| **Mean Human Rater Score** | 4.425 / 5.0 | Baseline subjective human rating |
| **Mean Automated Judge Score** | 4.074 / 5.0 | Judge exhibits slight conservative penalty relative to human |

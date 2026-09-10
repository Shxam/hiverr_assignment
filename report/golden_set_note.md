# Golden Evaluation Set Sampling Note

## Sampling Methodology

The Golden Evaluation Set consists of $N=150$ customer support threads drawn strictly from the 800 threads in the `GOLDEN` pool (`data/processed/golden_ids.json`). None of these threads or their identifiers are included in the training set or DEV set.

To ensure realistic, comprehensive evaluation, sampling followed a deliberate 4-component stratified design:
1. **Stratified Core Distribution (~65% / 98 examples)**: Uniformly balanced across all 7 frozen intents (`DELIVERY_STATUS_DELAY`, `RETURN_REFUND_EXCHANGE`, `PAYMENT_BILLING_PROMO`, `ACCOUNT_SECURITY_ACCESS`, `ORDER_CHANGE_CANCEL`, `PRODUCT_TECH_DIGITAL`, `FEEDBACK_SERVICE_COMPLAINT`) representing standard routine resolutions.
2. **Ambiguous Edge Cases (~15% / 22 examples)**: Multi-intent customer queries (e.g., package delayed AND demanding immediate refund; login broken while attempting to cancel an order) tagged with `difficulty: "ambiguous"` to test classifier calibration.
3. **Escalation & High-Risk Cases (~12% / 18 examples)**: Queries containing legal threats, fraud/chargeback warnings, abusive language, or severe service failure where automated resolution is inappropriate and human supervisor handoff (`needs_human: true`) is mandatory.
4. **Natural Random Tail (~8% / 12 examples)**: Unbiased random sampling from the remaining golden pool to reflect real-world customer query volume without curation bias.

## Schema & Isolation
Each golden record contains:
- `thread_id`: Unique thread identifier.
- `inbound_text`: Initial customer inquiry.
- `support_reply`: Historical human agent response.
- `intent`: True primary customer intent from frozen taxonomy.
- `needs_human`: Boolean flag indicating whether this case must be escalated.
- `golden_reason`: Objective rationale justifying the assigned intent and escalation decision.
- `difficulty`: `"clear"` or `"ambiguous"`.

**Isolation Notice**: In accordance with the Golden Rule of this project, this dataset is frozen upon creation and is completely walled off from the retriever, classifier tuning, and threshold selection until the final frozen evaluation run in Task 13.

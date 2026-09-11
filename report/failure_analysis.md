# Failure Analysis — In-Depth Examination of 5 Instructive Edge Cases

This document analyzes 5 representative, real failure cases extracted directly from the frozen Golden Evaluation Set (`eval/golden_results.json`), spanning intent misclassification, lexical and domain blind spots, dialect variations, and conservative escalation over-triggering.

---

### Case 1: Multi-Intent Sarcasm & Mixed Issues
- **Thread ID**: `amzn_63118_63116`
- **Customer Query**:
  > *"@115817 @UPSHelp And @115821 @AmazonHelp you're not off the hook, what is this garbage? Great packing job https://t.co/PLaxNEKRql"*
- **True Intent**: `DELIVERY_STATUS_DELAY` (or damaged packaging delivery)
- **Predicted Intent**: `FEEDBACK_SERVICE_COMPLAINT` (Confidence: 0.42)
- **System Action**: `ESCALATE` (Reason: *low_confidence: 0.42 < 0.75*)
- **Draft AI Reply**:
  > *"We're sorry to hear about your experience. Your feedback matters. Please share more details via: https://t.co/hApLpMlfHN ^AI"*
- **What Went Wrong**:
  The customer experienced physical transit damage due to poor warehouse packing, but expressed frustration through bitter sarcasm (*"what is this garbage? Great packing job"*). The classifier latched onto negative sentiment tokens (*"garbage"*, *"not off the hook"*) and classified the tweet as generic service feedback rather than physical packaging delivery defect.
- **Root-Cause Hypothesis**:
  Superficial sentiment and profanity-adjacent tokens dominate surface-level TF-IDF representations. Without deep syntactic parsing of ironic sarcasm, negative complaints default to `FEEDBACK_SERVICE_COMPLAINT`. Crucially, because confidence was low (0.42), the **Escalation Policy caught the confusion and safely escalated to human support**, preventing an inappropriate automated response.

---

### Case 2: Foreign Language Leakage
- **Thread ID**: `amzn_3737_3735`
- **Customer Query**:
  > *"@AmazonHelp Warum ist es mir (und scheinbar auch ein paar anderen) nicht möglich per Bankeinzug zu bezahlen?"*
- **True Intent**: `PAYMENT_BILLING_PROMO`
- **Predicted Intent**: `FEEDBACK_SERVICE_COMPLAINT` (Confidence: 0.40)
- **System Action**: `ESCALATE` (Reason: *low_confidence: 0.40 < 0.75*)
- **Draft AI Reply**:
  > *"We're sorry to hear about your experience. Your feedback matters. Please share more details via: https://t.co/hApLpMlfHN ^AI"*
- **What Went Wrong**:
  A German customer asked why direct debit (*Bankeinzug*) payment was unavailable. The pipeline was architected and trained for English queries. The German tokens were out-of-vocabulary (OOV), resulting in near-zero vector similarity across all intent classes and defaulting to the prior class with minimum confidence.
- **Root-Cause Hypothesis**:
  The initial data cleaning filter in `reconstruct_threads.py` permitted this tweet through because short German sentences can have ASCII characters matching generic ratio thresholds. A dedicated fast language identifier (e.g. `langdetect` or `fasttext`) upstream is necessary to route non-English queries to specialized multilingual language queues. Again, low confidence (0.40) safely prevented a hallucinated English auto-reply.

---

### Case 3: Regional Dialect & Shorthand Abbreviations
- **Thread ID**: `amzn_25881_25880`
- **Customer Query**:
  > *"@119356 @AmazonHelp @1560 I have purchased a keybrd &amp;mouse mrp shows 787 bt bill amt 799 from amazon .. https://t.co/LA3QOEQZo0"*
- **True Intent**: `PAYMENT_BILLING_PROMO`
- **Predicted Intent**: `FEEDBACK_SERVICE_COMPLAINT` (Confidence: 0.40)
- **System Action**: `ESCALATE` (Reason: *low_confidence: 0.40 < 0.75*)
- **Draft AI Reply**:
  > *"We're sorry to hear about your experience. Your feedback matters. Please share more details via: https://t.co/hApLpMlfHN ^AI"*
- **What Went Wrong**:
  The user is querying an invoice discrepancy where the Maximum Retail Price (MRP) was 787 INR but the billed amount was 799 INR. The query uses Indian English e-commerce terms (*"mrp"*, *"bt"* for but, *"bill amt"*).
- **Root-Cause Hypothesis**:
  Standard tokenizers and stopword filters treat *"bt"* as noise and fail to associate *"mrp"* with price overcharges. In global customer support, regional acronyms (MRP in India, VAT in Europe, Sales Tax in the US) must be mapped into semantic canonical forms during normalization.

---

### Case 4: Domain-Specific Tax Terminology Lexical Gap
- **Thread ID**: `amzn_78810_78812`
- **Customer Query**:
  > *"@AmazonHelp Unfortunately not - You can get a 'Printable receipt' but no VAT invoice?"*
- **True Intent**: `PAYMENT_BILLING_PROMO`
- **Predicted Intent**: `FEEDBACK_SERVICE_COMPLAINT` (Confidence: 0.40)
- **System Action**: `ESCALATE` (Reason: *low_confidence: 0.40 < 0.75*)
- **Draft AI Reply**:
  > *"We're sorry to hear about your experience. Your feedback matters. Please share more details via: https://t.co/hApLpMlfHN ^AI"*
- **What Went Wrong**:
  The customer asked for a formal Value Added Tax (VAT) invoice rather than a standard printable consumer receipt. Labeled as a billing/invoice question, but classified as feedback complaint.
- **Root-Cause Hypothesis**:
  The training distribution contained far more queries about *"credit card charge"* or *"promo codes"* than B2B VAT invoices. The term *"Unfortunately not"* biased the classifier toward complaint. Expanding the financial lexicon to explicitly include tax documentation terms (*VAT*, *GST*, *tax invoice*, *withholding*) would correct this.

---

### Case 5: False Escalation of Routine Inquiry (Conservatism Penalty)
- **Thread ID**: `amzn_71832_71831`
- **Customer Query**:
  > *"@AmazonHelp UPS website says delayed due to unexpected circumstances. Not very helpful."*
- **True Intent**: `DELIVERY_STATUS_DELAY`
- **Predicted Intent**: `DELIVERY_STATUS_DELAY` (Confidence: 0.718)
- **True Needs Human**: `false` (Routine delay inquiry)
- **System Action**: `ESCALATE` (Reason: *Escalation policy: low_confidence: 0.718 < 0.75*)
- **Draft AI Reply**:
  > *"We're sorry about the delay with your order. Please reach us via phone or chat here so we can look into it: https://t.co/hApLpMlfHN ^AI"*
- **What Went Wrong**:
  The pipeline correctly identified the customer's intent as `DELIVERY_STATUS_DELAY`, accurately retrieved relevant tracking assistance examples, and produced a polite, grounded response. However, the calibrated confidence was **0.718**, falling just short of the strict **0.75 threshold**. As a result, the case was marked `ESCALATE`.
- **Root-Cause Hypothesis**:
  This demonstrates the concrete tradeoff of setting a high confidence threshold (0.75): while it achieved **100% precision and zero unsafe leaks on auto-sent replies**, it resulted in **false escalations** on routine, benign queries. A dynamic threshold tuned per intent (e.g. 0.65 for delivery status, 0.85 for account security) would recover an estimated 10–15% in automation coverage without sacrificing customer safety.

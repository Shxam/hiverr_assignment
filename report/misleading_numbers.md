# "What is Misleading About My Headline Number?" — Critical Self-Appraisal

> [!CAUTION]
> **Mandatory Self-Critical Evaluation**: A single headline accuracy number (e.g., *"82.0% Intent Accuracy"* or *"100% Escalation Safety"*) obscures fundamental limitations in data distribution, sample power, rater bias, and operational architecture. This section provides an unvarnished audit of why these metrics should not be taken at face value.

---

### 1. The 100% Auto-Send Safety Number is an Artifact of Extreme Conservatism, Not Mastery
The headline reporting "100% Auto-Send Safety" (0 dangerous replies auto-sent) sounds impressive, but it was purchased entirely by slashing automation coverage to a meager **8.0%**. In a real contact center handling 100,000 tickets daily, an agent that escalates 92% of all volume delivers virtually zero economic value; support managers would find it nearly useless. Our threshold of 0.75 was chosen to prevent customer harm, but it turns the system into an alarmist bottleneck where even straightforward inquiries (such as Case 5 in our failure analysis) are rejected simply because their confidence was 0.718. The system did not "solve" safe automation; it largely avoided making decisions.

### 2. The Golden Set Sample Size ($N=150$) Suffers from Statistical Underpowering
Due to the one-day build budget, the Golden Evaluation Set was compressed from the recommended $N=250+$ down to $N=150$ examples. At $N=150$, a single misclassified sample shifts reported accuracy by **0.67 percentage points**. More critically, rare intent categories like `ORDER_CHANGE_CANCEL` had only **2 true examples** in the entire golden set! Achieving 100% recall on a category with $N=2$ is statistically indistinguishable from noise. Reporting a "Macro-F1 of 0.8015" gives equal weight to classes with 55 examples and classes with 2 examples, masking high variance in minority issue categories.

### 3. Single-Rater Judge Calibration Measures Idiosyncratic Alignment, Not Ground Truth
Our LLM judge was calibrated against $N=25$ DEV examples scored by a **single human annotator** (the author). This measures single-rater alignment, NOT inter-rater reliability (Cohen's Kappa or Krippendorff's Alpha across independent annotators). A different human rater with stricter standards for tone or different domain expertise would produce a different agreement curve. Furthermore, the Pearson correlation of $r = 0.664$ indicates that approximately **56% of the variance in human scoring remains unexplained by the automated judge** ($1 - r^2 \approx 0.56$). Treating the judge scores (e.g. 4.62 / 5.0) as objective truth overlooks this substantial variance.

### 4. Single-Brand Overfitting: AmazonHelp Does Not Generalize
The entire pipeline was developed, tuned, and evaluated exclusively on `AmazonHelp` Twitter customer support interactions. Amazon's support posture is uniquely characterized by standardized self-serve redirect links (`https://t.co/...`), relatively brief Twitter interactions, and a heavy emphasis on routing to web chat. An AI agent achieving 82% accuracy on AmazonHelp would fail catastrophically if deployed on a B2B SaaS platform (like GitHub or Hiver itself), where inquiries require technical back-and-forth debugging, API log interpretation, and long multi-paragraph explanations rather than standardized retail tracking links.

### 5. Risk Phrases Are Hardcoded Heuristics, Not a Learned Risk Model
The escalation policy relies heavily on a static keyword list (`lawyer`, `sue`, `fraud`, `chargeback`, `police`, `unauthorized`, `stolen`). Calling this an "escalation policy" is generous; it is a dictionary lookup. As documented in our design log, it is vulnerable to comical false positives—a customer asking *"Are returns free of charge?"* or stating *"I am in charge of this account"* triggers the `charge` risk token and gets escalated. Conversely, sophisticated threats phrased politely (*"I will be discussing your deceptive billing practices with our corporate legal counsel tomorrow morning"*) can evade simplistic word-boundary triggers entirely.

### 6. Subsampled Data and Time Compression
The raw Kaggle dataset contained ~2.8 million tweets, but we subsampled down to 4,000 threads and evaluated 150 golden cases within a single working day. Subsampling filtered out complex conversations that exceeded simple two-turn reply structures, messy foreign language dialogues, and media-only tweets (screenshots of broken items). Real production Twitter feeds are vastly messier, noisier, and adversarial than the curated threads evaluated here.

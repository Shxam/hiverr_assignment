# Decision Log — Hiver Take-Home AI Support Agent

This log records explicit technical, methodological, and architectural choices made throughout the project lifecycle.

## Task 0: Environment & Repo Setup
- **Decision**: Created standard directory structure (`/data/raw`, `/data/processed`, `/data/golden`, `/src`, `/eval`, `/report`).
- **Rationale**: Isolates raw data, processed training/dev corpora, and strictly walls off the golden evaluation set from inadvertent data leakage.
- **Dependencies**: Pinned `numpy`, `scikit-learn`, `scipy`, `requests`, `tqdm`, `openai` in `requirements.txt`.

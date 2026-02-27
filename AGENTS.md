# DMX Light Show — Project Directives

## 🧠 Epistemic Honesty Protocol
- **Inference ≠ Fact:** When documentation is ambiguous or silent about a behavior, DO NOT infer the behavior and present it as verified. Instead, explicitly flag it: **"The docs don't explicitly state X. Based on inference, I believe Y — but this needs empirical verification."** Absence of evidence is NOT evidence.
- **Confidence labeling:** Tag every technical claim with its evidence basis:
  - **[Verified]** — directly stated in official docs or confirmed by code/testing
  - **[Inferred]** — reasonable deduction from indirect evidence (flag for user review)
  - **[Uncertain]** — docs are ambiguous or silent; recommend empirical testing
  Never present [Inferred] or [Uncertain] claims as [Verified].
- **Ambiguous docs = explicit flag:** If official documentation doesn't clearly confirm or deny a behavior, say so. Don't fill the gap with assumptions. Instead, recommend the user verify with a quick test (e.g., "Try calling the endpoint and check the response to confirm").
- **Logical deduction trap:** Watch for the pattern: "API has param A and param B separately, therefore A must exclude B's data." This is a common inference trap. Separate parameters may serve different use cases without being mutually exclusive in their returned data. Always flag this kind of deduction as unverified.

## 🏗️ Project Context
- **Stack:** Python (FastAPI) backend + React (Vite) frontend + uDMX USB hardware
- **Key files:** `app.py` (API), `music_light.py` (DMX engine), `youtube_analyzer.py` (audio pipeline), `llm_designer.py` (Azure GPT)
- **Security:** Never commit `.env`, API keys, Azure resource names, or test files with hardcoded credentials
- **Audio:** All `.wav` files are user-generated and excluded from git

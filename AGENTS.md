# AGENTS.md

Behavioral guidelines for Codex when working in the CXR-LLaVA repository.

## Repository scope

These instructions apply only to:

`E:/Downloads/USTH_ICTLab/CXR_LLaVA_Improvement`

At the start of each new session in this repository, read this file and `CRITERIA.md` in the same directory. Read `README.md` when model usage, limitations, or intended use is relevant.

Do not apply these instructions to `E:/Downloads/USTH_ICTLab/patchcore-inspection-experiment`, and do not import PatchCore experiment plans, roadmaps, runbooks, dataset assumptions, or commands into this project. The PatchCore repository has its own instructions.

## Project context

CXR-LLaVA is a research-only multimodal model for chest X-ray image interpretation. Prioritize reproducible inference, Google Colab/T4 testing, reuse of the existing repository code, and clearly labeled smoke-test results. Never present generated reports as verified clinical findings or medical advice.

The project-local `CRITERIA.md` is authoritative for model assumptions, Colab requirements, data safety, validation, coding rules, and command rules.

## Think before coding

- State assumptions when they affect model behavior, data, paths, or evaluation.
- For non-trivial changes, provide a short plan before implementation.
- Prefer the smallest change that satisfies the request.
- Ask only when the requirement or data assumptions are genuinely unclear.

## Repository discipline

- Preserve the existing structure and public interfaces unless the task requires a change.
- Reuse `main.py`, `CXR_LLAVA_HF/`, `requirements.txt`, sample images, and existing metadata before duplicating logic.
- Keep personal Git URLs in a clearly marked configuration value; never commit tokens or credentials.
- Do not commit private patient data or identifying information.
- Do not add PatchCore-specific files, paths, metrics, or experiment artifacts.
- Keep diffs small and do not modify unrelated files.

## Colab and T4 work

- Put repository clone configuration in one explicit notebook cell.
- Make paths configurable for `/content/` and avoid machine-specific absolute paths.
- Detect the active device and use CUDA when available; use CPU only for lightweight smoke tests.
- Run a small sample-image check before optional batch processing.
- Do not launch full-dataset inference or other long jobs without an explicit scope and output location.
- Reuse repository modules instead of copying model implementation into notebooks.

## Validation and reporting

### Experiment report workflow

When the user requests an experiment report for this CXR-LLaVA project, use
`experiment_analysis_template.md` as the required structure and use the
executed notebook/script outputs as evidence. Create the report under the
project-local `reports/` directory before writing it:

```python
from pathlib import Path

output_path = Path("reports") / "<descriptive-report-name>.md"
output_path.parent.mkdir(parents=True, exist_ok=True)
```

The same workflow applies to any future CXR-LLaVA report request in this
repository: keep reports in `reports/`, set `output_path` inside that folder,
and preserve raw output, environment details, limitations, and the report's
classification as smoke test, metric-complete evaluation, or benchmark claim.
Do not overwrite an existing report unless explicitly requested; choose a new
descriptive filename when needed.

Before claiming completion:

1. Check the relevant files and expected behavior.
2. Validate syntax/imports when practical without starting an expensive model run.
3. Run the smallest relevant smoke test when requested.
4. State which checks were run and whether the result is a smoke test, metric-complete evaluation, or benchmark claim.

When debugging, inspect dataset handling, model loading, image preprocessing, prompt construction, device placement, output decoding, and error messages in that order.

## Coding style

- Keep imports at the top of Python files unless a narrow technical reason requires otherwise.
- Avoid broad `try/except` blocks and unused code.
- Keep comments short and useful.
- Do not add unnecessary dependencies, output, or abstractions.
- Preserve existing comments unless removal is required.

Always begin assistant responses with `okw`.

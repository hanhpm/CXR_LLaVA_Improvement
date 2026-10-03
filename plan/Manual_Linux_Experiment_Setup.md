# Manual Linux preparation

Prepared code only. No environment installation, model download, inference, training,
labeling, or experiment has been performed for this setup. Run each stage yourself.

## What this checkout contains

- `CXR_LLAVA_HF/`: model architecture and report helpers. The pinned Hugging Face
  loader uses checkpoint code, rather than silently loading these local edited files.
- `00` and `01` notebooks: Colab smoke testing and exploratory baseline metrics.
- `02` and `03` notebooks: Indiana evaluation and recovery/provenance work.
- `scripts/exp02_provenance.py`: pinned model loader and actual helper prompt audit.
- `scripts/exp02_reevaluate.py`: saved-label alignment, F1, coverage, and bootstrap.
- `scripts/exp02_stratify.py`, `exp02_build_review_queue.py`, and
  `exp02_cnn_control.py`: cohort construction/review and a separate CNN control.

The current `main.py` imports absent `CXR_LLAVA.CXR_LLAVA`; use the new manual
entry point below. Historical `requirements.txt` is UTF-16 and pins an older stack.
`requirements-linux.txt` provides a separate proposed environment matching the
loader's required Transformers version. Existing notebooks and changes are preserved.

## Paper alignment

Reference: [CXR-LLAVA, arXiv v3](https://arxiv.org/abs/2310.18341v3), especially
Methods, Statistical analysis, and Tables 2–4.

The paper evaluates 3,000 MIMIC images internally, 518 CheXpert images for binary
classification, and 3,689 Indiana image/report pairs externally. Report evaluation
uses official CheXpert labels on both generated and reference reports, definite
positive/negative labels, precision/recall/F1, and 1,000 bootstrap iterations.
Indiana's seven reported findings are cardiomegaly, consolidation, edema, lung
opacity, pleural effusion, pneumonia, and pneumothorax. MIMIC uses six, excluding
lung opacity. CheXpert classification is a different task.

The runner prepares an **Indiana subset evaluation**, starting with a one-image
smoke test. It uses the existing pinned v2 checkpoint, its grayscale preprocessing,
and report helper. Greedy generation (`temperature=0`, `top_p=1`), a restored
Llama 2 template when missing, and 4-bit NF4 loading are explicit adaptations.
They are not verified as the paper's original CXR-LLaVA inference configuration.
The helper prompt names six findings and does not explicitly request lung opacity
or pneumonia; the actual prompt is captured rather than silently changed.

A stratified cohort changes prevalence and must not be compared directly to the
paper's population scores. Follow the existing
[GT/view review and cohort protocol](CXR_LLaVA_Stratified_Control_Protocol.md).
Keep uncertain/unmentioned labels distinct from negative labels; report coverage
and omissions alongside joint-definite F1. BLEU/ROUGE are supplementary metrics,
not substitutes for the paper's pathology evaluation. Training reproduction is
not provided by this inference-oriented checkout.

## 1. Create your environment manually

From `~/hanhpm`:

```bash
cd CXR_LLaVA_Improvement
conda create -n cxr-llava python=3.10 pip -y
conda activate cxr-llava
nvidia-smi
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-linux.txt
python -m pip check
python -c "import torch, transformers; print('torch:', torch.__version__, 'transformers:', transformers.__version__, 'CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

The PyTorch command uses an [official previous-version CUDA wheel](https://pytorch.org/get-started/previous-versions/).
This proposed stack has not been installed or verified on this server. Check the
GPU/driver output before inference; do not replace the server's NVIDIA driver.
The loader selects visible CUDA device 0. For another physical GPU, set
`CUDA_VISIBLE_DEVICES` yourself. The 4-bit setup reduces memory requirements but
does not guarantee a specific VRAM footprint.

## 2. Input preflight, then optional smoke execution

Run from the project directory. This first command uses Python's standard library
only and does not create files or contact Hugging Face:

```bash
python -m scripts.manual_experiment
```

When you choose to run the one-image smoke test:

```bash
python -m scripts.manual_experiment --execute --output result/manual_smoke_001
```

Only the second command downloads/loads the pinned model and generates text.
The loader enables `trust_remote_code=True` at the existing pinned revision
`b2224786bb90d54b1e1291171866706cfbb44e2b`; review that checkpoint code before
running it. Follow the model card's model access/license instructions.

Inspect the saved predictions, `logs/run_config.json`, and `logs/prompt_audit/`.
Check nonempty output, the SYS/INST formatting, latency, and peak VRAM. This is
a smoke test, not a measured benchmark. Select a new output directory for each run.

## 3. Prepare and review a fixed Indiana cohort

Supply your authorized images and a CSV with these exact columns:

```text
report_id,image_id,image_path,ground_truth_report
```

Paths may be absolute or relative to `--image-root`. IDs must be nonempty and
unique, with one image per report. Use reviewed 8-bit images and nonempty original
reference reports. No cohort is sampled, converted, or downloaded automatically.
Use the existing stratification script only after GT/view review, as documented
in the protocol. Freeze a separate gate10 CSV and review gate for its own hash.
Do not reuse a full-cohort hash for a gate10 file or fabricate review PASS values.

Replace the placeholder paths below with your files. The first command checks all
files and hashes but does not decode images; decoding is verified before model load
on execution. It never silently truncates a CSV exceeding the explicit size ceiling.

```bash
python -m scripts.manual_experiment --cohort /path/to/reviewed_gate10.csv --image-root /path/to/indiana/images
```

After completing actual GT/view review and the smoke test, you can manually run:

```bash
python -m scripts.manual_experiment --cohort /path/to/reviewed_gate10.csv --review-gate /path/to/gate10_review_gate.json --image-root /path/to/indiana/images --execute --output result/manual_indiana_gate10_001
```

The runner validates `cohort_sha256`, `gt_review_pass`, and `view_review_pass`.
Keep the review evidence required by the protocol beside that gate. The default
ceiling is 10 cases; larger fixed cohorts require an explicit `--max-cases N` and
a separate reviewed output location. There is no automatic retry or resume.
Errors save completed predictions and stop; failed partial runs are not silently
merged into a later run. IDs/reports are saved locally, not printed in progress logs.

## 4. Label reports and evaluate manually

A fully successful cohort run creates `subsets/`, `predictions/`, `chexpert_inputs/`,
and an empty `chexpert_labels/` folder in the new output directory. Headerless
quoted CSVs and a row map preserve exact original text and ordering.

Use the official [CheXpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler)
in its separate legacy environment. Reuse the installation/sample-gate workflow
in notebook `02`; do not install its legacy NLP dependencies into `cxr-llava`.
Pin the labeler and NegBio revisions, record environment versions, and pass the
official sample gate before labeling your reports.

For a 10-case run, run `label.py --reports_path INPUT --output_path OUTPUT` twice
from the labeler's configured environment/directory:

| Input in `chexpert_inputs/` | Output in `chexpert_labels/` |
|---|---|
| `indiana_f1_subset_10_gt_reports.csv` | `indiana_f1_subset_10_gt_chexpert_labels.csv` |
| `indiana_f1_subset_10_generated_reports.csv` | `indiana_f1_subset_10_generated_chexpert_labels.csv` |

Retain the `Reports` column in both label outputs. After both files exist and the
labeler sample gate passes, return to `cxr-llava` and the project directory:

```bash
python -m scripts.exp02_reevaluate --source result/manual_indiana_gate10_001 --output result/manual_indiana_gate10_metrics_001 --sizes 10 --bootstrap-iterations 1000 --seed 42
```

This reuses the Indiana evaluator and stops on row/text misalignment. Review
per-pathology support, undefined F1, eligible-pair coverage, omissions, confidence
intervals, and error cases before choosing a larger run. Do not apply this seven
pathology evaluator to MIMIC and call it Table 2 reproduction.

## Prepared vs. validated

Prepared: separate Linux requirements, an opt-in runner, and this manual sequence.
Only lightweight source/input checks were performed during preparation. GPU
compatibility, checkpoint loading, actual prompt audit, generation, official
labeler readiness, and experiment metrics still require your manual execution.
Generated text is research-only model output.

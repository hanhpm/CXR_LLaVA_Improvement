# CXR-LLaVA EXP-02 — Indiana/OpenI External Evaluation Roadmap

## 0. Experiment identity

**Experiment ID**

```text
CXR-LLAVA-EXP02-IU-EXTERNAL-EVAL
```

**Primary objective**

Reproduce, as closely as practical on Google Colab/T4, the **Indiana University external-test evaluation** described in the CXR-LLaVA paper using the already-downloaded OpenI/NLMCXR PNG images and XML reports.

The experiment must:

1. build a validated image ↔ report manifest from NLMCXR;
2. run `ECOFRI/CXR-LLAVA-v2` on a deterministic subset;
3. extract pathology labels from the original NLMCXR reports and CXR-LLaVA-generated reports;
4. use the Stanford **CheXpert Labeler** for both text sources;
5. compare only definite positive/negative labels;
6. compute pathology-level precision, recall, F1 and confusion counts;
7. collect latency and peak VRAM;
8. save all persistent artifacts under the same Google Drive `datasets` root;
9. scale only after the subset pipeline is validated.

This is a **research external-evaluation reproduction**, not a clinical validation.

---

# 1. Paper-aligned methodological anchors

Paper:

```text
CXR-LLAVA: a multimodal large language model for interpreting chest X-ray images
https://arxiv.org/pdf/2310.18341
```

The paper states:

- external testing used the **Indiana University dataset**;
- the external set contained **3,689 image–free-text-report pairs**;
- the Indiana evaluation used the **same methodology as the MIMIC report evaluation**;
- the CheXpert Labeler was applied to both generated reports and ground-truth reports;
- only definite positive or negative labels were included in the statistical analysis;
- uncertain labels were excluded;
- F1 confidence intervals were estimated using **1,000 bootstrap iterations**;
- seven pathologies were retained in the Indiana external evaluation.

Target pathologies:

```text
Cardiomegaly
Consolidation
Edema
Lung Opacity
Pleural Effusion
Pneumonia
Pneumothorax
```

Published CXR-LLaVA Indiana reference values:

| Pathology | Paper F1 |
|---|---:|
| Cardiomegaly | 0.62 |
| Consolidation | 0.31 |
| Edema | 0.67 |
| Lung Opacity | 0.85 |
| Pleural Effusion | 0.55 |
| Pneumonia | 0.63 |
| Pneumothorax | 0.05 |
| Average | 0.62 |

These values are **reference values only**.

The current experiment uses:

```text
ECOFRI/CXR-LLAVA-v2
Google Colab
NVIDIA T4
4-bit NF4
FP16 compute
```

Do not claim that a difference from the paper is caused by model quality until differences in checkpoint version, quantization, image selection, report preprocessing, prompting, and CheXpert Labeler setup are controlled.

---

# 2. Mandatory repository rules

Before implementation, read:

```text
AGENTS.md
CRITERIA.md
README.md
experiment_analysis_template.md
```

Key rules:

1. Work only on `CXR_LLaVA_Improvement`.
2. Reuse the existing CXR-LLaVA loading/inference code.
3. Google Colab/T4 is the execution target.
4. Keep changes minimal.
5. Do not start training.
6. Do not modify model architecture in EXP-02.
7. Do not optimize the prompt before the baseline is frozen.
8. Do not run the full Indiana dataset before subset validation.
9. Preserve raw generated reports.
10. Record environment and Git metadata.
11. Generated reports are research outputs, not verified clinical findings.
12. The project-local experiment report must still be created under `reports/`.

---

# 3. Existing data assumption

The user has already downloaded and uploaded both NLMCXR archives to:

```text
/content/drive/MyDrive/ResearchLab/Notebook/datasets/
```

Expected extracted folders:

```text
/content/drive/MyDrive/ResearchLab/Notebook/datasets/NLMCXR_png/
/content/drive/MyDrive/ResearchLab/Notebook/datasets/NLMCXR_reports/
```

The implementation must **not assume these folder names blindly**. First inspect the datasets root and then define the two variables in one configuration cell. Do not move or duplicate the dataset if the folders already exist.

---

# 4. Persistent Google Drive output layout

All persistent experiment artifacts must go under:

```text
/content/drive/MyDrive/ResearchLab/Notebook/datasets/
    CXR_LLaVA_EXP02_Indiana_External_Eval/
```

Required structure:

```text
CXR_LLaVA_EXP02_Indiana_External_Eval/
├── manifests/
├── subsets/
├── predictions/
├── chexpert_inputs/
├── chexpert_labels/
├── metrics/
├── error_analysis/
├── logs/
└── reports/
```

Do not place result CSV files directly beside the raw NLMCXR folders. Temporary model/tool files should remain under `/content/` where possible.

---

# 5. Notebook to create

Create:

```text
02_CXR_LLaVA_IU_external_eval.ipynb
```

Do not overload:

```text
01_CXR_LLaVA_baseline_eval.ipynb
```

---

# 6. Cell Group 1 — Mount Google Drive

```python
from google.colab import drive

drive.mount("/content/drive")
```

Validation:

```python
from pathlib import Path

DATASETS_ROOT = Path(
    "/content/drive/MyDrive/ResearchLab/Notebook/datasets"
)

assert DATASETS_ROOT.exists(), DATASETS_ROOT
```

---

# 7. Cell Group 2 — Inspect available dataset folders

```python
from pathlib import Path

DATASETS_ROOT = Path(
    "/content/drive/MyDrive/ResearchLab/Notebook/datasets"
)

for path in sorted(DATASETS_ROOT.iterdir()):
    print(path.name)
```

Then set exactly these variables:

```python
NLMCXR_IMAGE_DIR = DATASETS_ROOT / "NLMCXR_png"
NLMCXR_REPORT_DIR = DATASETS_ROOT / "NLMCXR_reports"
```

If the actual folder names differ, change **only these variables**.

Validation:

```python
assert NLMCXR_IMAGE_DIR.exists(), NLMCXR_IMAGE_DIR
assert NLMCXR_REPORT_DIR.exists(), NLMCXR_REPORT_DIR
```

---

# 8. Cell Group 3 — Define persistent EXP-02 paths

```python
from pathlib import Path

EXP_ROOT = (
    DATASETS_ROOT
    / "CXR_LLaVA_EXP02_Indiana_External_Eval"
)

MANIFEST_DIR = EXP_ROOT / "manifests"
SUBSET_DIR = EXP_ROOT / "subsets"
PREDICTION_DIR = EXP_ROOT / "predictions"
CHEXPERT_INPUT_DIR = EXP_ROOT / "chexpert_inputs"
CHEXPERT_LABEL_DIR = EXP_ROOT / "chexpert_labels"
METRIC_DIR = EXP_ROOT / "metrics"
ERROR_DIR = EXP_ROOT / "error_analysis"
LOG_DIR = EXP_ROOT / "logs"
DRIVE_REPORT_DIR = EXP_ROOT / "reports"

for path in [
    MANIFEST_DIR,
    SUBSET_DIR,
    PREDICTION_DIR,
    CHEXPERT_INPUT_DIR,
    CHEXPERT_LABEL_DIR,
    METRIC_DIR,
    ERROR_DIR,
    LOG_DIR,
    DRIVE_REPORT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

print("EXP_ROOT:", EXP_ROOT)
```

---

# 9. Cell Group 4 — Repository setup

```python
REPO_URL = "https://github.com/hanhpm/CXR_LLaVA_Improvement.git"
PROJECT_DIR = Path("/content/CXR_LLaVA_Improvement")
```

If the repository is already present, do not clone another copy.

Record:

```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Persist to:

```text
logs/experiment_environment.txt
```

---

# 10. Cell Group 5 — Environment metadata

Record at minimum:

```text
date/time
Google Colab runtime
GPU
Python
CUDA
PyTorch
Transformers
Tokenizers
BitsAndBytes
Accelerate
model checkpoint
quantization
compute dtype
Git branch
Git commit
```

Example:

```python
import platform
import torch
import transformers

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

Target GPU:

```text
NVIDIA T4
```

CPU may be used for XML/manifest processing. Do not perform CXR-LLaVA batch inference on CPU.

---

# 11. Cell Group 6 — Inventory raw NLMCXR files

```python
png_files = sorted(NLMCXR_IMAGE_DIR.rglob("*.png"))
xml_files = sorted(NLMCXR_REPORT_DIR.rglob("*.xml"))

print("PNG:", len(png_files))
print("XML:", len(xml_files))
```

Save inventory summary:

```text
logs/nlmcxr_inventory.txt
```

Do not proceed if either count is zero.

---

# 12. Cell Group 7 — Parse NLMCXR XML reports

Use Python standard-library XML parsing unless the repository already provides a suitable parser.

```python
from xml.etree import ElementTree as ET


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def parse_nlmcxr_report(xml_path):
    root = ET.parse(xml_path).getroot()

    sections = {}

    for node in root.findall(".//AbstractText"):
        label = clean_text(node.attrib.get("Label")).upper()
        text = clean_text(" ".join(node.itertext()))
        if label:
            sections[label] = text

    images = []

    for node in root.findall(".//parentImage"):
        image_id = clean_text(node.attrib.get("id"))
        caption_node = node.find("caption")
        caption = ""
        if caption_node is not None:
            caption = clean_text(" ".join(caption_node.itertext()))

        if image_id:
            images.append({
                "image_id": image_id,
                "caption": caption,
            })

    return {
        "xml_path": str(xml_path),
        "findings": sections.get("FINDINGS", ""),
        "impression": sections.get("IMPRESSION", ""),
        "indication": sections.get("INDICATION", ""),
        "comparison": sections.get("COMPARISON", ""),
        "images": images,
    }
```

Do not throw away `FINDINGS`, `IMPRESSION`, image IDs, or image captions.

---

# 13. Ground-truth report text rule

The main paper does not fully specify the exact Indiana report-section preprocessing. For this reproduction use:

```text
ground_truth_report = FINDINGS + " " + IMPRESSION
```

Store separately:

```text
gt_findings
gt_impression
ground_truth_report
report_text_source
```

If one section is missing, use the available `FINDINGS` or `IMPRESSION` section and record one of:

```text
findings+impression
findings_only
impression_only
missing
```

Do not include `INDICATION` or `COMPARISON` in the default label-evaluation text.

This is an **implementation choice for this reproduction**, not a claim that the paper explicitly used the identical concatenation rule.

---

# 14. Cell Group 8 — Build the full NLMCXR manifest

Create one row per candidate image-report pair.

Required columns:

```text
report_id
xml_path
image_id
image_path
image_exists
caption
view_guess
view_evidence
gt_findings
gt_impression
ground_truth_report
report_text_source
```

Build the PNG index once:

```python
image_index = {
    path.stem: path
    for path in NLMCXR_IMAGE_DIR.rglob("*.png")
}
```

Then resolve:

```python
image_path = image_index.get(image_id)
```

Do not silently discard unmatched IDs.

Save issues to:

```text
manifests/indiana_manifest_issues.csv
```

---

# 15. View classification

The paper reports **3,689 Indiana image-report pairs** and its dataset table is framed around frontal CXRs. Therefore the manifest must retain view information.

Classify conservatively using caption evidence into:

```text
frontal
lateral
unknown
```

Search caption text for terms such as:

```text
PA
AP
frontal
posteroanterior
anteroposterior
lateral
```

Do not infer view solely from filename suffixes unless verified from the dataset.

Keep:

```text
view_guess
view_evidence
```

---

# 16. Paper-count validation

Target count reported by the paper:

```text
3,689 image-report pairs
```

Do **not** force the local manifest to this count by truncating, deleting, duplicating, or randomly selecting rows.

If the validated frontal manifest does not equal 3,689:

1. record the obtained count;
2. inspect missing reports/images;
3. inspect view classification;
4. document that the exact paper selection could not yet be reconstructed;
5. continue subset evaluation only if the subset itself is valid.

Use the term:

```text
paper-aligned external evaluation
```

unless exact set reconstruction is demonstrated.

---

# 17. Manifest outputs

Save:

```text
manifests/indiana_full_manifest.csv
manifests/indiana_frontal_manifest.csv
manifests/indiana_manifest_issues.csv
```

Required checks:

```python
assert manifest["image_id"].notna().all()
assert manifest["ground_truth_report"].notna().all()
assert manifest["image_path"].notna().all()
assert manifest["image_exists"].all()
```

Also print:

```text
all candidate pairs
frontal pairs
lateral pairs
unknown-view pairs
missing image count
missing report-text count
```

---

# 18. Cell Group 9 — Deterministic subsets

Use:

```python
SEED = 42
```

Create and persist:

```text
subsets/indiana_subset_10_seed42.csv
subsets/indiana_subset_50_seed42.csv
subsets/indiana_subset_100_seed42.csv
```

Once these exist, reuse them. Do not resample on every notebook run.

Default:

```python
RUN_SUBSET_SIZE = 10
RUN_FULL_DATASET = False
```

---

# 19. Cell Group 10 — Download official Stanford CheXpert Labeler

The paper explicitly uses the **CheXpert Labeler** for report-based evaluation.

Official repositories:

```text
https://github.com/stanfordmlgroup/chexpert-labeler
https://github.com/ncbi-nlp/NegBio.git
```

Use `/content/` for tools.

```python
from pathlib import Path
import subprocess

TOOLS_ROOT = Path("/content/cxr_exp02_tools")
CHEXPERT_LABELER_DIR = TOOLS_ROOT / "chexpert-labeler"
NEGBIO_DIR = TOOLS_ROOT / "NegBio"

TOOLS_ROOT.mkdir(parents=True, exist_ok=True)

if not CHEXPERT_LABELER_DIR.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/stanfordmlgroup/chexpert-labeler.git",
            str(CHEXPERT_LABELER_DIR),
        ],
        check=True,
    )

if not NEGBIO_DIR.exists():
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/ncbi-nlp/NegBio.git",
            str(NEGBIO_DIR),
        ],
        check=True,
    )

print("CheXpert Labeler:", CHEXPERT_LABELER_DIR)
print("NegBio:", NEGBIO_DIR)
```

Validate:

```python
assert (CHEXPERT_LABELER_DIR / "label.py").exists()
assert (CHEXPERT_LABELER_DIR / "environment.yml").exists()
assert NEGBIO_DIR.exists()
```

Record commits:

```bash
git -C /content/cxr_exp02_tools/chexpert-labeler rev-parse HEAD
git -C /content/cxr_exp02_tools/NegBio rev-parse HEAD
```

Persist to:

```text
logs/chexpert_labeler_version.txt
```

---

# 20. Important — isolate CheXpert Labeler dependencies

The official CheXpert Labeler environment pins old dependencies including Python 3.6.7, old NLTK, pandas, NumPy, BLLIP parser, and Stanford dependency parsing components.

Therefore:

```text
do NOT pip-install its legacy dependency set into the main CXR-LLaVA runtime
```

That could break PyTorch/Transformers/BitsAndBytes.

Run the labeler in an **isolated environment**.

---

# 21. Cell Group 11 — Create isolated CheXpert Labeler environment

Preferred Colab strategy:

```text
micromamba isolated environment
```

Install Java:

```bash
apt-get -qq update
apt-get -qq install -y default-jre
```

Install micromamba:

```bash
mkdir -p /content/micromamba-bin

curl -Ls \
  https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /content/micromamba-bin bin/micromamba
```

Create the environment from the official repository file:

```bash
/content/micromamba-bin/bin/micromamba \
  create \
  -y \
  -r /content/micromamba-root \
  -n chexpert-label \
  -f /content/cxr_exp02_tools/chexpert-labeler/environment.yml
```

Do not change the notebook's main Python environment.

---

# 22. Cell Group 12 — Complete official CheXpert prerequisites

The official README requires:

```text
NegBio in PYTHONPATH
NLTK universal_tagset
NLTK punkt
NLTK wordnet
GENIA+PubMed BLLIP parsing model
```

Run:

```bash
PYTHONPATH=/content/cxr_exp02_tools/NegBio \
/content/micromamba-bin/bin/micromamba \
run \
-r /content/micromamba-root \
-n chexpert-label \
python -m nltk.downloader universal_tagset punkt wordnet
```

Then:

```bash
PYTHONPATH=/content/cxr_exp02_tools/NegBio \
/content/micromamba-bin/bin/micromamba \
run \
-r /content/micromamba-root \
-n chexpert-label \
python -c \
"from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"
```

If the legacy environment no longer resolves on current Colab:

1. stop the clinical-label evaluation phase;
2. save the exact error to `logs/chexpert_setup_error.txt`;
3. do not silently replace the evaluator with CheXbert;
4. preserve the generated reports so the evaluator can be fixed independently.

A CheXbert fallback must be treated as a separate compatibility experiment.

---

# 23. Cell Group 13 — Verify CheXpert Labeler on its official sample

Run the repository's `sample_reports.csv` first:

```bash
PYTHONPATH=/content/cxr_exp02_tools/NegBio \
/content/micromamba-bin/bin/micromamba \
run \
-r /content/micromamba-root \
-n chexpert-label \
python \
/content/cxr_exp02_tools/chexpert-labeler/label.py \
--reports_path \
/content/cxr_exp02_tools/chexpert-labeler/sample_reports.csv \
--output_path \
/content/chexpert_sample_labeled.csv \
--verbose
```

Validation:

```python
from pathlib import Path

assert Path("/content/chexpert_sample_labeled.csv").exists()
```

Do not proceed to NLMCXR report labeling until this passes.

---

# 24. Cell Group 14 — Load CXR-LLaVA

Reuse the previously working Colab/T4 loading implementation.

Model:

```text
ECOFRI/CXR-LLAVA-v2
```

Deployment:

```text
4-bit NF4
FP16 compute
```

Record exact:

```text
BitsAndBytesConfig
device map
compute dtype
checkpoint
Transformers version
```

Do not duplicate model internals in notebook cells.

---

# 25. Cell Group 15 — Single-image inference smoke test

Use the first row of:

```text
indiana_subset_10_seed42.csv
```

Preferred baseline helper:

```python
model.write_radiologic_report(image)
```

Record:

```text
image_id
image_path
ground_truth_report
generated_report
latency_sec
peak_vram_gb
status
error_message
```

Persist immediately:

```text
predictions/indiana_smoke_1_prediction.csv
```

Stop if this fails.

---

# 26. Generation configuration

Freeze and record:

```text
report-generation helper/prompt
temperature
top_p
max_new_tokens
precision
quantization
device map
batch size
seed
```

Do not change the prompt between samples.

Do not mix report generation, differential diagnosis, or QA in EXP-02.

---

# 27. Cell Group 16 — Run the 10-image pipeline

For each sample save:

```text
row_id
report_id
image_id
image_path
view_guess
ground_truth_report
generated_report
latency_sec
peak_vram_gb
status
error_message
```

Output:

```text
predictions/indiana_subset_10_predictions.csv
```

Save incrementally after every image.

---

# 28. Cell Group 17 — Scale to 50 only after 10-image validation

Proceed only if:

```text
[ ] all image paths resolve
[ ] XML report pairing is correct
[ ] generated reports are not empty
[ ] output CSV persists
[ ] latency is recorded
[ ] peak VRAM is recorded
[ ] no obvious image/report mismatch is found
```

Then run:

```text
subsets/indiana_subset_50_seed42.csv
```

Output:

```text
predictions/indiana_subset_50_predictions.csv
```

This is the first meaningful **subset external evaluation**.

---

# 29. Cell Group 18 — Prepare CheXpert input CSVs

The official CheXpert Labeler expects a **headerless single-column CSV**. Reports containing commas/newlines must be quoted correctly.

Create in the same row order:

```text
chexpert_inputs/indiana_subset_50_gt_reports.csv
chexpert_inputs/indiana_subset_50_generated_reports.csv
```

```python
import csv
import pandas as pd

predictions = pd.read_csv(
    PREDICTION_DIR / "indiana_subset_50_predictions.csv"
)

successful = predictions[
    predictions["status"] == "success"
].reset_index(drop=True)

gt_input = (
    CHEXPERT_INPUT_DIR
    / "indiana_subset_50_gt_reports.csv"
)

pred_input = (
    CHEXPERT_INPUT_DIR
    / "indiana_subset_50_generated_reports.csv"
)

successful[["ground_truth_report"]].to_csv(
    gt_input,
    index=False,
    header=False,
    quoting=csv.QUOTE_ALL,
)

successful[["generated_report"]].to_csv(
    pred_input,
    index=False,
    header=False,
    quoting=csv.QUOTE_ALL,
)
```

Also save:

```text
chexpert_inputs/indiana_subset_50_row_map.csv
```

Minimum columns:

```text
chexpert_row
row_id
report_id
image_id
```

---

# 30. Cell Group 19 — Run CheXpert Labeler on ground truth

Use the persistent output path:

```text
chexpert_labels/indiana_subset_50_gt_chexpert_labels.csv
```

Command pattern:

```bash
PYTHONPATH=/content/cxr_exp02_tools/NegBio \
/content/micromamba-bin/bin/micromamba \
run \
-r /content/micromamba-root \
-n chexpert-label \
python \
/content/cxr_exp02_tools/chexpert-labeler/label.py \
--reports_path "$GT_REPORTS_CSV" \
--output_path "$GT_LABELS_CSV" \
--verbose
```

Do not leave the only output in `/content/`.

---

# 31. Cell Group 20 — Run CheXpert Labeler on generated reports

Persistent output:

```text
chexpert_labels/indiana_subset_50_generated_chexpert_labels.csv
```

Validation:

```python
gt_labels = pd.read_csv(GT_LABELS_CSV)
pred_labels = pd.read_csv(PRED_LABELS_CSV)

assert len(gt_labels) == len(successful)
assert len(pred_labels) == len(successful)
```

Attach `row_id`, `report_id`, and `image_id` from the saved row map after labeling.

---

# 32. CheXpert label handling

Expected semantic states:

```text
1      positive
0      negative
-1     uncertain
NaN    not mentioned / blank
```

For paper-aligned pathology metrics include only cases where **both** GT and generated label are definite:

```python
valid = (
    gt[pathology].isin([0, 1])
    & pred[pathology].isin([0, 1])
)
```

Exclude `-1` and `NaN` for that pathology.

Do not map uncertain or blank labels to positive/negative in EXP-02.

---

# 33. Primary Indiana pathology list

```python
TARGET_PATHOLOGIES = [
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Lung Opacity",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]
```

Do not use the six-label MIMIC list. Indiana external evaluation includes `Lung Opacity`.

---

# 34. Cell Group 21 — Compute pathology metrics

For every pathology calculate:

```text
valid sample count
positive GT support
negative GT support
TP
FP
TN
FN
precision
recall
F1
```

Output:

```text
metrics/indiana_subset_50_pathology_metrics.csv
```

Required columns:

```text
pathology
n_valid
n_gt_positive
n_gt_negative
tp
fp
tn
fn
precision
recall
f1
```

Also compute:

```text
mean_pathology_f1
```

as the mean of the seven per-pathology F1 values.

---

# 35. Cell Group 22 — Bootstrap confidence intervals

Use:

```python
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42
```

Resample rows with replacement and recompute pathology F1 values.

Output:

```text
metrics/indiana_subset_50_bootstrap_ci.csv
```

Columns:

```text
pathology
f1
ci_low_95
ci_high_95
bootstrap_iterations
seed
```

Do not interpret a 50-image CI as equivalent to the paper's full external-test CI.

---

# 36. Cell Group 23 — Save system metrics

Output:

```text
metrics/indiana_subset_50_system_metrics.csv
```

Record:

```text
number_requested
number_successful
number_failed
failure_rate
mean_latency_sec
median_latency_sec
min_latency_sec
max_latency_sec
peak_vram_gb_max
peak_vram_gb_mean
```

Optional if available:

```text
mean input tokens
mean generated tokens
```

---

# 37. Cell Group 24 — Automatic error cases

Create:

```text
error_analysis/indiana_subset_50_label_errors.csv
```

For each valid pathology/sample mismatch save:

```text
row_id
report_id
image_id
pathology
gt_label
pred_label
error_type
ground_truth_report
generated_report
review_notes
reviewer
```

Automatic `error_type` may only be:

```text
FP
FN
```

Do not automatically assign semantic failure categories.

---

# 38. Cell Group 25 — Manual error review

After the 50-image run sample up to:

```text
10 FP cases
10 FN cases
```

Manual categories may include:

```text
missing finding
hallucinated finding
negation error
location mismatch
laterality mismatch
severity mismatch
ambiguous ground truth
CheXpert labeler artifact
view mismatch
```

Create:

```text
error_analysis/indiana_subset_50_manual_review.csv
```

Do not overwrite the raw automatic error file.

---

# 39. Cell Group 26 — Paper reference comparison

Create:

```text
metrics/indiana_subset_50_vs_paper_reference.csv
```

Reference values:

```python
PAPER_REFERENCE_F1 = {
    "Cardiomegaly": 0.62,
    "Consolidation": 0.31,
    "Edema": 0.67,
    "Lung Opacity": 0.85,
    "Pleural Effusion": 0.55,
    "Pneumonia": 0.63,
    "Pneumothorax": 0.05,
}
```

Columns:

```text
pathology
current_nf4_f1
paper_cxr_llava_f1
difference
```

This comparison is descriptive only.

---

# 40. 100-image gate

Proceed only if:

```text
[ ] manifest validated
[ ] image/report pairing manually spot-checked
[ ] one-image inference passed
[ ] 10-image inference passed
[ ] 50-image inference completed
[ ] CheXpert sample test passed
[ ] GT labels generated
[ ] model-output labels generated
[ ] row alignment verified
[ ] pathology metrics generated
[ ] FP/FN cases inspected
```

Then produce equivalent artifacts prefixed with:

```text
indiana_subset_100_
```

---

# 41. Full external-set gate

Default:

```python
RUN_FULL_DATASET = False
```

Do not switch it to `True` until:

```text
[ ] 100-image pipeline is stable
[ ] validated candidate evaluation manifest is frozen
[ ] relation between local manifest and paper's 3,689 pairs is documented
[ ] prompt/helper is frozen
[ ] CheXpert Labeler version is frozen
[ ] partial-save/resume logic works
[ ] Drive storage is checked
```

---

# 42. Resume logic for larger runs

```python
if prediction_csv.exists():
    previous = pd.read_csv(prediction_csv)
    completed_ids = set(
        previous.loc[
            previous["status"] == "success",
            "image_id",
        ]
    )
else:
    completed_ids = set()

remaining = subset[
    ~subset["image_id"].isin(completed_ids)
]
```

Do not rerun successfully completed samples after a Colab disconnect.

---

# 43. Required Drive artifacts after first complete 50-image run

```text
CXR_LLaVA_EXP02_Indiana_External_Eval/
├── manifests/
│   ├── indiana_full_manifest.csv
│   ├── indiana_frontal_manifest.csv
│   └── indiana_manifest_issues.csv
├── subsets/
│   ├── indiana_subset_10_seed42.csv
│   ├── indiana_subset_50_seed42.csv
│   └── indiana_subset_100_seed42.csv
├── predictions/
│   ├── indiana_smoke_1_prediction.csv
│   ├── indiana_subset_10_predictions.csv
│   └── indiana_subset_50_predictions.csv
├── chexpert_inputs/
│   ├── indiana_subset_50_gt_reports.csv
│   ├── indiana_subset_50_generated_reports.csv
│   └── indiana_subset_50_row_map.csv
├── chexpert_labels/
│   ├── indiana_subset_50_gt_chexpert_labels.csv
│   └── indiana_subset_50_generated_chexpert_labels.csv
├── metrics/
│   ├── indiana_subset_50_pathology_metrics.csv
│   ├── indiana_subset_50_bootstrap_ci.csv
│   ├── indiana_subset_50_system_metrics.csv
│   └── indiana_subset_50_vs_paper_reference.csv
├── error_analysis/
│   ├── indiana_subset_50_label_errors.csv
│   └── indiana_subset_50_manual_review.csv
├── logs/
│   ├── experiment_environment.txt
│   ├── nlmcxr_inventory.txt
│   ├── chexpert_labeler_version.txt
│   └── chexpert_setup_error.txt   # only if setup fails
└── reports/
    └── 2026-09-17_exp02_indiana_external_eval.md
```

Do not create empty placeholder files.

---

# 44. Project-local experiment report

Create the canonical report under the repository, per `AGENTS.md`:

```text
CXR_LLaVA_Improvement/
reports/
2026-09-17_exp02_indiana_external_eval.md
```

Use:

```text
experiment_analysis_template.md
```

Then copy the same report to:

```text
CXR_LLaVA_EXP02_Indiana_External_Eval/
reports/
2026-09-17_exp02_indiana_external_eval.md
```

The repo copy satisfies repository discipline; the Drive copy persists across Colab resets.

---

# 45. Report classification

For 10 images:

```text
smoke evaluation
```

For 50 or 100 images:

```text
subset external evaluation
```

Only consider:

```text
paper-aligned external benchmark reproduction
```

after the full selected Indiana evaluation manifest is executed and its relation with the paper's 3,689 pairs is established.

Never call the subset a full benchmark or clinical validation.

---

# 46. Stop conditions

Stop rather than guess when:

```text
NLMCXR PNG folder cannot be found
NLMCXR XML folder cannot be found
XML parser cannot recover image IDs
image IDs cannot be paired with PNGs reliably
ground-truth report text is missing at high frequency
frontal-view selection is unclear
exact 3,689 paper subset cannot be reconstructed
CXR-LLaVA repeatedly OOMs
CheXpert Labeler environment cannot be created
CheXpert sample labeling fails
CheXpert output row count differs from input row count
label semantics cannot be determined
prediction/GT row alignment cannot be verified
```

Save diagnostic output before stopping.

---

# 47. Do not do in EXP-02

Do not:

```text
train or fine-tune CXR-LLaVA
run LoRA / QLoRA
modify architecture
change the prompt during the baseline
add RAG
add another VLM
replace CheXpert Labeler silently
evaluate differential diagnosis
evaluate QA
mix PadChest/Brixia/CXR8 into the Indiana metric table
claim clinical reliability
force local dataset count to 3,689
```

---

# 48. EXP-02 acceptance criteria

## Data

- [ ] Drive mounted
- [ ] NLMCXR PNG directory found
- [ ] NLMCXR report directory found
- [ ] image inventory recorded
- [ ] XML inventory recorded
- [ ] XML parsing succeeds
- [ ] image ↔ report mapping succeeds
- [ ] view information retained
- [ ] manifest saved
- [ ] manifest issues saved
- [ ] relation to paper's 3,689 pairs documented

## Reproducibility

- [ ] repository URL recorded
- [ ] branch recorded
- [ ] commit recorded
- [ ] GPU recorded
- [ ] Python recorded
- [ ] CUDA recorded
- [ ] PyTorch recorded
- [ ] Transformers recorded
- [ ] BitsAndBytes recorded
- [ ] model checkpoint recorded
- [ ] quantization recorded
- [ ] seed recorded

## CXR-LLaVA inference

- [ ] one-image smoke test succeeds
- [ ] 10-image run succeeds
- [ ] outputs persist to Drive
- [ ] 50-image fixed subset executed
- [ ] raw generated reports saved
- [ ] latency saved
- [ ] peak VRAM saved
- [ ] failures preserved

## CheXpert Labeler

- [ ] official repository cloned
- [ ] NegBio cloned
- [ ] tool commit recorded
- [ ] isolated environment used
- [ ] NLTK resources installed
- [ ] GENIA+PubMed model available
- [ ] official sample test passes
- [ ] GT reports labeled
- [ ] generated reports labeled
- [ ] row alignment validated

## Evaluation

- [ ] seven Indiana pathologies evaluated
- [ ] uncertain/blank handling documented
- [ ] P/R/F1 computed
- [ ] TP/FP/TN/FN computed
- [ ] mean pathology F1 computed
- [ ] 1,000-bootstrap CI implemented
- [ ] system metrics saved

## Analysis

- [ ] FP/FN file generated
- [ ] representative errors reviewed manually
- [ ] paper reference comparison saved
- [ ] limitations documented

## Reporting

- [ ] repo-local report created
- [ ] Drive report copy created
- [ ] report classified correctly
- [ ] no benchmark overclaim made
- [ ] no clinical claim made

---

# 49. Recommended execution sequence for a coding agent

Execute strictly in this order:

```text
STEP 1   Read AGENTS.md + CRITERIA.md + README.md
STEP 2   Mount Google Drive
STEP 3   Inspect datasets root
STEP 4   Resolve NLMCXR PNG + XML directories
STEP 5   Create EXP_ROOT output folders
STEP 6   Inventory PNG + XML files
STEP 7   Parse XML reports
STEP 8   Build image ↔ report manifest
STEP 9   Classify frontal/lateral/unknown views
STEP 10  Validate manifest and paper-count relation
STEP 11  Create deterministic 10/50/100 subsets
STEP 12  Clone official CheXpert Labeler + NegBio
STEP 13  Create isolated CheXpert environment
STEP 14  Run official CheXpert sample test
STEP 15  Reuse existing CXR-LLaVA NF4/T4 loader
STEP 16  Run 1-image smoke inference
STEP 17  Run 10 images
STEP 18  Verify saved outputs manually
STEP 19  Run fixed 50-image subset
STEP 20  Create GT + generated CheXpert input CSVs
STEP 21  Run CheXpert Labeler on both report sets
STEP 22  Validate row alignment
STEP 23  Compute 7-pathology P/R/F1
STEP 24  Compute 1,000-bootstrap CIs
STEP 25  Save system metrics
STEP 26  Generate FP/FN cases
STEP 27  Manually inspect representative errors
STEP 28  Create repository + Drive experiment reports
STEP 29  Decide whether to run 100 images
STEP 30  Only later decide whether full Indiana evaluation is justified
```

---

# 50. Short agent instruction

> Implement EXP-02 exactly according to this roadmap. The NLMCXR PNG images and XML reports are already available under the Google Drive datasets root. Do not redownload the dataset. First resolve the actual two dataset folders, build and validate the Indiana image–report manifest, preserve view information, create deterministic 10/50/100 subsets, and persist all experiment artifacts under `CXR_LLaVA_EXP02_Indiana_External_Eval/` in the same Drive datasets root. Clone the official Stanford CheXpert Labeler and NegBio into `/content/`, run the labeler in an isolated environment, and validate it on its official sample before evaluating project reports. Reuse the existing working NF4/T4 CXR-LLaVA inference code. Run 1 image, then 10, then 50. Apply CheXpert Labeler to both original and generated reports, evaluate the seven Indiana pathologies using only definite 0/1 labels, compute P/R/F1, confusion counts and 1,000-bootstrap CIs, save FP/FN cases, and create the experiment report. Do not train, change prompts, silently substitute another labeler, force the dataset to 3,689 rows, or run the full dataset before the 50-image pipeline is validated.

---

# 51. External references used to design this roadmap

## CXR-LLaVA paper

```text
https://arxiv.org/pdf/2310.18341
```

## Stanford CheXpert Labeler

```text
https://github.com/stanfordmlgroup/chexpert-labeler
```

## NegBio

```text
https://github.com/ncbi-nlp/NegBio.git
```

## NLMCXR/OpenI public dataset downloads

```text
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
```

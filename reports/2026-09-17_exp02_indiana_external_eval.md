# Experiment Analysis Log

> **SUPERSEDED ? 2026-09-21:** Tr?ng th?i th? nghi?m trong log 17/09 ?? ???c thay b?ng [timeline v? audit h?p nh?t](2026-09-21_exp02_consolidated_audit.md). Gi? nguy?n n?i dung d??i ??y l?m b?ng ch?ng l?ch s?; kh?ng d?ng l?m tr?ng th?i hi?n t?i.

## 1. Experiment Information

- **Experiment ID:** CXR-LLAVA-EXP02-IU-EXTERNAL-EVAL
- **Date:** 2026-09-17
- **Researcher:** USTH ICTLab
- **Project:** CXR_LLaVA_Improvement
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement
- **Branch:** `master`
- **Commit at report preparation:** `d49bda7f3010441715ad43b4cf4ad0ccc1ee6fa2`
- **Notebook / Script:** `02_CXR_LLaVA_IU_external_eval.ipynb`
- **Status:** Running; 10-image smoke/subset pipeline partially validated

---

## 2. Objective

Build a reproducible Indiana University/OpenI external-evaluation pipeline for CXR-LLaVA using the downloaded NLMCXR PNG/XML data, deterministic F1-based subsets, the official Stanford CheXpert Labeler, Drive-backed resume logic, and pathology-level metrics.

This work is a research external-evaluation reproduction. It is not clinical validation.

---

## 3. Survey / Technical Motivation

### References reviewed

- CXR-LLaVA paper: https://arxiv.org/abs/2310.18341
- CXR-LLaVA repository: https://github.com/hanhpm/CXR_LLaVA_Improvement
- Model: `ECOFRI/CXR-LLAVA-v2`
- Stanford CheXpert Labeler: https://github.com/stanfordmlgroup/chexpert-labeler
- NegBio: https://github.com/ncbi-nlp/NegBio
- Project roadmap: `plan/CXR_LLaVA_EXP02_Indiana_External_Eval_Roadmap.md`

### Key idea

Use the NLMCXR XML reports to build image-report pairs, select the XML `F1` image references as paper-aligned candidates, run CXR-LLaVA report generation with the established T4 NF4 configuration, label both reference and generated reports with the official CheXpert Labeler, and evaluate only joint definite 0/1 labels for the seven Indiana pathologies.

### Hypothesis

The existing CXR-LLaVA NF4/T4 inference path can be reused for an incremental 10/50/100-image external-evaluation pipeline, while all persistent artifacts survive Colab session loss through Google Drive.

---

## 4. Environment

| Component | Version / Configuration |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA T4 target; runtime-specific value recorded by notebook |
| Python | Runtime-specific; recorded by notebook |
| CUDA | Runtime-specific; recorded by notebook |
| PyTorch | Runtime-specific; recorded by notebook |
| Transformers | `4.46.3` |
| Tokenizers | `<0.21` |
| Hugging Face Hub | `0.36.0` |
| Accelerate | Installed and recorded by notebook |
| BitsAndBytes | Installed and recorded by notebook |
| Protobuf | `5.29.6` |
| Model | `ECOFRI/CXR-LLAVA-v2` |
| Precision | FP16 compute |
| Quantization | 4-bit NF4 |
| Device map | `{"": 0}` |

### Environment verification

The notebook writes environment metadata to:

```text
CXR_LLaVA_EXP02_Indiana_External_Eval/logs/experiment_environment.txt
```

The metadata includes Python, PyTorch, Transformers, Tokenizers, Hugging Face Hub, Accelerate, BitsAndBytes, CUDA, GPU, project branch, and project commit.

---

## 5. Dependency Setup

### Installation command

```bash
%pip install -q --upgrade \
    "transformers==4.46.3" \
    "tokenizers<0.21" \
    "huggingface-hub==0.36.0" \
    "protobuf==5.29.6" \
    sentencepiece accelerate "pillow>=11.0,<12" bitsandbytes
```

CheXpert Labeler dependencies are isolated in the official micromamba environment and are not installed into the CXR-LLaVA runtime.

### Dependency issues

1. A runtime encountered an inconsistent Pillow installation:

```text
ImportError: cannot import name '_Ink' from PIL._typing
```

The remedy is to force-reinstall the pinned Pillow range and restart the Colab runtime before importing Torch/Transformers.

2. The Hugging Face `HF_TOKEN` warning is not a blocking error because the selected model is public.

3. The remote-code warning from Hugging Face is retained as a reproducibility/security note. The notebook records the model configuration; an exact resolved model revision should be frozen before a full benchmark claim.

### Decision

Continue with the isolated official CheXpert environment and the existing NF4/T4 model configuration. Do not use CheXbert as a fallback.

---

## 6. Code / Notebook Changes

### Files changed

- `02_CXR_LLaVA_IU_external_eval.ipynb`
- `reports/2026-09-17_exp02_indiana_external_eval.md`

### Main changes

- Added Colab runtime dependency setup.
- Added XML parsing, PNG indexing, image-report manifest creation, and progress bars.
- Added `figure_id` and derived image suffix metadata directly inside the notebook.
- Added `image_metadata = manifest.copy()` so F1 selection does not depend on stale session state.
- Selected XML `F1` references as paper-aligned candidates instead of incorrectly calling paired `PA and lateral` captions frontal.
- Added deterministic persistent F1 subsets for 10, 50, and 100 images.
- Added Drive-backed incremental prediction files and resume behavior.
- Added independent `RUN_INFERENCE` and `RUN_CLINICAL_EVAL` gates.
- Added official CheXpert sample validation in an isolated micromamba environment.
- Added pathology metrics with explicit `defined`/`undefined` F1 status.
- Added bootstrap, system-metric, FP/FN, manual-review, and paper-reference artifact generation.

The changes are compatibility, reproducibility, and evaluation-pipeline fixes. They do not modify the model architecture or train the model.

---

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| Input data | NLMCXR/OpenI PNG images and XML reports |
| Local F1 image-report pairs | 1,251 / 3,689 paper-reference F1 pairs |
| Smoke stages | 10, 50, 100 |
| Current validated stage | 10-image subset |
| Image selection | XML `figure_id == F1`, not confirmed frontal-view classification |
| Report text | `FINDINGS + IMPRESSION` when available |
| Prompt | Existing `model.write_radiologic_report(image)` helper |
| Precision | FP16 compute |
| Quantization | 4-bit NF4 |
| Device map | `{"": 0}` |
| Batch size | 1; sequential inference |
| Seed | 42 |
| Bootstrap iterations | 1,000 when clinical artifact cell runs |
| Full dataset | Disabled |

---

## 8. Code Executed

Important gates:

```python
RUN_INFERENCE = False
RUN_CLINICAL_EVAL = False
RUN_SUBSET_SIZE = 10
RUN_FULL_DATASET = False
```

Persistent output root:

```text
/content/drive/MyDrive/USTH_Master/ResearchLab/Notebook/datasets/
CXR_LLaVA_EXP02_Indiana_External_Eval/
```

The inference artifact naming convention is:

```text
indiana_f1_subset_{N}_predictions.csv
```

The clinical-label and metric artifacts use the same `indiana_f1_subset_{N}` prefix.

---

## 9. Output

### Raw output

Dataset inventory:

```text
PNG files: 2541
XML files: 3955
XML image references: 7470
F1 XML references: 3689
F1 with local PNG: 1251
F1 without local PNG: 2438
```

CheXpert official sample:

```text
RETURN CODE: 0
Sample label rows: 4
CHEXPERT_READY: True
```

Subset-10 metrics artifact:

```text
metrics/indiana_f1_subset_10_pathology_metrics.csv
```

### Errors / warnings

- The initial clinical-label cell used an inconsistent prefix (`indiana_subset_10` instead of `indiana_f1_subset_10`); this was corrected.
- Re-enriching already enriched subset CSVs caused merge suffixes and a missing `ground_truth_report`; the cell was corrected to drop stale ground-truth columns before merging.
- The initial F1 display cell referenced an undefined `suffix` column; the notebook now derives it from `image_id`.
- A local dataset completeness limitation prevents full 3,689-pair reproduction.

---

## 10. Output Analysis

### What worked

- XML reports were parsed and persisted to Drive.
- F1 reference counting matched the paper's 3,689 reference count.
- 1,251 valid local F1 pairs were available for deterministic subsets.
- 10/50/100 F1 subset files were created from the available local pairs.
- The official CheXpert Labeler environment and sample test passed.
- Subset-10 CheXpert labels and pathology metrics were generated.
- Pathology F1 aggregation now reports undefined coverage instead of presenting a misleading complete mean.
- Prediction and label artifacts are designed to survive Colab session loss through Drive storage.

### What failed or remains incomplete

- Only 1,251 of the paper's 3,689 F1 image references have local PNG files.
- Full paper-aligned Indiana evaluation cannot be claimed from the current upload.
- Subset-10 pathology support is too small for meaningful external-performance conclusions.
- Several pathology F1 values are undefined because joint definite 0/1 support is insufficient.
- Exact resolved Hugging Face model revision is not yet frozen in the report.

### Root cause

The local NLMCXR image archive is incomplete relative to the XML report archive. Additionally, captions such as `PA and lateral` describe a paired study and do not identify the view of each individual `parentImage`; therefore the pipeline uses `F1` as a paper-aligned reference identifier without claiming it is a confirmed frontal view.

### Fix applied

- Added local-coverage accounting.
- Switched subset creation to valid local F1 references.
- Added explicit artifact prefixes and Drive persistence.
- Added independent inference and clinical-evaluation gates.
- Added undefined-F1 status and coverage reporting.

---

## 11. Quantitative Results

The following subset-10 values are descriptive only:

| Pathology | n_valid | TP | FP | TN | FN | F1 status |
|---|---:|---:|---:|---:|---:|---|
| Cardiomegaly | 2 | 0 | 0 | 1 | 1 | Undefined |
| Consolidation | 1 | 0 | 1 | 0 | 0 | Undefined |
| Edema | 1 | 0 | 0 | 1 | 0 | Undefined |
| Lung Opacity | 1 | 1 | 0 | 0 | 0 | Defined; F1=1.0 |
| Pleural Effusion | 6 | 0 | 0 | 6 | 0 | Undefined |
| Pneumonia | 0 | 0 | 0 | 0 | 0 | Undefined |
| Pneumothorax | 5 | 0 | 0 | 5 | 0 | Undefined |

The apparent `mean_pathology_f1 = 1.0` from the earlier implementation was misleading because pandas ignored undefined NaN values and averaged only the one defined pathology. The revised notebook does not report a complete mean unless all seven pathology F1 values are defined.

System latency and VRAM are saved only after successful inference rows are available; they should be interpreted as subset system measurements, not benchmark-wide performance.

---

## 12. Qualitative Results

### Input

Deterministic XML `F1` image-report pairs from the available local NLMCXR archive.

### Model output

Raw generated reports are retained in the Drive prediction CSV. They are not reproduced here as clinical findings.

### Observation

No manual semantic error conclusion is made from the subset-10 aggregate. FP/FN rows and a manual-review scaffold are generated for later review.

---

## 13. Comparison with Baseline

| Aspect | 2026-09-16 baseline | 2026-09-17 EXP-02 |
|---|---|---|
| Dataset | padChest, CXR8, Brixia subsets | Indiana/OpenI NLMCXR external data |
| Model | ECOFRI/CXR-LLAVA-v2 | ECOFRI/CXR-LLAVA-v2 |
| Runtime | Colab/T4, NF4/FP16 | Colab/T4 target, NF4/FP16 |
| Tasks | Report, differential, QA | Report only |
| Evaluator | BLEU/ROUGE baseline; clinical metrics pending | Official CheXpert Labeler sample passed |
| Persistence | Project/result artifacts | Drive-backed manifest, prediction, label, metric artifacts |
| Evaluation state | Small baseline subset | 10-image smoke/subset external pipeline |

### Interpretation

EXP-02 improves evaluation reproducibility and external-data handling, but it does not yet establish a quality improvement or a paper benchmark reproduction. The current result is limited by local dataset coverage and subset size.

---

## 14. Problems Encountered

### Problem 1: Incomplete local F1 image coverage

**Error / evidence**

```text
F1 XML references: 3689
F1 with local PNG: 1251
F1 without local PNG: 2438
```

**Cause**

The local PNG archive does not contain all image references represented by the XML reports.

**Solution**

Persist the coverage count, run only subsets from available valid F1 pairs, and do not claim the full paper benchmark.

**Status:** Unresolved dataset limitation

### Problem 2: Pillow import failure during model load

**Error**

```text
ImportError: cannot import name '_Ink' from PIL._typing
```

**Cause**

Inconsistent Pillow installation in the Colab runtime.

**Solution**

Force-reinstall a compatible Pillow range and restart the runtime before importing Torch/Transformers.

**Status:** Workaround documented

### Problem 3: CheXpert label command failure / hidden stderr

**Cause**

The initial cell did not set the CheXpert working directory or preserve subprocess diagnostics, and later cells used inconsistent artifact prefixes.

**Solution**

Use `cwd=CHEXPERT_LABELER_DIR`, capture stdout/stderr, save logs to Drive, and use `indiana_f1_subset_{N}` consistently.

**Status:** Fixed; official sample passed

### Problem 4: Undefined pathology mean

**Cause**

Pandas ignored NaN F1 values when computing the mean, producing an apparently perfect value from one defined pathology.

**Solution**

Add `f1_status` and suppress the complete mean unless all seven pathologies have defined F1.

**Status:** Fixed

---

## 15. Conclusion

The EXP-02 notebook now contains a Drive-backed, resumable Indiana/OpenI evaluation pipeline with deterministic F1 subsets, official CheXpert sample validation, independent inference/evaluation gates, and explicit undefined-F1 handling. The local data currently supports 1,251 of the paper's 3,689 F1 references, so the work is not a full paper benchmark reproduction. The subset-10 metrics are too sparse for quality conclusions, and the earlier apparent mean F1 of 1.0 was correctly identified as misleading. The next safe step is to validate the 10-image prediction/label alignment and then run the fixed 50-image subset, without changing the prompt or claiming clinical validity.

---

## 16. Next Experiment

- [ ] Run the notebook from a clean Colab runtime and save the final executed notebook state.
- [ ] Verify exact resolved Hugging Face model revision and record it in `experiment_environment.txt`.
- [ ] Complete/resume the 10-image inference and clinical-label artifact checks.
- [ ] Run the fixed 50-image F1 subset after the 10-image gate passes.
- [ ] Inspect representative FP/FN cases manually.
- [ ] Obtain the missing 2,438 F1 PNG files before considering full evaluation.
- [ ] Run the 100-image gate only after the 50-image pipeline is stable.

---

## 17. Reproducibility Checklist

- [x] Repository URL recorded
- [x] Branch recorded
- [x] Git commit recorded at report preparation
- [x] Notebook/script recorded
- [x] Environment configuration recorded in notebook
- [x] Dataset/input counts recorded
- [x] Random seed recorded
- [x] Exact subset naming and Drive output root recorded
- [x] Raw generated reports designed to be saved
- [x] Subset-10 metric artifact saved
- [x] Errors and fixes documented
- [ ] Clean Colab rerun completed after final notebook edits
- [ ] Exact Hugging Face model revision frozen
- [ ] Full 3,689-pair evaluation completed

---

## 18. Short Lab Update

> **Objective:** Build a reproducible Indiana/OpenI external-evaluation pipeline for CXR-LLaVA.  
> **Survey:** The pipeline follows the CXR-LLaVA paper's seven-pathology CheXpert methodology and the project EXP-02 roadmap.  
> **Implementation:** Added XML/PNG manifesting, F1-based deterministic subsets, Drive-backed resume, isolated official CheXpert labeling, and explicit metric coverage handling.  
> **Experiment:** The official CheXpert sample passed and subset-10 artifacts were generated.  
> **Result:** Only 1,251/3,689 F1 images are locally available; subset-10 pathology support is too sparse for a benchmark claim.  
> **Issue:** Missing local PNG coverage and incomplete final clean-runtime rerun.  
> **Next step:** Validate the 10-image gate, run the fixed 50-image subset, and preserve the paper-aligned limitation.

This report describes research outputs only. It is not a clinical diagnosis, treatment recommendation, or clinical validation.

# K80 report evaluation commands

The paper uses MIMIC report generation internally (3000 images), Indiana
externally (3689 pairs), and a separate CheXpert binary classification task
(518 images). This workflow implements a small report-generation evaluation
for MIMIC or Indiana, not the separate CheXpert image classification experiment.

Supply actual 8-bit CXR images and original reference reports. CSV columns:
`report_id,image_id,image_path,ground_truth_report`. One unique image per report,
nonempty reports, maximum 10 pairs by default. Relative image paths resolve
against `--image-root`. CSV quoting must preserve reference text. No reports,
review results, images or labels are fabricated or downloaded automatically.
Review pairing and image views before interpreting metrics. Existing MIMIC
metadata has only DICOM paths, and the old Indiana cohort paths are Colab paths;
those old artifacts are not ready local cohorts. The local OpenI collection was
subsequently located under `/storage/student4/hanhpm/datasets/NLMCRX`: 3955 XML
reports and 7470 PNG images, with 3826 eligible pairs. A fixed random cohort
of ten reports (seed 42) was created in `data/local_indiana_10_seed42.csv`
using `scripts.prepare_local_indiana`. Each report uses its first existing image
in XML order, which is not a verified frontal-view selection. All ten images
decoded successfully and runner preflight passed; no cohort inference was run
during this data preparation. The JSON beside the CSV records selection/hashes.

```bash
cd /storage/student4/hanhpm/CXR_LLaVA_Improvement
source /home/student4/anaconda3/etc/profile.d/conda.sh
conda activate cxr-llava-k80
python -m pip install -r requirements-k80.txt
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATASET=indiana
export COHORT="$PWD/data/local_indiana_10_seed42.csv"
export IMAGE_ROOT=/storage/student4/hanhpm/datasets/NLMCRX/NLMCXR_png
export RUN_DIR="$PWD/result/${DATASET}_k80_$(date +%Y%m%d_%H%M%S)"
python -m scripts.k80_original --dataset "$DATASET" --cohort "$COHORT" --image-root "$IMAGE_ROOT"
python -u -m scripts.k80_original --dataset "$DATASET" --cohort "$COHORT" --image-root "$IMAGE_ROOT" --execute --output "$RUN_DIR"
```

## Official labeler setup (separate legacy environment)

This installation follows https://github.com/stanfordmlgroup/chexpert-labeler .
The exact legacy Conda package solve and parser downloads have not been run in
this session. Stop on setup or sample failure; no replacement labeler is used.
Java is required and was found on this server. Clone only into new directories.

```bash
export TOOLS_ROOT="$PWD/model_cache/chexpert_tools"
mkdir -p "$TOOLS_ROOT"
git clone https://github.com/stanfordmlgroup/chexpert-labeler.git "$TOOLS_ROOT/chexpert-labeler"
git -C "$TOOLS_ROOT/chexpert-labeler" checkout 44ddeb363149aa657296237f18b5472a73c1756f
git clone https://github.com/ncbi-nlp/NegBio.git "$TOOLS_ROOT/NegBio"
git -C "$TOOLS_ROOT/NegBio" checkout 073199e2792824740e89844a59c13d3d40ce4d23
CONDA_CHANNEL_PRIORITY=flexible conda env create -n chexpert-label -f "$TOOLS_ROOT/chexpert-labeler/environment.yml"
conda activate chexpert-label
export PYTHONPATH="$TOOLS_ROOT/NegBio"
java -version
python -m nltk.downloader universal_tagset punkt wordnet
python -c "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"
export LABELER_PYTHON="$(command -v python)"
conda activate cxr-llava-k80
python -m scripts.label_cohort --source "$RUN_DIR" --labeler-dir "$TOOLS_ROOT/chexpert-labeler" --negbio-dir "$TOOLS_ROOT/NegBio" --labeler-python "$LABELER_PYTHON"
```

The wrapper requires exact pinned commits, checks successful full-cohort inference,
runs the official sample and compares its actual labels to the official reference,
then labels both GT and generated reports. Logs, package versions and provenance
are saved under `labeler_logs/`. Existing label/log outputs are preserved. On a
failed labeling attempt, inspect its log; move failed outputs aside before retrying.

## Metrics

For a CSV of exactly 10 pairs:

```bash
python -m scripts.exp02_reevaluate --dataset "$DATASET" --source "$RUN_DIR" --output "${RUN_DIR}_metrics" --sizes 10 --bootstrap-iterations 1000 --seed 42
cat "${RUN_DIR}_metrics/metrics/${DATASET}_f1_subset_10_pathology_metrics_v2.csv"
cat "${RUN_DIR}_metrics/metrics/${DATASET}_f1_subset_10_summary.json"
cat "${RUN_DIR}_metrics/metrics/${DATASET}_f1_subset_10_bootstrap_ci_v2.csv"
```

Use `DATASET=mimic` and a MIMIC CSV for the internal test. MIMIC uses six findings
(cardiomegaly, consolidation, edema, pleural effusion, pneumonia, pneumothorax);
Indiana adds lung opacity. Definite 0/1 labels on both sides are eligible;
uncertain -1 and unmentioned NaN are excluded, not converted to negative.
Outputs include per-pathology precision/recall/F1, confusion counts, eligible-pair
coverage, omission/error queues, 1000-iteration bootstrap intervals and macro means.
Undefined metrics remain undefined and are excluded from means, with their count
reported. Row counts, report text/order and image/report IDs are checked.

The K80 workflow retains the official helper settings (temperature 0.2/top_p 0.8)
and prompt but changes BF16 to FP32 and dispatches across four GPUs. These are
hardware adaptations; exact paper inference settings and table scores are not
claimed. A ten-case subset is a pipeline check with metrics, not a benchmark.

Validation: existing nine evaluator tests plus synthetic end-to-end cohort export,
alignment and known precision/recall/F1 tests for both datasets passed. Test
fixtures are not clinical labels. No real new cohort inference or official labeler
setup/sample/labeling was performed for this code change.

Legacy environment troubleshooting: strict channel priority excluded required
old builds. The unchanged official YAML solved successfully in a dry run with
`CONDA_CHANNEL_PRIORITY=flexible` and the Miniforge LibMamba solver on this
server. This applies only to that command; global Conda settings are unchanged.
Actual package/Pip installation and the official sample gate remain separate checks.

from pathlib import Path

import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip() + "\n")


def code(text):
    return nbf.v4.new_code_cell(text.strip() + "\n")


cells = [
    md(
        """
# EXP-02 Recovery / Reevaluation C-D-F Notebook

Research-only notebook for continuing EXP-02 after offline reevaluation. Run on Google Colab with a T4 GPU and Drive-mounted NLMCXR/OpenI data.

Order:

1. Mount Drive.
2. Run C1 dataset/root/inventory preflight. Stop if it fails.
3. Enable model provenance and run C2/C3.
4. Enable diagnostic retry for the two historical empty-output images.
5. Enable gate10 only after C and D evidence are acceptable.

Generated text is experimental model output, not a clinical finding.
"""
    ),
    code(
        """
# Cell 1 - Mount Drive.
from google.colab import drive

drive.mount('/content/drive', force_remount=True)
"""
    ),
    code(
        """
# Cell 2 - Configuration. Change only paths/flags here.
from pathlib import Path
from datetime import datetime, timezone
import csv
import gc
import hashlib
import inspect
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid

import pandas as pd
from PIL import Image

SEED = 42
random.seed(SEED)

DATASETS_ROOT_CANDIDATES = [
    Path('/content/drive/MyDrive/ResearchLab/Notebook/datasets'),
    Path('/content/drive/MyDrive/USTH_Master/ResearchLab/Notebook/datasets'),
]

EXP02_V2_ROOT_NAME = 'CXR_LLaVA_EXP02_Indiana_External_Eval_v2'
PARENT_OFFLINE_RUN_ID = '20260921T031045Z_a2ca54af'
RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]

RUN_MODEL_PROVENANCE = False
RUN_DIAGNOSTIC_RETRY = False
RUN_GATE10 = False
RUN_STAGE50 = False
RUN_STAGE100 = False
RUN_FULL_DATASET = False

MODEL_ID = 'ECOFRI/CXR-LLAVA-v2'
FAILED_IMAGE_IDS = ['CXR159_IM-0382-1001', 'CXR1706_IM-0466-1001']
CONTROL_IMAGE_ID = 'CXR1019_IM-0015-1001'

print('RUN_ID:', RUN_ID)
"""
    ),
    code(
        """
# Cell 3 - Helpers.
from xml.etree import ElementTree as ET


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def clean_text(value):
    if value is None:
        return ''
    return ' '.join(str(value).split())


def sha256_file(path, chunk_size=1024 * 1024):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text):
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\\n', encoding='utf-8')


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + '\\n')


def parse_nlmcxr_report(xml_path):
    root = ET.parse(xml_path).getroot()
    sections = {}
    for node in root.findall('.//AbstractText'):
        label = clean_text(node.attrib.get('Label')).upper()
        text = clean_text(' '.join(node.itertext()))
        if label:
            sections[label] = text
    images = []
    for node in root.findall('.//parentImage'):
        image_id = clean_text(node.attrib.get('id'))
        caption_node = node.find('caption')
        caption = clean_text(' '.join(caption_node.itertext())) if caption_node is not None else ''
        if image_id:
            images.append({'image_id': image_id, 'caption': caption})
    findings = sections.get('FINDINGS', '')
    impression = sections.get('IMPRESSION', '')
    if findings and impression:
        source = 'findings+impression'
    elif findings:
        source = 'findings_only'
    elif impression:
        source = 'impression_only'
    else:
        source = 'missing'
    return {
        'report_id': Path(xml_path).stem.replace('CXR', ''),
        'xml_path': str(xml_path),
        'findings': findings,
        'impression': impression,
        'ground_truth_report': clean_text(f'{findings} {impression}'),
        'report_text_source': source,
        'images': images,
    }


def classify_view(caption):
    value = str(caption).lower()
    has_lateral = 'lateral' in value
    has_frontal = any(term in value for term in ['pa', 'ap', 'frontal', 'posteroanterior', 'anteroposterior'])
    if has_lateral and not has_frontal:
        return 'lateral', caption
    if has_frontal and not has_lateral:
        return 'frontal', caption
    return 'unknown', caption


def environment_snapshot():
    result = {'date_utc': utc_now(), 'python': platform.python_version(), 'platform': platform.platform()}
    for name in ['torch', 'transformers', 'bitsandbytes', 'accelerate', 'tokenizers', 'huggingface_hub', 'PIL', 'numpy', 'pandas']:
        try:
            module = __import__(name)
            result[name] = getattr(module, '__version__', 'unknown')
        except Exception as exc:
            result[name] = f'import_error: {type(exc).__name__}: {exc}'
    try:
        import torch
        result['cuda_available'] = bool(torch.cuda.is_available())
        result['gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        result['cuda_version'] = getattr(torch.version, 'cuda', None)
    except Exception as exc:
        result['cuda_error'] = f'{type(exc).__name__}: {exc}'
    return result


def resolve_paths():
    candidate_records = []
    valid = []
    for datasets_root in DATASETS_ROOT_CANDIDATES:
        for base in [datasets_root, datasets_root / 'NLMCRX']:
            png_dir = base / 'NLMCXR_png'
            report_dir = base / 'NLMCXR_reports'
            record = {
                'datasets_root': str(datasets_root),
                'base': str(base),
                'png_dir': str(png_dir),
                'report_dir': str(report_dir),
                'png_exists': png_dir.exists(),
                'report_exists': report_dir.exists(),
            }
            if png_dir.exists():
                record['png_count'] = sum(1 for _ in png_dir.rglob('*.png'))
            if report_dir.exists():
                record['xml_count'] = sum(1 for _ in report_dir.rglob('*.xml'))
            candidate_records.append(record)
            if record['png_exists'] and record['report_exists']:
                valid.append(record)
    if not valid:
        raise RuntimeError('No dataset candidate has both NLMCXR_png and NLMCXR_reports. Check Drive mount and DATASETS_ROOT_CANDIDATES.')
    selected = valid[0]
    datasets_root = Path(selected['datasets_root'])
    return {
        'datasets_root': datasets_root,
        'nlmcxr_image_dir': Path(selected['png_dir']),
        'nlmcxr_report_dir': Path(selected['report_dir']),
        'exp_root': datasets_root / EXP02_V2_ROOT_NAME / RUN_ID,
        'historical_root': datasets_root / 'CXR_LLaVA_EXP02_Indiana_External_Eval',
        'candidate_records': candidate_records,
        'selected': selected,
    }
"""
    ),
    code(
        """
# Cell 4 - Phase C1: dataset root, manifest and historical cohort preflight. Stop here if it fails.
paths = resolve_paths()
EXP_ROOT = paths['exp_root']
for subdir in ['logs', 'diagnostics', 'manifests', 'subsets', 'attempts', 'predictions', 'metrics', 'chexpert_inputs', 'chexpert_labels', 'review']:
    (EXP_ROOT / subdir).mkdir(parents=True, exist_ok=True)

write_json(EXP_ROOT / 'diagnostics' / 'dataset_root_candidates.json', {
    'candidate_records': paths['candidate_records'],
    'selected': paths['selected'],
})
write_json(EXP_ROOT / 'logs' / 'environment.json', environment_snapshot())

png_files = sorted(paths['nlmcxr_image_dir'].rglob('*.png'))
xml_files = sorted(paths['nlmcxr_report_dir'].rglob('*.xml'))
image_index = {p.stem: p for p in png_files}
rows = []
issues = []
for xml_path in xml_files:
    parsed = parse_nlmcxr_report(xml_path)
    for image in parsed['images']:
        image_id = image['image_id']
        image_path = image_index.get(image_id)
        view_guess, view_evidence = classify_view(image.get('caption', ''))
        row = {
            'report_id': parsed['report_id'],
            'xml_path': str(xml_path),
            'image_id': image_id,
            'image_path': str(image_path) if image_path else '',
            'image_exists': bool(image_path),
            'caption': image.get('caption', ''),
            'view_guess': view_guess,
            'view_evidence': view_evidence,
            'gt_findings': parsed['findings'],
            'gt_impression': parsed['impression'],
            'ground_truth_report': parsed['ground_truth_report'],
            'report_text_source': parsed['report_text_source'],
        }
        rows.append(row)
        if not image_path:
            issues.append({**row, 'issue': 'missing_png'})
        if not parsed['ground_truth_report']:
            issues.append({**row, 'issue': 'missing_report_text'})

manifest = pd.DataFrame(rows)
issues_df = pd.DataFrame(issues)
manifest.to_csv(EXP_ROOT / 'manifests' / 'indiana_full_manifest.csv', index=False)
issues_df.to_csv(EXP_ROOT / 'manifests' / 'indiana_manifest_issues.csv', index=False)
valid_f1 = manifest[(manifest['image_exists']) & (manifest['ground_truth_report'].fillna('').astype(str).str.strip().astype(bool))].copy()
valid_f1.to_csv(EXP_ROOT / 'manifests' / 'indiana_valid_manifest.csv', index=False)

summary = {
    'png_count': len(png_files),
    'xml_count': len(xml_files),
    'manifest_rows': len(manifest),
    'valid_f1_rows': len(valid_f1),
    'missing_png_rows': int((~manifest['image_exists']).sum()) if len(manifest) else 0,
    'missing_report_text_rows': int((manifest['ground_truth_report'].fillna('').astype(str).str.strip() == '').sum()) if len(manifest) else 0,
    'view_counts': manifest['view_guess'].value_counts(dropna=False).to_dict() if len(manifest) else {},
}
write_json(EXP_ROOT / 'diagnostics' / 'phase_c1_inventory_summary.json', summary)
print(json.dumps(summary, indent=2, ensure_ascii=False))

if not paths['historical_root'].exists():
    raise RuntimeError(f'Historical EXP-02 root not found: {paths[\"historical_root\"]}')
for size in [10, 50, 100]:
    subset_path = paths['historical_root'] / 'subsets' / f'indiana_f1_subset_{size}_seed42.csv'
    if not subset_path.exists():
        raise RuntimeError(f'Missing historical subset: {subset_path}')
print('PASS C1 dataset/root preflight. EXP_ROOT:', EXP_ROOT)
"""
    ),
    code(
        """
# Cell 5 - Verify control and failed images before model load.
historical_root = paths['historical_root']
subset100 = pd.read_csv(historical_root / 'subsets' / 'indiana_f1_subset_100_seed42.csv')
needed_ids = [CONTROL_IMAGE_ID] + FAILED_IMAGE_IDS
verification_rows = []
for image_id in needed_ids:
    matches = subset100[subset100['image_id'].astype(str) == image_id]
    if matches.empty:
        verification_rows.append({'image_id': image_id, 'status': 'missing_from_historical_subset100'})
        continue
    row = matches.iloc[0].to_dict()
    old_path = Path(str(row['image_path']))
    remapped_path = paths['nlmcxr_image_dir'] / old_path.name
    status = {'image_id': image_id, 'old_path': str(old_path), 'remapped_path': str(remapped_path), 'exists': remapped_path.exists()}
    try:
        with Image.open(remapped_path) as img:
            img.verify()
        with Image.open(remapped_path) as img:
            status.update({'width': img.width, 'height': img.height, 'mode': img.mode, 'sha256': sha256_file(remapped_path)})
    except Exception as exc:
        status.update({'image_error': f'{type(exc).__name__}: {exc}'})
    status['gt_hash'] = sha256_text(row.get('ground_truth_report', ''))
    status['report_id'] = row.get('report_id', '')
    verification_rows.append(status)

verification = pd.DataFrame(verification_rows)
verification.to_csv(EXP_ROOT / 'diagnostics' / 'failed_and_control_image_verify.csv', index=False)
print(verification.to_string(index=False))
if not verification['exists'].all():
    raise RuntimeError('At least one control/failed image is not readable from the resolved dataset root.')
"""
    ),
    code(
        """
# Cell 6 - Install once, restart session, then rerun Cells 1-5 and 7.
if RUN_MODEL_PROVENANCE or RUN_DIAGNOSTIC_RETRY or RUN_GATE10:
    from importlib.metadata import version
    # Preserve the runtime's Torch/CUDA stack and already imported numerical packages.
    protected = ['torch', 'torchvision', 'torchaudio', 'numpy', 'pandas', 'pillow']
    constraints = EXP_ROOT / 'logs' / 'runtime_constraints.txt'
    constraints.write_text(''.join(name + '==' + version(name) + '\n' for name in protected), encoding='utf-8')
    install_cmd = [
        sys.executable, '-m', 'pip', 'install', '--upgrade-strategy', 'only-if-needed',
        '--constraint', str(constraints),
        'transformers==4.46.3', 'tokenizers==0.20.3', 'huggingface_hub>=0.26,<1',
        'accelerate>=1.0', 'bitsandbytes>=0.45.5', 'sentencepiece>=0.2.1', 'protobuf',
    ]
    subprocess.run(install_cmd, check=True)
    raise RuntimeError('Install complete. Restart session NOW; rerun Cells 1-5 and 7. Skip Cell 6 after restart.')
else:
    print('Skipping dependency install because all model/gate flags are False.')
"""
    ),
    code(
        """
# Cell 7 - Phase C2/C3: pin model revision, load NF4/T4 model and record template/generation provenance.
if not (RUN_MODEL_PROVENANCE or RUN_DIAGNOSTIC_RETRY or RUN_GATE10):
    print('Skipping model load. Set RUN_MODEL_PROVENANCE=True, RUN_DIAGNOSTIC_RETRY=True, or RUN_GATE10=True in Cell 2.')
else:
    import torch
    import transformers
    import inspect
    from transformers import AutoModel, BitsAndBytesConfig
    from huggingface_hub import HfApi
    from unittest.mock import patch

    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Colab GPU before loading CXR-LLaVA.')

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    model_info = HfApi().model_info(MODEL_ID)
    model_revision = model_info.sha
    write_json(EXP_ROOT / 'logs' / 'model_revision.json', {'model_id': MODEL_ID, 'revision': model_revision})

    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    original_from_pretrained = transformers.AutoTokenizer.from_pretrained
    def tokenizer_compat(*args, **kwargs):
        kwargs.pop('token', None)
        kwargs.pop('use_auth_token', None)
        kwargs.setdefault('use_fast', False)
        return original_from_pretrained(*args, **kwargs)

    with patch.object(transformers.AutoTokenizer, 'from_pretrained', tokenizer_compat):
        model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=model_revision,
            trust_remote_code=True,
            quantization_config=q_config,
            device_map={'': 0},
            torch_dtype=torch.float16,
        )
    model.eval()

    provenance = {
        'model_id': MODEL_ID,
        'revision': model_revision,
        'load_finished_utc': utc_now(),
        'quantization': '4bit NF4 double quant, FP16 compute',
        'device_map': {'': 0},
        'model_class': type(model).__name__,
        'module': type(model).__module__,
        'has_write_radiologic_report': hasattr(model, 'write_radiologic_report'),
        'has_generate_cxr_repsonse': hasattr(model, 'generate_cxr_repsonse'),
    }
    for attr in ['write_radiologic_report', 'generate_cxr_repsonse', 'apply_chat_template']:
        obj = getattr(model, attr, None)
        if obj is not None:
            try:
                source = inspect.getsource(obj)
                source_path = EXP_ROOT / 'diagnostics' / f'{attr}_source.py'
                source_path.write_text(source, encoding='utf-8')
                provenance[f'{attr}_source_sha256'] = sha256_file(source_path)
            except Exception as exc:
                provenance[f'{attr}_source_error'] = f'{type(exc).__name__}: {exc}'

    synthetic_chat = [
        {'role': 'system', 'content': 'You are a helpful radiologist.'},
        {'role': 'user', 'content': '<image>\\nWrite a radiologic report on the given chest radiograph.'},
    ]
    try:
        rendered = model.tokenizer.apply_chat_template(synthetic_chat, tokenize=False)
    except Exception as exc:
        rendered = f'APPLY_CHAT_TEMPLATE_ERROR: {type(exc).__name__}: {exc}'
    (EXP_ROOT / 'diagnostics' / 'synthetic_prompt.txt').write_text(str(rendered), encoding='utf-8')
    provenance['synthetic_prompt_sha256'] = sha256_file(EXP_ROOT / 'diagnostics' / 'synthetic_prompt.txt')
    write_json(EXP_ROOT / 'logs' / 'model_template_generation_provenance.json', provenance)
    print(json.dumps(provenance, indent=2, ensure_ascii=False))
"""
    ),
    code(
        """
# Cell 8 - Phase D: diagnostic retry for control + two historical failed images.
if not RUN_DIAGNOSTIC_RETRY:
    print('Skipping diagnostic retry. Set RUN_DIAGNOSTIC_RETRY=True after C2/C3 passes.')
else:
    import torch
    if 'model' not in globals():
        raise RuntimeError('Run Cell 7 first so model is loaded and provenance is recorded.')

    attempts_path = EXP_ROOT / 'attempts' / 'attempts.jsonl'
    subset100 = pd.read_csv(paths['historical_root'] / 'subsets' / 'indiana_f1_subset_100_seed42.csv')
    id_to_row = {str(row['image_id']): row for _, row in subset100.iterrows()}

    def run_one_attempt(image_id, attempt_index, role):
        row = id_to_row[image_id]
        image_path = paths['nlmcxr_image_dir'] / Path(str(row['image_path'])).name
        attempt = {
            'attempt_uuid': uuid.uuid4().hex,
            'run_id': RUN_ID,
            'role': role,
            'image_id': image_id,
            'attempt_index': attempt_index,
            'start_utc': utc_now(),
            'seed': SEED,
            'image_path': str(image_path),
            'image_sha256': sha256_file(image_path),
            'ground_truth_report': row.get('ground_truth_report', ''),
        }
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            with Image.open(image_path) as img:
                pil_image = img.convert('L')
            with torch.inference_mode():
                raw = model.write_radiologic_report(pil_image)
            text = '' if raw is None else str(raw)
            attempt.update({
                'raw_response_type': type(raw).__name__,
                'raw_text': text,
                'stripped_text': text.strip(),
                'status': 'success' if text.strip() else 'failed',
                'error_message': '' if text.strip() else 'Model returned an empty report.',
            })
        except Exception as exc:
            attempt.update({'status': 'failed', 'error_message': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        finally:
            attempt['elapsed_sec'] = time.time() - start
            attempt['peak_vram_gb'] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            attempt['end_utc'] = utc_now()
            append_jsonl(attempts_path, attempt)
        return attempt

    results = [run_one_attempt(CONTROL_IMAGE_ID, 1, 'control')]
    for image_id in FAILED_IMAGE_IDS:
        first = run_one_attempt(image_id, 1, 'diagnostic_failed_historical')
        results.append(first)
        if first['status'] != 'success':
            print('STOP before attempt 2 for', image_id, 'because attempt 1 failed and no single-change hypothesis has been selected.')

    pd.DataFrame(results).to_csv(EXP_ROOT / 'attempts' / 'diagnostic_retry_summary.csv', index=False)
    print(pd.DataFrame(results)[['role', 'image_id', 'attempt_index', 'status', 'error_message', 'elapsed_sec', 'peak_vram_gb']].to_string(index=False))
"""
    ),
    code(
        """
# Cell 9 - Phase F gate10: fixed configuration on the historical 10-image cohort only when enabled.
if not RUN_GATE10:
    print('Skipping gate10. Set RUN_GATE10=True only after C and D evidence are acceptable.')
else:
    import torch
    if 'model' not in globals():
        raise RuntimeError('Run Cell 7 first so model is loaded and config is fixed.')

    subset10 = pd.read_csv(paths['historical_root'] / 'subsets' / 'indiana_f1_subset_10_seed42.csv').copy()
    subset10['remapped_image_path'] = subset10['image_path'].map(lambda p: str(paths['nlmcxr_image_dir'] / Path(str(p)).name))
    prediction_path = EXP_ROOT / 'predictions' / 'fixed_config_gate10_predictions.csv'
    existing = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()
    completed = set(existing.loc[existing.get('status', pd.Series(dtype=str)).eq('success'), 'image_id'].astype(str)) if len(existing) else set()
    records = existing.to_dict('records') if len(existing) else []

    for _, row in subset10.iterrows():
        image_id = str(row['image_id'])
        if image_id in completed:
            continue
        image_path = Path(row['remapped_image_path'])
        record = {'run_id': RUN_ID, 'image_id': image_id, 'report_id': row.get('report_id', ''), 'image_path': str(image_path), 'ground_truth_report': row.get('ground_truth_report', ''), 'start_utc': utc_now()}
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            with Image.open(image_path) as img:
                pil_image = img.convert('L')
            with torch.inference_mode():
                raw = model.write_radiologic_report(pil_image)
            text = '' if raw is None else str(raw)
            record.update({'generated_report': text, 'status': 'success' if text.strip() else 'failed', 'error_message': '' if text.strip() else 'Model returned an empty report.'})
        except Exception as exc:
            record.update({'generated_report': '', 'status': 'failed', 'error_message': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        finally:
            record['latency_sec'] = time.time() - start
            record['peak_vram_gb'] = torch.cuda.max_memory_allocated() / (1024 ** 3)
            record['end_utc'] = utc_now()
            records.append(record)
            pd.DataFrame(records).to_csv(prediction_path, index=False)
            print(image_id, record['status'], record.get('error_message', ''))
        if record['status'] != 'success':
            raise RuntimeError(f'Gate10 stopped on failed output: {image_id}')

    gate = pd.DataFrame(records)
    checks = {
        'rows': len(gate),
        'unique_image_id': int(gate['image_id'].nunique()) if len(gate) else 0,
        'success': int((gate['status'] == 'success').sum()) if len(gate) else 0,
        'failed': int((gate['status'] != 'success').sum()) if len(gate) else 0,
        'all_nonempty': bool(gate['generated_report'].fillna('').astype(str).str.strip().astype(bool).all()) if len(gate) else False,
    }
    checks['status'] = 'PASS_TECHNICAL_GATE10' if checks['rows'] == 10 and checks['unique_image_id'] == 10 and checks['success'] == 10 and checks['all_nonempty'] else 'FAIL'
    write_json(EXP_ROOT / 'logs' / 'gate10.json', checks)
    print(json.dumps(checks, indent=2))
"""
    ),
    code(
        """
# Cell 10 - Handoff summary. Run after any stop/failure before closing Colab.
summary = {
    'run_id': RUN_ID,
    'exp_root': str(EXP_ROOT) if 'EXP_ROOT' in globals() else None,
    'flags': {
        'RUN_MODEL_PROVENANCE': RUN_MODEL_PROVENANCE,
        'RUN_DIAGNOSTIC_RETRY': RUN_DIAGNOSTIC_RETRY,
        'RUN_GATE10': RUN_GATE10,
        'RUN_STAGE50': RUN_STAGE50,
        'RUN_STAGE100': RUN_STAGE100,
        'RUN_FULL_DATASET': RUN_FULL_DATASET,
    },
    'environment': environment_snapshot(),
    'next_rule': 'If C1 passes, enable RUN_MODEL_PROVENANCE. Enable diagnostic retry only after model provenance is saved. Enable RUN_GATE10 only after C and D evidence are acceptable.',
}
if 'EXP_ROOT' in globals():
    write_json(EXP_ROOT / 'logs' / 'handoff_summary.json', summary)
print(json.dumps(summary, indent=2, ensure_ascii=False))
"""
    ),
]


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nb.cells = cells
    output = Path("03_CXR_LLaVA_IU_recovery_eval.ipynb")
    if output.exists():
        raise FileExistsError('Recovery notebook has manual fixes. Edit it directly; do not overwrite from this historical scaffold.')
    nbf.write(nb, output)
    print(output)


if __name__ == "__main__":
    main()

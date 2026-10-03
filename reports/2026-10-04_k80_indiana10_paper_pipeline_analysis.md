# Experiment Analysis Log

## 1. Experiment Information

- **Experiment ID:** indiana_k80_20261004_004350
- **Date:** 2026-10-04, Asia/Ho_Chi_Minh. Metrics chạy lúc 01:01:29 giờ Việt Nam (18:01:29 UTC ngày 03/10).
- **Researcher:** hanhpm / tài khoản thực thi student4.
- **Project:** CXR_LLaVA_Improvement
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement
- **Branch:** master (kiểm tra lúc viết báo cáo).
- **Commit:** `d49bda7f3010441715ad43b4cf4ad0ccc1ee6fa2` (HEAD hiện tại; checkout có thay đổi chưa commit, không đại diện đầy đủ code thực thi).
- **Notebook / Script:** `scripts/k80_original.py`, `scripts/prepare_local_indiana.py`, `scripts/label_cohort.py`, `scripts/exp02_reevaluate.py`.
- **Status:** Completed — metric-complete evaluation trên subset 10 ca; không phải benchmark claim hay tái lập chính xác bảng paper.
- **Nguồn bằng chứng:** [inference log](../result/indiana_k80_20261004_004350.log), [config inference](../result/indiana_k80_20261004_004350/run_config.json), [config metrics](../result/indiana_k80_20261004_004350_metrics/run_config.json).

## 2. Objective

Chạy local pipeline CXR-LLaVA theo hướng đánh giá báo cáo của paper: ảnh + báo cáo tham chiếu → sinh báo cáo → CheXpert chính thức cho cả hai phía → precision/recall/F1 theo bệnh lý và bootstrap. Kiểm tra khả năng sử dụng server Tesla K80 bắt buộc, không dùng NF4.

## 3. Survey / Technical Motivation

### References reviewed

- [CXR-LLAVA paper v3](https://arxiv.org/pdf/2310.18341v3): ViT-L/16, Llama-2 7B; MIMIC 3.000 ảnh/báo cáo, Indiana 3.689 cặp; CheXpert 518 ảnh là bài toán phân loại riêng.
- [Model phát hành](https://huggingface.co/ECOFRI/CXR-LLAVA-v2), revision `b2224786bb90d54b1e1291171866706cfbb44e2b`; model card tại `README.md`.
- [CheXpert labeler chính thức](https://github.com/stanfordmlgroup/chexpert-labeler).
- Implementation và hướng dẫn local: `plan/K80_Unquantized_Setup.md`, `plan/K80_Paper_Subset_Evaluation.md`.

### Key idea

Giữ checkpoint, preprocessing và prompt của code phát hành; thích nghi precision/device placement để chạy trên K80. Đánh giá bệnh lý từ báo cáo bằng official labeler thay vì suy luận chất lượng từ vài câu sinh ra.

### Hypothesis

PyTorch CUDA 10.2 + FP32 chia model qua bốn GPU sẽ giải quyết lỗi driver và thiếu hỗ trợ BF16/NF4, cho phép hoàn tất pipeline nhỏ. Không đặt giả thuyết đạt điểm benchmark paper trên 10 ca.

## 4. Environment

| Component | Version / Configuration |
|---|---|
| Platform | Local Linux, ictserver6 |
| OS | Linux-4.19.0-14-amd64-x86_64-with-glibc2.36 (config metrics) |
| GPU | 4 Tesla K80 được chọn từ 8 GPU; mỗi GPU khoảng 11,2 GiB |
| Driver | 440.33.01; nvidia-smi báo CUDA 10.2, từ kiểm tra trong phiên |
| Python | 3.10.21 cho inference/evaluation |
| CUDA runtime | 10.2 từ torch wheel cu102 |
| PyTorch | 1.12.1+cu102 (config inference) |
| Transformers | 4.36.2 (config inference) |
| tokenizers | 0.15.2 (kiểm tra hiện tại; không có snapshot đầy đủ lúc inference) |
| huggingface-hub | 0.36.0 (kiểm tra hiện tại; không có snapshot đầy đủ lúc inference) |
| accelerate | 0.25.0 (kiểm tra hiện tại; không có snapshot đầy đủ lúc inference) |
| protobuf | 5.29.6 (kiểm tra hiện tại; không có snapshot đầy đủ lúc inference) |
| pillow | 10.4.0 (kiểm tra hiện tại; không có snapshot đầy đủ lúc inference) |
| jinja2 | 3.1.6 (kiểm tra hiện tại; không có snapshot đầy đủ lúc inference) |
| BitsAndBytes | Không sử dụng |
| NumPy / Pandas metrics | 1.26.4 / 2.2.3 |
| Labeler | Python 3.6 executable, NLTK 3.3, NumPy 1.15.4, Pandas 0.23.4 |
| Model | ECOFRI/CXR-LLAVA-v2, FP32, không lượng tử hóa |

### Environment verification

```text
PyTorch 1.12.1+cu102: CUDA available=True; supported sm_37.
Tensor operations passed on four Tesla K80 GPUs, compute capability (3,7).
No broken requirements found.
Official CheXpert sample labels match reference: PASS.
Offline evaluator final_status: PASS_B_OFFLINE.
```

Thông tin tensor/driver lấy từ kiểm tra trước trong phiên. [Labeler pip freeze](../result/indiana_k80_20261004_004350/labeler_logs/pip_freeze.txt) lưu phiên bản thực tế của môi trường legacy.

## 5. Dependency Setup

### Installation command

```bash
conda create -n cxr-llava-k80 python=3.10 pip -y
conda activate cxr-llava-k80
python -m pip install torch==1.12.1+cu102 --index-url https://download.pytorch.org/whl/cu102
python -m pip install -r requirements-k80.txt
python -m pip check
CONDA_CHANNEL_PRIORITY=flexible conda env create -n chexpert-label -f "$TOOLS_ROOT/chexpert-labeler/environment.yml"
conda activate chexpert-label
export PYTHONPATH="$TOOLS_ROOT/NegBio"
python -m nltk.downloader universal_tagset punkt wordnet
python -c "from bllipparser import RerankingParser; RerankingParser.fetch_and_load('GENIA+PubMed')"
```

### Dependency issues

Môi trường ban đầu torch 2.5.1+cu118 không khởi tạo CUDA với driver 440.33.01. NF4 không hỗ trợ K80. Thiếu Jinja2 làm dựng prompt thất bại; đã bổ sung 3.1.6. Conda strict channel priority loại các package legacy và báo OpenSSL không giải được; dry-run với flexible thành công.

### Decision

Tách inference và labeler thành hai môi trường; giữ YAML official labeler và áp dụng flexible chỉ cho lệnh cài, không đổi Conda toàn cục. Việc gán nhãn đã thực sự hoàn tất, không còn chỉ là kiểm tra dry-run.

## 6. Code / Notebook Changes

### Files changed

- `scripts/k80_original.py`: FP32, phân bố 32 lớp Llama qua 4 GPU; batch cohort tuần tự, lưu predictions và CheXpert input.
- `scripts/prepare_local_indiana.py`: ghép XML/PNG, lấy FINDINGS + IMPRESSION, chọn seed 42 và ghi hashes.
- `scripts/label_cohort.py`: pin revision, official sample gate, chạy nhãn GT/generated và lưu provenance.
- `scripts/exp02_reevaluate.py`: chọn Indiana/MIMIC, summary macro metrics, giữ kiểm tra alignment/coverage/bootstrap.
- `requirements-k80.txt`, `.gitignore`, hướng dẫn K80 và `tests/test_k80_cohort.py`.

### Main change

Thay các hardcoded cast `torch.bfloat16` bằng `torch.float32` trong bản code compatibility riêng. Weights gốc/config gốc được giữ trong `model_cache/cxr-llava-v2-original`; bản thích nghi dùng symlink weights, không retrain và không quantize.

### Why this change is necessary

Đây là compatibility fix và thay đổi thực thi cho phần cứng K80. Giữ KV cache trên GPU của từng lớp để tránh lỗi khác device trong cached decoding; dùng synchronous generation để không vướng streamer timeout 15 giây. Không khẳng định precision/decoding tương đương chính xác experiment gốc của tác giả.

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| Input data | Local OpenI/Indiana: datasets/NLMCRX |
| Number of samples | 10, một ảnh cho mỗi báo cáo |
| Pool | 3.955 XML, 7.470 PNG; 3.826 cặp đủ điều kiện |
| Sampling | random.Random(42), không stratify; ảnh đầu tiên tồn tại theo thứ tự XML |
| Reference reports | FINDINGS + IMPRESSION nguyên bản |
| Image size | Preprocessing 512×512 grayscale, input PNG 8-bit RGB được giải mã thành công |
| Normalization | mean 0.5518136078431373; std 0.3821719215686275 |
| Prompt | Official report helper; nội dung lưu trong prompt_audit/report_prompt.txt |
| Precision | FP32 |
| Quantization | None |
| Device map | Llama lớp 0–7/8–15/16–23/24–31 → GPU 0/1/2/3; vision/projector/embed/head → GPU 0, final norm → GPU 3 |
| Max new tokens | Helper giới hạn tối đa 512, có tính giới hạn context |
| Batch size | 1, tuần tự 10 ca |
| Generation | temperature=0.2, top_p=0.8, sampling, KV cache enabled |
| Seed | 42 |
| Bootstrap | 1.000 lần, seed 42; mỗi bệnh lý trên các cặp eligible |
| Label policy | Cả GT/pred phải là 0/1; -1 và NaN bị loại, không chuyển thành âm tính |

Cohort SHA256: `96b177648569840ea2df604fecc1f90836fd0b9b500551f263d6cc6f6d43099f`. GT/view review chưa PASS theo JSON cohort; chưa xác minh toàn bộ là frontal. Prompt mặc định nêu sáu nhóm bệnh nhưng không yêu cầu rõ lung opacity/pneumonia như các tên riêng trong bảng Indiana. Defaults 0.2/0.8 lấy từ code phát hành, chưa được xác nhận là decoding dùng trong bảng paper.

## 8. Code Executed

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -u -m scripts.k80_original --dataset indiana --cohort data/local_indiana_10_seed42.csv --image-root /storage/student4/hanhpm/datasets/NLMCRX/NLMCXR_png --execute --output result/indiana_k80_20261004_004350
python -m scripts.label_cohort --source result/indiana_k80_20261004_004350 --labeler-dir model_cache/chexpert_tools/chexpert-labeler --negbio-dir model_cache/chexpert_tools/NegBio --labeler-python /home/student4/anaconda3/envs/chexpert-label/bin/python3.6
python -m scripts.exp02_reevaluate --dataset indiana --source result/indiana_k80_20261004_004350 --output result/indiana_k80_20261004_004350_metrics --sizes 10 --bootstrap-iterations 1000 --seed 42
```

Lệnh được tổng hợp từ cấu hình/output đã lưu và lệnh trong phiên; không phải shell history nguyên vẹn. Để chạy lại, dùng output directory mới vì code bảo vệ các run cũ.

## 9. Output

### Raw output

```text
Loading original unquantized weights as FP32 across 4 GPUs
Loading checkpoint shards: 100% 3/3 [02:24]
1/10 saved; research-only output
...
10/10 saved; research-only output
Inference complete: .../result/indiana_k80_20261004_004350.
CheXpert sample_gate: PASS
Evaluator alignment: pass; all five label/input/map row counts = 10
Evaluator final_status: PASS_B_OFFLINE
```

### Errors / warnings

```text
No chat template is defined for this tokenizer - using the default template for the LlamaTokenizer class.
```

Đây là cảnh báo fallback template trong Transformers 4.36.2; prompt audit cho thấy cấu trúc `<s>[INST] <<SYS>> ... <</SYS>> ... [/INST]`. [Prompt thực tế](../result/indiana_k80_20261004_004350/prompt_audit/report_prompt.txt), [labeler provenance](../result/indiana_k80_20261004_004350/labeler_logs/provenance.json), [alignment](../result/indiana_k80_20261004_004350_metrics/audit/indiana_f1_subset_10_alignment_checks.json).

## 10. Output Analysis

### What worked

Toàn bộ 10 ca có báo cáo không rỗng, không thất bại. Official sample gate khớp reference; hai CSV nhãn đủ 10 dòng. Evaluator xác nhận text/order/ID alignment. Khi viết báo cáo, đối chiếu lại cả bảy source hashes với `source_hashes.json` và sample labels với reference official: PASS.

### What failed

Môi trường cu118/NF4 ban đầu không chạy trên K80. Những lỗi dependency/KV cache đã được sửa trước cohort thành công. Chất lượng bệnh lý chưa tốt: 3 false positives cardiomegaly và 1 false positive edema trong các cặp eligible; không có eligible pairs cho consolidation/pneumonia.

### Root cause

Lỗi chạy là tương thích driver, dtype và multi-GPU cache placement. Kết quả metrics bị giới hạn bởi cohort nhỏ, ít ca dương tính và báo cáo không đề cập nhiều nhãn. Chưa có kiểm chứng đủ để quy nguyên nhân sai bệnh lý cho FP32, prompt hoặc dataset shift.

### Fix applied

CUDA 10.2, FP32 không lượng tử hóa, multi-GPU KV cache, Jinja2 và flexible Conda priority. Không thay nhãn missing/uncertain để làm tăng support.

### Result after fix

Pipeline inference → official labeling → metrics hoàn tất. Chưa có cơ sở tuyên bố cải thiện clinical accuracy.

## 11. Quantitative Results

| Metric | Baseline | Current Experiment | Difference |
|---|---|---|---|
| Completion | Một ảnh smoke test thành công | 10/10, 100% | Mở rộng pipeline, không phải tăng accuracy |
| Model shard loading | Không có đo tương đương | Khoảng 144 giây trong tqdm | Không xác định; chưa bao gồm mọi overhead |
| Mean inference latency | 27,59 giây trên IMG/img.jpg | 21.68 giây/ảnh | Khác ảnh, không dùng để kết luận speedup |
| Median latency | Không đo | 21.42 giây | N/A |
| Min–max latency | Không đo | 18.32–25.85 giây | N/A |
| Throughput | Không so sánh | 0.0461 ảnh/giây, khoảng 2.77 ảnh/phút | Tính từ latency, không tính model load/labeling |
| Peak allocated GPU memory | 8,87 GiB max-device | 8.87 GiB max-device | Không phải tổng VRAM 4 GPU hay nvidia-smi usage |
| Macro precision | Chưa đo | 0,3333, 3 bệnh lý có precision xác định | N/A |
| Macro recall | Chưa đo | 1,0000, chỉ Lung Opacity có recall xác định | Không diễn giải là sensitivity tổng thể 100% |
| Macro F1 | Chưa đo | 0,3333, chỉ 3/7 bệnh lý có F1 xác định | Không tương đương average 7 bệnh lý trong paper |

### Per-pathology metrics

| pathology | n_valid | coverage | tp | fp | tn | fn | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cardiomegaly | 3 | 0.3000 | 0 | 3 | 0 | 0 | 0.0000 | N/A | 0.0000 |
| Consolidation | 0 | 0.0000 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| Edema | 1 | 0.1000 | 0 | 1 | 0 | 0 | 0.0000 | N/A | 0.0000 |
| Lung Opacity | 2 | 0.2000 | 1 | 0 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Pleural Effusion | 7 | 0.7000 | 0 | 0 | 7 | 0 | N/A | N/A | N/A |
| Pneumonia | 0 | 0.0000 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| Pneumothorax | 5 | 0.5000 | 0 | 0 | 5 | 0 | N/A | N/A | N/A |

`coverage` = eligible pairs / 10. Tổng 18/70 cặp bệnh lý eligible (25,7%); không phải 18 ca độc lập. Pleural Effusion và Pneumothorax chỉ có TN, nên F1 undefined; không được gọi là F1=1. Recall cardiomegaly/edema undefined do không có GT positive eligible. Lung Opacity chỉ có 2 eligible pairs, gồm 1 positive.

### Bootstrap 95% intervals

| pathology | ci_low_95 | ci_high_95 | bootstrap_defined | bootstrap_undefined |
| --- | --- | --- | --- | --- |
| Cardiomegaly | 0.0000 | 0.0000 | 1000 | 0 |
| Consolidation | N/A | N/A | 0 | 1000 |
| Edema | 0.0000 | 0.0000 | 1000 | 0 |
| Lung Opacity | 1.0000 | 1.0000 | 746 | 254 |
| Pleural Effusion | N/A | N/A | 0 | 1000 |
| Pneumonia | N/A | N/A | 0 | 1000 |
| Pneumothorax | N/A | N/A | 0 | 1000 |

CI Lung Opacity [1,1] chỉ tính trên 746 resamples có F1 xác định; 254/1.000 resamples undefined bị loại. Khoảng hẹp này không chứng minh độ chính xác chắc chắn trên quần thể. Tất cả bệnh lý được evaluator đánh dấu low support.

## 12. Qualitative Results

### Input

CXR1429_IM-0275-1001, một ảnh trong cohort. Chỉ đối chiếu văn bản reference, chưa có radiologist xác nhận trực tiếp ảnh.

### Model output

Reference:

```text
Mediastinal contours are normal. Heart size is within normal limits. Lungs are clear. There is no pneumothorax or large pleural effusion. No acute cardiopulmonary abnormality.
```

Generated:

```text
The radiologic report reveals an enlarged cardiac silhouette. There is no evidence of acute pneumonia, vascular congestion, or pleural effusion.
```

### Observation

- Correct findings: văn bản hai phía cùng phủ nhận pleural effusion; không khẳng định đã xác minh lâm sàng.
- Missing findings: generated không nhắc pneumothorax trong ví dụ này.
- Incorrect findings: enlarged cardiac silhouette mâu thuẫn reference heart size within normal limits, phù hợp false positive cardiomegaly.
- Hallucinated findings: mô tả tim to là ứng viên cần review ảnh, không kết luận hallucination chắc chắn chỉ từ text.
- Language / formatting issues: báo cáo tiếng Anh không rỗng, đọc được; nội dung reference/pred khác độ bao phủ.

Ca CXR3478_IM-1690-1001: reference nêu scoliosis; generated không nhắc scoliosis và ghi no acute osseous abnormalities. Đây là khác biệt cần review, ngoài bảy bệnh lý đánh giá.

## 13. Comparison with Baseline

| Aspect | Previous | Current |
|---|---|---|
| Model | v2 official, một ảnh smoke test | Cùng revision v2, 10 cặp Indiana |
| Environment | FP32 K80/cu102 sau thích nghi | Cùng stack inference, thêm legacy labeler |
| Method | Sinh báo cáo đơn lẻ | Inference → CheXpert GT/pred → metrics/bootstrap |
| Memory | Khoảng 8,87 GiB max-device | Khoảng 8,87 GiB max-device |
| Latency | 27,59 giây/ảnh demo | Trung bình 21,68 giây/ảnh khác |
| Output quality | Chưa đo bệnh lý | Macro F1 0,3333 trên 3 bệnh lý defined |

### Interpretation

Cải thiện là hoàn thiện khả năng chạy và đo pipeline local. Chưa thể kết luận model tốt hơn hoặc nhanh hơn baseline do dữ liệu khác nhau và không có so sánh paired. Không dùng kết quả Colab cũ 10/50/100 để thay metrics của cohort K80 này; các run khác cohort/method.

## 14. Problems Encountered

### Problem 1 — Driver / GPU

**Error:** CUDA initialization: driver too old với cu118. **Cause:** driver hỗ trợ CUDA 10.2; K80 không hỗ trợ NF4/BF16 như runner ban đầu. **Solution:** torch 1.12.1+cu102, FP32 unquantized trên 4 GPU. **Status:** Workaround, đã chạy thành công.

### Problem 2 — Prompt dependency

**Error:** apply_chat_template requires jinja2. **Cause:** dependency thiếu. **Solution:** thêm Jinja2 3.1.6. **Status:** Fixed. Cảnh báo default template vẫn tồn tại nhưng định dạng đã audit.

### Problem 3 — Cache placement

**Error:** tensors on cuda:0 and cuda:1 khi concatenate KV cache. **Cause:** dispatch di chuyển cache không phù hợp với layer GPU. **Solution:** skip device placement cho past_key_values/past_key_value. **Status:** Fixed.

### Problem 4 — Legacy Conda

**Error:** LibMambaUnsatisfiableError, packages excluded by strict repo priority và OpenSSL. **Solution:** CONDA_CHANNEL_PRIORITY=flexible với YAML official, môi trường riêng. **Status:** Fixed cho pipeline này; official labeling/sample đều có output.

### Problem 5 — Evaluation support / views

**Cause:** chỉ 10 reports, chọn ảnh đầu tiên theo XML, view/GT chưa review và nhãn unmentioned nhiều. **Solution:** cần review pairing/views và tăng cohort có đủ support. **Status:** Unresolved; không chỉnh nhãn hay làm benchmark claim.

## 15. Conclusion

Pipeline CXR-LLaVA chạy thành công local trên bốn K80, không NF4, và hoàn tất gán nhãn/evaluation cho 10 cặp Indiana. Tỷ lệ hoàn tất là 100%, latency trung bình 21,68 giây/ảnh và peak allocated memory max-device khoảng 8,87 GiB. Macro F1 0,3333 chỉ tổng hợp ba bệnh lý có F1 xác định; coverage và positive support rất thấp. FP32, multi-GPU và cohort/view chưa review khiến kết quả không phải exact paper reproduction. Có thể tiếp tục dùng pipeline để mở rộng đánh giá sau khi kiểm tra data/views và support.

## 16. Next Experiment

- [ ] Review ảnh frontal/lateral và cặp report/image; ghi bằng chứng review thực tế.
- [ ] Phân tích 3 FP cardiomegaly và 1 FP edema, cùng các omission, đối chiếu ảnh/reference.
- [ ] Đóng băng cohort lớn hơn với support dương/âm đủ cho từng bệnh lý; nếu stratify, báo riêng prevalence và không so trực tiếp điểm population của paper.
- [ ] Lưu full environment freeze, git status/commit và dtype/device config lúc chạy.
- [ ] Đánh giá thêm MIMIC sáu bệnh lý khi có ảnh/báo cáo được phép dùng; không thay thế bằng CheXpert binary classification.

## 17. Reproducibility Checklist

- [x] Repository URL recorded
- [x] Branch recorded (hiện tại)
- [x] Git commit recorded (hiện tại; run chưa snapshot commit/dirty diff đầy đủ)
- [x] Notebook/script recorded; runner hash trong config inference
- [x] Environment versions recorded (core inference/metrics + legacy freeze; một số phiên bản inference kiểm tra hiện tại)
- [x] Dataset/input version recorded qua cohort/XML/image hashes
- [x] Random seed recorded
- [x] Exact effective dtype/device/generation configuration recorded
- [x] Raw output saved
- [x] Metrics saved
- [x] Errors and fixes documented
- [x] Final conclusion recorded
- [ ] Full executed checkout và inference pip freeze đã lưu tại thời điểm chạy
- [ ] GT/view review đã hoàn tất

Official labeler revision: `44ddeb363149aa657296237f18b5472a73c1756f`; NegBio revision: `073199e2792824740e89844a59c13d3d40ce4d23`. Source hashes và official sample đã được kiểm tra lại lúc viết báo cáo. [Metrics CSV](../result/indiana_k80_20261004_004350_metrics/metrics/indiana_f1_subset_10_pathology_metrics_v2.csv), [bootstrap CSV](../result/indiana_k80_20261004_004350_metrics/metrics/indiana_f1_subset_10_bootstrap_ci_v2.csv), [summary](../result/indiana_k80_20261004_004350_metrics/metrics/indiana_f1_subset_10_summary.json).

## 18. Short Lab Update

> **Objective:** kiểm tra pipeline đánh giá báo cáo CXR-LLaVA trên server K80 bắt buộc. **Survey:** paper v3 dùng ViT-L/16 + Llama-2 7B và CheXpert labeler trên GT/generated reports. **Implementation:** FP32 unquantized/cu102, chia model qua 4 GPU, xuất cohort và official labeling. **Experiment:** 10 cặp Indiana, seed 42, 1.000 bootstrap iterations. **Result:** inference 10/10, sample/alignment PASS; latency 21,68 giây/ảnh, macro F1 0,3333 trên 3/7 bệnh lý defined. **Issue:** support/coverage thấp và view chưa review, chưa thể so benchmark paper. **Next step:** review views/errors, đóng băng cohort lớn hơn và lưu snapshot môi trường/code đầy đủ.

### Cập nhật kết quả 50 ảnh — 2026-10-04

Đã hoàn tất run 50 ảnh: 50/50, sample/alignment PASS; macro F1 0,3949 trên 5/7 bệnh lý defined, latency trung bình 21,99 giây/ảnh. Cohort chứa đủ 10 IDs trước; thay đổi F1 chưa thể diễn giải là cải thiện model vì support/tập bệnh lý defined khác nhau. Phân tích đầy đủ theo 18 mục: [báo cáo 50 ảnh](2026-10-04_k80_indiana50_paper_pipeline_analysis.md). Kết quả 10 ảnh phía trên được giữ nguyên.

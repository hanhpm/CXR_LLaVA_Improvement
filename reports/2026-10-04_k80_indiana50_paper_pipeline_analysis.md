# Experiment Analysis Log

## 1. Experiment Information

- **Experiment ID:** indiana_k80_50_20261004_013132_1238264
- **Date:** 2026-10-04, Asia/Ho_Chi_Minh; 01:31:32–01:53:38 (~22 phút 06 giây toàn pipeline).
- **Researcher:** hanhpm / student4.
- **Project:** CXR_LLaVA_Improvement.
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement
- **Branch:** master (từ phiên trước; không snapshot branch riêng trong run).
- **Commit:** `d49bda7f3010441715ad43b4cf4ad0ccc1ee6fa2`; checkout chưa sạch, có git status/diff và script snapshot đi kèm.
- **Notebook / Script:** scripts/run_indiana50_k80.sh và các Python scripts được snapshot.
- **Status:** Completed — metric-complete subset evaluation; không phải original paper benchmark.

[Run results](../result/indiana_k80_50_20261004_013132_1238264/RESULTS.md), [pipeline log](../result/indiana_k80_50_20261004_013132_1238264/pipeline.log), [metrics summary](../result/indiana_k80_50_20261004_013132_1238264/metrics/metrics/indiana_f1_subset_50_summary.json), [alignment](../result/indiana_k80_50_20261004_013132_1238264/metrics/audit/indiana_f1_subset_50_alignment_checks.json).

---

## 2. Objective

Mở rộng đánh giá local từ 10 lên 50 cặp Indiana, chạy tự động không cần giữ terminal: tạo cohort → inference K80 → official CheXpert cho GT/generated → precision/recall/F1 → 1.000 bootstrap iterations → RESULTS.md.

---

## 3. Survey / Technical Motivation

### References reviewed

[Paper v3](https://arxiv.org/pdf/2310.18341v3), [model phát hành](https://huggingface.co/ECOFRI/CXR-LLAVA-v2), [official CheXpert](https://github.com/stanfordmlgroup/chexpert-labeler), hướng dẫn local và báo cáo run 10 ảnh.

### Key idea

Tái sử dụng checkpoint/prompt/preprocessing đã chạy được trên K80, đánh giá nhãn trên original reference reports. Không thay missing/uncertain bằng negative.

### Hypothesis

Pipeline có thể hoàn tất 50 ảnh với cùng stack và bộ nhớ; tăng N có thể tăng support nhưng không bảo đảm F1 cao hơn.

---

## 4. Environment

| Component | Version / Configuration |
| --- | --- |
| Platform | ictserver6; Linux 4.19.0-14-amd64 |
| GPU | 4 Tesla K80; CUDA_VISIBLE_DEVICES=0,1,2,3 |
| Driver | 440.33.01 (kiểm tra trước trong phiên) |
| Python | 3.10.21 |
| CUDA runtime | 10.2 |
| PyTorch | 1.12.1+cu102 |
| Transformers | 4.36.2 |
| tokenizers | 0.15.2 |
| huggingface-hub | 0.36.0 |
| accelerate | 0.25.0 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| pillow | 10.4.0 |
| protobuf | 5.29.6 |
| Jinja2 | 3.1.6 |
| BitsAndBytes | Không sử dụng |
| Labeler | Python 3.6; legacy pip freeze riêng |
| Model | ECOFRI/CXR-LLAVA-v2 |

### Environment verification

Preflight PASS cho GPU/checkpoint/revisions; legacy labeler imports PASS. Official sample gate và evaluator alignment PASS. Full inference freeze: [pip freeze](../result/indiana_k80_50_20261004_013132_1238264/inference_pip_freeze.txt); labeler freeze: [legacy freeze](../result/indiana_k80_50_20261004_013132_1238264/inference/labeler_logs/pip_freeze.txt).

---

## 5. Dependency Setup

### Installation command

```bash
conda activate cxr-llava-k80
python -m pip install torch==1.12.1+cu102 --index-url https://download.pytorch.org/whl/cu102
python -m pip install -r requirements-k80.txt
CONDA_CHANNEL_PRIORITY=flexible conda env create -n chexpert-label -f "$TOOLS_ROOT/chexpert-labeler/environment.yml"
```

Các môi trường đã được cài từ run trước; bash 50 ảnh dùng executable trực tiếp, không cài lại package. Labeler giữ revision official đã pin.

### Dependency issues

Các lỗi cu118/driver, NF4/BF16, thiếu Jinja2, KV cache khác GPU và strict Conda priority đã xử lý ở giai đoạn trước. Run 50 không có lỗi làm dừng pipeline.

### Decision

Giữ hai môi trường riêng, giữ checkpoint unquantized và không thay official labeler bằng phương pháp khác.

---

## 6. Code / Notebook Changes

### Files changed

`scripts/run_indiana50_k80.sh` gộp toàn quy trình; dùng lại `prepare_local_indiana.py`, `k80_original.py`, `label_cohort.py`, `exp02_reevaluate.py` và các dependency đã chuẩn bị.

### Main change

Bash hỗ trợ nohup/background, preflight, output directory mới cho mỗi run, status RUNNING/SUCCESS/FAILED theo stage, logs, script snapshot, pip freeze, git commit/status/diff và RESULTS.md.

### Why this change is necessary

Tự động hóa thực nghiệm dài, giảm thao tác thủ công và giữ bằng chứng khi tắt máy cá nhân. Không thay đổi weights hay thuật toán model giữa run 10 và 50.

---

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| Dataset | Local Indiana/OpenI, datasets/NLMCRX |
| Samples | 50 unique report/image pairs; chứa đủ 10 image IDs của run trước |
| Pool | 3.955 XML / 7.470 PNG / 3.826 eligible pairs |
| Sampling | random.Random(42); không stratify; ảnh đầu tiên tồn tại theo thứ tự XML |
| Reference | FINDINGS + IMPRESSION nguyên bản |
| Input/preprocessing | PNG 8-bit; 512×512 grayscale, normalization config gốc |
| Precision / Quantization | FP32 / None |
| Model revision | b2224786bb90d54b1e1291171866706cfbb44e2b |
| Device map | Llama 8 lớp/GPU 0–3; vision/projector/embed/head trên GPU 0, final norm GPU 3 |
| Prompt | Official report helper, audited Llama-2 SYS/INST |
| Batch size | 1, tuần tự |
| Generation | temperature 0.2, top_p 0.8, sampling, KV cache |
| Max new tokens | Tối đa 512 theo helper/context |
| Seed / Bootstrap | 42 / 1.000 lần |
| Label policy | Joint-definite 0/1; exclude -1 và NaN |

Cohort SHA256: `ca1263f68fbd9e170a533770635b06c5dad8b1d8d6f7f27c1c7eaedd5f55c988`. GT/view review vẫn false; chưa xác minh frontal. Số ca/view và FP32 khác điều kiện benchmark gốc; không gọi đây là tái lập Table 4.

---

## 8. Code Executed

```bash
cd /storage/student4/hanhpm/CXR_LLaVA_Improvement
bash scripts/run_indiana50_k80.sh --background
```

PID khởi chạy người dùng ghi nhận: 1238270. Script tự đặt max-cases=50, tạo cohort, chạy official sample gate và labeling hai phía, evaluator sizes=50/bootstrap=1000/seed=42. Snapshot từng script nằm trong scripts_snapshot/ của run. Không ghép output 10 ảnh cũ; toàn bộ 50 ca được inference lại trong cùng một run.

---

## 9. Output

### Raw output

```text
50/50 saved; research-only output
Official CheXpert sample labels match reference: PASS
gt: labeling complete
generated: labeling complete
Completed: 50/50
Macro precision: 0.3364
Macro recall: 1.0000
Macro F1: 0.3949
Defined F1 pathologies: 5/7
Final state: SUCCESS
```

### Errors / warnings

Cảnh báo tokenizer không có chat template riêng vẫn xuất hiện; Transformers 4.36.2 dùng default Llama template. Actual prompt được lưu ở [prompt audit](../result/indiana_k80_50_20261004_013132_1238264/inference/prompt_audit/report_prompt.txt). Không có lỗi làm thất bại run.

---

## 10. Output Analysis

### What worked

50/50 báo cáo thành công, cả hai label files đủ 50 dòng, alignment text/order/IDs PASS, final evaluator PASS_B_OFFLINE. Kiểm tra lại bảy source hashes và official sample labels khi viết báo cáo: PASS.

### What failed

Không có execution failure. Có 12 FP trong joint-definite pairs: 9 cardiomegaly, 1 consolidation, 1 edema, 1 pleural effusion. Không có FN trong phần eligible nhưng có 13 dòng omission queue cần review; omission không tự tính thành FN trong chính sách hiện tại.

### Root cause

Support nhỏ và missing/uncertain làm coverage thấp. Không đủ bằng chứng để quy các lỗi nội dung cho dtype, prompt hay dataset shift.

### Fix applied / Result after fix

Không sửa model dựa trên test output. Các compatibility fix từ run 10 giữ nguyên; pipeline 50 hoàn tất.

---

## 11. Quantitative Results

| pathology | n_valid | coverage | n_gt_positive | tp | fp | tn | fn | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Cardiomegaly | 14 | 0.2800 | 2 | 2 | 9 | 3 | 0 | 0.1818 | 1.0000 | 0.3077 |
| Consolidation | 9 | 0.1800 | 0 | 0 | 1 | 8 | 0 | 0.0000 | N/A | 0.0000 |
| Edema | 2 | 0.0400 | 0 | 0 | 1 | 1 | 0 | 0.0000 | N/A | 0.0000 |
| Lung Opacity | 2 | 0.0400 | 1 | 1 | 0 | 1 | 0 | 1.0000 | 1.0000 | 1.0000 |
| Pleural Effusion | 30 | 0.6000 | 1 | 1 | 1 | 28 | 0 | 0.5000 | 1.0000 | 0.6667 |
| Pneumonia | 0 | 0.0000 | 0 | 0 | 0 | 0 | 0 | N/A | N/A | N/A |
| Pneumothorax | 20 | 0.4000 | 0 | 0 | 0 | 20 | 0 | N/A | N/A | N/A |

Macro precision 0,3364; macro recall 1,0000; macro F1 0,3949. F1 chỉ defined cho 5/7 bệnh lý. Recall chỉ defined cho cardiomegaly/lung opacity/pleural effusion, tổng cộng 4 cặp GT positive eligible; không phải sensitivity 100% trên tất cả bất thường.

Eligible pairs: 77/350 = 22.0%; các cặp bệnh lý không độc lập. Pneumonia không có cặp eligible; pneumothorax chỉ TN nên F1 undefined, không phải F1=1. Tất cả bệnh lý low_support=True.

| System metric | Value |
|---|---|
| Completion | 50/50, 100% |
| Mean / median latency | 21.99 / 21.59 giây/ảnh |
| Min–max latency | 15.63–28.31 giây |
| Throughput (generation only) | 2.73 ảnh/phút |
| Sum per-image latency | 18.32 phút |
| Whole pipeline wall time | ~22 phút 06 giây theo log |
| Peak allocated VRAM (max-device) | 8.87 GiB, không phải tổng VRAM 4 GPU |

### Bootstrap 95% intervals

| pathology | ci_low_95 | ci_high_95 | bootstrap_defined | bootstrap_undefined |
| --- | --- | --- | --- | --- |
| Cardiomegaly | 0.0000 | 0.6154 | 1000 | 0 |
| Consolidation | 0.0000 | 0.0000 | 622 | 378 |
| Edema | 0.0000 | 0.0000 | 746 | 254 |
| Lung Opacity | 1.0000 | 1.0000 | 746 | 254 |
| Pleural Effusion | 0.0000 | 1.0000 | 868 | 132 |
| Pneumonia | N/A | N/A | 0 | 1000 |
| Pneumothorax | N/A | N/A | 0 | 1000 |

CI tính conditional trên các resamples defined; các resamples undefined bị loại. Lung Opacity [1,1] chỉ dựa trên 2 eligible pairs/1 GT positive và 746 resamples defined, không chứng minh tính tổng quát. CI Pleural Effusion [0,1] thể hiện support rất thấp.

---

## 12. Qualitative Results

### Input

Image ID `CXR1429_IM-0275-1001`, ví dụ false positive `Cardiomegaly` từ error queue.

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

Đối chiếu text/nhãn cho thấy FP Cardiomegaly; chưa có radiologist xác minh trên ảnh. Không gọi generated report là chẩn đoán. Missing/uncertain cần đọc cùng omission queue thay vì coi mọi omitted finding là negative. Các bệnh ngoài bảy target chưa được chấm trong pipeline này.

---

## 13. Comparison with Baseline

| Aspect | Run 10 | Run 50 |
|---|---|---|
| Model/precision | v2, FP32, no quantization | Không đổi |
| Completion | 10/10 | 50/50 |
| Mean latency | 21,68 giây | 21,99 giây |
| Peak max-device VRAM | 8,87 GiB | 8,87 GiB |
| Macro precision | 0,3333 | 0,3364 |
| Macro recall | 1,0000; 1 pathology defined | 1,0000; 3 pathologies defined |
| Macro F1 | 0,3333; 3/7 defined | 0,3949; 5/7 defined |
| Coverage | 18/70 = 25,7% | 77/350 = 22,0% |

### Interpretation

Macro F1 tăng khoảng 0,0615 nhưng cohort và tập bệnh lý defined khác nhau; không phải model improvement hay paired accuracy gain. Cohort 50 có đủ 10 IDs trước, nhưng 40 ca thêm thay đổi support/prevalence. Latency tăng khoảng 0,30 giây/ảnh, chưa có kiểm định để gọi regression. Đánh giá ổn định hơn về execution, vẫn yếu về statistical support.

---

## 14. Problems Encountered

### Problem 1 — Coverage / rare positives

**Error:** Không có eligible pneumonia; nhiều bệnh có 0–2 GT positives. **Cause:** subset nhỏ và loại unmentioned/uncertain. **Solution:** review omissions, tăng cohort phù hợp; không đổi blank thành negative. **Status:** Unresolved.

### Problem 2 — Benchmark mismatch

**Cause:** 50 ca local, first XML image chưa xác minh frontal; paper dùng 3.689 ảnh frontal; chưa có official manifest/aggregation rõ ràng. **Solution:** lưu đúng classification subset evaluation và xác minh views/method trước benchmark. **Status:** Unresolved.

### Problem 3 — Background operation

**Solution:** nohup, logs/status/error trap. **Status:** Fixed, run SUCCESS sau khi người dùng khởi chạy. Không phải công cụ tự phục hồi khi reboot server.

---

## 15. Conclusion

Pipeline tự động đã hoàn tất 50/50 ảnh và official CheXpert/evaluation với đủ bằng chứng alignment và hashes. Mean latency là 21,99 giây/ảnh, peak max-device allocated VRAM 8,87 GiB và macro F1 0,3949 trên 5 bệnh lý defined. Có 12 FP và 13 omission entries; coverage chỉ 22%, mọi bệnh lý có support thấp. So với run 10, quy mô tăng nhưng chưa chứng minh chất lượng model cải thiện. Tiếp tục review lỗi/views và thiết kế đánh giá lớn hơn trước khi so benchmark paper.

---

## 16. Next Experiment

- [ ] Review 9 FP cardiomegaly và các FP còn lại trên ảnh/reference.
- [ ] Review 13 omission entries, giữ tách missing/uncertain khỏi negative.
- [ ] Xác minh frontal/lateral và report/image pairing; ghi review evidence thực tế.
- [ ] Đóng băng cohort lớn hơn, báo support/coverage và mọi exclusions; nếu stratify phải ghi thay đổi prevalence.
- [ ] Xác minh manifest/decoding/aggregation gốc trước tuyên bố paper reproduction.

---

## 17. Reproducibility Checklist

- [x] Repository URL và executed scripts được ghi.
- [x] Git commit/status/diff và scripts snapshot được lưu trong run.
- [x] Inference/labeler package freeze đã lưu.
- [x] Cohort IDs/text/source hashes, seed và config lưu.
- [x] Raw output, sample gate, row/text alignment, metrics và bootstrap lưu.
- [x] Bảy source hashes và official sample được đối chiếu lại lúc báo cáo.
- [x] Kết luận/limitations ghi rõ.
- [ ] GT/view review thực tế.
- [ ] Official paper split và exact inference/average-F1 method xác minh.

Snapshot không tự chứng minh checkout đầy đủ đã được commit; xem git_status.txt và scripts_snapshot/ để tái kiểm tra.

---

## 18. Short Lab Update

> **Objective:** mở rộng pipeline K80 từ 10 lên 50 ảnh Indiana. **Survey:** official v2 và CheXpert report-label evaluation theo paper. **Implementation:** bash chạy nền toàn quy trình, lưu logs/status/snapshots. **Experiment:** 50 cặp, FP32 trên 4 K80, seed 42, 1.000 bootstrap. **Result:** 50/50, official sample/alignment PASS; mean 21,99 giây/ảnh, macro F1 0,3949 trên 5/7 bệnh lý defined. **Issue:** 12 FP, 13 omission entries, coverage 22%, view chưa review. **Next step:** review errors/views và tăng support; chưa tuyên bố benchmark reproduction.

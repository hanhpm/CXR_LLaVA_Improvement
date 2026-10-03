# Experiment Analysis Log

## 1. Experiment Information

- **Experiment ID:** CXR-LLAVA-EXP02-IU-EXTERNAL-EVAL — phân tích output 10/50/100.
- **Date:** ngày lập báo cáo 2026-09-21. Log môi trường ghi `2026-09-18T10:31:02Z`, tương ứng 17:31:02 ngày 18/09/2026 tại Việt Nam. Đây là thời điểm ghi môi trường, không phải bằng chứng ngày bắt đầu/kết thúc riêng của cả ba lượt chạy.
- **Researcher:** chưa ghi tên người thực hiện trong artifacts.
- **Project:** CXR_LLaVA_Improvement.
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement.
- **Branch / Commit trong log:** `master` / `d49bda7f3010441715ad43b4cf4ad0ccc1ee6fa2`. HEAD khi phân tích trùng commit này, nhưng notebook 02 có thay đổi chưa commit; không thể coi HEAD là snapshot chính xác của mã đã chạy.
- **Notebook / Script:** [02_CXR_LLaVA_IU_external_eval.ipynb](../02_CXR_LLaVA_IU_external_eval.ipynb), đối chiếu notebook [00](../00_CXR_LLaVA_smoke_test.ipynb) và [01](../01_CXR_LLaVA_baseline_eval.ipynb).
- **Status:** hoàn thành phân tích artifacts; inference 10/10 và 50/50 thành công, 98/100 thành công ở bộ 100. EXP-02 còn vấn đề cần giải quyết trước khi mở rộng.
- **Phân loại:** smoke test và đánh giá mô tả trên subset, chưa phải metric-complete evaluation hay benchmark reproduction.

Nguồn chính là thư mục [result/CXR_LLaVA_EXP02_Indiana_External_Eval](../result/CXR_LLaVA_EXP02_Indiana_External_Eval/), ký hiệu **R/** bên dưới. Báo cáo này phân tích output đã lưu và chạy kiểm tra CSV nhẹ trên máy local; không chạy lại model, training hoặc CheXpert. Giữ nguyên notebook, dữ liệu nguồn và báo cáo cũ.

## 2. Objective

Đánh giá khả năng mở rộng pipeline sinh báo cáo CXR-LLaVA trên Indiana/OpenI từ 10 lên 50 và 100 ảnh: hoàn thành inference, latency, VRAM, tính nhất quán image–report–label và chất lượng nhãn bệnh lý. Đồng thời xác định các khó khăn thực tế, phân biệt biện pháp đã có bằng chứng thành công với phương án mới đề xuất.

## 3. Survey / Technical Motivation

### References reviewed

- Quy định local: [AGENTS.md](../AGENTS.md), [CRITERIA.md](../CRITERIA.md), [README.md](../README.md).
- Cấu trúc báo cáo: [experiment_analysis_template.md](../experiment_analysis_template.md).
- Thiết kế EXP-02: [roadmap](../plan/CXR_LLaVA_EXP02_Indiana_External_Eval_Roadmap.md).
- Implementation: notebook 00, 01, 02 và `CXR_LLAVA_HF/CXR_LLAVA_HF.py` trong repository.
- Bằng chứng thực thi: CSV và log trong R/. Giá trị paper trong `*_vs_paper_reference.csv` chỉ là tham chiếu được lưu trong dự án; báo cáo này không kiểm chứng lại bài báo bên ngoài.

### Key idea

Ghép PNG với báo cáo XML, lấy `figure_id == "F1"`, tạo subset seed 42, sinh báo cáo bằng NF4/FP16 trên T4, rồi dùng official CheXpert Labeler cho cả văn bản tham chiếu và văn bản sinh. Bảy bệnh lý được đánh giá chỉ trên các cặp mà cả hai nhãn đều là 0 hoặc 1; nhãn không đề cập và không chắc chắn bị loại.

### Hypothesis

Pipeline có thể mở rộng đến 100 ảnh với bộ nhớ ổn định. Tuy nhiên, số ảnh tăng không đồng nghĩa mô hình được cải thiện; độ tin cậy của metric phụ thuộc số cặp nhãn hợp lệ, số ca dương tính và tính nhất quán evaluator.

## 4. Environment

| Component | Version / Configuration | Mức bằng chứng |
|---|---|---|
| Platform | Google Colab, lưu artifacts trên Google Drive | Notebook và đường dẫn `/content/drive/` trong log |
| OS | Môi trường Linux của Colab; chưa lưu phiên bản OS | Suy ra từ lệnh Linux; thiếu OS fingerprint |
| GPU | Tesla T4 | `logs/experiment_environment.txt` |
| Python | 3.13.15 | Log môi trường |
| CUDA / PyTorch | 12.8 / 2.11.0+cu128 | Log môi trường |
| Transformers | 4.46.3 | Log môi trường |
| Tokenizers | 0.20.3 | Log môi trường |
| Hugging Face Hub | 0.36.0 | Log môi trường |
| Accelerate | 1.15.0 | Log môi trường |
| BitsAndBytes | 0.50.2 | Log môi trường |
| Protobuf | Cấu hình cài 5.29.6 | Không có version thực tế trong environment log |
| Model | ECOFRI/CXR-LLAVA-v2 | Revision ghi `main`, chưa pin resolved SHA |
| Precision | 4-bit NF4, FP16 compute | Log và notebook |
| CheXpert Labeler | `44ddeb363149aa657296237f18b5472a73c1756f` | `logs/chexpert_labeler_version.txt` |
| NegBio | `073199e2792824740e89844a59c13d3d40ce4d23` | Cùng file, theo thứ tự ghi ở notebook |

### Environment verification

```text
date_utc: 2026-09-18T10:31:02Z
cuda_available: true
gpu: Tesla T4
quantization: 4-bit NF4
compute_dtype: float16
png_count: 2541
xml_count: 3955
```

Chỉ có một environment log dùng chung, có thể bị ghi đè khi chạy cell môi trường. Không có timestamp theo ảnh hoặc theo subset trong predictions. Vì vậy ngày chạy cụ thể và cấu hình runtime độc lập của 10/50/100 đều chưa được chứng minh đầy đủ; không dùng thời gian sửa file local thay ngày chạy.

## 5. Dependency Setup

### Installation command

Lệnh hiện có trong notebook 02, chưa có pip transcript riêng cho từng subset:

```python
%pip install -q --upgrade \
    "transformers==4.46.3" \
    "tokenizers<0.21" \
    "huggingface-hub==0.36.0" \
    "protobuf==5.29.6" \
    sentencepiece accelerate pillow bitsandbytes
```

Official CheXpert chạy trong environment micromamba riêng. Notebook cài `default-jre`, tạo environment từ `chexpert-labeler/environment.yml`, tải NLTK `universal_tagset`, `punkt`, `wordnet` và parser `GENIA+PubMed`; đặt `PYTHONPATH` trỏ đến NegBio.

### Dependency issues

`logs/chexpert_setup_error.txt` giữ một `CalledProcessError(1, ...)` từ sample test. Log này không chứa traceback gốc đủ để xác định dependency nào gây lỗi. Ngược lại, `chexpert_sample_test_error.txt` dù có chữ “error” trong tên lại ghi **return code 0**, xử lý 4 sample thành công. Các log subset đều có hai return code 0, cho GT và generated.

### Decision

Có bằng chứng official labeler hoạt động ở các output cuối. Giữ log lỗi cũ để truy vết, nhưng không coi sự tồn tại của file có tên “error” là bằng chứng evaluator vẫn lỗi. Cảnh báo Java cgroup xuất hiện cả khi chạy thành công; chưa thấy nó chặn labeling. Không có bằng chứng sử dụng CheXbert thay thế.

## 6. Code / Notebook Changes

### Files changed

Lần phân tích này chỉ tạo báo cáo hiện tại. Notebook 02 đã ở trạng thái modified trước khi bắt đầu; không chỉnh sửa hoặc ghi đè.

### Main change

Các cơ chế đang có trong notebook: chọn dataset root, dựng manifest, giữ subset seed 42, kiểm tra ảnh đọc được, NF4 với `device_map={"": 0}`, bỏ đối số tokenizer `add_special_tokens`, lưu từng prediction lên Drive, retry output rỗng và tách môi trường CheXpert.

```python
generated_text = str(generated).strip()
if not generated_text:
    raise ValueError('Model returned an empty report.')
```

### Why this change is necessary

Đây là cơ chế tương thích và kiểm soát chất lượng pipeline, không phải thay đổi trọng số mô hình. Hai điểm chưa đồng nhất cần lưu ý:

- Cell load model đang hoạt động của notebook 02 fallback bằng cách nối `message['content']`; notebook 00/01 có template Llama 2 với `[INST]` và `<<SYS>>`. Đây là khác biệt mã hiện tại, chưa đủ bằng chứng khẳng định fallback nào đã được dùng cho từng output, hay nó là nguyên nhân báo cáo rỗng.
- Inference có khử trùng trước retry nhưng lại append bản ghi mới. Cell cleanup cần chạy sau retry; metric hệ thống hiện vẫn đếm toàn bộ dòng CSV. Artifacts 100 cho thấy cleanup cuối chưa được phản ánh trong file đã lưu.

## 7. Experiment Configuration

| Parameter | Value |
|---|---|
| Input data | Indiana/OpenI, NLMCXR PNG và XML; snapshot Drive trong artifacts |
| Number of samples | 10, 50, 100 ảnh yêu cầu; 10, 50, 98 báo cáo thành công |
| Selection | `figure_id == "F1"`, PNG có trong manifest, báo cáo tham chiếu không rỗng |
| Seed | 42 cho lấy mẫu; bootstrap seed 42 |
| Image | `.convert('L')`; README mô tả đầu vào 512×512; chưa lưu kích thước tensor runtime |
| Prompt | Gọi `model.write_radiologic_report(image)` |
| Precision / Quantization | FP16 compute / 4-bit NF4; double quant trong mã |
| Device map | `{"": 0}` trong notebook |
| Batch size | 1, vòng lặp theo ảnh |
| Generation | Không truyền temperature/top_p/max_new_tokens tại call site; remote-code revision chưa pin |
| Local helper reference | Mã local mặc định temperature 0.2, top_p 0.8, tối đa 512 token còn phụ thuộc context; không coi đây là runtime dump của remote model |
| Clinical evaluator | Official CheXpert Labeler; chỉ joint definite labels 0/1 |
| Bootstrap | 1.000 lần/pathology |
| Full dataset | `RUN_FULL_DATASET=False` |

Seed lấy mẫu không đủ đảm bảo tái lập token generation; cần lưu seed RNG của mô hình và cấu hình generation thực tế.

Manifest đã lưu có 7.470 tham chiếu ảnh, trong đó 2.541 `image_exists=True`, 4.929 `False`; 3.689 tham chiếu F1, 1.251 F1 có PNG và 1.238 F1 có PNG cùng báo cáo không rỗng. `manifest_issues.csv` có 4.969 dòng: 4.929 missing PNG và 40 missing report text; đây là số issue, không cộng thành số ảnh độc lập. Số này mô tả snapshot đã lưu, không khẳng định bộ Indiana gốc thiếu ảnh hay dataset local hiện tại thiếu ảnh.

Ba tập có quan hệ bao hàm xác nhận bằng `image_id`: 10 ⊂ 50 ⊂ 100. Chúng không phải ba thử nghiệm độc lập. File `indiana_subset_10_seed42.csv` không có tiền tố F1 là artifact khác; không dùng thay tập tương ứng với predictions F1.

## 8. Code Executed

Logic quan trọng trong notebook, dùng để giải thích artifacts:

```python
with torch.inference_mode():
    generated = model.write_radiologic_report(image)

valid = gt[pathology].isin([0, 1]) & pred[pathology].isin([0, 1])
```

Trong lần phân tích local, đọc CSV bằng thư viện chuẩn Python, so khớp thứ tự `image_id` qua row map, đối chiếu nguyên văn GT/generated giữa predictions, input CSV không header và cột `Reports` ở label CSV. Tính lại TP/FP/TN/FN từ nhãn cho 21 tổ hợp subset–pathology; mọi confusion count khớp file metric. Kiểm tra trạng thái cuối mỗi ảnh bằng quy tắc giữ dòng cuối, chỉ trong bộ nhớ, không sửa dữ liệu.

## 9. Output

### Raw output

| Artifact | 10 | 50 | 100 |
|---|---:|---:|---:|
| Dòng predictions | 10 | 50 | 102 |
| `image_id` độc lập | 10 | 50 | 100 |
| Dòng success, báo cáo không rỗng | 10 | 50 | 98 |
| Dòng failed/rỗng | 0 | 0 | 4 |
| Ảnh failed sau giữ dòng cuối | 0 | 0 | 2 |
| Dòng GT labels / generated labels / row map | 10 / 10 / 10 | 50 / 50 / 50 | 98 / 98 / 98 |
| Label-error records | 2 | 13 | 28 |
| Manual-review records | 2 | 13 | 20 |
| Review có ghi reviewer hoặc notes | 0 | 0 | 0 |

Nguồn: `R/predictions/`, `R/chexpert_inputs/`, `R/chexpert_labels/`, `R/error_analysis/`. Các input report CSV **không có header**; đọc chúng bằng mặc định header sẽ làm mất một mẫu.

### Errors / warnings

```text
Empty report retained for retry.
ValueError('Model returned an empty report.')
```

Hai ảnh còn lỗi: `CXR159_IM-0382-1001`, `CXR1706_IM-0466-1001`. Mỗi ảnh có hai dòng failed; retry đã diễn ra nhưng chưa khắc phục được output rỗng. Không có bằng chứng CUDA OOM trong các artifacts được kiểm tra.

## 10. Output Analysis

### What worked

- Hoàn thành 10 và 50 ảnh, mở rộng đến 98 báo cáo không rỗng ở bộ 100.
- Official CheXpert xử lý thành công các báo cáo có output; row alignment và văn bản đều khớp.
- Peak memory và latency không tăng mạnh khi số mẫu tăng; lưu Drive hỗ trợ tiếp tục sau gián đoạn.

### What failed

- Bộ 100 chưa hoàn tất 100/100, metric hệ thống trộn số dòng retry với số ảnh.
- Chưa có manual review hoàn thành. Các file review là scaffold, không phải kết quả bác sĩ xác nhận.
- Điểm F1 lưu NaN cả trong một số trường hợp F1 theo confusion count phải bằng 0; bootstrap lại gán 0 cho cả resample không xác định.
- Chỉ một environment log chung; notebook 02 hiện có 33 cell nhưng không có saved cell output hoặc execution count, nên không khôi phục được lịch thực thi chi tiết.

### Root cause

Tỷ lệ lỗi 100 bị sai đơn vị vì cell system metrics dùng `.mean()` trên 102 dòng predictions thay vì trên 100 ảnh duy nhất. F1 bị NaN ngoài ý muốn vì công thức đi qua precision/recall và chỉ tính khi `precision + recall > 0`; điều này loại cả ca TP=0 nhưng FP+FN>0. Riêng nguyên nhân model sinh báo cáo rỗng chưa được log token/prompt đủ để xác định.

### Fix applied / Result after fix

Code đã có guard báo cáo rỗng và retry; output cho thấy lỗi được ghi đúng nhưng hai ảnh vẫn thất bại. Sample CheXpert cuối trả code 0. Báo cáo này hiệu chỉnh cách diễn giải tỷ lệ lỗi thành 2/100 và trình bày F1 tính trực tiếp ở bảng audit riêng; không sửa metric gốc và không tuyên bố các lỗi code đã được sửa.

## 11. Quantitative Results

### Hiệu năng hệ thống

| Metric | Subset 10 | Subset 50 | Subset 100 |
|---|---:|---:|---:|
| Success theo ảnh | 10/10 | 50/50 | 98/100 |
| Failure theo ảnh, giữ dòng cuối | 0% | 0% | **2,00%** |
| Failure rate lưu trong CSV | 0% | 0% | 3,9216% = 4/102 dòng |
| Mean latency, giây/ảnh thành công | 6,0005 | 5,5350 | 5,7941 |
| Median latency, giây | 5,6340 | 5,3356 | 5,7354 |
| Min–max latency, giây | 4,1070–7,8529 | 2,5702–8,7605 | 2,4723–9,5733 |
| Peak VRAM lớn nhất, GiB | 5,4282 | 5,4397 | 5,4655 |
| Tổng latency ảnh thành công, giây | 60,0050 | 276,7492 | 567,8209 |
| Throughput suy ra = 1/mean latency, ảnh/giây | 0,1667 | 0,1807 | 0,1726 |

Nguồn: `R/metrics/*_system_metrics.csv`, đối chiếu predictions. Cột tên `peak_vram_gb` thực tế chia `1024**3`, nên báo cáo ghi GiB. Đây là peak allocated memory của PyTorch theo mã, không phải toàn bộ GPU memory qua nvidia-smi. Tổng latency và throughput không gồm model load, cài dependency, Drive I/O toàn pipeline, CheXpert hoặc thời gian các lần failed; không dùng làm end-to-end runtime. Không có model-load-time metric, BLEU/ROUGE hoặc FP16 đối chứng cho EXP-02.

### F1 theo file gốc và độ phủ nhãn

Mỗi ô ghi **F1 / n_valid / số GT dương tính trong n_valid**; `NA` là giá trị trống trong CSV gốc.

| Pathology | 10 | 50 | 100 |
|---|---|---|---|
| Cardiomegaly | NA / 2 / 1 | 0,5455 / 13 / 6 | 0,5833 / 28 / 10 |
| Consolidation | NA / 1 / 0 | NA / 4 / 0 | NA / 11 / 0 |
| Edema | NA / 1 / 0 | NA / 3 / 0 | NA / 3 / 0 |
| Lung Opacity | 1,0000 / 1 / 1 | 0,6667 / 4 / 2 | 0,7273 / 7 / 4 |
| Pleural Effusion | NA / 6 / 0 | NA / 27 / 0 | NA / 59 / 0 |
| Pneumonia | NA / 0 / 0 | NA / 0 / 0 | NA / 2 / 1 |
| Pneumothorax | NA / 5 / 0 | NA / 16 / 0 | NA / 36 / 0 |

Nguồn: `R/metrics/*_pathology_metrics.csv`. Ở bộ 100, Cardiomegaly chỉ có 28/98 cặp hợp lệ (28,57%), Lung Opacity chỉ 7/98 (7,14%). Không diễn giải số GT dương tính trong tập hợp lệ thành prevalence trên toàn bộ subset. Vì mask phụ thuộc cả nhãn generated, model không đề cập bệnh có thể làm mẫu bị loại; đây là giới hạn quan trọng khi đánh giá omission.

### Audit công thức F1

Theo `F1 = 2TP / (2TP + FP + FN)`, kết quả bằng 0 khi mẫu số dương nhưng TP=0. Các ô sau bị lưu NA bởi công thức hiện tại:

| Subset | Pathology | TP / FP / FN | F1 tính trực tiếp |
|---|---|---|---:|
| 10 | Cardiomegaly | 0 / 0 / 1 | 0 |
| 10 | Consolidation | 0 / 1 / 0 | 0 |
| 50 | Consolidation | 0 / 3 / 0 | 0 |
| 50 | Pleural Effusion | 0 / 3 / 0 | 0 |
| 100 | Consolidation | 0 / 4 / 0 | 0 |
| 100 | Pleural Effusion | 0 / 9 / 0 | 0 |
| 100 | Pneumonia | 0 / 1 / 1 | 0 |

Đây là phép kiểm tra bổ sung, không phải metric đã được pipeline sửa và xuất lại. Các trường hợp `2TP+FP+FN=0` vẫn không xác định nếu không áp dụng quy ước zero-division riêng. **Không báo cáo macro-F1 đầy đủ bảy bệnh**, kể cả sau sửa các ô trên, vì vẫn có bệnh không xác định và độ phủ quá thấp.

### Bootstrap và error records

| Pathology | 10: CI 95% lưu | 50: CI 95% lưu | 100: CI 95% lưu |
|---|---|---|---|
| Cardiomegaly | [0; 0], point NA | [0; 0,8571] | [0,2854; 0,7827] |
| Lung Opacity | [1; 1], n_valid=1 | [0; 1] | [0,2500; 0,9231] |

Nguồn: `R/metrics/*_bootstrap_ci.csv`. CI [1;1] của một mẫu không chứng minh tổng quát hóa tốt. Bootstrap hiện thay trường hợp không xác định bằng 0, khác logic point estimate; không diễn giải CI [0;0] đi cùng point NA như một kết quả hoàn chỉnh. Cần thống nhất định nghĩa trước khi dùng CI để so sánh.

Error records lần lượt là 1 FP + 1 FN, 10 FP + 3 FN, 24 FP + 4 FN; tương ứng 2, 11, 23 ảnh có ít nhất một mismatch nhãn hợp lệ. Bộ 100 có FP ở Cardiomegaly 7, Consolidation 4, Lung Opacity 3, Pleural Effusion 9, Pneumonia 1; FN ở Cardiomegaly 3 và Pneumonia 1. Đây là mismatch của text labeler, không phải tổng số lỗi lâm sàng, và không bao gồm các cặp bị mask loại.

## 12. Qualitative Results

### Input / Model output

Ví dụ từ predictions, không dùng để kết luận chẩn đoán ảnh:

| Subset / image_id | Báo cáo tham chiếu, trích | Model-generated output, trích | Nhận xét văn bản |
|---|---|---|---|
| 10 / CXR1188_IM-0127-1001 | “Both lungs are clear and expanded. Heart and mediastinum normal. No active disease.” | “No evidence of pneumonia.” | Output ngắn, thiếu nội dung tim/trung thất có trong GT |
| 50 / CXR1038_IM-0029-1001 | “Heart size normal. Lungs are clear.” | “The cardiac silhouette is borderline in size.” | Cách mô tả kích thước tim khác GT; cần review ảnh |
| 100 / CXR1019_IM-0015-1001 | “could be secondary to a small effusion versus scarring” | “a small right pleural effusion” | Model chuyển câu còn phân biệt effusion/scarring thành khẳng định effusion |

### Observation

- **Correct findings:** chỉ có thể nói một số mô tả trùng nội dung GT; chưa đánh giá ảnh để xác nhận đúng.
- **Missing findings:** ví dụ subset 10 bỏ nội dung tim và trung thất.
- **Incorrect / hallucinated findings:** ví dụ kích thước tim và mức chắc chắn về effusion là ứng viên cần review, chưa đủ bằng chứng gọi là hallucination đã xác nhận.
- **Language / formatting:** output tiếng Anh; một số quá ngắn so với mục tiêu báo cáo nhiều bệnh lý. Hai ảnh không có output.
- Không có reviewer/notes nào được điền trong manual-review CSV; cần review có hệ thống trước khi đưa kết luận chất lượng chi tiết.

## 13. Comparison with Baseline

| Aspect | Notebook 00 / 01 | Notebook 02 |
|---|---|---|
| Vai trò | Smoke load/inference; baseline đa dataset | External subset Indiana 10/50/100 |
| Bằng chứng notebook | 00 có 9 cell chứa output; 01 có 15 | 02 không giữ cell output; dựa vào CSV/log bên ngoài |
| Model/runtime | Cùng family; output 00/01 ghi T4 | Environment log ghi T4, NF4/FP16 |
| Chat-template fallback hiện tại | Legacy Llama 2 với `[INST]`, `<<SYS>>` | Cell load hoạt động nối content |
| Metric | 01 có saved output BLEU-4 0,001199, ROUGE-L 0,020943 | CheXpert pathology metrics trên Indiana |
| So sánh chất lượng | Khác dataset và tiêu chí | Không suy ra cải thiện từ chênh metric khác loại |

### Interpretation

Từ 50 lên 100, F1 lưu của Cardiomegaly tăng 0,5455 → 0,5833 và Lung Opacity tăng 0,6667 → 0,7273. Đây là thay đổi thành phần/số lượng mẫu hợp lệ trên các tập lồng nhau, không phải cải tiến mô hình. Latency 5,5–6,0 giây/ảnh và peak 5,43–5,47 GiB hỗ trợ tính khả thi của inference tuần tự, nhưng chưa phải đo tốc độ có kiểm soát prompt, độ dài output và phiên runtime.

Các file paper-reference ghi Cardiomegaly 0,62 và Lung Opacity 0,85; chênh số học ở subset 100 là −0,0367 và −0,1227. Không dùng chúng để kết luận kém paper hoặc suy giảm do NF4: còn khác cohort, checkpoint/revision, prompting, số mẫu, masking và công thức F1.

## 14. Problems Encountered

| Problem / bằng chứng | Cause hoặc mức chắc chắn | Solution | Status |
|---|---|---|---|
| Manifest chỉ thấy 2.541 PNG trong 7.470 tham chiếu | Snapshot đường dẫn Drive chưa phủ hết tham chiếu; chưa phân biệt được thiếu file thực sự hay sai layout chỉ từ artifacts | Kiểm kê root đang dùng và đối chiếu XML–PNG; nếu đổi cohort, dùng EXP_ROOT mới | Chưa giải quyết trong snapshot này |
| F1 bị hiểu nhầm thành frontal | F1 là figure reference; caption có thể mô tả nhiều view | Giữ tên F1 subset, xác minh view riêng nếu cần; không gọi confirmed frontal | Giới hạn phương pháp |
| Sample CheXpert từng exit 1 | Log gốc không đủ nguyên nhân | Environment riêng, PYTHONPATH NegBio, sample gate và stdout/stderr | Output cuối exit 0; nguyên nhân ban đầu chưa xác định |
| Hai ảnh trả report rỗng sau retry | Chưa đủ token/prompt diagnostics | Giữ failed; kiểm tra ảnh, prompt sau template, input/output token, EOS và generation config trước retry có kiểm soát | Unresolved |
| 102 dòng cho 100 ảnh, failure rate 4/102 | Append retry, system metrics không deduplicate | Giữ attempt log riêng; tính trạng thái cuối theo image_id trước metric | Đã xác định; code chưa sửa |
| NA F1 dù FP+FN>0 | Công thức qua precision/recall xử lý zero sai | Dùng confusion-count formula với quy ước denominator=0 rõ ràng; xuất lại trong phiên bản riêng | Đã xác định; code chưa sửa |
| Point F1 và CI khác zero policy | Bootstrap ép undefined thành 0 | Thống nhất policy và ghi số bootstrap hợp lệ | Chưa sửa |
| Template 02 khác 00/01 | Fallback nối content thay legacy Llama 2 | Kiểm tra actual tokenizer template, pin revision, log rendered prompt; kiểm tra formatter trước GPU run | Rủi ro tái lập, chưa chứng minh quan hệ nhân quả |
| Thiếu ngày/config từng stage; manual review trống | Log chung và scaffold chưa hoàn thành | Run ID + UTC start/end + môi trường từng stage; review theo ảnh với người đánh giá | Chưa hoàn tất |

Frontal manifest đã lưu chỉ có 21 ảnh và vẫn chứa caption “PA and LAT”; không coi tên file là ground truth về view. Không thay subset cũ bằng cohort khác rồi tiếp tục append vào cùng predictions.

## 15. Conclusion

Pipeline đã chứng minh khả năng inference và official CheXpert labeling trên 10, 50 và 98 ảnh thành công của stage 100 với Tesla T4/NF4. Latency trung bình khoảng 5,53–6,00 giây/ảnh thành công và peak allocated memory tối đa khoảng 5,47 GiB. Bộ 100 còn hai ảnh sinh báo cáo rỗng, trong khi retry làm metric lỗi hệ thống đếm nhầm đơn vị. F1 và CI hiện cần thống nhất lại cách xử lý zero/undefined, và độ phủ nhãn chưa đủ để đưa ra kết luận bảy bệnh hoặc benchmark. Nên tiếp tục sau khi sửa evaluator, hoàn thiện provenance và review mẫu; kết quả chỉ dành cho nghiên cứu.

## 16. Next Experiment

- [ ] Ưu tiên 1: thống nhất dedup/status, F1 và bootstrap policy; tái tính từ labels có sẵn trong output version mới, không cần tải model.
- [ ] Ưu tiên 2: kiểm tra hai ảnh failed cùng prompt/token diagnostics, thử lại có kiểm soát và lưu từng attempt.
- [ ] Ưu tiên 3: xác minh dataset root/inventory và template thực tế; pin model revision, lưu environment và timestamp riêng cho từng run.
- [ ] Review 23 ảnh có mismatch ở bộ 100 cùng một nhóm không mismatch; ghi findings bị bỏ sót kể cả nhãn bị mask.
- [ ] Chỉ mở rộng cohort sau khi gate 10 ảnh của cấu hình đã cố định đạt: ảnh/report khớp, output không rỗng, label alignment đúng, metric thống nhất. Nếu chọn lại cohort, bắt đầu thư mục thí nghiệm mới.

Trình tự tái lập sau khi xử lý các lỗi đã nêu, cell tính từ 1: cấu hình ở cell 5 → mount/paths ở cell 6–9 → dependencies cell 11 → environment/manifest/subset cell 12–16 → official CheXpert setup/sample cell 18–20 → model load cell 23 → inference cell 24 → kiểm tra/cleanup cell 26–28 sau retry → labeling cell 29 → pathology metrics cell 30 → bootstrap/system/error analysis cell 33. Giữ full dataset tắt và đi lần lượt 10 → 50 → 100; không dùng “Run all” hiện tại như một tái lập đã được xác nhận.

## 17. Reproducibility Checklist

- [x] Repository URL và branch có trong log.
- [x] Commit được ghi; đã ghi rõ giới hạn notebook modified.
- [x] Notebook và raw output xác định được.
- [x] Subset seed 42 và quan hệ bao hàm 10/50/100 được kiểm tra.
- [x] Predictions–row map–input reports–label Reports khớp từng dòng cho cả ba stage.
- [x] Tính lại 21 confusion matrices từ saved labels, khớp metric gốc.
- [x] Đếm unique image/status, empty reports và review records.
- [x] Lỗi, giải pháp và kết luận được ghi; giữ nguyên artifacts nguồn.
- [ ] Timestamp bắt đầu/kết thúc và môi trường riêng từng stage.
- [ ] Resolved model revision, actual chat template và generation RNG/config.
- [ ] Dataset fingerprint đầy đủ và image-readability audit hiện tại trên runtime nguồn.
- [ ] Hoàn thành 100/100, dedup metric và F1/CI thống nhất.
- [ ] Manual review và benchmark-comparable protocol.

## 18. Short Lab Update

> **Objective:** đánh giá CXR-LLaVA trên Indiana/OpenI theo stage 10–50–100. **Survey:** dùng thiết kế EXP-02 và official CheXpert Labeler trong dự án. **Implementation:** NF4/FP16 trên Colab T4, lưu predictions và labels trên Drive, hỗ trợ retry. **Experiment:** 10/10, 50/50 và 98/100 ảnh có báo cáo không rỗng. **Result:** trung bình 5,53–6,00 giây/ảnh, peak allocated memory tối đa 5,47 GiB; subset 100 có F1 lưu Cardiomegaly 0,5833 và Lung Opacity 0,7273 trên lần lượt 28 và 7 cặp nhãn hợp lệ. **Issue:** hai ảnh rỗng, dòng retry làm sai failure rate, F1/CI chưa nhất quán và chưa có manual review. **Next step:** sửa evaluator từ labels đã lưu, xác minh prompt/dataset provenance và chẩn đoán hai ảnh lỗi trước khi mở rộng.


## Ph? l?c b? sung 21/09 ? ?nh v? d? ??ng g?i

?? b? sung [b? 9 ?nh v? ch? th?ch](exp02_example_images_2026-09-21/index.html) ([Markdown](exp02_example_images_2026-09-21/examples.md)), g?m ba v? d? ? m?c 12, nh?m control v? finding b? mask/nh?n nghi v?n. C?c k?t qu? review m?i ???c ph?n bi?t v?i tr?ng th?i l?ch s? trong b?o c?o n?y; xem [audit h?p nh?t](2026-09-21_exp02_consolidated_audit.md).

| V? d? | ?nh | ? ngh?a |
|---|---|---|
| Subset 10 | ![CXR1188](exp02_example_images_2026-09-21/images/CXR1188_IM-0127-1001.png) | Output ng?n, b? nhi?u n?i dung GT |
| Subset 50 | ![CXR1038](exp02_example_images_2026-09-21/images/CXR1038_IM-0029-1001.png) | Normal vs borderline cardiac silhouette |
| Subset 100 | ![CXR1019](exp02_example_images_2026-09-21/images/CXR1019_IM-0015-1001.png) | Effusion/scarring ch?a ch?c ch?n th?nh kh?ng ??nh effusion |

[T?i ZIP](exp02_example_images_2026-09-21.zip). Gi?i n?n r?i m? `index.html` ?? ??c offline. ZIP c? PNG nguy?n g?c, CSV ch?a GT/output/ch? th?ch/hash v? hai b?o c?o tham chi?u. ??y l? v? d? minh h?a c? ch? ??ch, kh?ng ph?i clinical validation.

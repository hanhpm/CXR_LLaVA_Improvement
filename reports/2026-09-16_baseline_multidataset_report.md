# Báo cáo Phân tích Baseline Đa Dataset

## 1. Thông tin thí nghiệm

- **Mã thí nghiệm:** CXR-LLaVA-BASELINE-MULTIDATASET-2026-09-16
- **Ngày:** 2026-09-16
- **Dự án:** CXR_LLaVA_Improvement
- **Notebook:** `01_CXR_LLaVA_baseline_eval.ipynb`
- **Mô hình:** `ECOFRI/CXR-LLAVA-v2`
- **Thiết bị:** Google Colab GPU; output notebook ghi nhận đường dẫn Drive và GPU T4 ở smoke-test trước đó
- **Cấu hình:** 4-bit NF4 quantization, FP16 compute; không thay đổi architecture
- **Phân loại:** Baseline evaluation trên subset nhỏ; chưa phải benchmark đầy đủ

## 2. Mục tiêu

Đánh giá pipeline CXR-LLaVA gốc trên các task report, differential diagnosis và question answering với dữ liệu padChest và CXR8; đồng thời ghi nhận latency, VRAM và output để làm baseline cho các thí nghiệm sau.

## 3. Dữ liệu và phạm vi

| Dataset | Số output | Task | Ground truth |
|---|---:|---|---|
| padChest | 24 mẫu baseline report; 24 × 3 task trong multi-task output | Report, differential diagnosis, QA | Có `Report` |
| CXR8 | 25 × 3 task trong multi-task output | Report, differential diagnosis, QA | Không có radiology report; chỉ có label metadata |
| Brixia-score-COVID-19 | 25 × 3 task = 75 | Report, differential diagnosis, QA | Có Brixia score, không có report reference |

Tổng file `result/dataset_task_outputs.csv` có 147 dòng: 72 padChest và 75 CXR8; mỗi dataset có 25 hoặc 24 ảnh hợp lệ tương ứng, với ba task.

## 4. Các thay đổi notebook

- Dùng dataset từ Google Drive:

```text
/content/drive/MyDrive/ResearchLab/Notebook/datasets/
```

- Giữ pipeline load model đã xác nhận trên T4 với 4-bit NF4.
- Thêm task report, differential diagnosis và question answering.
- Thêm đo latency và peak GPU VRAM.
- Thêm cell clone Brixia repository và đọc annotation path.
- Thêm cell Brixia baseline riêng và lưu output cho 25 ảnh Cohen subset.

## 5. Kết quả padChest

### 5.1 Report metrics

Kết quả từ `result/padchest_datasample_baseline_metrics.csv` trên 24 mẫu:

| Metric | Mean |
|---|---:|
| BLEU-4 | 0.00121 |
| ROUGE-L | 0.01567 |
| BERTScore | Chưa tính |
| RadGraph F1 | Chưa tính |
| CheXbert F1 | Chưa tính |
| GREEN | Chưa tính |

Các điểm BLEU-4 và ROUGE-L rất thấp trong subset này. Đây là kết quả baseline cần phân tích, không được diễn giải trực tiếp thành kết luận về chất lượng lâm sàng.

### 5.2 System metrics

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| Latency/image (s) | 6.88 | 4.34 | 11.87 |
| Peak VRAM (GB) | 5.45 | 5.41 | 5.53 |
| Input tokens | 28.5 | - | - |
| Generation tokens | 67.5 | - | - |

## 6. Kết quả multi-task

`result/dataset_task_outputs.csv` ghi nhận đầy đủ ba task:

- `report`
- `differential_diagnosis`
- `question_answering`

Latency trung bình trong cell multi-task khoảng 15.94 giây cho mỗi ảnh khi chạy ba task tuần tự. Giá trị này không nên so sánh trực tiếp với latency report-only 6.88 giây vì phạm vi tính toán khác nhau.

### Giới hạn với CXR8

CXR8 chỉ cung cấp label/classification metadata, không có `ground_truth_report`. Vì vậy output report/differential/QA trên CXR8 chỉ là inference qualitative; chưa có metric report hợp lệ.

## 7. Brixia

Brixia repository đã được clone và annotation file được nhận diện:

```text
data/public-annotations.csv
data/public-cohen-subset/
```

Đã tạo `result/brixia_baseline_outputs.csv` với 75 dòng trên 25 ảnh, gồm ba task: report, differential diagnosis và question answering. Output giữ `S-Global` và `J-Global` cho từng ảnh.

| Metric | Mean | Min | Max |
|---|---:|---:|---:|
| S-Global | 13.68 | 8 | 18 |
| J-Global | 14.56 | 13 | 17 |
| Latency/image-task (s) | 16.25 | 14.34 | 18.31 |
| Peak VRAM (GB) | 5.44 | 5.42 | 5.50 |

Task counts: 25 report, 25 differential diagnosis và 25 question answering. Brixia score mới được lưu làm annotation để phân tầng severity; notebook chưa có bộ chuyển đổi đáng tin cậy từ văn bản tự do của CXR-LLaVA sang score 0–18, nên chưa báo cáo correlation hoặc accuracy. Không dùng BLEU/ROUGE vì Brixia không có report reference.

## 8. Phân tích lỗi

File `result/padchest_datasample_error_analysis.csv` đã được tạo nhưng các cột `error_categories`, `review_notes` và `reviewer` hiện chưa được annotate. Chưa có đủ bằng chứng để kết luận các failure mode như hallucination, missing finding, wrong location, severity error hoặc negation error.

## 9. Kết luận

Pipeline baseline CXR-LLaVA đã chạy được trên subset padChest, CXR8 và Brixia với ba task inference. padChest có report reference và cho BLEU-4 trung bình 0.00121, ROUGE-L trung bình 0.01567 trên 24 mẫu; các metric clinical nâng cao chưa được cấu hình. CXR8 chỉ hỗ trợ inference/label-oriented analysis trong setup hiện tại vì không có ground-truth report. Brixia đã có output trên 25 ảnh, nhưng kết quả hiện mới là severity-annotation-aligned inference, chưa phải score prediction benchmark.

## 10. Bước tiếp theo

- [x] Chạy cell Brixia với `RUN_BRIXIA_BASELINE = True` trên 25 ảnh.
- [x] Lưu output Brixia và đối chiếu metadata với `S-Global`/`J-Global`.
- [ ] Thiết kế evaluator severity riêng trước khi tính correlation/accuracy với Brixia score.
- [ ] Hoàn thiện BERTScore và các clinical evaluators có môi trường reproducible.
- [ ] Annotate thủ công error categories cho 24 report padChest.
- [ ] Chỉ mở rộng lên toàn bộ dataset sau khi subset pipeline đã được kiểm tra.

## 11. Artifact

- `result/dataset_task_outputs.csv`
- `result/padchest_datasample_baseline_predictions.csv`
- `result/padchest_datasample_baseline_metrics.csv`
- `result/padchest_datasample_error_analysis.csv`
- `01_CXR_LLaVA_baseline_eval.ipynb`

Đây là output nghiên cứu thử nghiệm của mô hình, không phải chẩn đoán, tư vấn điều trị hoặc xác nhận lâm sàng.

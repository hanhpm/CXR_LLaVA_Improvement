# Indiana: cohort phân tầng và đối chứng CNN

Ngày 21/09/2026. Yêu cầu hiện tại của người dùng thay thứ tự A–G cũ: audit prompt/revision trước retry; sửa evaluator offline; chốt cohort phân tầng trước inference mới. Quy định cũ “không thêm VLM” không cấm đối chứng CNN đã được người dùng yêu cầu. Không thay official CheXpert bằng CNN: CNN là mô hình đối chứng riêng.

## Trạng thái và điều kiện đầu vào

Chưa có cohort phân tầng đạt chuẩn. Nhãn GT đã lưu chỉ phủ 98 ảnh bộ 100; positive counts thô của bảy bệnh lần lượt 15, 1, 1, 15, 0, 2, 0. Một số positive label trái với câu phủ định trong GT. Không được nhân bản ca hiếm, đổi uncertain/blank thành positive hoặc lấy prediction để đạt quota.

Local có 7.470 PNG / 3.955 XML. Đây là inventory local ngày 21/09, khác snapshot Drive 2.541 PNG; không suy ra Drive hiện đã đầy đủ. Ảnh F1 có cả lateral: không gọi F1 là frontal.

## Cohort mới

1. Dựng lại manifest từ XML/PNG hiện có, giữ report ID, image ID, caption/view, report text, hashes. Một ảnh cho mỗi study/report; dùng patient grouping nếu có patient ID đáng tin cậy. Không suy patient ID từ tên file.
2. Cohort chính dùng ảnh frontal đã xác nhận. Mixed/unknown phải review view; lateral chỉ được đưa vào một phân tích phụ riêng. Chọn ảnh trước khi xem model output. GT report có thể mô tả cả hai view: ghi hạn chế này; không sửa GT để khớp ảnh.
3. Label GT của candidate pool bằng official CheXpert đã pin và qua sample gate. Lưu row map, headerless reports và Reports column; đối chiếu nguyên văn trước join. Review các positive và negation/uncertain bất thường trước dùng làm sampling strata; giữ raw label và adjudicated label riêng kèm người review/evidence.
4. Target **25 positive và 25 definite negative mỗi bệnh**, hard minimum 20 mỗi loại. Một study đa bệnh có thể đáp ứng nhiều quota. Tổng N do quota quyết định, không cố định 100. Đây là quota thiết kế, không phải bảo đảm power hay CI hẹp.
5. `python -m scripts.exp02_stratify --pool reviewed_gt_pool.csv --output NEW_SELECTION_DIR --target 25 --negatives 25 --seed 42`. Pool phải có `image_id,report_id,image_path,ground_truth_report` và bảy cột GT. Script chỉ chọn từ GT; nếu thiếu support, xuất bảng thiếu hụt và không tạo cohort.
6. Lưu CSV/hash cố định, distribution GT/view, excluded IDs và lý do. Tạo `review_gate.json` chứa `cohort_sha256`, `gt_review_pass`, `view_review_pass` và đường dẫn bằng chứng review. Chỉ đặt PASS khi review thật đã xong. Không có file gate mẫu tự đánh PASS.
7. Lấy gate10 bằng seed42 từ cohort đã đóng băng. Sau gate kỹ thuật/label alignment/review, chạy phần còn lại cùng cấu hình. Không chọn lại cohort hoặc threshold dựa trên điểm test.

Enrichment thay prevalence: F1/precision chỉ mô tả cohort phân tầng, không dùng làm prevalence/benchmark population. Báo từng bệnh, support và coverage; không so trực tiếp F1 với cohort lịch sử hoặc paper.

## Đối chứng đã thêm

`scripts/exp02_cnn_control.py`: TorchXRayVision DenseNet121, `densenet121-res224-chex` (CheXpert-trained checkpoint). Bảy target có trong metadata; map duy nhất `Pleural Effusion -> Effusion`. Tham chiếu [mã nguồn chính thức](https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/models.py) và [preprocessing chính thức](https://github.com/mlmed/torchxrayvision#getting-started).

- Giữ cùng cohort CSV/hash, ảnh gốc/hash, GT và exclusion policy với CXR-LLaVA. CNN dùng preprocessing của checkpoint: grayscale 8-bit, normalize 255, center crop, resize 224. Không ép CNN dùng preprocessing VLM.
- Dùng released operating thresholds theo từng bệnh; không tune trên test. Script tắt operating-point normalization trước khi áp threshold trên sigmoid scores, tránh threshold hai lần. Lưu thresholds, package versions và checkpoint SHA256.
- Chạy preflight trước bằng `python -m scripts.exp02_cnn_control --cohort COHORT.csv --review-gate REVIEW_GATE.json --image-root PNG_DIR --output NEW_CNN_DIR`; chỉ thêm `--execute` sau gate10 và khi scope/output đã chốt. Chạy trên CUDA, không tự fallback CPU. Chưa cài/chạy CNN trong lượt audit này.
- CNN đánh giá classification, không sinh report; không gán ROUGE/BLEU cho CNN.

## So sánh công bằng và masking

Giữ metric joint-definite lịch sử để audit, nhưng không so F1 CNN trên tất cả GT-definite với F1 VLM trên subset phụ thuộc prediction rồi gọi là paired improvement.

So sánh mới phải báo hai bảng:

1. **Common eligible pairs:** GT 0/1 và VLM 0/1; CNN và VLM dùng đúng cùng ID/pathology mask. Báo coverage, confusion counts, F1 và paired bootstrap theo study trên mask cố định. Đây là selective comparison, không đo hết omission.
2. **GT-definite coverage:** tất cả GT 0/1, giữ VLM uncertain/unmentioned/failed thành abstention riêng. Báo tỷ lệ abstention, positive omissions, negative unknowns; không âm thầm biến abstention thành negative. CNN F1 trên GT-definite được ghi rõ denominator khác bảng 1.

Point/bootstrap cùng `2TP/(2TP+FP+FN)`; mẫu số 0 là undefined, bootstrap bỏ undefined khỏi percentile và báo số lượng. Macro bảy bệnh chỉ có khi đủ bảy F1. Không gọi enriched cohort là benchmark reproduction hoặc clinical validation.

## Thứ tự tiếp tục

1. Runtime prompt audit trên checkpoint đã pin trong notebook 03; giữ log static hiện có làm đối chiếu, không giả định SHA hiện tại là revision lịch sử.
2. GT/view review và label đủ candidate pool; chốt quota/cohort.
3. Gate10 CXR-LLaVA, official labeler và alignment; rồi inference cohort cùng CNN.
4. Tạo bảng paired và abstention, review các ca mới; chỉ sau đó bàn cải tiến mô hình.

Hai output rỗng lịch sử chưa có bằng chứng nguyên nhân. Không retry chúng chỉ để làm đẹp completion rate; diagnostic retry là run riêng, sau actual prompt audit, không ghép vào kết quả cũ.

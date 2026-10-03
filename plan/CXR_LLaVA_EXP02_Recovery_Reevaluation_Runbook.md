# EXP-02: Runbook sửa evaluator, tái đánh giá và kiểm tra khả năng tái lập

Ngày soạn: 2026-09-21. Trạng thái: **kế hoạch thực thi, chưa chạy các bước trong tài liệu này**.

Tài liệu dành cho agent thực thi tuần tự, dựa trên [báo cáo output 10–50–100](../reports/2026-09-21_exp02_indiana_10_50_100_output_analysis.md). Đọc cùng [roadmap EXP-02](CXR_LLaVA_EXP02_Indiana_External_Eval_Roadmap.md), không thay thế quy định dự án.

## 1. Mục tiêu, phạm vi và thứ tự

Hoàn thành năm việc: sửa cách tính metric từ labels đã có; chẩn đoán hai ảnh sinh output rỗng; cố định dataset/model/template và log từng run; review lỗi cùng mẫu đối chứng; chỉ mở rộng sau gate 10 ảnh.

Thứ tự bắt buộc:

1. A — bảo toàn output cũ, kiểm kê bằng chứng.
2. B — sửa evaluator, kiểm thử và tái tính offline; không tải model.
3. C — xác minh dataset, revision, prompt và environment trên Colab.
4. D — retry có kiểm soát hai ảnh lỗi.
5. E — review văn bản và ảnh, gồm lỗi bị mask bỏ qua.
6. F — chạy gate mới 10 ảnh; chỉ sau PASS mới chạy 50 rồi 100.
7. G — lập báo cáo và bàn giao.

C phải xong trước D dù phần chẩn đoán hai ảnh là ưu tiên nghiên cứu thứ hai. Không thể chẩn đoán prompt đáng tin cậy khi chưa xác định model/template thực tế.

**Ngoài phạm vi:** training, full-dataset inference, đổi kiến trúc, tự thay CheXpert bằng CheXbert, kết luận lâm sàng hoặc benchmark. Việc tạo runbook không có nghĩa các thử nghiệm đã chạy.

## 2. Quy tắc bắt buộc cho agent

- Bắt đầu phản hồi bằng `okw`. Đọc bản hiện hành của [AGENTS.md](../AGENTS.md), [CRITERIA.md](../CRITERIA.md), [README.md](../README.md), roadmap và báo cáo nguồn trước khi làm.
- Làm việc tại root `CXR_LLaVA_Improvement`; không áp dụng lệnh, runbook hay dữ liệu PatchCore.
- Nếu usage window 5 giờ còn ≤6%, không bắt đầu lệnh/công việc mới; ghi handoff theo mục 12.
- Trước mỗi phase ghi: input, output directory, phạm vi mẫu, tiêu chí PASS/STOP. Sau mỗi phase ghi kết quả thực tế; không đánh dấu hoàn thành chỉ vì đã viết code.
- Không sửa/xóa raw output cũ, không reset notebook đang modified, không commit dữ liệu ảnh/báo cáo riêng tư hoặc token. Không chạy `Run all` notebook hiện tại.
- Agent được giao thực thi kế hoạch có thể tự làm các bước đã rõ trong phạm vi này. Chỉ hỏi khi thiếu dữ liệu/quyết định bắt buộc; không xin phép lại cho thao tác read-only hoặc sửa lỗi đã được giao.
- Nếu không có Colab GPU hoặc dataset, hoàn thành phần offline độc lập rồi bàn giao đúng bước bị chặn. Không chạy model lớn bằng CPU để thay thế T4.

## 3. Output layout và phân biệt các loại run

Input cũ, chỉ đọc:

```text
result/CXR_LLaVA_EXP02_Indiana_External_Eval/
```

Mỗi lần chạy mới dùng UTC run ID, ví dụ `20260921T090000Z_<8 ký tự UUID>`. Tạo thư mục với `exist_ok=False` để không ghi đè. Giá trị ví dụ không được hard-code cho lần thực thi thật.

```text
result/CXR_LLaVA_EXP02_Indiana_External_Eval_v2/<run_id>/
  run_config.json
  source_hashes.json
  audit/                    # alignment, dedup, excluded labels, kiểm tra
  canonical_predictions/    # mỗi image_id đúng một dòng
  metrics/                  # point estimates và bootstrap mới
  review/                   # danh sách, checklist, kết quả review
  logs/                     # commands, stdout/stderr, environment, phase status
```

Trên Colab tạo cấu trúc tương tự dưới `DATASETS_ROOT`, thêm `attempts/`, `diagnostics/`, `chexpert_inputs/`, `chexpert_labels/`, `subsets/`. Mọi đường dẫn qua biến cấu hình. Lưu `parent_run_id`, `run_kind`, source hashes và cohort hash để liên kết run.

| run_kind | Ý nghĩa | Có được dùng như historical output đã hoàn tất? |
|---|---|---|
| `offline_reevaluation` | Tái tính cùng predictions/labels cũ | Có, nhưng ghi rõ evaluator v2 |
| `diagnostic_retry` | Thử cấu hình kiểm soát cho 2 ảnh | Không tự ghép vào metric cũ |
| `fixed_config_gate10` | Chạy lại 10 ảnh với cấu hình cố định | Kết quả mới, không ghi đè stage cũ |
| `fixed_config_50` / `fixed_config_100` | Mở rộng cấu hình đã qua gate | Liên kết gate và cấu hình bằng hash |

Nếu thay prompt/template/revision/decoding, đó là cấu hình mới. Không ghép 98 báo cáo cũ với 2 báo cáo cấu hình mới rồi gọi là một evaluation đồng nhất.

## 4. Phase A — preflight và bảo toàn nguồn

### A1. Kiểm tra workspace

Chạy trong terminal PowerShell tại root dự án:

```powershell
Get-Location
git status --short
git rev-parse HEAD
Get-Content -Encoding UTF8 AGENTS.md
Get-Content -Encoding UTF8 CRITERIA.md
```

Ghi HEAD, branch, danh sách dirty files. Khi có notebook modified, ghi hash notebook và diff; không tự kết luận code tại HEAD là code tạo output. Tạo output root mới theo mục 3 trước khi ghi bất kỳ artifact nào.

### A2. Tạo script kiểm kê trước, rồi mới chạy

**Các script CLI dưới đây là deliverable cần agent tạo, chưa có sẵn chỉ vì được nêu trong runbook.** Dùng Python chuẩn, `numpy` và `pandas` khi có. Đặt script trong `scripts/`, tests trong `tests/`; kiểm tra file hiện có trước khi chọn tên để không ghi đè.

Script `scripts/exp02_reevaluate.py` nhận:

```text
--source PATH --output PATH --sizes 10 50 100
--bootstrap-iterations 1000 --seed 42
```

Script không import transformers/torch, không tải model, không gọi mạng. Nó ghi SHA256 từng source CSV/log/notebook được dùng, giữ ordinal của dòng CSV. SHA256 tính trên bytes gốc, không trên bản CSV đã parse rồi export.

### A3. Đối chiếu snapshot kỳ vọng

| Check | 10 | 50 | 100 |
|---|---:|---:|---:|
| Raw prediction rows | 10 | 50 | 102 |
| Unique image_id | 10 | 50 | 100 |
| Success không rỗng | 10 | 50 | 98 |
| Canonical failed | 0 | 0 | 2 |
| Label rows mỗi phía | 10 | 50 | 98 |

Hai ảnh lỗi kỳ vọng: `CXR159_IM-0382-1001`, `CXR1706_IM-0466-1001`.

**PASS:** đủ subset, predictions, row map, GT/generated input và label CSV cho từng size. **STOP:** thiếu file, ID ngoài subset, GT khác nhau giữa các bản ghi cùng ID, hoặc snapshot khác bảng mà chưa giải thích được. Không sửa dữ liệu cho khớp số kỳ vọng; ghi sai khác và xác minh phiên bản nguồn.

## 5. Phase B — dedup, F1, bootstrap và tái tính offline

### B1. Quy tắc canonical status

1. Đọc ID bằng string. Giữ `source_row_index` trước mọi sort/filter.
2. Với CSV lịch sử không timestamp, thứ tự vật lý trong file là thứ tự duy nhất có bằng chứng. Giữ dòng cuối mỗi `image_id`; ghi rõ giả định này trong audit.
3. `success` chỉ hợp lệ khi status là success **và** `generated_report.strip()` không rỗng. Dòng success rỗng chuyển thành failed trong canonical copy, có reason; không sửa raw.
4. Không đổi dòng failed thành success chỉ vì còn text. Nếu các attempt mâu thuẫn như success trước, failed sau, canonical vẫn theo dòng cuối và xuất conflict audit; không tự chọn attempt có metric đẹp hơn.
5. Left join canonical lên subset. ID chưa từng được thử nhận `not_attempted`. ID ngoài subset là STOP.
6. `n_requested = n_success + n_failed + n_not_attempted`. Failure rate = n_failed/n_requested, completion rate = n_success/n_requested; báo not_attempted riêng.
7. Latency/VRAM chỉ tính canonical success với giá trị hợp lệ. Không coi latency thiếu là 0. Ghi số mẫu có timing và số thiếu timing.

Với snapshot hiện tại phải thu được 98/100 success và failure rate 0,02. Bốn raw failed rows không chứng minh bốn lượt inference độc lập: hai dòng là trạng thái cleanup, nên gọi là “raw failed records”, không gọi attempt count đã xác nhận.

### B2. Alignment labels, không dựa vào số dòng đơn thuần

Với từng size:

1. Đọc `*_row_map.csv`, kiểm tra `chexpert_row` liên tục từ 0, image_id duy nhất và thuộc subset.
2. Input `*_gt_reports.csv` và `*_generated_reports.csv` **không header**: dùng `header=None` hoặc `csv.reader`.
3. GT/generated label CSV có header, cột `Reports` phải khớp nguyên văn input cùng dòng.
4. Join row map với canonical theo image_id; đối chiếu cả report_id, GT text và generated text. Không sort labels riêng khỏi row map.
5. Phải có đủ label cho mọi canonical success, và không có label tương ứng output khác canonical. Nếu không khớp, STOP việc tái sử dụng labels; xuất danh sách cần relabel bằng official CheXpert.
6. Không nối labels của hai cohort hoặc hai prompt config khác nhau.

Xuất `audit/alignment_checks.json`, `audit/dedup_conflicts.csv`, `canonical_predictions/*_predictions.csv`.

### B3. Chính sách metric v2

Giữ chính sách chính của dự án: mỗi pathology chỉ tính cặp có cả GT và prediction trong `{0,1}`. Blank/NaN = unmentioned, −1 = uncertain; không tự đổi thành negative.

```python
def f1_from_counts(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator > 0 else float('nan')
```

- Precision = TP/(TP+FP) nếu mẫu số dương, ngược lại NaN.
- Recall = TP/(TP+FN) nếu mẫu số dương, ngược lại NaN.
- F1 tính trực tiếp từ counts, không tính qua precision/recall.
- Nếu denominator=0: `f1_status=undefined_no_positive_reference_or_prediction`.
- Nếu n_valid=0: `f1_status=undefined_no_valid_pairs`.
- Nếu denominator>0: `f1_status=defined`, bao gồm F1=0.
- Xuất n_requested, n_success, n_valid, coverage=n_valid/n_success, n_gt_positive, TP/FP/TN/FN, precision, recall, F1 và status.
- Không xuất macro-F1 bảy bệnh nếu bất kỳ bệnh nào undefined. Không dùng `mean(skipna=True)` rồi gọi là macro-F1 đầy đủ.

Xuất thêm `audit/*_label_coverage.csv`: cho mỗi bệnh ghi bảng chéo GT/pred trong bốn trạng thái positive, negative, uncertain, unmentioned. Liệt kê các mẫu GT positive nhưng pred uncertain/unmentioned để review omission; chúng vẫn bị loại khỏi metric chính, không âm thầm thay đổi protocol.

### B4. Chính sách bootstrap v2

1. Dùng bảy bệnh theo thứ tự cố định trong notebook 02. Mỗi bệnh reset RNG `numpy.random.default_rng(42)` để kết quả không phụ thuộc bệnh chạy trước.
2. Lấy joint definite pairs của bệnh đó; resample n_valid cặp có hoàn lại, 1.000 lần. Resample nguyên cặp GT/pred, không tách hai phía.
3. Mỗi resample tính cùng `f1_from_counts`. Denominator=0 giữ NaN, không ép bằng 0.
4. Lưu `bootstrap_total`, `bootstrap_defined`, `bootstrap_undefined`, seed và version numpy.
5. Nếu point estimate undefined hoặc không có bootstrap hợp lệ: CI=NaN, có reason.
6. Nếu point defined và có bootstrap hợp lệ: percentile 2,5/97,5 trên các resample defined, dùng nội suy linear; gắn nhãn `conditional_on_defined_resamples`. Đây là CI có điều kiện, không tuyên bố tương đương CI paper.
7. Gắn cờ `low_support` nếu n_valid<20 hoặc GT positive<5; cờ mô tả do runbook quy định, không phải ngưỡng bảo đảm thống kê. Luôn giữ counts và số resample undefined để người đọc tự đánh giá.

CI v2 có thể khác CI cũ do đổi policy và RNG. Không dùng CI [1;1] từ một cặp làm bằng chứng chắc chắn.

### B5. Kiểm thử trước khi tái tính

Tạo `tests/test_exp02_reevaluate.py` bằng `unittest`, kiểm tra ít nhất:

| Input | Expected |
|---|---|
| TP=0, FP=1, FN=0 | F1=0 |
| TP=0, FP=0, FN=1 | F1=0 |
| TP=0, FP=1, FN=1 | F1=0 |
| TP=FP=FN=0 | F1 NaN |
| TP=7, FP=7, FN=3 | F1=14/24 |
| Success nhưng text whitespace | Canonical failed |
| 2 failed records cùng ID | Một canonical failed |
| Label Reports hoặc row map bị đảo | Alignment phải fail |
| Cùng input và seed | Bootstrap giống nhau |
| Tất cả nhãn negative | Point và CI undefined; 1.000 undefined resamples |
| Không có valid pair | Không resample mảng rỗng; ghi reason |

Chỉ chạy sau khi script/test đã tồn tại:

```powershell
python -m unittest discover -s tests -p test_exp02_reevaluate.py
python scripts/exp02_reevaluate.py --help
```

Tạo run ID bằng Python hoặc UTC + UUID, lưu vào biến `$Exp02Output`, rồi chạy:

```powershell
python scripts/exp02_reevaluate.py --source result/CXR_LLaVA_EXP02_Indiana_External_Eval --output $Exp02Output --sizes 10 50 100 --bootstrap-iterations 1000 --seed 42
```

`$Exp02Output` phải là thư mục run mới theo mục 3; script từ chối output đã có. Không copy nguyên placeholder chưa gán vào terminal.

### B6. Kiểm chứng kết quả thực tế

- 21 confusion matrices phải giữ nguyên so với nguồn khi labels và mask không đổi.
- F1 bộ 100: Cardiomegaly 0,5833333333; Lung Opacity 0,7272727273; Consolidation/Pleural Effusion/Pneumonia bằng 0; Edema/Pneumothorax undefined.
- Bộ 10: Cardiomegaly và Consolidation bằng 0, Lung Opacity bằng 1; các bệnh còn lại undefined.
- Bộ 50: Cardiomegaly 0,5454545455; Lung Opacity 0,6666666667; Consolidation/Pleural Effusion bằng 0; còn lại undefined.
- Lưu bảng old/new metric với reason cho mỗi thay đổi; tính lại tổng TP+FP+TN+FN=n_valid.
- Hash raw sources sau chạy phải trùng trước chạy.

**PASS B:** tests qua, alignment qua, confusion counts giữ nguyên, canonical counts đúng, bootstrap policy nhất quán và không ghi đè nguồn. Nếu fail, sửa nguyên nhân và chỉ chạy lại check bị ảnh hưởng trước khi rerun output mới.

## 6. Phase C — dataset, model revision, template và provenance

### C1. Dataset preflight trên Colab, chưa tải model

1. Mount Drive; đặt `DATASETS_ROOT` bằng đường dẫn thực có. Notebook cũ dùng nhánh `MyDrive/USTH_Master/ResearchLab/Notebook/datasets`; không giả định đường dẫn này tồn tại trong runtime khác.
2. Kiểm tra hai candidate `DATASETS_ROOT/NLMCRX` và `DATASETS_ROOT`, mỗi candidate cần `NLMCXR_png` và `NLMCXR_reports`. Nếu nhiều candidate hợp lệ, không tự chọn theo thứ tự; đối chiếu inventory và cấu hình người dùng.
3. Dùng parser hiện có ở notebook 02 để đối chiếu từng XML reference với PNG. Ghi tổng PNG/XML, matched/missing, empty report, F1 eligible, duplicate ID và mixed caption.
4. Đừng đặt số 7.470 PNG thành điều kiện phải đạt bằng mọi giá. Snapshot cũ chỉ thấy 2.541 PNG; nếu tìm thấy root đầy đủ hơn, lưu khác biệt và không thay đổi cohort cũ.
5. Với cùng cohort cũ, remap path theo image_id, xác nhận basename duy nhất, GT text và report_id không đổi. Với cohort mới, tạo EXP_ROOT/subset mới và bắt đầu lại gate 10.
6. `Image.open(...).verify()` cho mọi ảnh định chạy; mở lại để đọc width/height/mode. Ghi SHA256 ảnh, XML, subset CSV và canonical image–report mapping đã sort. Không gọi F1 là frontal được xác nhận.

**STOP:** ảnh không đọc được, ID/path không duy nhất, GT khác nguồn, hoặc không xác định được root. Chỉ việc retry/gate bị chặn; không hủy kết quả offline B đã hoàn thành.

### C2. Pin model và remote code

1. Trước khi load, resolve `ECOFRI/CXR-LLAVA-v2` một lần thành commit SHA bằng metadata Hugging Face, lưu SHA thật vào `run_config.json`; không dùng chuỗi `main` như pin.
2. Kiểm tra API đang cài và model metadata trước khi truyền revision. Mọi lần tải config/weights/tokenizer/remote code phải quy về SHA đã ghi. Đặc biệt đọc custom loader vì tokenizer có thể được tải bằng lệnh riêng bên trong remote code.
3. Nếu tokenizer dùng repo khác, pin SHA repo đó riêng. Nếu custom loader không cho pin dependency nội bộ, ghi blocker và sửa tối thiểu loader/wrapper trước khi tuyên bố tái lập.
4. Sau load, ghi resolved paths/revisions và hash các tệp remote Python thực dùng; đối chiếu với SHA yêu cầu. Không tuyên bố khôi phục revision lịch sử vì log cũ chỉ ghi `main`.
5. Giữ NF4, double quant, FP16 compute, `device_map={"": 0}` và tokenizer compatibility fix đã kiểm chứng. Không gọi `model.to(...)` sau load quantized model.

### C3. Xác minh chat template và decoding

1. Đọc function `write_radiologic_report`, `generate_cxr_repsonse`, `apply_chat_template` của code thực load, không chỉ mã local.
2. Ghi template thực tế trước fallback. Nếu thiếu, tái sử dụng nguyên template legacy Llama 2 đã có trong notebook 00/01; không dùng fallback nối content của notebook 02.
3. Render một chat synthetic với system/user và `<image>` mà không inference. Xác nhận có `[INST]`, `[/INST]`, `<<SYS>>`, nội dung user và marker ảnh đúng theo helper; lưu text + SHA256 template/prompt.
4. Ghi actual temperature, top_p, do_sample, max_new_tokens, EOS/pad/BOS IDs và input token count. Không suy diễn default từ một file chưa được load.
5. Chọn cấu hình diagnostic deterministic: temperature=0, top_p=1 nếu helper hỗ trợ và xác nhận `do_sample=False`. Đây là cấu hình mới, không tái hiện chắc chắn output lịch sử.
6. Seed Python, numpy, torch/CUDA bằng 42. Ghi giới hạn: cùng seed không bảo đảm bitwise-identical qua GPU/package khác nhau.

### C4. Mỗi run phải có hồ sơ riêng

Ghi UTC start trước khi làm và UTC end trong `finally`, cả khi lỗi. Environment gồm OS/Python, CUDA/driver, GPU, torch/transformers/tokenizers/hub/accelerate/bitsandbytes/protobuf/Pillow/numpy/pandas, `pip freeze`, Git HEAD/dirty diff, notebook/script hashes, model/tokenizer/code revisions, template hash, generation config, cohort hash, seed, run kind, parent run, artifact paths, final status.

**PASS C:** inventory của scope rõ, ảnh đọc được, revision/template/generation config có bằng chứng thực tế và environment riêng đã lưu. Không ghi `PASS` nếu chỉ mới điền giá trị mong muốn.

## 7. Phase D — hai ảnh failed và attempt log

Scope chính xác: `CXR159_IM-0382-1001`, `CXR1706_IM-0466-1001`; tối đa **hai diagnostic attempts mỗi ảnh**, không retry vô hạn.

1. Kiểm tra image/report/path/hash và ảnh hiển thị trước load; ghi nhận ảnh trắng/hỏng hoặc hình học bất thường nếu có, không sửa pixel nguồn.
2. Dùng model đã qua C. Chạy một ảnh mẫu đã thành công làm control để xác nhận helper hoạt động; lưu output riêng.
3. Tạo `attempts.jsonl` append-only, mỗi attempt có UUID, run_id, image_id, attempt_index, start/end UTC, config hash, seed, elapsed time kể cả failed, peak memory, raw response type, raw text, stripped text, status và exception traceback.
4. Instrument code generation thực dùng để lưu prompt render, input token IDs/count, output IDs/count, phần token mới tách khỏi prompt, EOS xuất hiện ở đâu và decoded text trước/sau postprocess. Bọc instrument bằng `try/finally` để khôi phục method gốc. Không thay cách decode mà chưa lưu output trước thay đổi.
5. Không coi `None` là chuỗi thành công `"None"`; yêu cầu response là text hợp lệ và không whitespace. Lưu return type bất thường như lỗi.
6. Attempt 1 dùng cấu hình cố định của C cho từng ảnh. Nếu thành công, dừng retry ảnh đó.
7. Nếu failed: xác định lỗi ở image, input/template, placement, generation hay postprocess. Attempt 2 chỉ được chạy khi có giả thuyết cụ thể và một thay đổi duy nhất được log. Nếu chưa có giả thuyết, dừng ảnh đó sau attempt 1.
8. Nếu output còn rỗng hoặc OOM, giữ failed và diagnostics. Không tăng token limit, đổi precision hay thay prompt hàng loạt để cố tạo success.
9. Báo riêng kết quả control, từng ảnh/attempt và điều đã thay đổi. Nếu cần labels cho output mới, chạy official CheXpert sample gate rồi label lại; không reuse labels của text cũ.

**PASS D về xử lý:** có chẩn đoán/attempt evidence đầy đủ, dù ảnh vẫn failed. **PASS D về inference:** cả hai ảnh có report không rỗng. Phân biệt hai trạng thái trong handoff; chỉ trạng thái thứ hai đóng vấn đề empty-output.

## 8. Phase E — review 23 ảnh mismatch và đối chứng

### E1. Chọn mẫu, không thay theo kết quả đẹp

1. Dùng bộ 100 lịch sử, dedup label errors theo image_id để lấy **23 ảnh**, chứa 28 mismatch records.
2. Lấy 10 ảnh đối chứng từ canonical success chưa có mismatch joint-definite; sort image_id rồi lấy mẫu numpy seed 42, lưu danh sách một lần. “Không mismatch” không có nghĩa không lỗi hoặc normal.
3. Xuất thêm danh sách tất cả ảnh GT positive nhưng generated uncertain/unmentioned từ B3. Union với 33 ảnh trên để tạo review queue; ghi rõ phần mở rộng bao nhiêu ảnh, không chỉ review 23 ảnh rồi bỏ omission.
4. Giữ riêng historical cohort và outputs mới của D/F; không thay text trong review lịch sử.

### E2. Schema review

Mỗi dòng image_id/pathology có: cohort/run_id, image hash/path, GT text, generated text, raw GT/pred labels, valid-mask/reason, error_type, GT evidence span, generated evidence span, omission candidate, uncertainty shift, laterality/location mismatch, unsupported statement candidate, reviewer_type, reviewer, review_date_utc, image_viewed, notes, resolution và clinical_validation_status.

### E3. Trình tự review mỗi ảnh

1. Đọc GT và generated đầy đủ, không chỉ câu có label error.
2. Trích evidence spans chứng minh nhận xét. Nếu GT positive mà generated không nói tới, ghi omission candidate dù metric đã loại cặp đó.
3. Kiểm tra mức chắc chắn, bên trái/phải, vị trí và mô tả bình thường/bất thường có mâu thuẫn văn bản hay không.
4. Nếu có quyền truy cập ảnh, mở ảnh và ghi `image_viewed=true`; nếu không, để false và ghi giới hạn.
5. Agent ghi `reviewer_type=agent_text_review`; không giả danh bác sĩ, không đánh dấu clinical_validation_status=validated. Nhận định về đúng/sai trên ảnh cần người đủ chuyên môn xác nhận riêng.
6. Ghi `unresolved` khi bằng chứng chưa đủ. Không coi ô trống là đã review.

**PASS E:** toàn bộ 23 mismatch + 10 đối chứng được text-review có evidence, và queue omission đã được xử lý hoặc ghi rõ từng mục còn chờ. Review lâm sàng là trạng thái riêng, không chặn việc hoàn thành audit kỹ thuật khi chưa có người chuyên môn.

## 9. Phase F — gate 10 và mở rộng có kiểm soát

Tạo notebook phục hồi riêng nếu cần, ví dụ `03_CXR_LLaVA_IU_recovery_eval.ipynb`; tái sử dụng loader/helper từ 00/01 và pipeline 02, không copy model architecture. Đặt inference/clinical flags False mặc định, full dataset False.

### F1. Gate 10

1. Cố định configuration hash: model/tokenizer/code revisions, template, preprocessing, generation, precision và evaluator policy.
2. Nếu giữ cohort, dùng đúng 10 image_id của historical F1 subset, nhưng **chạy mới cả 10** dưới cấu hình mới. Không resume từ predictions cấu hình cũ.
3. Nếu cohort đổi, tạo subset mới từ manifest mới, lưu seed/hash và ghi `cohort_changed=true`; không gọi kết quả là paired comparison với historical 10.
4. Load model một lần, inference batch size 1, lưu mỗi attempt lên Drive. GPU OOM, ảnh không đọc được hoặc output rỗng: dừng mở rộng, giữ diagnostics.
5. Official CheXpert setup + sample test phải return 0 và có output đúng schema. Sau đó label GT/generated của gate; check row map và Reports text như B2.
6. Tính metric bằng evaluator đã PASS B. Record timing, memory và valid-label coverage; không yêu cầu F1 cao để “qua gate”.

Gate PASS chỉ khi tất cả đúng:

- 10/10 unique image/report pairs đúng cohort, ảnh đọc được.
- 10/10 generated report không rỗng; mọi attempt thất bại trước đó vẫn được giữ trong log.
- Nhãn và predictions khớp ID, thứ tự và text; official labeler thành công.
- Metric formula/bootstrap đúng policy; undefined có reason, không cần ép tất cả F1 defined.
- Có đầy đủ provenance/config/hash/start/end và không có silent fallback.
- Review nhanh 10 cặp không phát hiện lỗi pipeline như đảo GT/generated, prompt bị lộ nguyên văn hoặc output chỉ lặp prompt. Sai khác nội dung bệnh lý được ghi lại, không dùng điểm số đẹp làm điều kiện lựa chọn mẫu.

Ghi `logs/gate10.json` với từng check và artifact evidence. Agent không tự đánh PASS bằng một cờ boolean không có evidence.

### F2. Stage 50 rồi 100

Chỉ bắt đầu 50 khi gate10 PASS, 100 khi 50 PASS cùng các tiêu chí kỹ thuật tương ứng. Không chạy full dataset.

Để các stage mới lồng nhau có chủ đích, tạo một permutation seed 42 của eligible manifest đã sort và lấy prefix 10/50/100; nếu phải giữ cohort lịch sử, dùng các CSV đã lưu và kiểm tra quan hệ bao hàm thay vì sample lại. Ghi phương pháp đã chọn.

Có thể reuse success từ stage trước **chỉ khi** image hash, GT hash và toàn bộ config hash trùng; ghi `origin_run_id` và timestamp gốc cho mỗi prediction reuse. Báo riêng inference mới, reuse và tổng cohort; không ghi latency reuse là thời gian chạy stage hiện tại. Retry chỉ failed/not_attempted của cùng config.

**STOP mở rộng:** bất kỳ preflight/alignment/provenance check fail, output rỗng chưa giải quyết, OOM, hoặc labeler sample fail. Không thay policy giữa stage rồi so sánh như cùng phương pháp.

## 10. Phase G — báo cáo, kiểm tra cuối và bàn giao

1. Tạo báo cáo mới trong `reports/` theo đủ 18 mục của `experiment_analysis_template.md`; không overwrite báo cáo 21/09.
2. Tách ba nhóm kết quả: historical metrics nguyên bản, historical reevaluation v2, fixed-config inference mới. Ghi số ảnh requested/success/failed/not_attempted và scope metric.
3. Ghi chính xác command, run ID, UTC, platform, versions, hashes, changed settings, tests và limitations. Không biến diagnostic retry thành benchmark.
4. Nếu sửa notebook, kiểm tra JSON bằng `python -m json.tool <notebook> > $null` trên PowerShell; không parse `%pip` như Python AST thông thường.
5. Chạy tests evaluator, `git diff --check`, kiểm tra raw source hashes không đổi và links report tồn tại. Kiểm tra dirty files cuối so với đầu; không stage thay đổi ngoài phạm vi.
6. Không tick checklist cũ “đã hoàn tất” khi mới xong một phần; bổ sung liên kết evidence trong báo cáo mới hoặc nhật ký trạng thái.

## 11. Checklist hoàn thành cho agent

- [ ] A: rules, Git state, hashes và snapshot audit.
- [ ] B: evaluator/tests triển khai, offline run thực thi, metric v2 và old/new diff kiểm chứng.
- [ ] C: dataset root/inventory xác minh trên runtime; revision/template/config thực dùng được ghi.
- [ ] D: control và hai ảnh lỗi có diagnostic attempts; kết quả success/failed được báo riêng.
- [ ] E: 23 mismatch + 10 đối chứng + omission queue có review trạng thái rõ.
- [ ] F: gate10 có evidence; chỉ chạy 50/100 nếu đủ điều kiện, không full dataset.
- [ ] G: báo cáo mới, kiểm tra cuối, bàn giao phần còn thiếu.

## 12. Mẫu handoff khi dừng hoặc hết quota

```text
Stage hiện tại:
Run ID / run_kind / output root:
Rules và source files đã đọc:
Files đã tạo/sửa:
Commands và checks đã thực sự chạy:
Artifacts chứng minh PASS:
Counts requested/success/failed/not_attempted:
Blocker cụ thể + error/log path:
Phần chưa chạy:
Lệnh tiếp theo hoặc quyết định cần người dùng:
Có cần GPU/dataset/clinical reviewer hay không:
```

Đừng ghi “hoàn thành” nếu chưa thực thi. Nếu chỉ có máy local và saved labels, kết quả hợp lệ có thể là **B hoàn thành, C–F đang chờ runtime**, với script, metric và bằng chứng đã lưu đầy đủ.

# Nhật ký Phân tích Thí nghiệm

## 1. Thông tin thí nghiệm

- **Mã thí nghiệm:** CXR-LLaVA-T4-SMOKE-2026-09-15
- **Ngày:** 2026-09-15
- **Nhà nghiên cứu:**
- **Dự án:** CXR_LLaVA_Improvement
- **Repository:** https://github.com/hanhpm/CXR_LLaVA_Improvement
- **Branch:** Không được ghi nhận trong output notebook
- **Commit:** Không được ghi nhận trong output notebook
- **Notebook / Script:** `CXR_LLaVA_Colab_T4_smoke_test.ipynb`
- **Trạng thái:** Hoàn thành (smoke test)

## 2. Mục tiêu

Kiểm tra CXR-LLaVA v2 có thể được nạp và tạo một báo cáo X-quang từ ảnh X-quang ngực mẫu trong quy trình Colab/T4 hay không.

## 3. Khảo sát / Động lực kỹ thuật

Notebook sử dụng các hàm suy luận hiện có của CXR-LLaVA và có phần tương thích cho định dạng hội thoại Llama 2 cũ.

## 4. Môi trường

Output notebook ghi nhận quá trình thiết lập thiết bị/mô hình và suy luận thành công. Phiên bản môi trường, tên GPU, branch và commit chính xác chưa được ghi lại; cần bổ sung khi tái hiện.

## 5. Cài đặt phụ thuộc

Xem các cell cài đặt và kiểm tra môi trường trong notebook. Không suy luận thêm quyết định về dependency từ output đã lưu.

## 6. Thay đổi code / notebook

- `CXR_LLaVA_Colab_T4_smoke_test.ipynb`

Notebook chứa phần xử lý tương thích và các cell smoke test tạo báo cáo.

## 7. Cấu hình thí nghiệm

| Tham số | Giá trị |
|---|---|
| Dữ liệu đầu vào | Ảnh mẫu trong `IMG/` |
| Số lượng mẫu | 1 |
| Tác vụ | Tạo báo cáo X-quang |
| Mô hình | ECOFRI/CXR-LLAVA-v2 |
| Phân loại | Smoke test |

## 8. Code đã thực thi

Cell tạo báo cáo gọi `model.write_radiologic_report(sample_image)`.

## 9. Output

### Output thô

```text
MODEL-GENERATED REPORT (research only):
The radiologic report reveals a large area of consolidation in the right upper lobe, likely indicative of pneumonia. The left lung appears clear. Cardiomediastinal and hilar silhouettes are normal. Pleural surfaces are also normal with no signs of pneumothorax.
```

### Lỗi / cảnh báo

Notebook ghi nhận vấn đề tương thích định dạng prompt cho các helper report/Q&A và áp dụng fallback trước khi suy luận. Output đã lưu cho thấy suy luận hoàn tất sau đó.

## 10. Phân tích output

### Những phần hoạt động tốt

- Notebook đã thực hiện đến bước tạo báo cáo.
- Output văn bản do mô hình tạo đã được lưu trong notebook.

### Những phần chưa đạt / giới hạn

- Đây chỉ là bằng chứng smoke test trên một mẫu.
- Metadata về phiên bản môi trường, GPU, branch và commit còn thiếu.
- Văn bản được tạo là output thử nghiệm của mô hình, không phải kết luận lâm sàng đã được xác minh hay tư vấn y tế.

## 11. Kết quả định lượng

Không thu thập metric định lượng vì đây là smoke test.

## 12. Kết quả định tính

Mô hình tạo ra một đoạn văn theo phong cách báo cáo X-quang cho ảnh mẫu. Tính chính xác y khoa chưa được đánh giá độc lập.

## 13. So sánh với baseline

Chưa thực hiện so sánh với baseline.

## 14. Vấn đề gặp phải

Notebook ghi nhận vấn đề thiếu template/định dạng prompt cũ và áp dụng fallback trước khi suy luận. Output đã lưu cho thấy suy luận hoàn tất.

## 15. Kết luận

Notebook smoke test Colab/T4 đã tạo thành công một báo cáo do mô hình sinh ra từ ảnh X-quang ngực mẫu. Kết quả chỉ xác nhận quy trình có thể chạy, không chứng minh độ chính xác lâm sàng, hiệu năng benchmark hay khả năng tổng quát hóa. Cần bổ sung metadata môi trường và Git trong lần chạy tái lập tiếp theo.

## 16. Thí nghiệm tiếp theo

- [ ] Ghi lại GPU, Python, CUDA, PyTorch, Transformers và Git commit.
- [ ] Chạy lại trên một tập validation cố định và được cấp phép.
- [ ] Xác định metric chất lượng báo cáo trước khi thực hiện so sánh nghiên cứu.

## 17. Checklist tái lập

- [x] Đã ghi nhận notebook/script
- [x] Đã lưu output thô
- [ ] Đã ghi nhận phiên bản môi trường
- [ ] Đã ghi nhận phiên bản dữ liệu đầu vào
- [ ] Đã ghi nhận cấu hình đầy đủ
- [ ] Đã lưu metric
- [x] Đã ghi rõ giới hạn

## 18. Cập nhật ngắn cho phòng thí nghiệm

> **Mục tiêu:** Kiểm tra quy trình suy luận CXR-LLaVA trên Colab/T4. **Triển khai:** Sử dụng fallback tương thích trong notebook và chạy tạo báo cáo trên một ảnh mẫu. **Kết quả:** Đã tạo được output theo định dạng báo cáo X-quang. **Vấn đề:** Metadata môi trường và Git còn thiếu; kết quả hiện chỉ là smoke test. **Bước tiếp theo:** Bổ sung metadata và đánh giá trên một tập dữ liệu cố định, được cấp phép.

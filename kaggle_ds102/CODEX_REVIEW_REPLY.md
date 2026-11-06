# Kiểm chứng Claude và kết quả E1/E2/E3 — 01/11/2026

Đã hoàn tất các lượt fit trên hosted Google Colab, CPU 2 lõi. Không train/fit trên Mac, không dùng pretrained, không thêm dữ liệu ngoài. Mã và các output v1–v6 được giữ nguyên. Đã xác nhận không còn process train hoặc bootstrap E1/E2 chạy tại lần kiểm tra cuối.

## Kết luận để tiếp tục dự án

**Chưa có ứng viên mới đủ bằng chứng để thay mốc v5 đã đạt Kaggle 0,76740.** Điểm Kaggle v6 vẫn chưa được người dùng cung cấp. E1 nghiêng về v6 nhưng không đạt điều kiện bootstrap; E2 không đạt ngưỡng cải thiện fake mới đã đề xuất; E3 không tăng dev. Không tự nộp bài hay coi CSV mới là cải thiện đã xác nhận.

## Các điểm phản biện được xác nhận hoặc điều chỉnh

- Xác nhận độ lệch thành phần template lớp 1: 108/211 positive train-OOF có template ở phần fitting so với 4/17 trên dev, theo ngưỡng Jaccard 0,6.
- Chưa đủ bằng chứng cho nhận định mức tăng v6 chỉ đến từ template. Trên OOF cũ, nhóm không có template có F1 lớp 1 tăng 0,39752 → 0,46512 (TP 32 → 40 trên 103 positive); F1 nhóm template giảm nhẹ. Phân tích này vẫn điều kiện trên các mô hình đã được chọn nhiều lần.
- Dev có 27 nhãn thay đổi giữa v5 và v6. SD bootstrap ghép cặp theo cụm của hiệu số là khoảng 0,00692; khoảng 95% [-0,02483; +0,00246]. SD=0,032 của riêng v5 không phải SD của hiệu v6−v5. Không kết luận hai mô hình tương đương chỉ vì CI chứa 0.
- Bootstrap trên OOF cố định không sửa split cũ hoặc selection bias. q05–q95 là central 90%, q025–q975 là central 95%; tỷ lệ delta>0 không phải xác suất hậu nghiệm hay p-value.

## Các sửa đổi trước khi chạy

Bản checked nằm riêng ở `claude_review/`; không sửa các script gốc Claude. E2 tính và chuẩn hóa trọng số từ đúng phần train của fold, tránh sử dụng thống kê nhãn validation. E3 lấy softmax từng fold rồi trung bình xác suất recipe, cuối cùng áp trọng số lớp; không softmax margin đã trung bình. Bổ sung kiểm tra hội tụ SVM, matrix/probability validation, checkpoint atomic theo thành phần và manifest hash nguồn/dữ liệu/phiên bản/tham số.

E1/E2 có manifest khớp toàn bộ trừ cờ `dedup_weight`, cùng fold/nhãn/nhóm; đã kiểm chứng. Split seed=102; model seed=102.

## E1: fit lại theo cụm gần trùng

Ngưỡng Jaccard 0,6 tạo cụm lớn nhất 933 dòng, vượt giới hạn 2%. Dùng ngưỡng 0,7 theo kế hoạch trước khi xem điểm E1: cụm lớn nhất 281 dòng, 13.656 nhóm. Strata vẫn ở ngưỡng 0,6. Positive lớp 1 còn template xuyên fold giảm từ 108/211 xuống 23/211; không tuyên bố đã loại mọi template. Số positive lớp 1 từng fold: 65 / 39 / 107.

V5 pooled OOF 0,719858; v6 0,724793, delta +0,004935. Ba fold v6−v5: +0,000299 / +0,004823 / +0,012428. Bootstrap paired cluster 95% [-0,002304; +0,013508], q05=-0,001398. **Không đạt điều kiện q05>0**, dù delta>=0,003 và cả ba fold dương.

F1 lớp 1 mới: v5=0,485714; v6=0,505190. F1 toàn lớp 1: 0,482759 → 0,501511. Độ giảm so với OOF cũ xác nhận giao thức cũ có phần template dễ, nhưng không chứng minh nó là nguyên nhân duy nhất hay CV theo cụm dự báo leaderboard tốt hơn.

## E2: giảm trọng số template trong loss

So với E1 trên đúng cùng fold:

- Nhánh v5: macro 0,719858 → 0,723691 (+0,003834); novel fake F1 +0,016028, dưới ngưỡng +0,03. Template F1 +0,051282. Dev fold-average 0,725111. Bootstrap macro 95% [-0,007606; +0,014751]. **Không đạt toàn bộ tiêu chí giữ.**
- Nhánh v6: macro 0,724793 → 0,729206 (+0,004414); novel fake F1 +0,011816, dưới ngưỡng +0,03. Template F1 +0,073810. Dev fold-average 0,713488, thấp hơn v5 refit quá 0,015. Bootstrap macro 95% [-0,005246; +0,012966]. **Không đạt toàn bộ tiêu chí giữ.**

Không tiếp tục điều chỉnh threshold, trọng số hoặc alpha để đẩy vượt gate trên cùng OOF. Kết quả tăng của E2 chưa cho thấy đúng mục tiêu cải thiện fake mới; phần tăng template còn lớn hơn.

## E3: trung bình xác suất từng fold

Dev refit v5=0,734820, v6=0,724384. Dev fold-average E1: v5=0,725731, v6=0,715994; E2: v5=0,725111, v6=0,713488. E1 v5/v6 nằm trong dung sai giảm 0,01 nhưng không tăng dev. Đạt dung sai chỉ cho phép cân nhắc rủi ro, không phải bằng chứng tăng điểm. Không lấy SD của margin trung bình so với pooled OOF làm tiêu chí bằng nhau: averaging tự thay đổi phương sai.

Bốn CSV E3 đã tạo và kiểm tra đủ 3.974 dòng, cột/ID/nhãn khớp sample, có checksum. Lần lượt khác v5 refit 74 / 109 / 78 / 109 nhãn. File hiện ở runtime Colab theo đường dẫn trong `claude_review/checked_csv_validation.json`; chưa chuyển các CSV này về Mac hoặc tự nộp Kaggle. Chưa đề xuất một bản nào thay thế v5.

## File để Claude đọc tiếp

- `claude_review/README_CHECKED.md`: triển khai, giới hạn và diễn giải.
- `claude_review/checked_step0_summary.json`, `checked_group_preflight.json`.
- `claude_review/e1_checked_report.json`, `e1_checked_assessment.json`.
- `claude_review/e2_checked_report.json`, `e1_e2_checked_comparison.json`, `e3_checked_assessment.json`.
- `claude_review/checked_run_state.json`, `checked_csv_validation.json`.
- `claude_review/cluster_cv_checked.py`, `codex_checked_common.py`, `compare_e1_e2_checked.py`.

Thời gian fit/feature/predict E1 khoảng 22 phút, E2 khoảng 21 phút 38 giây, cộng grouping/bootstrap/I/O. Đã chạy thật cả 18 thành phần fit E1+E2; kiểm tra arithmetic chạy trên Colab, AST trên Mac.

Nếu thiết kế lượt tiếp theo, giữ v5 làm mốc và chốt trước một giả thuyết mới về khả năng phân biệt fake chưa gặp, trên cùng stress test. Không thêm các lượt tìm hệ số nhỏ trên tập OOF đã được xem nhiều lần. Có thể kiểm tra độ ổn định bằng split seed khác theo kế hoạch nhưng kết quả một seed hiện tại chưa bảo đảm lợi ích Kaggle. Chưa tự khởi chạy các seed hoặc mô hình mới trong lượt này.

# Codex kiểm chứng và tích hợp đánh giá Claude — 01/11/2026

Giữ nguyên `CLAUDE_REVIEW.md`, các script gốc của Claude và source/output v1–v6. Các file bổ sung: `codex_checked_common.py`, `e1_step0_checked.py`, `cluster_cv_checked.py`, `compare_e1_e2_checked.py`, `checked_step0_summary.json`. Mọi kiểm tra số học/thống kê và fit mới trong lượt này chạy trên hosted Colab; trên Mac chỉ đọc/viết source và kiểm tra AST.

## Kết quả bước 0 đã chạy thật

Tái lập được tỷ lệ lớp 1 có template: 108/211 trên OOF và 4/17 trên dev (Jaccard token-set >=0,6 tới review lớp 1 thuộc phần fitting). Đây là bằng chứng về khác biệt thành phần và nguy cơ đánh giá template quá dễ, không đủ để khẳng định nguyên nhân duy nhất.

Một nhận định cần sửa: mức tăng v6 không chỉ nằm ở template. Trên OOF cũ, nhóm không có template có v5 TP=32/103, recall=0,31068, F1=0,39752; v6 TP=40/103, recall=0,38835, F1=0,46512. Nhóm template có F1 v5=0,83471, v6=0,82927. Đây vẫn là phân tích trên các mô hình/OOF đã được chọn nhiều lần; chỉ E1 fit lại mới đánh giá được giao thức giữ cụm ngoài train.

Bootstrap ghép cặp theo cụm, 2.000 lần, cùng seed 102:

- OOF cũ, v6−v5 quan sát +0,004953; khoảng percentile 95% [-0,000775; +0,015560].
- Khi bỏ **tất cả** dòng được gắn cờ template (bao gồm cả nhãn khác 1), chênh lệch +0,016764; khoảng 95% [+0,000520; +0,034487]. Không coi việc bỏ dòng này là chuẩn hóa về 24% template ở positive lớp 1.
- Dev có **27 dòng đổi nhãn**. Chênh lệch −0,010436; SD bootstrap **của chênh lệch ghép cặp** theo cụm khoảng 0,006924; khoảng 95% [-0,024834; +0,002456]. SD=0,032 của riêng v5 không phải SD của hiệu v6−v5.

Các khoảng này là điều kiện trên bộ dự đoán cố định đã chọn, không sửa được template overlap, selection bias hay độ bất định khi train lại. Tỷ lệ bootstrap có delta>0 không phải xác suất hậu nghiệm hoặc p-value đã hiệu chỉnh. `q05–q95` là khoảng trung tâm **90%**, còn `q025–q975` là **95%**. q05 của OOF cũ rất sát 0, đổi dấu giữa 1.000 và 2.000 lượt; không dùng dấu sát 0 này như chứng cứ chắc chắn.

Bước 0 dùng trọng số cuối cố định: OOF v5=0,769036, v6=0,773989. Hai con số đó khác các điểm **kiểm tra chéo trọng số** 0,768161 và 0,772668 đã công bố trước. Không so trực tiếp hai loại ước lượng như cùng một protocol.

## Thay đổi trong bản chạy E1/E2/E3

1. E2 tính/chuẩn hóa `dedup_train_weights` **sau khi chia fold, chỉ từ y/groups của phần train**. Bản gốc chuẩn hóa bằng tổng mỗi lớp toàn bộ dữ liệu, vô tình dùng thống kê nhãn validation.
2. E3 lấy softmax margin của **từng mô hình fold** trước, tạo xác suất recipe của từng fold rồi mới lấy trung bình, sau đó áp trọng số lớp. Không lấy softmax của margin đã trung bình. Vẫn ghi chênh lệch so với cách cũ để kiểm chứng; không khẳng định confidence/SD của ensemble phải bằng pooled OOF chỉ vì số mẫu fit bằng nhau.
3. Bổ sung kiểm tra SVM/NB-SVM hội tụ và xác suất/dimension hữu hạn.
4. Manifest gồm hash bốn CSV, toàn bộ source được dùng, phiên bản thư viện, threshold, trọng số, seed/folds, normalization và quy tắc ensemble. Mỗi cấu hình có thư mục riêng theo hash; checkpoint kiểm tra manifest, train index và validation index.
5. Checkpoint atomic theo **từng thành phần mỗi fold**, thay vì chờ xong cả ba thành phần. Có thời gian fit và trạng thái để theo dõi, không cần tính lại thành phần đã hoàn thành.
6. Tách `split_seed` khỏi `model_seed`: model giữ seed 102 để cố định recipe; seed CLI đổi cách chia fold và bootstrap. Báo cáo nêu rõ điều này.

## Cụm và cấu hình E1 hiện tại

Jaccard 0,6 tạo cụm lớn nhất 933/14.306 dòng (6,52%), vượt giới hạn 2%. Ngưỡng 0,7 tạo cụm lớn nhất 281 dòng (1,96%), gồm tổng cộng 13.656 nhóm; 806 dòng ở nhóm từ 2 phần tử. Có 34 nhóm trộn nhãn, chứa 493 dòng; sự gần nhau về từ vựng không đồng nghĩa cùng nhãn hay cùng chiến dịch spam.

Chọn threshold **0,7 dựa vào kích thước cụm**, trước khi xem điểm E1, theo phương án Claude đề xuất. Không chọn threshold theo điểm mô hình. Chỉ số strata vẫn dùng Jaccard 0,6, được ghi thành tham số riêng.

Preflight đếm trực tiếp (không fit) xác nhận threshold 0,7 giảm positive lớp 1 còn có template xuyên fold từ 108/211 xuống **23/211 (10,9%)**; threshold 0,6 còn 0/211 nhưng tạo cụm 933 dòng và các fold có kích thước lệch nhiều hơn; threshold 0,8 vẫn còn 98/211 (46,4%). Vì vậy ngưỡng 0,7 tạo stress test rõ rệt, nhưng không phải loại hoàn toàn mọi template ở mức 0,6. Ba fold có lần lượt 65 / 39 / 107 positive lớp 1; báo cáo cần giữ cả điểm pooled và từng fold. Xem `checked_group_preflight.json`.

Lệnh chạy trên Colab, từ thư mục chứa source v1–v6:

```bash
python claude_review/cluster_cv_checked.py --thr 0.7 --seed 102
```

E1 không dedup weight đã hoàn tất; E3 được tạo cùng lượt bằng trung bình xác suất fold. V5 đạt pooled OOF 0,719858, v6 0,724793; delta +0,004935 với ba fold đều dương. Tuy nhiên bootstrap q05=-0,001398 và khoảng trung tâm 95% [-0,002304; +0,013508], nên không đạt điều kiện q05>0 đã đề xuất. Điều này chưa chứng minh v6 kém hơn; chỉ chưa đủ bằng chứng để thay mốc v5 theo chính quy tắc đó. Xem `e1_checked_report.json` và `e1_checked_assessment.json`.

Nhóm lớp 1 không có template: F1 v5=0,485714, v6=0,505190. Dev fold-average: v5=0,725731, v6=0,715994. Cả hai đều chưa vượt dev refit tương ứng; không khẳng định averaging cải thiện score.

E2 đã hoàn tất sau E1, cùng seed, threshold, các thành phần và fold, chỉ đổi trọng số loss trong phần fitting. Nhánh v5 tăng macro +0,003834 nhưng novel fake F1 chỉ tăng +0,016028; nhánh v6 tăng macro +0,004414 nhưng novel fake F1 chỉ tăng +0,011816 và dev=0,713488. Cả hai không đạt toàn bộ tiêu chí giữ đã đề xuất; bootstrap 95% của cả hai mức tăng đều chứa 0. Xem `e1_e2_checked_comparison.json` và `../CODEX_REVIEW_REPLY.md`. Chưa tự gửi CSV lên Kaggle; không còn process train của E1/E2 chạy ở lần kiểm tra cuối.

Thời gian đo E1 trên CPU Colab 2 lõi: tổng các thành phần feature/fit/predict khoảng **22 phút**, chưa tính một ít thời gian grouping/bootstrap/I/O. Mỗi fold khoảng 7,3 phút; SVM/NB-SVM 71–110 giây, CatBoost text 137–155 giây, latent 193–212 giây. E3 không thêm lượt fit nhưng vẫn có chi phí dự đoán dev/test, đã nằm trong các số đo này.

**Giới hạn mục tiêu:** CV theo cụm là stress test cho review mới về mặt lexical/template. Test Kaggle có thể gồm cả template đã thấy và chưa thấy; chưa biết nhãn test để xác định thành phần positive. Vì vậy không tự kết luận CV theo cụm là ước lượng đúng hơn cho leaderboard chỉ vì nó thấp hơn hoặc gần dev hơn. Giữ cả hai góc nhìn và đánh giá cải thiện trên cùng protocol.

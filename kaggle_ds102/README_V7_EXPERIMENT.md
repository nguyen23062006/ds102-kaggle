# V7 — đã chạy và loại ứng viên

Ngày 02/11/2026: đã xác minh gói người dùng upload và chạy đủ ba fold exact-group trên hosted Colab, exit code 0. OOF v5=0.7690359812, v7=0.7664955251 (giảm 0.0025404561). Ba fold đều giảm: -0.00293454 / -0.00244002 / -0.00241252. Novel fake F1 giảm 0.00757842. Bootstrap paired cluster 95% [-0.00548814, +0.00005018]. Không đạt gate nên dừng theo kế hoạch trước cluster CV/dev/full refit; không tạo submission v7. Không có tiến trình train còn chạy tại thời điểm kiểm tra cuối. Kết quả đầy đủ: output_v7_experiment/report.json. Đây là điểm OOF, không phải điểm leaderboard.

Gói `ds102_v7_bundle.zip` chứa đúng bốn CSV cuộc thi, cache v3 đã được cho phép dùng và chín mục nguồn/cache cần thiết, kèm checksum. Không có token, mật khẩu hay pretrained model. Kéo gói vào Files của Colab để có `/content/ds102_v7_bundle.zip`, hoặc chạy cell setup đã thêm để chọn file. Notebook độc lập: `DS102_Colab_v7_experiment.ipynb`.

Giả thuyết: nhánh Logistic Regression học loss khác LinearSVC, có thể bổ sung những lỗi của v5. Đặc trưng FeaturesV2 fit trong từng fold; bốn binary OVR logistic C=4, sqrt-balanced từ train của fold; xác suất OVR được chuẩn hóa. Trộn cố định 20% với 80% v5, giữ hệ số lớp v5. Không tìm thêm trọng số.

Thứ tự: exact-group CV với cache baseline đã kiểm tra fold/nhãn; nếu đạt gate mới chạy near-duplicate cluster CV, dựng lại baseline cùng fold; nếu đạt cả hai mới fit nhánh logistic trên toàn train và kiểm tra dev. Các gate được chốt trước khi chạy: macro +0.003, ít nhất 2/3 fold dương, bootstrap cụm 95% của hiệu số có cận dưới dương, novel fake F1 không giảm quá 0.01. Dev không giảm quá 0.005 so với v5. Chỉ xuất CSV candidate nếu đạt gate. Nếu không đạt thì lưu report và dừng, không tự nộp bài.

Các gate không bảo đảm tăng leaderboard; OOF và dev đã được xem ở nhiều vòng trước, bootstrap không loại được selection bias. V5 đạt 0.76740 vẫn là mốc hiện tại. Runtime phải là hosted Colab; guard từ chối fit trên Mac.

Thư viện dùng scikit-learn (BSD-3-Clause), CatBoost (Apache-2.0), NumPy/SciPy/pandas. Source pipeline đi kèm; chưa tự gán giấy phép công khai cho toàn bộ dự án.

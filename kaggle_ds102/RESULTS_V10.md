# V10 — đã hoàn thành, không giữ ứng viên

Chạy thật đủ 9 mô hình latent trên hosted Google Colab CPU: ba seed [102,17,2026] ở mỗi fold. Không pretrained, không dữ liệu ngoài, không fit dev. Huấn luyện khoảng 31 phút (1864 giây), process exit 0. MCP có ngắt giữa chừng nhưng tiến trình nền không bị dừng; sau ghép nối đã kiểm tra checkpoint và tiếp tục theo dõi, không chạy trùng.

V6 baseline refit seed 102 trên cùng fold: OOF macro F1 0.7731457846. V10 ba seed trung bình: 0.7732828488. Delta +0.0001370642, dưới ngưỡng +0.001 đã chốt. Delta từng fold +0.0010249943 / +0.0018957394 / -0.0024740110. Novel fake F1 delta 0.

Bootstrap ghép cặp theo cụm Jaccard 0.7, 2000 resample: central 95% [-0.0020951435, +0.0023766128]; q05=-0.0017177545. Gate macro và bootstrap không đạt, gate ít nhất 2/3 fold dương và novel fake không giảm đạt. Không sửa gate hay chọn lại seed sau kết quả.

Baseline refit hiện tại không tái lập bit-for-bit OOF v6 cũ 0.7739889142. Kiểm tra ép nhánh latent về float32 vẫn cho 0.7731457846; không thể quy chênh lệch này chỉ cho rounding. Nguyên nhân chính xác chưa được xác định; không tuyên bố đã tái lập hoàn toàn mô hình nộp cũ. Phép so sánh seed ensemble dùng cùng baseline refit hiện tại ở cả hai nhánh. Manifest phiên bản, dữ liệu, source đã lưu để kiểm chứng tiếp.

Dừng theo kế hoạch trước ba lượt fit toàn train: không xem thêm dev, không tạo CSV v10, không nộp Kaggle. V6 có điểm leaderboard do người dùng báo 0.77269 vẫn là bản tốt nhất đã biết. Những điểm OOF ở trên không phải điểm leaderboard. Bootstrap điều kiện trên các dự đoán đã fit và không loại selection bias từ nhiều lần thử trước.

File: output_v10_experiment/report.json; output_v10_experiment/manifest.json; experiment_v10.py; DS102_Colab_v10_experiment.ipynb. Chín checkpoint và oof.npz hiện ở /content/ds102_v7_bundle/output_v10_experiment trên Colab; chưa tải các mảng dự đoán này về Mac. Không còn process experiment_v10.py chạy tại lần kiểm tra cuối.

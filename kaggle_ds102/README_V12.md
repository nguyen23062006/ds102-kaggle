# V12 — kết hợp cố định hai kích thước latent

Đề xuất sau khi V11 tăng OOF nhưng không đạt đủ gate: lấy trung bình 50/50 xác suất latent64 và latent128, rồi giữ nguyên .7 v3 + .3 latent, hệ số lớp v6. Đây là một ứng viên cố định; không thử lưới hệ số để chọn tỷ lệ tốt nhất. Sàng lọc OOF từ cache V11/V10 đã xác minh data/source/versions/indices trước khi fit mới.

Gate định trước, dùng như seed ensemble V10: OOF tăng >=0.001, >=2/3 fold dương, novel fake F1 không giảm quá 0.01, paired cluster bootstrap q05>0. Nếu đạt mới fit full train các nhánh64/128, dev không giảm quá0.005 so baseline64 thì xuất CSV candidate. Không tự nộp Kaggle.

Việc đề xuất sau xem V11 là nghiên cứu thích nghi, có selection bias; bootstrap không loại được bias này. Baseline là v6 fit lại hiện tại, chưa tái lập bit-for-bit submission v6 cũ đạt0.77269. Không tuyên bố tăng OOF bảo đảm tăng leaderboard.

Mọi tính toán/fitting chạy hosted Colab; không pretrained hoặc dữ liệu ngoài, không fit dev. Source experiment_v12.py tái sử dụng hàm fit_latent trong experiment_v11.py; source V11 không bị sửa. Runtime output /content/ds102_v7_bundle/output_v12_experiment.

## Trạng thái cuối

Đã hoàn thành trên Colab, exit0; ứng viên không đạt đủ gate. Không tạo CSV mới và không còn process huấn luyện. Xem RESULTS_V11_V12.md và các report/manifest cục bộ.

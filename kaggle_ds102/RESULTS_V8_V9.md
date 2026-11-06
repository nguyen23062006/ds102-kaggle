# Kết quả lượt V8/V9

Đã thực sự huấn luyện trên hosted Colab CPU, không train trên Mac, không pretrained hoặc dữ liệu ngoài. V8 gồm 6 lượt fit binary CatBoost (hai specialist, ba fold); V9 gồm 3 lượt fit multiclass CatBoost có thông tin sản phẩm. Cả hai process exit code 0 và đã kết thúc.

V5 baseline cùng OOF: 0.7690359812. Đây là điểm validation từ dự đoán đã lưu với hệ số lớp cố định, khác điểm leaderboard 0.76740 người dùng báo.

- V8: 0.7689366823, delta -0.0000992989. Delta ba fold +0.00098027 / -0.00434541 / +0.00369686. Novel fake F1 delta 0. Paired cluster bootstrap 95% [-0.00428817, +0.00533031]. Không đạt gate macro và khoảng bootstrap.
- V9: 0.7675364493, delta -0.0014995319. Delta ba fold +0.00022853 / -0.00142476 / -0.00368397. Novel fake F1 delta -0.01714743. Paired cluster bootstrap 95% [-0.00635231, +0.00249114]. Không đạt gate macro, số fold dương, novel fake, và khoảng bootstrap.

Theo kế hoạch chốt trước, dừng cả hai ở exact-group CV; không chạy cluster CV, không fit đầy đủ trên toàn train hoặc xem dev cho hai ứng viên, không tạo submission mới. Không chỉnh lại tỷ lệ blend sau khi xem kết quả để vượt gate.

Kết luận: chưa có bằng chứng đề xuất thay v5. Các khoảng bootstrap điều kiện trên dự đoán cố định và không khử selection bias do đã thử nhiều cấu hình. Không suy ra hai ứng viên chắc chắn thấp hơn trên leaderboard; chỉ xác nhận thử nghiệm hiện tại không chứng minh lợi ích theo tiêu chí đã chọn.

Nguồn: experiment_v8.py, experiment_v9.py, PLAN_V8_V9.md, source_v8_v9.zip. Báo cáo máy đọc: output_v8_experiment/final_report.json và output_v9_experiment/final_report.json. Checkpoint dự đoán từng fold còn trong runtime Colab dưới /content/ds102_v7_bundle/output_v8_experiment và output_v9_experiment; không mặc định tồn tại sau khi runtime hết hạn.

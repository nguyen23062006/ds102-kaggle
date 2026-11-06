# Yêu cầu đã xác nhận

- Không dùng pretrained model (người dùng xác nhận ngày 31/10/2026).
- Các mô hình/thư viện phải có mã nguồn mở (người dùng xác nhận ngày 31/10/2026). Cung cấp đủ source pipeline trong script/notebook. Không tự coi việc có source là đã phát hành dự án công khai với giấy phép nguồn mở.
- Huấn luyện tiếp theo chạy trên hosted Google Colab, không chạy trên máy cá nhân vì máy nóng.
- Metric cuộc thi: F1-macro, nhãn 0–3.
- Chỉ dùng bốn file dữ liệu cuộc thi đã cung cấp; giữ dev riêng khi đánh giá, không gộp dev vào train khi chưa rõ quy định.
- Điểm Kaggle người dùng báo: v2 0,75946; v3 0,76360; v4 0,76636; v5 0,76740. v6 0,77269. SVD128 experimental 0,77436. Mốc tốt nhất hiện tại là SVD128 experimental (người dùng báo ngày 03/11/2026).

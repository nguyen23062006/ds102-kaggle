# Lượt thử V8 và V9

Chốt trước khi xem kết quả: hai giả thuyết, chạy tuần tự trên hosted Colab; không pretrained, không thêm dữ liệu, không fit dev.

V8: hai CatBoost binary specialist 0/1 và 0/2, mỗi mô hình 500 cây depth=5, learning_rate=0.1, SqrtBalanced. Mỗi mô hình phân chia lại xác suất trong cặp lớp tương ứng. Kết hợp 60% v5 + 20% specialist01 + 20% specialist02; giữ hệ số lớp v5.

V9: hoàn tất hướng product identity mà thử nghiệm rich CatBoost v5 trước đây chưa chạy đủ. Cấu hình gọn: 500 cây depth=5, thêm product_name/categories dạng categorical và sold_per_review, max_ctr_complexity=1 để giới hạn thời gian. Trộn 25% nhánh mới với 75% v5; giữ hệ số lớp v5. Không dùng nhãn của review khác làm feature thủ công; CatBoost tính categorical statistics từ phần train của fold.

Cả hai dùng gate chốt trước như v7: exact-group macro +0.003, ít nhất 2/3 fold dương, novel fake F1 không giảm quá 0.01, paired cluster bootstrap 95% cận dưới >0. Chỉ ứng viên đạt gate mới sang cluster-group CV với baseline fit cùng split; sau đó mới full refit và kiểm tra dev không giảm quá 0.005. Nếu bị loại thì lưu kết quả, không sửa hệ số để vượt gate. Hai lần thử vẫn tạo selection bias; bootstrap không loại được bias này và không bảo đảm tăng leaderboard.

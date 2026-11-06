# Kết quả V11/V12

Baseline: cùng cache v6 refit 64D từ V10, OOF 0.7731457846. Submission v6 cũ đạt Kaggle 0.77269 do người dùng báo, chưa tái lập bit-for-bit ở runtime này. Không nhầm các điểm OOF dưới đây với leaderboard.

V11 — tăng SVD64 lên128, giữ các tham số khác: chạy thật ba mô hình CatBoost800 trên hosted Colab CPU, khoảng1158giây, exit0. OOF 0.7749451431, delta +0.0017993585. Ba fold +0.0040247003 / +0.0028846297 / -0.0016802264. Novel fake F1 delta +0.00629275. Bootstrap cụm ghép cặp 95% [-0.00154781,+0.00529150], q05=-0.00097733. Không đạt gate delta>=0.002 và q05>0, dù 2/3 fold tăng và novel fake không giảm. Không fit full train, không xem thêm dev, không tạo CSV V11.

V12 — đề xuất sau V11, lấy trung bình50/50 latent64 và128, không tìm hệ số: chỉ đánh giá lại cache trên Colab, không fit thêm. OOF 0.7734060545, delta +0.0002602700. Ba fold +0.0004055208 / +0.0024136617 / -0.0020267964. Novel fake F1 delta0. Bootstrap95% [-0.00191967,+0.00246490], q05=-0.00152112. Không đạt gate ensemble delta>=0.001 và q05>0. Không fit full train, không tạo CSV V12.

Không dùng pretrained, không dữ liệu ngoài, không train trên Mac, không gộp dev. Bootstrap có điều kiện trên mô hình đã fit và không loại selection bias của các lần thử thích nghi. Những gate này là quy tắc nghiên cứu đã chốt, không chứng minh V11 chắc chắn kém hơn trên Kaggle. Chưa có bằng chứng đủ để đề xuất thay submission v6 hiện tại.

Mọi process V11/V12 đã kết thúc exit0; kiểm tra pgrep cuối trả rỗng. Checkpoint V11 và OOF còn ở /content/ds102_v7_bundle/output_v11_experiment trên runtime Colab; chưa tải các mảng dự đoán về Mac. Báo cáo/manifest đã lưu cục bộ trong output_v11_experiment và output_v12_experiment. Source: experiment_v11.py, experiment_v12.py; giao thức: README_V11.md, README_V12.md.

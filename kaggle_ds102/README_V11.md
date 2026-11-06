# V11 — kiểm tra SVD 128 chiều

Thử nghiệm duy nhất, chốt trước kết quả: tăng số chiều latent của v6 từ 64 lên 128. Giữ nguyên vectorizer, seed=102, CatBoost 800 cây depth=5 lr=0.06, tỷ lệ v3/latent 70/30, trọng số lớp [1,1.2,1.2,0.9]. Mục tiêu là kiểm tra khả năng giữ thêm tín hiệu review/tên sản phẩm. Không pretrained, không dữ liệu ngoài, không gộp dev, mọi fit trên hosted Colab.

Baseline lấy đúng checkpoint seed102 ba fold từ V10, kiểm tra source/data hash, phiên bản thư viện, train/validation indices và manifest ID trước tái sử dụng. Baseline này là v6 fit lại hiện tại, không bit-identical với submission v6 cũ đã đạt Kaggle 0.77269. Hai nhánh so sánh có cùng cache v3 và cùng fold.

Gate: macro OOF tăng >=0.002, ít nhất 2/3 fold dương, novel fake F1 không giảm quá 0.01, paired cluster bootstrap q05>0. Nếu đạt mới fit toàn train nhánh64 và128 để so dev; không giảm dev quá 0.005 mới xuất CSV candidate. Nếu không đạt, lưu report rồi dừng; không điều chỉnh ngưỡng hoặc tỷ lệ blend sau khi xem kết quả. Không tự nộp Kaggle.

Quá trình này vẫn có selection bias do OOF/dev đã được dùng trong nhiều vòng trước. Bootstrap có điều kiện trên các dự đoán đã fit và không bảo đảm điểm leaderboard tăng. Không tuyên bố điểm OOF là điểm Kaggle.

Source: experiment_v11.py. Runtime output: /content/ds102_v7_bundle/output_v11_experiment. Trạng thái cục bộ: v11_run_state.json. Checkpoint atomic có manifest ID và tr/va indices; source v1-v10 không bị sửa.

## Trạng thái cuối

Đã hoàn thành trên Colab, exit0; ứng viên không đạt đủ gate. Không tạo CSV mới và không còn process huấn luyện. Xem RESULTS_V11_V12.md và các report/manifest cục bộ.

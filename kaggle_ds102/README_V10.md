# V10 — lấy trung bình các seed của v6

Người dùng báo submission_v6.csv đạt Kaggle 0.77269 ngày 03/11/2026; dùng v6 làm baseline mới. Đây là điểm do người dùng cung cấp, chưa truy cập trang submission riêng để xác minh.

MCP ghép nối thành công nhưng runtime Colab mới đã xóa dữ liệu. Đã đặt source v10 lên Colab và thêm cell upload. Chưa fit v10, chưa có điểm hoặc CSV mới. Dùng lại ds102_v7_bundle.zip (~18 MB); không cần tạo gói dữ liệu khác.

Giả thuyết chốt trước kết quả: giảm phương sai bằng cách trung bình ba latent CatBoost với seeds [102,17,2026]. SVD và CatBoost của từng nhánh dùng cùng seed tương ứng; baseline là nhánh seed 102 tái lập v6. Giữ nguyên feature 64D, 800 cây, depth 5, lr 0.06, v3/latent 70/30 và trọng số lớp [1,1.2,1.2,0.9]. Không chọn seed hoặc hệ số theo dev. V3 cache dùng chung ở cả baseline/candidate, kiểm tra nhãn và fold.

9 lượt fit ở ba fold exact-group. Mỗi seed có checkpoint prediction atomic theo train/validation indices và manifest hash. Gate trước full refit: macro F1 tăng ít nhất 0.001, >=2/3 fold dương, novel fake F1 không giảm quá 0.01, paired cluster bootstrap q05>0. Đây là ngưỡng nghiên cứu cho giả thuyết seed ensemble, không bảo đảm leaderboard. Nếu đạt, fit ba nhánh trên toàn train, đánh giá dev với dung sai giảm <=0.005 so với baseline v6 tái lập, rồi mới xuất submission_v10_candidate.csv. Không tự nộp Kaggle.

Tất cả feature fitting và training bị chặn ngoài hosted Colab. Không pretrained hoặc dữ liệu ngoài; không gộp dev. Script dùng sklearn/CatBoost nguồn mở; source pipeline kèm notebook. Việc dùng lại OOF/dev sau nhiều vòng vẫn có selection bias. Bootstrap chỉ điều kiện trên các dự đoán đã fit.

Source: experiment_v10.py; notebook độc lập: DS102_Colab_v10_experiment.ipynb. Trạng thái: v10_run_state.json. Sau khi upload, Codex cần xác minh dữ liệu, cài catboost==1.2.10 và scikit-learn==1.7.2, chạy script ở /content/ds102_v7_bundle với OMP/OPENBLAS/MKL_NUM_THREADS=2. Runtime hiện không có tác vụ train v10 đang chạy.

## Kết quả hoàn tất

Đã chạy đủ 9 mô hình trên Colab, exit 0. OOF refit v6=0.7731457846, v10=0.7732828488; delta +0.0001371 không đủ ngưỡng. Fold thứ ba giảm 0.0024740; bootstrap 95% chứa 0. Đã loại ứng viên, không tạo submission mới và không còn train. Baseline refit không bit-identical với OOF v6 cũ; xem RESULTS_V10.md và manifest để biết giới hạn tái lập.

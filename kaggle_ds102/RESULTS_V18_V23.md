# Kết quả V18–V23 trên hosted Colab

Mốc leaderboard do người dùng báo: submission_v11_experimental.csv đạt 0.77436. Các nhánh dưới đây so với baseline SVD128 này trên cùng ba fold train. Dev không gộp vào train; không dùng pretrained hoặc dữ liệu bên ngoài. Các số chênh lệch là macro-F1 trên từng fold, không phải điểm Kaggle.

- V18: LightGBM trên đặc trưng SVD128, trộn 20% vào baseline. Fold1 −0.000857; fold2 −0.020184. Dừng sau hai fold do vi phạm mức giảm tối đa 0.004. Chẩn đoán fold2: LightGBM đạt F1 train=1.0 nhưng validation kém hơn CatBoost, đặc biệt lớp2.
- V19: dùng lại dự đoán V18, thay nhánh latent bằng công thức 0.7*v3+0.3*LightGBM. Fold1 −0.003410; fold2 −0.019335. Dừng sau hai fold; không huấn luyện thêm.
- V20: LightGBM ít lá hơn, regularization mạnh hơn, class weighting balanced. Fold1 −0.000122; fold2 −0.007574. Cải thiện so với V18 nhưng vẫn kém baseline; dừng sau hai fold.
- V21: dùng lại V20 trong công thức thay nhánh latent. Fold1 −0.001916; fold2 −0.011605. Dừng sau hai fold; không huấn luyện thêm.
- V22: fit SVD128 với trọng số hàng theo nhãn train, nhưng mọi phép transform không dùng nhãn. Fold1 −0.005485; dừng sau một fold.
- V23: thêm 100 đặc trưng mô tả sản phẩm trong chính không gian TF-IDF/SVD128 của baseline, giữ CatBoost và công thức trộn. Fold1 −0.000779; fold2 −0.002242. Hai fold đều âm nên không thể đáp ứng tiêu chí ít nhất hai trên ba fold dương. Dừng chủ động khi bước chuẩn bị fold3 bắt đầu; mã thoát −15 là kết quả của việc dừng có chủ đích.

V23 đã kiểm tra fit/transform nhất quán, thứ tự dòng và mô tả rỗng trên Colab. Ở fold1, F1 lớp2 tăng từ 0.619916 lên 0.623596 nhưng F1 lớp1 giảm từ 0.693333 xuống 0.684564, khiến điểm tổng giảm. Feature importance không được dùng thay thế điểm validation để chọn mô hình.

Không nhánh nào có OOF đủ ba fold hoặc file submission mới. Không suy diễn điểm trung bình từ phần OOF chưa chạy và không xem chênh lệch trên hai fold là điểm đánh giá đầy đủ. Reports/manifests nằm trong output_v18_experiment tới output_v23_experiment; source các phiên bản trước giữ nguyên.

V24 tiếp tục bằng một thay đổi cố định sát baseline: SVD 128→192 chiều. Đây là giả thuyết tiếp theo, chưa phải kết quả cải thiện.

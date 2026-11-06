# V24 — SVD 192 chiều, giữ nguyên pipeline tốt nhất

Giả thuyết: lần tăng SVD 64 lên 128 chiều trước đây đã cải thiện leaderboard từ 0.77269 lên 0.77436. Thử đúng một mức tiếp theo là 192 chiều để kiểm tra khả năng giữ thêm tín hiệu trong văn bản. Không giả định điểm sẽ tăng tiếp.

Dùng đúng LatentFeatures và CatBoost của baseline; chỉ đổi n_components từ 128 thành 192. Giữ TF-IDF word/char, chuẩn hóa L2, SVD n_iter=5 và seed102; vector review/title, tích, hiệu tuyệt đối, cosine và 62 thống kê. Tổng 831 cột số và hai cột categorical. CatBoost 800 cây, depth5, learning_rate0.06, l2=8, SqrtBalanced, hai luồng. Điểm cuối bằng 0.7*v3 + 0.3*latent192, nhân lớp [1,1.2,1.2,0.9].

Từ vựng, IDF và SVD chỉ fit trên train của từng fold. Không thêm mô tả sản phẩm, không cân bằng các hàng khi fit SVD. Không dùng pretrained, dữ liệu ngoài hoặc nhãn dev để train. Mọi fit chạy trên hosted Colab. Nguồn: latent_rank_v24.py và experiment_v24.py; source baseline giữ nguyên.

So với OOF và full-fit SVD128 đã đóng băng; xác minh hash dữ liệu, mã nguồn gốc, phiên bản thư viện, nhãn và chỉ số fold. Cache đặc trưng mới có fingerprint của cấu hình/phân hoạch; cache dự đoán có fingerprint manifest. Seed và ba fold giống baseline.

Điều kiện giữ: pooled OOF tăng ít nhất 0.001; ít nhất hai trên ba fold tăng; không fold nào giảm quá 0.004; F1 fake review không có template không giảm quá 0.01. Dừng sớm nếu ngưỡng giảm bị vi phạm hoặc số fold còn lại không thể đáp ứng hai fold dương. Khi dừng sớm không báo điểm OOF đầy đủ. Bootstrap báo độ bất định, không dùng như phép thử độc lập sau nhiều lần chọn mô hình.

Chỉ khi qua CV mới fit toàn bộ train. Giữ dev riêng; yêu cầu dev không giảm quá 0.005 rồi mới xuất submission_v24.csv và kiểm tra đúng 3974 ID/nhãn. Không tự submit Kaggle. Việc dùng lại OOF/dev nhiều lần tạo thiên lệch lựa chọn; đây vẫn là thử nghiệm và không bảo đảm leaderboard tăng.

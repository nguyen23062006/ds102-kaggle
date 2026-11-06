# V16 — chuyên gia phát hiện sao chép mô tả cho fake review

Sau hai cấu hình TextCNN-rand chưa bổ sung tốt, thử một giả thuyết khác từ các lỗi fake bị bỏ sót: review có thể sao chép một đoạn dài từ mô tả sản phẩm, với thay đổi dấu câu/dấu tiếng Việt. Không gán nhãn bằng quy tắc hoặc sửa nhãn; mô hình CatBoost binary học fake-vs-rest trong từng phần train.

Features: 62numeric cũ +20row-local features đối chiếu comment với title/description: RapidFuzz ratio/partial_ratio có và bỏ dấu, longest contiguous word block (SequenceMatcher), exact normalized containment, tỷ lệ độ dài và tương tác độ giống/độ dài. HTML tags bị bỏ trong kênh so sánh; giữ tối đa6000ký tự comment và18000ký tự context. Số đo chuỗi dài là heuristic theo RapidFuzz, không chứng minh một review là fake. Docs chính thức: https://rapidfuzz.github.io/RapidFuzz/Usage/fuzz.html . RapidFuzz3.14.6 đã cài trên Colab, licenseMIT; không chứa mô hình pretrained.

Features chỉ phụ thuộc inputs của chính dòng, không vocabulary/corpus statistics/nhãn, nên có thể precompute trên train trước split. CatBoost category/rating statistics vẫn được fit riêng từ phần train của fold. Đã kiểm tra bỏ spam_label không đổi features; phép trộn bảo toàn xác suất, kể cả trường hợp xác suất0/1.

Cấu hình cố định: CatBoost700cây depth5 lr0.06 l2=8, SqrtBalanced, seed102,2threads. Expert tạo P(fake); phân phối các lớp khác giữ theo tỷ lệ baseline. Trộn15%expert +85%SVD128 tốt nhất (Kaggle do người dùng báo0.77436), weights[1,1.2,1.2,0.9]. Baseline/hash/versions/fold/y được xác minh.

Sàng lọc: OOF+0.001, >=2/3fold dương, worstfold>=-0.004, novel fakeF1 không giảm. Dừng sớm nếu một fold đã vi phạm worstfloor không thể sửa bằng các fold sau. Bootstrap được báo như độ bất định, không làm gate significance. Nếu đạt CV mới fit full train và so dev, dev giảm không quá0.005 mới xuất CSV. Không fit dev/test, không pretrained/dữ liệu ngoài, mọi ML tính trên hosted Colab. Không tự submit Kaggle.

Đây là nghiên cứu thích nghi sau nhiều lần xem OOF/dev và các lỗi; selection bias vẫn còn. Không bảo đảm leaderboard tăng. Source experiment_v16.py; runtime output /content/ds102_v7_bundle/output_v16_experiment.

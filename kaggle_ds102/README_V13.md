# V13 — TF-IDF có ngữ cảnh sản phẩm

Giả thuyết từ phân tích lỗi OOF: nhầm lớp0/2 chiếm nhiều lỗi, fold3 có F1 lớp2=0.57471 so fold1=0.61474/fold2=0.59718. Không sửa nhãn hay viết quy tắc từ một review cụ thể. Bổ sung cách biểu diễn tổng quát: mỗi token review được gắn hit/new theo việc xuất hiện trong tên sản phẩm, mô tả, danh mục; thêm kênh token kèm rating. Như vậy SVM có thể học token nào liên quan sản phẩm, không chỉ tỷ lệ overlap tổng hợp vốn đã có ở v2.

Nhánh mới: FeaturesV2 giữ nguyên + context TF-IDF word1-2, min_df2, max120k; LinearSVC C0.5, balanced, max_iter30000, temp3. Trộn15% nhánh mới với85% v6 refit; giữ trọng số lớp [1,1.2,1.2,0.9]. Một cấu hình cố định, không tìm lưới hệ số. Mọi vocabulary/scaler/SVM fit bên trong train của fold, không pretrained hoặc dữ liệu ngoài. Tất cả tính toán trên hosted Colab. Đã kiểm tra context feature không phụ thuộc cột spam_label trên một mẫu nhỏ tại Colab.

Baseline kiểm tra source/data/versions từ V11/V10 và dùng cùng y/fold; v6 refit hiện tại không bit-identical submissionv6 cũ đạtKaggle0.77269. Gate trước fullfit: deltaOOF>=0.001, >=2/3fold dương, novel fake F1 không giảm quá0.01, bootstrapq05>0. Nếu đạt mới fullfit contextSVM và latent64; dev không giảm quá0.005 mới xuất CSV candidate. Không tự nộp Kaggle.

Đây là thử nghiệm thích nghi sau nhiều lần xem OOF. Bootstrap không khử selection bias và không bảo đảm leaderboard tăng. Source experiment_v13.py; output runtime /content/ds102_v7_bundle/output_v13_experiment.

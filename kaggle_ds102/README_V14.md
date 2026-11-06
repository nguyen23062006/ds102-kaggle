# V14 — TextCNN khởi tạo ngẫu nhiên bổ sung cho SVD128

Người dùng báo SVD128 experimental đạt Kaggle0.77436 ngày03/11/2026; đây là baseline mới, thay cho v6=0.77269. Không sửa các báo cáo lịch sử vốn chưa biết điểm này.

Hướng mới: CNN-rand-inspired, tham khảo bài Kim2014 https://aclanthology.org/D14-1181/. Bài có biến thể khởi tạo ngẫu nhiên; mô hình này được bổ sung quan hệ review/title và numeric, không phải tái lập kết quả bài báo. Không tải word2vec, PhoBERT, checkpoint hoặc corpus ngoài. Embedding và mọi tham số mạng khởi tạo ngẫu nhiên, vocabulary/scaler chỉ fit phần train của fold. PyTorch đang có trên Colab, license BSD-3-Clause.

Cấu hình cố định trước kết quả: vocabulary tối đa30k, mincount2 từ comment/product_name train, embedding64, Conv1d48channels với kernels2/3/4, review160tokens (đầu128+đuôi32 khi dài), title40tokens; masked maxpool, masked means và tương tác review/title;62numeric đã có, StandardScaler train-only. MLP128, dropout0.3. AdamW lr0.0015 decay0.01, batch64,12epoch, trung bình probability epoch10/11/12 (không chọn epoch bằng validation). Loss dùng inverse-sqrt class-frequency tính từ fitting labels. Seed102, threads2; hosted Colab CPU hiện không có GPU.

Kết hợp cố định80% baselineSVD128 +20% CNN; giữ weights[1,1.2,1.2,0.9]. Dùng đúng fold/y/cache và kiểm tra hash/versions. Baselinefull128 cache phải tái lập SHA CSV tốt nhất05bffeff... trước xuất ứng viên. Mọi fit trên hosted Colab, không gộp dev.

Sàng lọc trước full fit: OOF tăng>=0.001, >=2/3fold dương, fold tệ nhất không giảm quá0.004, novel fakeF1 không giảm quá0.01. Bootstrap vẫn báo đủ95%CI/q05, nhưng không làm gate nhị phân: dữ liệu Kaggle mới cho thấy V11 tăng thật dù CI từng chứa0. Đây là thay đổi giao thức được chốt trước khi thấy kết quả V14, không sửa gate cũ. Không tuyên bố CI chứa0 là bằng chứng không hiệu quả, hay CI dương bảo đảm leaderboard tăng.

Nếu qua CV, fit CNN từ toàn train, so dev với baselineSVD128 cached (không fit dev), dung sai dev giảm<=0.005. Chỉ khi đạt mới tạo submission_v14.csv. Không tự nộp Kaggle. Repeated/adaptive OOF+dev selection vẫn tạo bias; bootstrap điều kiện không khử bias này.

Đã kiểm tra trên Colab forward/gradient finite cho câu rỗng/ngắn/đủ160tokens. Source: scratch_textcnn_v14.py và experiment_v14.py. Chỉ kiểm tra AST/đóng gói trên Mac. Snapshot weights/vocabulary/scaler và checkpoint predictions lưu trên Colab, không tự phát hành ra ngoài. Runtime output /content/ds102_v7_bundle/output_v14_experiment.

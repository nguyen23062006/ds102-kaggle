# DS102 v4 — thử nghiệm không dùng pretrained

**Điểm Kaggle v4 do người dùng báo ngày 01/11/2026: 0,76636**, so với v3 **0,76360**. CV tăng từ **0,76512 lên 0,76868**, nhưng dev giảm từ **0,73316 xuống 0,72248**. Các số Kaggle là thông tin người dùng cung cấp, chưa truy xuất độc lập từ Kaggle.

## File

- `output_v4/submission_v4.csv`: 3.974 dòng dự đoán đã kiểm tra, đúng hai cột, ID duy nhất và đúng thứ tự sample.
- `output_v4/metrics_v4.json`: CV, điểm dev và nguồn dự đoán.
- `DS102_Colab_v4_no_pretrained.ipynb`: notebook tự chứa source, train lại từ đầu trên hosted Colab. Không cần upload các file Python phụ; chỉ cần bốn CSV hoặc `colab_data.zip`.
- `train_submit_v4.py`: script cho cấu hình đã chọn, có chặn huấn luyện ngoài hosted Colab.

## Cách chạy notebook

Mở notebook trong Colab, dùng hosted runtime CPU và Run All. Chọn `colab_data.zip` hoặc bốn CSV tại ô upload. Notebook sẽ huấn luyện các mô hình từ train và xuất `/content/ds102_v4/submission_v4.csv`. Không bật local runtime. Quá trình có thể lâu trên CPU 2 lõi.

Notebook cài scikit-learn 1.7.2 và CatBoost 1.2.10. Không dùng pretrained model, pretrained embedding hoặc dữ liệu ngoài cuộc thi. Các bản v1/v2/v3 trước đây cũng không dùng pretrained.

## Phương pháp và đánh giá

V4 kết hợp 90% điểm v3 và 10% phiếu dự đoán của SVM phi tuyến. Kernel gồm phần bậc hai trên cosine TF-IDF và RBF trên metadata đã chuẩn hóa, clip outlier; `C=1`, trọng số lớp bằng căn của trọng số cân bằng. Tất cả dictionary, IDF, scaler và bộ phân loại đều học từ train.

Đã thử bảy cấu hình kernel trên CPU Colab. SVM phi tuyến riêng đạt CV tốt nhất **0,75755**, thấp hơn v3. Sau đó thử bốn tỷ lệ kết hợp 0/5/10/20% bằng OOF train, chọn 10%. CV từng fold v4: **0,78107 / 0,76293 / 0,76145**, cao hơn v3 ở cả ba fold. Tuy nhiên, khi đánh giá cấu hình đã chốt trên dev, kết quả giảm; đây là tín hiệu không nhất quán cần kiểm chứng bằng bài nộp mới.

Sử dụng cùng 3-fold StratifiedGroupKFold, seed 102; comment giống nhau sau chuẩn hóa nằm cùng fold. Không gộp dev vào train hoặc fit trên test. CV đã được dùng chọn mô hình, dev được dùng nhiều lần trong quá trình phát triển; không coi các điểm này là test độc lập.

## Nơi thực thi và dữ liệu tham chiếu

Huấn luyện kernel và các thí nghiệm mới đã chạy trên Colab Linux, Python 3.13.18, CPU 2 lõi, RAM khoảng 12,67 GiB. Không khởi chạy huấn luyện mới trên máy Mac. Checksum train/dev/test trên Colab khớp file gốc.

Theo sự cho phép cụ thể của người dùng, lần tạo CSV v4 hiện tại dùng lại các dự đoán v3 đã lưu để tiết kiệm thời gian; phần tính lại tham chiếu không cần thiết đã dừng ở điểm an toàn. Người dùng cũng đã cho phép tải CSV và metrics từ Colab về dự án. Notebook sạch được giao không phụ thuộc file cache này và train lại toàn bộ từ đầu.

Không gửi bài lên Kaggle tự động. Người dùng đã nộp v4 và xác nhận điểm 0,76636; giữ file này để đối chiếu các ứng viên sau.

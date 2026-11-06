# DS102 Kaggle baseline

Code dự đoán `spam_label` từ `comment`, chạy bằng CPU và không dùng mô hình pretrained. Ý nghĩa nhãn và F1-macro được xác nhận từ hai ảnh Overview/Evaluation người dùng cung cấp ngày 30/10/2026. Link cuộc thi hiện không truy cập được trong phiên này; chưa xác minh quy định về thư viện hoặc giới hạn phương pháp.

Lần chạy ngày 30/10/2026: chọn `C=1`, `class_weight=balanced`; F1-macro OOF trên train **0,72544**, F1-macro trên dev riêng **0,68948**, accuracy dev **0,89057**. F1 từng nhãn dev: nhãn 0 **0,9340**, nhãn 1 **0,4000**, nhãn 2 **0,5766**, nhãn 3 **0,8473**. Điểm yếu hiện tại là hai nhãn 1 và 2. File submission đã tạo và kiểm tra có **3.974 dòng**; đây chưa phải kết quả leaderboard.

## Chạy trên Kaggle

1. Tạo Notebook trong tài khoản Kaggle, import `ds102_kaggle_baseline.ipynb`.
2. Dùng **Add Input** để thêm dữ liệu cuộc thi chứa bốn CSV đã cung cấp.
3. Chạy **Run All**. Giữ `DATA_DIR = None` để tự tìm dữ liệu; nếu có nhiều bản dữ liệu, điền chính xác thư mục chứa bốn file.
4. Lấy `/kaggle/working/submission.csv` trong Output, nộp ở mục **Submit Predictions** của cuộc thi.

Không cần GPU. Thư viện numpy, pandas, SciPy và scikit-learn phải có trong môi trường; phiên bản tương thích nằm trong `requirements.txt`.

## Chạy trên máy cá nhân

Dùng Python 3.10 trở lên, tạo môi trường riêng, cài thư viện rồi chạy:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r kaggle_ds102/requirements.txt
python kaggle_ds102/train_submit.py \
  --data-dir "/Users/ngien2306/Downloads/ds-102-r-11-cnvn" \
  --output-dir kaggle_ds102/output
```

Trong workspace hiện tại, môi trường `.venv-kaggle` đã được chuẩn bị. Có thể chạy lại bằng `.venv-kaggle/bin/python` thay cho `python`.

## Phương pháp và cách đọc kết quả

- Chuẩn hóa Unicode NFC, đưa về chữ thường, chuẩn hóa khoảng trắng. Giữ dấu tiếng Việt, emoji và ký tự lặp để không bỏ mất dấu hiệu spam.
- TF-IDF từ 1–2 gram, ký tự 2–5 gram, cộng đặc trưng độ dài và tỷ lệ ký tự.
- Thử 6 cấu hình LinearSVC: `C` bằng 1 hoặc 4, với trọng số lớp mặc định, căn bậc hai của trọng số cân bằng hoặc `balanced`.
- Chọn cấu hình có F1-macro OOF cao nhất qua 3-fold StratifiedGroupKFold trên **train**. Comment giống nhau sau chuẩn hóa được xếp cùng fold; vocabulary, IDF và scaler được fit lại riêng ở từng fold.
- Huấn luyện cấu hình đã chọn trên toàn bộ **train**, đánh giá trên **dev**, rồi dự đoán **test**. Không dùng dev hoặc test để fit bộ biến đổi hay mô hình; không gộp dev vào train.
- Do có comment trùng giữa train và dev, báo cáo thêm điểm dev sau khi bỏ các comment đã xuất hiện trong train.

`output/metrics.json` chứa điểm từng fold, cấu hình được chọn, F1 từng lớp, confusion matrix và số dòng đã kiểm tra. `output/dev_predictions.csv` dùng để xem các lỗi phân loại. Điểm cross-validation dùng chọn cấu hình có thể lạc quan; điểm dev riêng phù hợp hơn để đánh giá bản baseline này. Nhãn 1 có rất ít mẫu dev nên F1 của lớp này dễ dao động.

`output/submission.csv` có đúng hai cột `ID,spam_label`, không có cột index. Code ánh xạ dự đoán theo `test.id`, sắp lại theo `sample.ID`, kiểm tra ID duy nhất, đủ số hàng và nhãn nguyên thuộc 0–3, rồi đọc lại file để xác minh.

Kết quả này là baseline đầu tiên; không bảo đảm điểm leaderboard. Trước khi nộp, đối chiếu Rules của môn học về các thư viện/phương pháp được phép. Không có thao tác gửi bài lên Kaggle trong code.

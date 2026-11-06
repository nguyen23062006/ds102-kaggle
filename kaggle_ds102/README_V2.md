# DS102 — bản cải thiện v2

Bản v2 chạy CPU, dùng dữ liệu cuộc thi đã cung cấp. File để nộp là `output_v2/submission_v2.csv`; notebook để chạy lại trên Kaggle là `ds102_kaggle_v2.ipynb`. Bản cũ vẫn ở `output/submission.csv`.

## Kết quả chạy thực tế

- F1-macro OOF train: **0,72544 → 0,75898**.
- F1-macro trên cùng tập dev: **0,68948 → 0,72981** (tăng **0,04033**).
- F1 nhãn 1: **0,4000 → 0,5000**; nhãn 2: **0,5766 → 0,5826**; nhãn 3: **0,8473 → 0,8964**.
- Accuracy dev: **0,89057 → 0,90314**.
- Đã tạo và kiểm tra độc lập **3.974 dự đoán**, đúng header và thứ tự ID của sample submission. Điểm OOF từ pipeline cuối khớp với cấu hình đã chọn trong thí nghiệm.

Đây là số liệu local. File v2 chưa được gửi lên Kaggle trong phiên này; chưa biết điểm leaderboard mới.

## Cách chạy trên Kaggle

1. Import `ds102_kaggle_v2.ipynb` vào Kaggle Notebook.
2. Add Input bộ dữ liệu có đủ `ds102_train.csv`, `ds102_dev.csv`, `ds102_test.csv`, `ds102_samplesubmission.csv`.
3. Run All với CPU. Không cần upload các file `.py`; notebook đã chứa đủ code. Không cần GPU, pretrained model hoặc dữ liệu ngoài.
4. Lấy `/kaggle/working/submission_v2.csv` trong Output để nộp. Giữ lại điểm và file bản cũ để đối chiếu.

Nếu không tự tìm được dữ liệu, đặt `DATA_DIR` thành đường dẫn thư mục chứa bốn CSV. Thư viện cần cài nằm trong `requirements.txt`; notebook đã kiểm tra với scikit-learn 1.7.2.

## Chạy local

Từ thư mục dự án hiện tại:

```bash
.venv-kaggle/bin/python kaggle_ds102/train_submit_v2.py \
  --data-dir "/Users/ngien2306/Downloads/ds-102-r-11-cnvn" \
  --output-dir kaggle_ds102/output_v2
```

Nếu dùng máy khác, tạo môi trường Python 3.10 trở lên và cài `requirements.txt` trước. Giữ `train_submit.py`, `improved_features.py`, `select_v2.py`, `train_submit_v2.py` trong cùng thư mục khi chạy script.

## Những thay đổi

- TF-IDF từ 1–3 gram và ký tự 2–5 gram để nhận dạng cách viết và các mẫu câu.
- Đo tỷ lệ cụm 1–4 từ trong review xuất hiện ở tên/mô tả sản phẩm. Các đặc trưng này giúp mô hình phát hiện bình luận sao chép nội dung mô tả.
- Thêm độ dài từ, tỷ lệ lặp, phân bố ký tự, thông tin `product_name`, `rating`, `category`, `num_sold`, `num_review`. `ID` không được dùng làm đặc trưng.
- Kết hợp 75% điểm quyết định SVM và 25% điểm quyết định NB-SVM, cùng `C=0.5`. NB-SVM học trọng số log-count ratio riêng cho từng lớp từ phần train của fold, rồi học bốn bộ phân loại one-vs-rest. Tham khảo phương pháp [Wang & Manning, 2012](https://aclanthology.org/P12-2018/); ở đây dùng biến thể TF-IDF và bổ sung đặc trưng sản phẩm.
- Kiểm tra điều chỉnh ngưỡng cho các lớp bằng OOF train. Chỉ áp dụng khi việc kiểm tra chéo cải thiện hơn 0,002 F1; cấu hình được chọn hiện không áp dụng điều chỉnh này.

## Cách đánh giá

Thử nghiệm trên **train** gồm baseline cũ, sáu cấu hình SVM v2 và ba cấu hình NB-SVM. NB-SVM có `C=1/2` gặp cảnh báo chưa hội tụ nên không được chọn. Sau đó so sánh thêm ba tỷ lệ kết hợp cho SVM/NB-SVM `C=0.5`.

Các cấu hình dùng cùng 3-fold StratifiedGroupKFold, seed 102. Comment giống nhau sau chuẩn hóa nằm cùng fold. Vocabulary, IDF, scaler và log-count ratio đều học trong phần huấn luyện của fold. Các điểm OOF và bước kiểm tra chéo calibration là số liệu dùng chọn mô hình, không phải test độc lập hoặc nested CV đầy đủ.

Chốt cấu hình bằng train trước khi đánh giá v2 trên dev. Dev trước đó đã được dùng xem lỗi baseline, nên vẫn là tập phục vụ phát triển. Không gộp dev vào train; không học vocabulary/scaler/mô hình từ test. Báo cáo thêm điểm dev sau khi loại các comment đã xuất hiện trong train.

`output_v2/metrics_v2.json` lưu điểm, seed, phiên bản thư viện, checksum dữ liệu và cấu hình. `dev_predictions_v2.csv` cho phép kiểm tra các lỗi. `scores_v2.npz` lưu điểm dự đoán phục vụ tái kiểm tra. Chương trình kiểm tra header, số dòng, ID, thứ tự và giá trị nhãn sau khi lưu CSV.

Điểm Kaggle của bản cũ khoảng 0,73 theo người dùng; điểm bản mới chỉ biết sau khi nộp. Chưa xác minh được Rules của môn học về thư viện/phương pháp; bản này dùng scikit-learn và không sử dụng mô hình pretrained.

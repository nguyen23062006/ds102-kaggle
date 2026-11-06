# DS102 — bản v3

Bản v2 đạt **0,75946 trên Kaggle theo người dùng**. Bản v3 là lần thử cải thiện tiếp theo, chạy CPU và không dùng mô hình pretrained hoặc dữ liệu ngoài. File v2 được giữ nguyên để so sánh.

## Kết quả đã đo

- F1-macro OOF train: v2 **0,75898** → v3 **0,76512**.
- F1-macro dev: v2 **0,72981** → v3 **0,73316**.
- F1-macro dev bỏ comment trùng train: **0,72023** → **0,72353**.
- V3 tốt hơn trên cả ba fold CV: **0,77566**, **0,76198**, **0,75702**; v2 tương ứng **0,76963**, **0,75531**, **0,75123**.
- Nhãn 1 trên dev: F1 **0,50000 → 0,51852**; nhãn 2 **0,58261 → 0,58559**. Nhãn 3 giảm **0,89642 → 0,88847**, nhưng F1-macro chung tăng nhẹ.
- Đã kiểm tra **3.974 dòng**, hai cột `ID,spam_label`, ID duy nhất và đúng thứ tự sample, nhãn nguyên 0–3. V3 thay đổi **115 dự đoán** so với v2.

Mức tăng nhỏ; chưa có điểm Kaggle cho v3 và không bảo đảm leaderboard sẽ tăng. CV được dùng chọn mô hình, dev đã được dùng trong nhiều vòng phát triển, nên các điểm này không phải một lần đánh giá độc lập hoàn toàn.

## File cần dùng

- `output_v3/submission_v3.csv`: file nộp ở Submit Predictions.
- `ds102_kaggle_v3.ipynb`: notebook tự chứa code, import vào Kaggle rồi chạy lại.
- `output_v3/metrics_v3.json`: cấu hình, phiên bản thư viện, checksum dữ liệu và kết quả dev.
- `output_v3/model_selection.json`: các cấu hình được so sánh bằng OOF train.

## Chạy trên Kaggle

1. Import `ds102_kaggle_v3.ipynb`.
2. Add Input bộ dữ liệu cuộc thi chứa bốn CSV.
3. Run All bằng CPU. Không cần upload các file Python phụ. Notebook cần numpy, pandas, scipy, scikit-learn và catboost; nếu thiếu CatBoost, cài bằng `%pip install catboost` trong một cell trước khi chạy.
4. Lấy `/kaggle/working/submission_v3.csv` trong Output để nộp.

Tham số đã được chốt qua CV. Notebook fit các mô hình trên train, đánh giá dev và tạo CSV; không chạy lại toàn bộ grid search. Các điểm CV tham chiếu chỉ được đưa vào báo cáo nếu checksum train khớp dữ liệu đã dùng trong thí nghiệm.

## Chạy local

```bash
python -m pip install -r kaggle_ds102/requirements_v3.txt
python kaggle_ds102/train_submit_v3.py \
  --data-dir "/Users/ngien2306/Downloads/ds-102-r-11-cnvn" \
  --output-dir kaggle_ds102/output_v3
```

Trong workspace này có thể thay `python` bằng `.venv-kaggle/bin/python`. Đã kiểm tra với scikit-learn **1.7.2**, CatBoost **1.2.10**, seed **102**. Giữ các module Python của v2/v3 trong cùng thư mục khi chạy script local.

## Phương pháp

Giữ SVM/NB-SVM của v2, chuẩn hóa điểm quyết định bằng `softmax(3 * scores)`. Kết hợp 50% điểm này với 50% đầu ra CatBoost. Đây là các điểm dùng chọn nhãn; không khẳng định chúng là xác suất đã được hiệu chỉnh.

CatBoost dùng 500 cây, depth 5, learning rate 0,1, l2_leaf_reg 5, trọng số lớp `SqrtBalanced`; đầu vào gồm đặc trưng review/sản phẩm của v2, category, rating và nội dung review. Xử lý text bằng dictionary unigram/bigram, BoW, NaiveBayes và BM25, theo [tài liệu CatBoost](https://catboost.ai/docs/en/features/text-features).

So sánh hai chế độ trọng số CatBoost và các tỷ lệ kết hợp cố định bằng 3-fold StratifiedGroupKFold trên train, cùng split với v2. Những comment giống nhau sau chuẩn hóa nằm cùng fold. Dictionary, IDF, scaler và mô hình chỉ học từ phần train của fold. Không gộp dev vào train; không fit trên test. Chọn phương án trước khi đánh giá v3 trên dev.

Các thử nghiệm SVM tăng trọng số metadata gặp cảnh báo chưa hội tụ nên không được chọn. Thử nghiệm bộ sửa lỗi trên các điểm OOF không tốt hơn ở bước sàng lọc, nên bị loại; không dùng điểm sàng lọc đó làm kết quả CV cuối.

Chưa xác minh Rules của môn học về thư viện/phương pháp. V3 thêm thư viện CatBoost; vẫn không dùng mô hình pretrained. Không có thao tác tự gửi bài lên Kaggle trong code.

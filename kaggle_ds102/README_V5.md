# DS102 v5 — ứng viên để thử nộp

File: `output_v5/submission_v5.csv`, 3.974 dòng, đúng hai cột `ID,spam_label`, đúng thứ tự sample, nhãn 0–3. Có 58 nhãn khác v4. **Điểm Kaggle v5 người dùng báo ngày 01/11/2026: 0,76740**, cao hơn v4 **0,76636**. Các số leaderboard do người dùng cung cấp, chưa được truy xuất độc lập.

V5 giữ các mô hình SVM/NB-SVM/CatBoost của v3, nhiệt độ SVM = 3 và tỷ lệ CatBoost = 50%, rồi nhân xác suất bốn lớp lần lượt với `[1, 1, 1.2, 0.9]` trước argmax. Các hệ số được chọn từ một grid nhỏ trên OOF train, kiểm tra chéo trên các fold OOF còn lại. Không gộp dev vào train.

CV sau kiểm tra chéo trọng số của v5: **0,76816**. V3 trước điều chỉnh: **0,76512**; v4: **0,76868**. V5 tăng so với v3 ở cả ba fold, nhưng chưa vượt CV v4. Dev v5: **0,73482**, so với v3 **0,73316** và v4 **0,72248**. Vì vậy đây là một ứng viên có kết quả dev tốt hơn và CV gần v4, không phải mức tăng leaderboard đã được xác nhận. OOF/dev đã được dùng nhiều lần trong phát triển; đánh giá calibration trên OOF không phải nested CV đầy đủ.

Lần xuất CSV này tính toán trên hosted Colab và dùng lại các dự đoán đã lưu, được người dùng cho phép chuyển riêng sang runtime mới. Checksum cache, nhãn/fold và cả bốn CSV đã được kiểm tra. Người dùng cũng đã cho phép tải riêng CSV/metrics v5 về dự án. Không khởi chạy huấn luyện mới trên Mac.

## Mã nguồn và chạy lại

- `DS102_Colab_v5_no_pretrained.ipynb`: notebook tự chứa source, chạy hosted Colab CPU; upload ZIP hoặc bốn CSV. Notebook train lại mô hình từ train, không phụ thuộc cache cũ, và xuất CSV/metrics. Cấu hình cố định từ lần chọn OOF, không chạy lại tìm kiếm CV.
- `finalize_v5.py`: fit/predict/xuất file; `temperature_v5.py`, `selection_v5.py`: chọn cấu hình/trọng số bằng train OOF.
- `experiment_v5.py`, `fasttext_scratch.py`, `stacking_v5.py`: mã các hướng khác đã thử; các hướng này không được chọn trong CSV v5.
- `output_v5/chosen_config.json`, `metrics_v5.json`: cấu hình và kết quả đánh giá.

Không dùng pretrained model, pretrained embedding hoặc dữ liệu ngoài cuộc thi. Mô hình cuối dùng scikit-learn 1.7.2 ([BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING)) và CatBoost 1.2.10 ([Apache-2.0](https://github.com/catboost/catboost/blob/master/LICENSE)). fastText là thử nghiệm riêng, học từ khởi tạo ngẫu nhiên với source chính thức v0.9.2/MIT; không dùng trong file nộp này. Có source pipeline không đồng nghĩa dự án đã được công khai hoặc tự động mang một giấy phép nguồn mở mới.

Đã chạy thật phần thử nghiệm và xuất CSV trên Colab, kiểm tra CSV độc lập tại máy và parse tất cả cell Python của notebook. Chưa chạy toàn bộ notebook sạch từ đầu đến cuối; lần chạy đó sẽ train lại các mô hình nền đã có cache, tốn thêm thời gian. Notebook JSON/cell được kiểm tra cấu trúc cơ bản; môi trường local hiện thiếu `nbformat` để chạy trình kiểm định schema riêng.

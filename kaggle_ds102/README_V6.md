# DS102 v6 — TF-IDF/SVD và CatBoost, bản thử nghiệm

**Chưa biết điểm Kaggle v6.** Mốc người dùng báo của v5 là **0,76740**. V6 tăng CV nhưng giảm dev; không khẳng định v6 sẽ tăng leaderboard. Giữ nguyên file v5 để đối chiếu.

Ứng viên: `output_v6/submission_v6.csv`, 3.974 dòng `ID,spam_label`, 64 nhãn khác v5. Chỉ nộp CSV này; notebook là mã để train lại. `metrics_v6.json` lưu kết quả và nguồn dự đoán.

## Mô hình cuối

V6 kết hợp 70% xác suất v3 (SVM/NB-SVM + CatBoost text) và 30% CatBoost mới trên TF-IDF/SVD. Sau đó nhân điểm bốn lớp với `[1, 1.2, 1.2, 0.9]` trước argmax.

Nhánh mới học TF-IDF từ review trong train và fit TruncatedSVD 64 chiều. Review và tên sản phẩm được chiếu vào cùng không gian; classifier nhận hai vector, tích từng chiều, hiệu tuyệt đối, cosine TF-IDF và các thống kê review/sản phẩm. CatBoost: 800 cây, depth=5, learning_rate=0.06, l2=8, SqrtBalanced, max_ctr_complexity=1, seed=102.

Không dùng pretrained model/embedding hoặc dữ liệu ngoài cuộc thi. SVD được học từ đầu, ngay trong phần train của mỗi fold. Các dictionary, IDF, scaler, SVD và bộ phân loại không fit trên dev/test. Không gộp dev vào train.

## Đánh giá

3-fold StratifiedGroupKFold, seed 102, nhóm theo comment đã chuẩn hóa. Cross-validation sau kiểm tra chéo trọng số: **v5 0,76816 → v6 0,77267**. Ba fold v6: **0,77834 / 0,77478 / 0,76405**, cao hơn v5 ở cả ba fold. Raw OOF trước trọng số v6: **0,76721**. Trọng số cuối được fit trên toàn bộ OOF, không phải trên dev.

Dev: **v5 0,73482 → v6 0,72438**. Đây là tín hiệu không nhất quán với CV, nên v6 vẫn là thử nghiệm. OOF đã được dùng nhiều lần để tìm mô hình; kiểm tra calibration trên OOF không phải nested CV đầy đủ. Dev cũng đã được dùng để so sánh các ứng viên và định hướng thử nghiệm, vì vậy không coi là test độc lập.

Đã thử thêm SVM kernel có xác suất (CV 0,77113; dev 0,72311) và dò review gần trùng (CV 0,77001; dev 0,73482 nhưng chỉ đổi 3 nhãn test). Nhánh SVD có CV cao nhất, được chốt trước khi đánh giá dev của chính nhánh đó. Không dùng điểm Kaggle hoặc nhãn test để chọn tham số.

## Thực thi, source và kiểm tra

- Các fit mới chạy thật trên hosted Google Colab CPU 2 lõi. Không khởi chạy huấn luyện mới trên Mac.
- CSV lần này dùng lại cache v3 đã được người dùng cho phép đưa sang Colab. Cache/nhãn/fold và bốn CSV đã được kiểm tra; metadata ghi rõ nguồn cache.
- `DS102_Colab_v6_no_pretrained.ipynb` chứa toàn bộ source, train lại từ đầu trên hosted Colab mà không cần cache cũ. Cấu hình cố định từ lần chọn OOF; notebook không chạy lại tìm kiếm CV.
- Source chính: `latent_v6.py`, `train_submit_v6.py`; thí nghiệm khác: `experiment_v6.py`. Cấu hình/đánh giá CV nằm trong `output_v6`.
- Thư viện mô hình: [scikit-learn 1.7.2 — BSD-3-Clause](https://github.com/scikit-learn/scikit-learn/blob/1.7.2/COPYING) và [CatBoost 1.2.10 — Apache-2.0](https://github.com/catboost/catboost/blob/master/LICENSE). Source pipeline được cung cấp; chưa tự phát hành dự án công khai hay gán giấy phép mới cho toàn bộ dự án.

Đã chạy toàn bộ thí nghiệm mới và fit nhánh SVD trên toàn bộ train để xuất CSV trên Colab. Notebook JSON/Python được kiểm tra cấu trúc và cú pháp. Chưa chạy lại notebook sạch từ đầu đến cuối vì sẽ fit lại các mô hình nền đã có cache; không coi kiểm tra cú pháp là thay cho chạy thực tế.

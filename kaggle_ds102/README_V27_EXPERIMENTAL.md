# V27 experimental — file để nộp thử Kaggle

File nộp: `output_v27_experimental/submission_v27_experimental.csv`.

Đã kiểm tra độc lập trên máy: 3.974 dòng, đúng hai cột `ID,spam_label`, ID duy nhất và đúng thứ tự sample, nhãn nguyên 0–3. File tải về có SHA256 `12cea6f01707e34ebd3cd660780ea78dfc7ba25ed5f6fbbbc2450d791668179b`, khớp Colab. Có 8 nhãn khác file SVD128 đang đạt Kaggle 0.77436.

## Kết quả và giới hạn

- Baseline SVD128: OOF macro-F1 **0.77494514**.
- V27: OOF macro-F1 **0.77587035**, tăng **0.00092521**.
- Chênh lệch ba fold: **+0.00306404 / −0.00103164 / +0.00070234**.
- F1 fake review toàn OOF giữ nguyên **0.68421053**; F1 fake review không có template cũng giữ nguyên.
- Dev baseline và V27 cùng **0.72634901**. Có 6 dòng dev đổi dự đoán, nhưng confusion matrix và điểm tổng bằng nhau.
- Khoảng bootstrap 95% có chứa 0. OOF/dev đã được xem nhiều lần trong dự án nên còn thiên lệch lựa chọn.

**Đây là ứng viên thử nghiệm, chưa xác nhận tăng leaderboard.** Mức tăng OOF nhỏ và chưa đạt ngưỡng chọn chính thức +0.001. Báo cáo V27 gốc vẫn giữ `rejected_on_cv`; lần full-fit/xuất CSV được ghi riêng trong `output_v27_experimental`. Không coi ứng viên này là đã vượt mọi tiêu chí ban đầu. Điểm Kaggle tốt nhất được người dùng xác nhận vẫn là **0.77436**.

## Thay đổi mô hình

V25 ExtraTrees không cải thiện. V26 tăng BoW của CatBoost text từ 700 lên 2.000 nhưng thay cả phân phối xác suất làm giảm điểm OOF tổng. Từ chẩn đoán theo lớp, V27 chỉ dùng thay đổi xác suất lớp 2:

1. Giữ baseline SVM + CatBoost text700 + SVD128.
2. Fit thêm CatBoost text2000 với cùng 500 cây, depth5, learning_rate0.1, l2=5, SqrtBalanced, seed102.
3. Tính `delta = clip(0.175*(new_cat2-old_cat2), -baseline2, baseline0)`.
4. Cộng delta vào xác suất lớp2 và trừ delta khỏi lớp0; giữ nguyên xác suất thô lớp1/3. Điều này không bảo đảm giữ nguyên nhãn argmax; kết quả được kiểm tra thực tế.
5. Nhân trọng số lớp [1,1.2,1.2,0.9] và lấy argmax.

Toàn bộ fit mới chạy trên hosted Colab, không chạy trên Mac. Không dùng pretrained, embedding pretrained, API phân loại hoặc dữ liệu ngoài bốn CSV được cung cấp. Dev không gộp vào train. Không tự nộp Kaggle. Thư viện dùng scikit-learn (BSD-3-Clause) và CatBoost (Apache-2.0); không tự công khai dự án hay gán giấy phép mới.

## Notebook và source

`DS102_Colab_v27_experimental.ipynb` chứa source đầy đủ để train lại pipeline từ bốn CSV trên hosted Colab. Notebook tự kiểm tra hash dữ liệu và có chặn local training. Có thể dùng CPU, không cần GPU. Run All, upload bốn CSV hoặc ZIP chứa chúng, rồi lấy file CSV ở cell cuối.

Các cell và module nhúng đã được kiểm tra cú pháp. Trên Colab đã kiểm tra công thức tính điểm trích trực tiếp từ source notebook bằng dự đoán thật đã lưu: nhãn dev/test khớp hoàn toàn file V27; sai khác xác suất tối đa khoảng 1.2e−7 do số học. **Chưa chạy lại toàn bộ notebook sạch từ đầu đến cuối**; CSV hiện tại dùng lại các thành phần baseline đã được cho phép và fit mới nhánh text2000. Khi fit lại mọi thành phần, có thể có sai khác số học.

Source inference được xuất thành `train_submit_v27_experimental.py`; dùng notebook làm điểm bắt đầu chạy. Source thực nghiệm: `experiment_v27.py`, `text_capacity_v26.py`, `finalize_v27_experimental.py`. Các script thực nghiệm dùng cache baseline; notebook inference chạy từ CSV và không yêu cầu cache dự đoán.

Các kiểm tra nằm trong `output_v27_experimental/local_validation.json` và `generated_source_validation.json`. Báo cáo thực thi, nguồn và fingerprint nằm trong metrics/manifest cùng thư mục.

Người dùng đã cho phép sao lưu checkpoint ngày 03/11/2026. Bản sao đã tải về `ds102_v27_prediction_checkpoints.zip` (2.780.931 bytes); checksum khớp Colab và cả 19 file dữ liệu bên trong đã được kiểm tra. Archive chứa dự đoán đã lưu, nhãn train và metadata để tiếp tục nghiên cứu khi Colab ngắt phiên. CSV bên trong khớp đúng file đã giao. Báo cáo kiểm tra: `output_v27_experimental/checkpoint_backup_verification.json`. Chỉ sao lưu và kiểm tra file; không chạy huấn luyện trên Mac.

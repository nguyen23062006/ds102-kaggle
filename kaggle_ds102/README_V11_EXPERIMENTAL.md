# Bản SVD128 thăm dò — không phải thăng hạng dưới gate V11

Người dùng tiếp tục yêu cầu tìm cách cải thiện leaderboard sau V11/V12. V13 được thử trước: context-tagged TF-IDF/SVM không tăng OOF, nên bị loại. Trong lượt này, đánh giá thêm dev và xuất một bản SVD128 để có ứng viên thăm dò thực tế, vì V11 đã tăng pooled OOF và 2/3 fold nhưng chưa đủ chắc theo gate ban đầu.

Không sửa kết quả hoặc gate V11: OOF 0.77314578→0.77494514, delta+0.00179936, fold3 âm, CI95% chứa0, gate gốc không đạt. Việc tạo CSV riêng không có nghĩa đã chứng minh cải thiện. V6 đạt0.77269 vẫn là mốc leaderboard; điểm Kaggle của SVD128 chưa biết. Chỉ người dùng tự quyết định nộp để kiểm chứng, không tự nộp Kaggle.

Follow-up: fit latent64 và128 từ đúng train, giữ seed102,800cây,.7v3+.3latent và trọng số lớp v6. Dev/test chỉ predict, không fit hoặc chọn hệ số. Báo cáo cả dev, dev bỏ exact-overlap, số nhãn test thay đổi so baseline64 fit lại và guardrail dev. File được ghi nhãn experimental; nếu dev guardrail không đạt thì phải nói rõ, không khuyến nghị như bản nâng cấp. Baseline fit lại không bit-identical submission v6 cũ.

Source chạy trên Colab: finalize_v11_experimental.py, dùng cachev3 đã được cho phép từ bốn CSV gốc. Mọi train trên hosted Colab, không pretrained/dữ liệu ngoài. Giữ nguyên source/report V11 đã bị loại trước đây; output follow-up tách riêng /content/ds102_v7_bundle/output_v11_experimental.

Notebook DS102_Colab_SVD128_experimental.ipynb nhúng đầy đủ source, có thể train toàn bộ pipeline từ bốn CSV trên hosted Colab, không cần cache dự đoán. Notebook dùng bản sao source SVD128 trong Colab, không sửa v1-v6 ở dự án. File hiện tại dùng cache v3, còn notebook fit lại v3; sai khác số học có thể làm một số nhãn khác. Notebook mới chỉ được kiểm tra cú pháp/source, chưa chạy lại trọn pipeline vì sẽ lặp lại nhiều lượt fit.

## Kết quả hoàn tất

Đã fit full64/128 trên Colab, exit0. Dev0.724384→0.726349; dev bỏ exact-overlap0.715254→0.716697. CSV đã tải về output_v11_experimental/submission_v11_experimental.csv, kiểm tra3974dòng và checksum khớp. Khác38nhãn so v6 gốc; baseline full64 tái lập đúng hash CSV v6. Đây vẫn là bản thử nghiệm, điểm Kaggle chưa biết. Không còn process train. Xem RESULTS_V13_AND_EXPERIMENTAL_SUBMISSION.md.

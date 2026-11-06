# Bàn giao DS102 cho Claude Code

Cập nhật 01/11/2026. Tài liệu này để Claude Code cùng người dùng và Codex làm dự án Kaggle. Chưa có kết nối gọi Claude trực tiếp từ phiên Codex; việc chia sẻ hiện thực hiện qua file dự án. Đây là bản mô tả trạng thái và đề xuất phân công, không phải xác nhận Claude đã tham gia hoặc đã chạy thí nghiệm.

## Mục tiêu và ràng buộc người dùng

Cải thiện F1-macro của cuộc thi https://www.kaggle.com/competitions/ds-102-r-11-cnvn bằng một pipeline có đầy đủ mã nguồn và xuất CSV đúng định dạng.

- Không dùng pretrained model, pretrained embedding hoặc API mô hình để suy luận/pseudo-label dữ liệu thi. Claude và Codex hỗ trợ phân tích/viết mã, không đóng vai trò bộ phân loại review.
- Các mô hình/thư viện phải có mã nguồn mở. Hiện dùng scikit-learn 1.7.2 (BSD-3-Clause), CatBoost 1.2.10 (Apache-2.0). Chưa tự phát hành toàn bộ dự án công khai hoặc gán giấy phép mới.
- Mọi fit/training/hiệu chỉnh mô hình mới phải chạy trên hosted Google Colab. Không train, fit vectorizer/SVD/scaler/calibrator trên Mac; người dùng đã yêu cầu vì máy nóng.
- Chỉ dùng bốn CSV đã cung cấp. Giữ dev riêng, chưa gộp train+dev vì chưa xác minh quy định. Không dùng nhãn test.
- Nhóm comment giống nhau sau chuẩn hóa trong cùng fold; fit dictionary/IDF/scaler/SVD trong phần train của từng fold.
- Giữ nguyên các submission/mốc cũ để đối chiếu. Không tự nộp Kaggle, gửi dữ liệu ra dịch vụ khác hoặc khởi động lại runtime Colab.

## Workspace và dữ liệu

Project: `/Users/ngien2306/Documents/ChatGPT/Đồ án DS102`.

Thư mục làm việc ML: `kaggle_ds102/`. Bốn file gốc ở `/Users/ngien2306/Downloads/ds-102-r-11-cnvn/`: `ds102_train.csv`, `ds102_dev.csv`, `ds102_test.csv`, `ds102_samplesubmission.csv`.

Train 14.306; dev 1.590; test 3.974. Cột chính: `comment`, `rating`, `product_name`, `description`, `category`, `categories`, `num_sold`, `num_review`; nhãn `spam_label`; ID test là `id`, ID submission là `ID`.

Nhãn: 0 no-spam; 1 fake review; 2 review on brand only; 3 non-review. Train lần lượt có 10.512 / 211 / 1.103 / 2.480 mẫu. Dev lớp 1 chỉ có 17 mẫu nên F1 lớp này dễ dao động. Có 13 comment dev trùng train sau chuẩn hóa.

CSV nộp đúng hai cột `ID,spam_label`, 3.974 dòng, nhãn nguyên 0–3, ID duy nhất theo đúng thứ tự sample. Đây là yêu cầu kỹ thuật của bài nộp.

## Trạng thái đã xác nhận

Điểm Kaggle do người dùng báo, chưa truy xuất độc lập: baseline khoảng 0,73; v2 0,75946; v3 0,76360; v4 0,76636; **v5 0,76740**. V6 đã giao CSV nhưng **chưa biết điểm Kaggle v6**; không gọi v6 là mốc leaderboard tốt nhất.

V3: TF-IDF word/char + thống kê/ngữ cảnh sản phẩm, SVM/NB-SVM kết hợp CatBoost text. Grouped OOF 0,765123; dev 0,733156.

V4: thêm 10% phiếu SVM kernel phi tuyến. OOF 0,768679; dev 0,722476; Kaggle người dùng báo 0,76636.

V5: điểm v3 nhân theo lớp `[1,1,1.2,0.9]`. OOF khi kiểm tra chéo trọng số 0,768161; dev 0,734820; Kaggle 0,76740.

V6 cuối: 70% điểm v3 + 30% CatBoost trên TF-IDF/SVD và quan hệ review/tên sản phẩm; sau đó nhân theo lớp `[1,1.2,1.2,0.9]`. SVD 64 chiều học từ train; latent review/title, tích từng chiều, hiệu tuyệt đối, cosine và thống kê. CatBoost 800 cây, depth 5, learning_rate 0.06, l2 8, SqrtBalanced, max_ctr_complexity 1, seed 102. OOF kiểm tra chéo trọng số **0,772668**; ba fold **0,778341 / 0,774778 / 0,764053**, đều cao hơn mốc v5 tương ứng. **Dev giảm xuống 0,724384**, nên chưa chứng minh tăng điểm Kaggle. Có 64 dự đoán test khác v5.

`output_v6/submission_v6.csv` đã tải từ Colab về với sự cho phép cụ thể của người dùng. Đã kiểm tra độc lập số dòng, ID, nhãn và checksum; SHA-256: `b6fc3149fdf5ca0544b51bac8f762623b202c1330ca797a35da972a0e8e25f13`.

## Thí nghiệm khác và giới hạn đánh giá

Đã thử TF-IDF char_wb/không dấu: không vượt tham chiếu. fastText v0.9.2 học từ khởi tạo ngẫu nhiên: OOF khoảng 0,68, không chọn. Stacking logistic trên dự đoán OOF cũ không cải thiện rõ. CatBoost thêm categorical product_name rất chậm, dừng an toàn trước fit fold tiếp theo, CV không đầy đủ nên không dùng kết luận thắng/thua.

Lần v6: kernel có hiệu chỉnh xác suất đạt CV 0,771125 nhưng dev 0,723109. Dò review gần trùng đạt CV 0,770014, dev 0,734820 nhưng chỉ đổi 3 nhãn test. SVD/CatBoost có CV cao nhất trong lần này nhưng dev vẫn giảm.

OOF được dùng nhiều lần để chọn mô hình. Hiệu chỉnh trọng số trên các fold OOF không tương đương nested CV đầy đủ: các dự đoán dùng ở tầng hiệu chỉnh có phụ thuộc vào các fit tầng nền. Dev đã được xem nhiều lần và dùng so sánh các ứng viên; không coi nó là test độc lập. Cần phản biện nguy cơ overfit quy trình chọn mô hình, không chỉ tìm thêm hệ số làm OOF tăng rất nhỏ.

## File nên đọc theo thứ tự

1. `CONSTRAINTS.md`, `leaderboard_history.json`, `README_V5.md`, `README_V6.md`.
2. `output_v5/metrics_v5.json`, `output_v6/metrics_v6.json`, `output_v6/cv_results.json`, `output_v6/chosen_config.json`.
3. `train_submit.py`, `improved_features.py`, `train_submit_v2.py`, `catboost_v3.py`.
4. `selection_v5.py`, `experiment_v6.py`, `latent_v6.py`, `train_submit_v6.py`.
5. `DS102_Colab_v6_no_pretrained.ipynb` và `build_notebook_v6.py`.

Notebook v6 chứa toàn bộ source và cấu hình, train lại từ đầu trên Colab không cần cache. Các cell Python/JSON/source nhúng đã kiểm tra cú pháp và đồng bộ. Thí nghiệm mới/full fit nhánh mới chạy thật trên Colab; notebook sạch chưa chạy lại end-to-end vì baseline đã có cache.

Các output/cache/venv bị gitignore; một worktree Git mới có thể không chứa các file này. Kiểm tra file hiện diện thực tế trước khi kết luận thiếu dữ liệu. Đọc file gốc theo đường dẫn project khi người dùng đã mở/cấp quyền thư mục đó.

Một số script thí nghiệm đời đầu (`experiment_v2.py`, `experiment_v3.py`, `experiment_linear_v3.py`) có code fit ở top-level. Chỉ đọc source/AST trên Mac, không import hoặc chạy chúng để khám phá dự án.

## Phân công đề xuất cho Claude Code

Đầu tiên thực hiện một lượt phản biện độc lập, chưa chạy training:

1. Kiểm tra nguồn của OOF, grouping, fit/transform và calibration; chỉ ra nơi đánh giá có thể lạc quan hoặc sai khác giữa CV và inference.
2. Phân tích vì sao các nhánh mới tăng CV nhưng giảm dev. Phân biệt lỗi triển khai, độ lệch phân phối, nhiễu nhãn, mất cân bằng và phương sai lớp ít mẫu; dẫn chứng từ source/metrics thay vì đoán.
3. Đề xuất tối đa ba thí nghiệm ưu tiên, mỗi thí nghiệm có giả thuyết, thay đổi cụ thể, chi phí Colab, cách đánh giá và điều kiện giữ/loại.
4. Nếu viết code mới, lưu file riêng trong `claude_review/` bên dưới `kaggle_ds102/`; không ghi đè source/output v1–v6. Dùng guard hosted Colab trước mọi fit và seed rõ ràng.

Ghi kết quả vào `kaggle_ds102/CLAUDE_REVIEW.md` để Codex đọc, kiểm chứng rồi tích hợp/chạy trên Colab. Chỉ một phía điều khiển runtime Colab tại một thời điểm; phối hợp bàn giao trước khi nhận quyền điều khiển. Không chia sẻ token ghép nối MCP qua file/chát.

Tài liệu kỹ thuật: https://code.claude.com/docs/en/quickstart ; https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.TruncatedSVD.html .

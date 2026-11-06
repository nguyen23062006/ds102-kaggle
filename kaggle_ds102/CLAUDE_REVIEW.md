# Phản biện độc lập pipeline DS102 (Claude, 01/11/2026)

Phạm vi: đọc source/metrics v1–v6 và chạy chẩn đoán **chỉ đọc** trên Mac (stdlib + numpy: đếm, argmax, bootstrap trên dự đoán đã cache và CSV gốc). **Không fit** mô hình, vectorizer, SVD, scaler hay calibrator nào trên Mac. Code mới nằm trong `claude_review/`; không sửa file nào của v1–v6.

## 1. Kết luận chính

1. **Không thấy lỗi triển khai rò nhãn.** Vectorizer/SVD/scaler/NB-ratio được fit trong phần train của từng fold (`fit_predict`, `LatentFeatures.transform(fit=True)`), dev/test chỉ được transform. Cách chọn trọng số lớp `cross_weight` đúng là chọn trên 2 fold rồi chấm fold còn lại.
2. **Khoảng cách OOF–dev gần như chỉ đến từ lớp 1 (fake review), và nguyên nhân là các mẫu viết theo template gần trùng bị chia sang nhiều fold.** Các fold chỉ gộp nhóm theo *comment trùng khít sau chuẩn hóa*, nên các review fake gần trùng nhau vẫn rơi vào các fold khác nhau. Khi chấm OOF, mô hình gần như "nhớ" được các template này. Dev có ít template như vậy hơn nhiều.
3. **Hiệu số dev giữa v5 và v6 (−0,0104) nằm trong vùng nhiễu.** Hiệu số này tương đương khoảng chục dòng dev đổi nhãn. Mức tăng CV +0,0045 cũng nhỏ so với nhiễu do đã chọn trong rất nhiều ứng viên. Cả hai con số đều chưa đủ để kết luận v6 tốt hơn hay kém hơn v5.
4. Các nhánh mới làm CV tăng (kernel SVM, neighbours, SVD latent) đều là bộ học **dựa trên độ tương đồng**. Chúng tăng điểm đúng ở phần template mà OOF thổi phồng. Kèm theo đó, lượt chọn trọng số trên OOF đẩy hệ số lớp 1 từ 1,0 (v5) lên 1,2–1,4 (v6, kernel). Trên dev, v6 có thêm 1 false positive lớp 1 và không có thêm true positive nào.

## 2. Bằng chứng

Số liệu nằm trong `claude_review/diagnostics*.json`. Có thể chạy lại bằng `python3 claude_review/diagnose_*.py`; script chỉ đọc dữ liệu, chạy dưới 15 giây.

### 2.1 Khoảng cách OOF–dev nằm ở lớp 1

| F1 theo lớp | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| OOF v3-ref (3 fold) | 0,935–0,938 | **0,638–0,671** | 0,573–0,615 | 0,876–0,879 |
| Dev v5 | 0,940 | **0,519** | 0,597 | 0,883 |

Lớp 1 chênh khoảng 0,13. Chia cho 4 lớp được khoảng 0,033 điểm macro, gần bằng toàn bộ khoảng cách OOF 0,768 so với dev 0,735. Ba lớp còn lại giữa OOF và dev gần như bằng nhau.

### 2.2 Template lớp 1 bị chia sang nhiều fold

Một dòng được coi là có "template" nếu Jaccard trên tập token với một review lớp 1 khác ≥ 0,6. Chỉ so với review nằm trong phần dữ liệu dùng để fit:

| | OOF (so với các fold khác) | Dev (so với toàn bộ train) |
|---|---|---|
| Tỷ lệ positive lớp 1 có template | **51%** (108/211) | **24%** (4/17) |
| Median Jaccard tới review lớp 1 gần nhất | 0,65 | 0,23 |

Recall lớp 1 của v5, tách theo hai nhóm (`diagnostics_class1.json`):

| | Có template | Mới (không có template) |
|---|---|---|
| OOF | 0,94 (101/108), F1 0,83 | **0,31** (32/103), F1 0,40 |
| Dev | 0,75 (3/4) | **0,31** (4/13), F1 0,44 |

Trên các mẫu lớp 1 mới, mô hình làm **như nhau** ở OOF và dev. OOF cao hơn chỉ vì trong OOF có khoảng một nửa positive là template dễ nhận. Đây là độ lệch thành phần giữa train-OOF và dev. Đây không phải lỗi code, và cũng không phải mô hình suy giảm trên dev.

Với test, tỷ lệ dòng gần trùng template lớp 1 trong train là 61/3974 = 1,5%, gần với dev (23/1590 = 1,4%) và OOF (234/14306 = 1,6%). Chưa biết nhãn test nên **không xác định được** test giống OOF hay giống dev ở thành phần lớp 1. Kaggle 0,767 gần OOF hơn dev, nhưng public LB chỉ là một phần của test và lớp 1 trong đó có khoảng vài chục mẫu.

### 2.3 Dev quá nhiễu để phân định v5 và v6

- Bootstrap dev của v5: SD = **0,032**, khoảng tin cậy 95% là [0,666; 0,794].
- Tách mức giảm −0,0104 của v6 theo lớp (từ confusion matrix): lớp 0 −0,0010; lớp 1 −0,0046 (đúng 1 false positive); lớp 2 −0,0031; lớp 3 −0,0018. Tổng cộng chỉ khoảng 10–15 dòng đổi nhãn.
- Chỉ đổi riêng hệ số lớp trên cùng mô hình v3, trong grid 48 điểm: dev dao động 0,718–0,736 và OOF dao động 0,760–0,769. Spearman giữa thứ hạng OOF và thứ hạng dev chỉ là **0,42**. Như vậy riêng việc đổi hệ số đã làm dev biến động mạnh hơn hiệu số v5–v6.

### 2.4 Quá trình chọn mô hình lạc quan

- `cross_weight` chỉ lồng (nested) cho bước chọn hệ số lớp. Họ mô hình, alpha (0,15/0,3/0,5), số cây (400/800), nhiệt độ… đều được chọn trên **cùng** OOF. Riêng lần v6 đã có hơn chục cấu hình (kernel, neighbours, latent × alpha × số cây), mỗi cấu hình lại có 48 tổ hợp hệ số. Người thắng sẽ chịu "winner's curse": mức tăng theo fold so với v5 là +0,0018 / +0,0106 / +0,0006, tức gần như toàn bộ đến từ fold 2.
- Quy tắc "cần tăng ≥ 0,002 CV" có ngưỡng thấp hơn độ dao động giữa các fold (độ chênh giữa các fold khoảng 0,01–0,014).

### 2.5 Lệch thang điểm giữa CV và inference (ảnh hưởng nhỏ)

Nhiệt độ 3, tỷ lệ 50/50 với CatBoost và các hệ số lớp được chọn trên điểm của mô hình fit với 2/3 dữ liệu. Lúc inference, mô hình được fit trên 100% dữ liệu. Với cùng C, LinearSVC fit trên nhiều dữ liệu hơn sẽ cho margin lớn hơn: SD margin OOF 1,357, dev 1,485 (+9%), test 1,442. CatBoost latent dùng cố định 800 cây trên tập dữ liệu lớn gấp 1,5 lần. Entropy của điểm kết hợp gần như không đổi (0,407 so với 0,401), nên đây là rủi ro phụ, không phải nguyên nhân chính.

### 2.6 Những điểm đã kiểm tra và loại trừ

- **Tên sản phẩm không phải nguồn lệch phân phối:** 99,4% dòng dev và test có `product_name` đã xuất hiện trong train. **Không** nên gộp nhóm theo sản phẩm, vì làm vậy sẽ khiến CV bi quan hơn so với test.
- **Mất cân bằng nhãn:** tỷ lệ dự đoán lớp 1 trên OOF là 1,34% (thực tế 1,47%), trên dev 0,63% (thực tế 1,07%), trên test 1,13%. Dev dự đoán thiếu lớp 1 phù hợp với giải thích "dev ít template" ở trên. Chưa thấy dấu hiệu cần tăng thêm hệ số lớp.
- **Nhiễu nhãn:** trong dev, 23 dòng giống template lớp 1 thì 19 dòng mang nhãn khác (thường là review ngắn, chung chung). Điều này gợi ý có cả review ngắn dễ trùng lẫn nhãn không nhất quán. Chưa đủ bằng chứng để coi là nguyên nhân chính.

## 3. Đề xuất: ba thí nghiệm ưu tiên (chạy hosted Colab)

Quy ước chung: seed 102 (E1 chạy thêm seed 202 và 302 nếu còn thời gian), 3 fold `StratifiedGroupKFold`. Thay nhóm theo comment trùng khít bằng **cụm gần trùng**: thành phần liên thông của "trùng khít" hoặc "Jaccard ≥ 0,6 và cả hai có ≥ 4 token". Dev chỉ là tín hiệu phụ, dùng để kiểm tra không tệ hơn quá nửa SD. Không chọn tham số theo dev.

### E1. Chấm lại v5 và v6 bằng CV gộp theo cụm gần trùng (ưu tiên cao nhất)

- **Giả thuyết:** mức tăng CV của v6 chủ yếu đến từ template lớp 1. Khi gộp theo cụm, hiệu v6−v5 về khoảng 0 hoặc âm, và OOF lớp 1 tiến gần dev.
- **Bước 0, chi phí ≈ 0:** `claude_review/e1_step0_rescore_oof.py`. Chấm lại OOF đã có trên Colab (`colab_reference.npz` và `/content/ds102_v6_latent/oof.npz`) theo hai nhóm template/mới, kèm bootstrap theo cụm. Bước này không fit gì.
- **Bước 1:** `python claude_review/cluster_cv.py --seed 102`. Chạy lại SVM/NB-SVM, CatBoost text và latent CatBoost trên fold gộp theo cụm, công thức giống hệt v5/v6. Mỗi fold được lưu checkpoint và có thể chạy tiếp nếu bị ngắt.
- **Chi phí:** khoảng một lượt CV của v3 cộng một lượt CV latent của v6, cộng thêm phần dự đoán dev/test cho mỗi fold. Đây là ước lượng, chưa đo; Codex nên đối chiếu với thời gian log của v6. Mỗi seed thêm tốn chi phí tương đương.
- **Giữ v6 chỉ khi:** hiệu trung bình ≥ +0,003, ít nhất 2/3 fold dương, bootstrap theo cụm có q05 > 0, **và** F1 lớp 1 nhóm "mới" không giảm quá 0,02. Nếu không đạt, giữ v5 làm ứng viên chính. Đồng thời dùng CV gộp theo cụm làm giao thức chọn mô hình mặc định nếu F1 lớp 1 trên OOF giảm về gần dev, tức tỷ lệ template trong OOF giảm.
- **Lưu ý:** script dừng nếu cụm lớn nhất vượt 2% dữ liệu do bị nối chuỗi. Khi đó tăng `--thr` lên 0,7.

### E2. Giảm trọng số mẫu template khi huấn luyện

- **Giả thuyết:** mô hình đang học thuộc chiến dịch template thay vì học tín hiệu fake chung. Giảm trọng số các cụm lớn sẽ tăng recall lớp 1 trên mẫu mới, vốn chỉ khoảng 0,31 ở cả OOF và dev. Đây là phần đáng cải thiện nhất cho F1-macro.
- **Thay đổi:** `cluster_cv.py --dedup-weight`. Trọng số mẫu là `1/sqrt(kích thước cụm)`, chuẩn hóa lại để tổng trọng số mỗi lớp không đổi. Như vậy SqrtBalanced/balanced giữ nguyên ý nghĩa. Áp cho cả 3 thành phần; NB-ratio giữ nguyên để tách riêng tác động. Không đổi đặc trưng hay siêu tham số.
- **Chi phí:** bằng một seed của E1.
- **Giữ khi:** so với E1 trên cùng fold, macro OOF ≥ +0,003, F1 lớp 1 nhóm "mới" ≥ +0,03, F1 lớp 1 nhóm template giảm không quá 0,05, và dev (lấy trung bình các fold) không thấp hơn v5 quá 0,015. Nếu không đạt thì bỏ.

### E3. Inference bằng trung bình các mô hình fold thay vì refit toàn bộ train

- **Giả thuyết:** nhiệt độ, tỷ lệ blend và hệ số lớp được chọn trên thang điểm của mô hình fit với 2/3 dữ liệu. Lấy trung bình K mô hình fold để dự đoán dev/test sẽ giữ đúng thang điểm đó (bỏ lệch margin khoảng 9%) và giảm phương sai.
- **Thay đổi:** không cần fit thêm. `cluster_cv.py` đã lưu dự đoán dev/test của từng fold và xuất `submission_{v5,v6}_foldavg.csv` trong `/content/claude_review_out/<tag>/`. Các file này không ghi đè `output_v*` và **không tự nộp**.
- **Chi phí:** 0 ngoài E1/E2.
- **Đánh giá:** so sánh SD margin của dev lấy trung bình fold với OOF (kỳ vọng xấp xỉ nhau), dev macro so với bản refit, và số nhãn test khác so với v5. **Giữ khi** dev không thấp hơn bản refit quá 0,01. Đây là lựa chọn giảm rủi ro chứ không phải để tăng điểm, vì CV không phân biệt được hai cách inference này.

## 4. Không khuyến nghị lúc này

- Thêm hệ số lớp, alpha, ensemble hoặc kernel mới để vắt thêm khoảng 0,002 OOF trên giao thức gộp theo comment. Như mục 2.2–2.4 đã cho thấy, các mức tăng này phần lớn đo khả năng nhớ template.
- Chọn giữa v5 và v6 dựa trên dev, vì hiệu số chỉ bằng khoảng 1/3 SD.
- Gộp nhóm theo sản phẩm, vì CV sẽ bi quan sai lệch so với test.

## 5. File trong `claude_review/`

| File | Chạy ở đâu | Fit? |
|---|---|---|
| `diagnose_cached.py`, `diagnose_neardup.py`, `diagnose_class1.py` | Mac (đã chạy) | Không: đếm/argmax/bootstrap trên cache |
| `diagnostics*.json` | — | Kết quả ở trên |
| `review_common.py` | dùng chung | Không: hàm Jaccard, cụm, metric, bootstrap |
| `e1_step0_rescore_oof.py` | Colab (cần OOF latent) | Không |
| `cluster_cv.py` | **Chỉ Colab**, có `hosted_guard()` trước mọi fit | Có (E1/E2/E3) |

Cách chạy trên Colab: đặt các module v1–v6 trong `SRC/` và thư mục này ở `SRC/claude_review/`. Script tự thêm cả hai đường dẫn vào `sys.path`. Thư viện vẫn là scikit-learn 1.7.2 và CatBoost 1.2.10; không dùng pretrained.

**Chưa kiểm chứng:** `cluster_cv.py` và `e1_step0_rescore_oof.py` mới qua kiểm tra cú pháp bằng AST. Chưa chạy thật vì Mac thiếu scipy/pandas và không được phép fit trên Mac. Codex cần chạy thử bước 0 trước, rồi mới chạy E1.

# H3 nested: chọn trọng số lớp bằng nested CV đầy đủ

Trạng thái: **mã mới chỉ được kiểm tra cú pháp trên Mac** (AST + `py_compile`). Chưa chạy, chưa fit, chưa có kết quả. Codex sẽ rà soát và chạy trên hosted Colab. Các file cũ trong `antigravity_review/` giữ nguyên, kể cả `h3_threshold_calibration.py`.

## 1. Kiểm tra lại H3 cũ và đính chính

`antigravity_review/h3_threshold_calibration.py` **không phải nested CV**, và con số 0,76637 trong `ANTIGRAVITY_REVIEW.md` không được dùng để đánh giá. Có ba lý do:

1. **Rò nhãn qua mô hình.** Khi chọn trọng số cho fold k, script dùng OOF của các fold j ≠ k. Nhưng mỗi dự đoán OOF ở fold j lại đến từ mô hình đã fit trên dữ liệu có fold k. Vì vậy nhãn của fold k vẫn ảnh hưởng tới xác suất dùng để chọn trọng số cho chính fold k.
2. **Sai giao thức và sai baseline.** Script dùng `colab_reference.npz` (fold nhóm theo comment trùng khít, chỉ có thành phần v3). Đây không phải các fold cụm gần trùng, và cũng không phải V11 có nhánh latent SVD128.
3. **Diễn giải quá mức.** Mức chênh −0,0018 so với trọng số cố định không chứng minh được "winner's curse ở fold 2".

Tôi cũng đính chính một số nhận định trong `ANTIGRAVITY_REVIEW.md`, đồng ý với `antigravity_h1_review_checks.json`:

- Không biết nhãn test, cũng không biết test chia public/private thế nào. Việc tôi đọc các review test bị đổi nhãn ở V27 **không** cho biết nhãn thật và không giải thích được mức giảm leaderboard. Câu "V11 đã đoán đúng lớp 2" là sai.
- Định nghĩa chính thức của lớp 2 không phải "vận chuyển/chưa dùng". Logistics/deferral chỉ có thể là đặc trưng dự báo.
- Khoảng tin cậy chứa 0 là bất định, không phải "bất khả thi về thống kê". Ở V57, dự đoán lớp 1 giữ nguyên, nên hiệu F1 lớp 1 bằng 0.
- "Đổi hơn 30 nhãn test" không phải tiêu chí ý nghĩa thống kê. Tiêu chí này bị bỏ.

## 2. Thiết kế (chốt trước khi chạy)

| Thành phần | Thiết lập cố định |
|---|---|
| Outer | Fold cụm gần trùng giống V53/V56/V57/V60: `groups.npz` của V53, `StratifiedGroupKFold(3, shuffle=True, random_state=2053)`. Script kiểm tra chỉ số khớp `indices.npz` của V53. |
| Xác suất outer | V11 recipe gốc `.35·SVM + .35·CatText + .30·latent`, nạp lại cache `svm/cat/latent.npz` của V53 và kiểm tra `manifest_id`. Script assert baseline pooled = 0,7240841964761913. |
| Inner | Chỉ trong outer-train: `StratifiedGroupKFold(3, shuffle=True, random_state=4053+f)` trên cùng cluster id. Mỗi inner split fit lại **từ đầu** trên inner-train: FeaturesV2 TF-IDF, SVD128, LinearSVC + NB-SVM, CatBoost text 500, latent CatBoost 800. Tham số giống V53 (`experiment_v53.fit_svm`, `make_cat`, `tree`). |
| Không gian tìm kiếm | w0 = 1; w1 ∈ {0,8; 1,0; …; 2,0} (7 giá trị); w2 ∈ {0,8; 0,9; …; 1,6} (9 giá trị); w3 ∈ {0,7; …; 1,1} (5 giá trị). Tổng 315 điểm, trong đó có trọng số V11 (1; 1,2; 1,2; 0,9). |
| Quy tắc chọn | Chọn điểm có macro-F1 cao nhất trên inner OOF. Nếu hòa, chọn điểm có L1 gần trọng số V11 nhất, sau đó theo thứ tự từ điển. Không làm mượt, không dùng quy tắc phụ. |
| Đánh giá | Trọng số được ghi ra `selected_weights.json` **trước** khi đọc nhãn outer-validation. Sau đó áp đúng một lần lên xác suất outer-validation. Baseline dùng cùng xác suất đó với trọng số V11. |
| Dev/test | Không đọc, không fit, không chấm. Không gộp dev vào train. Không tạo CSV. |

### Tiêu chí (giống gate cluster của V53–V60, không nới)

- Delta macro pooled ≥ +0,003.
- Ít nhất 2/3 fold dương.
- Fold tệ nhất ≥ −0,005.
- Hiệu F1 lớp 1 nhóm novel ≥ 0 (template theo Jaccard ≥ 0,6 với lớp 1 ở outer-train).
- Paired cluster bootstrap 2000 lần, seed 2053, q025 > 0.

Nếu đạt đủ gate, script chỉ tính thêm trọng số cuối bằng cùng quy tắc trên outer OOF pooled và ghi vào báo cáo. Script không áp trọng số này lên dev/test. Việc đánh giá dev, nếu có, là bước riêng và chỉ mang tính chẩn đoán.

### Lưu ý diễn giải

- Trọng số V11 được chọn trong lịch sử dự án trên chính các dòng train này. Vì vậy so sánh đang **thiên về baseline**.
- Inner model chỉ fit trên khoảng 44% dữ liệu, outer model khoảng 67%. Thang xác suất có thể lệch nhau (Claude đã ghi nhận margin SVM tăng khi fit trên nhiều dữ liệu hơn). Lúc triển khai thật cũng có cùng độ lệch: chọn trên OOF rồi áp cho mô hình fit trên toàn bộ train. Đây là một phần của thứ đang được đo, không phải lỗi.
- Lượt chạy pseudo-nested cũ cho kết quả âm. Kỳ vọng tiên nghiệm là H3 tăng ít hoặc không tăng. Nếu bị loại thì giữ V11.

## 3. Chi phí dự kiến (ước lượng, chưa đo)

- Gồm 9 inner partition, mỗi partition khoảng 6.360 dòng fit và 3.180 dòng dự đoán, chạy 3 thành phần. Outer dùng lại cache nên không fit thêm.
- Căn cứ: V53 mất khoảng 2.931 giây cho 3 fold cluster với 4 thành phần, tức khoảng 980 giây/fold trên khoảng 9.540 dòng, đã gồm NMF + CatBoost topics. Bỏ topics và giảm còn khoảng 2/3 số dòng thì ước tính **khoảng 6–8 phút mỗi inner partition**.
- Tổng ước tính **khoảng 55–75 phút** trên CPU 2 lõi của Colab, cộng khoảng 1–3 phút cho bootstrap và chọn grid (315 × 3 lần argmax trên khoảng 9,5k dòng, không đáng kể). Nếu cache V53 thiếu và phải dùng `--allow-outer-refit`, cộng thêm khoảng 30–45 phút.
- Mỗi inner partition có checkpoint riêng (`outer{f}/inner{k}.npz`, ghi atomic, kiểm tra manifest và chỉ số), nên có thể chạy tiếp nếu bị ngắt.

## 4. File

| File | Vai trò | Trạng thái trên Mac |
|---|---|---|
| `h3_selection.py` | Grid cố định, macro-F1, quy tắc chọn (chỉ dùng NumPy) | Chỉ kiểm tra cú pháp |
| `h3_nested_cv.py` | Driver Colab: inner OOF → chọn → chấm outer, gate, bootstrap. Có `guard()` trước khi import ML | Chỉ kiểm tra cú pháp |
| `test_h3_selection.py` | Unit test cho quy tắc chọn: grid, công thức macro, hòa → baseline, tính tất định | Chỉ kiểm tra cú pháp, **chưa chạy** |

SHA256 lúc bàn giao:

- `h3_nested_cv.py`: `f2cdda43511e4e1eb98be69481974d3135d9bfb8f4655da0e2713f4025686c1b`
- `h3_selection.py`: `725573d194374b84259a163dcec70be939a5e923e6393f831fbb6837af4a52d1`
- `test_h3_selection.py`: `7d2cf1e4dbe293a49834db1d6ce6a54babc34f9838786422622b34ebfd6201a0`

## 5. Cách chạy trên Colab (dành cho Codex)

Yêu cầu: có `/content/ds102_v53/source`, `/content/ds102_v53/run` (gồm `run_manifest.json`, `groups.npz`, `cluster/fold{0,1,2}/{indices,svm,cat,latent}.npz`) và `/content/ds102_v53/inputs`. Dùng venv của V53 để phiên bản thư viện khớp manifest.

```bash
cd /content/antigravity_review/h3_nested
PY=/content/ds102_v53/venv/bin/python
$PY test_h3_selection.py                                   # unit test của quy tắc chọn
$PY h3_nested_cv.py --inputs /content/ds102_v53/inputs \
    --output /content/ds102_h3_nested/smoke --smoke          # 600 dòng, 20 cây, SVD16, không đủ điều kiện để chọn
$PY h3_nested_cv.py --inputs /content/ds102_v53/inputs \
    --output /content/ds102_h3_nested/run                    # lượt chạy thật
```

Kết quả nằm ở `metrics_h3_nested.json` (gate, fold delta, trọng số chọn và điểm inner theo từng fold, số nhãn outer bị đổi), `outer_oof.npz`, và `outer{f}/selected_weights.json`.

## 6. Điểm đề nghị Codex kiểm tra rò rỉ

1. Assert `inner rows must exclude outer validation` và kiểm tra không có cluster chung giữa inner-train và inner-val.
2. `fit_components` chỉ nhận `fit_frame`. `predict_frame` đã bỏ `spam_label`, có assert.
3. `selected_weights.json` được ghi trước khi tính bất kỳ metric nào trên `y[va]`.
4. Baseline pooled khớp V53 (assert 1e-9). Nếu lệch, nghĩa là fold hoặc xác suất không giống thiết lập đối chứng.
5. `load_data` có đọc file dev/test, nhưng script không `prepare`, fit hay chấm chúng.

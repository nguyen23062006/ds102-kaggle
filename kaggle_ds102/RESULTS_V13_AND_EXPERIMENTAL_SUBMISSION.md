# Kết quả V13 và CSV SVD128 thăm dò

V13: ba lượt fit context-tagged TF-IDF/LinearSVC trên hosted Colab. OOF0.7731457846→0.7729950586, delta-0.0001507259. Delta fold +0.0020202962 / -0.0038148727 / +0.0014513025. Novel fake F1 giảm0.01834734. Không đạt gate macro/novel/bootstrap; loại nhánh, không fit full V13.

Follow-up riêng: fit full-train latent64 và128 theo recipe V11 để có ứng viên thử nghiệm thực tế. Không sửa quyết định/gate V11 cũ; V11 tăng pooled OOF0.00179936 nhưng CI95% chứa0 và không đạt ngưỡng delta0.002. Do đó CSV SVD128 là bản thăm dò để người dùng cân nhắc nộp, không phải cải thiện leaderboard đã chứng minh.

Dev full64:0.7243841306; SVD128:0.7263490140, delta+0.0019648835. Dev bỏ13exact-overlap còn1577dòng:0.7152543900→0.7166965127. Dữ liệu dev/test không được fit; không dùng hệ số mới theo điểm dev. Tất cả model/source nguồn mở, không pretrained, không dữ liệu ngoài, không train trên Mac. Follow-up fullfit khoảng598giây, exit0.

Đã tạo và tải về output_v11_experimental/submission_v11_experimental.csv, đủ3974dòng, đúng hai cột ID,spam_label, đúng thứ tự sample, ID duy nhất, nhãn nguyên0–3. SHA25605bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059, khớp Colab và Mac. Khác38nhãn so CSV v6 gốc. Baseline full64 tái lập đúng SHA256 CSV v6 gốc b6fc3149fdf5ca0544b51bac8f762623b202c1330ca797a35da972a0e8e25f13. Việc tái lập full-inference không giải thích được khác biệt OOF cũ/fit lại đã ghi ở V10; không tuyên bố toàn bộ CV tái lập bit-for-bit.

Điểm Kaggle SVD128 chưa biết. V6 gốc0.77269 vẫn là mốc đã biết; chưa nộp tự động. Thử nhiều lần trên OOF/dev làm selection bias, vì vậy các mức tăng nhỏ chỉ là tín hiệu thử nghiệm. Có thể tăng hoặc giảm leaderboard sau khi nộp.

Notebook source-complete: DS102_Colab_SVD128_experimental.ipynb, train lại toàn pipeline từ4CSV trên hosted Colab, không phụ thuộc cache. Đã kiểm tra AST các cell/module; chưa chạy lại notebook trọn vẹn vì CSV hiện tại đã được tạo với cache v3 được phép dùng. Fresh-fit v3 có thể gây khác biệt số học so CSV dùng cache. Source archive: SVD128_experimental_source.zip.

Báo cáo: output_v13_experiment/report.json; output_v11_experimental/metrics_v11_experimental.json, manifest.json, local_validation.json, baseline_reproduction.json. Mọi process V13/finalize đã kết thúc; kiểm tra cuối pgrep rỗng.

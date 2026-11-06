# V19 — thay nhánh CatBoost latent bằng LightGBM, giữ tỷ lệ70/30

Đề xuất sau kết quả fold1 của V18, trước khi xem đủ V18. Không gọi đây là validation độc lập. Một công thức cố định, không grid trọng số:

- Baseline tốt nhất:0.7v3 +0.3CatBoostSVD128.
- V18:0.8baseline+0.2LightGBM, tương đương0.56v3+0.24CatBoostSVD128+0.2LightGBM.
- V19:0.7v3+0.3LightGBM, giữ tỷ trọng text/base so với baseline và thay thuật toán cho nhánh latent.

Giữ weights[1,1.2,1.2,0.9], cùng LightGBM600cây/31leaves/lr0.05 và train-onlySVD128 của V18. Tái sử dụng prediction checkpoints khi manifestID/tr/va/source hash/versions khớp; nếu thiếu fold mới fit cùng cấu hình và có thể sao chép feature cache sang thư mục mới. Không sửa source/report/model files của V18. Chạy sau V18 để tránh tranh CPU hoặc fit full trùng.

Gate giữ nguyên:OOF+0.001,>=2/3fold dương,worstfold>=-0.004,novel fakeF1 không giảm quá0.01. Dừng sớm nếu worstfloor không thể sửa. Bootstrap báouncertainty,khônggatep-value. QuaCV mới fulltrain,dev không giảm quá0.005 mới xuấtCSV. FullLightGBM của V18 nếu có được tái sử dụng với manifest kiểm chứng, không fit lại vô ích. BaselineCSV phải khớp bản Kaggle0.77436.

Không pretrained/dữ liệu ngoài, không gộp dev, mọi fit hostedColab. Lựa chọn sau một phần kết quả và repeatedOOF/dev use tạo selection bias; không coi tăng CV là bảo đảm tăng leaderboard. Source experiment_v19.py dùng lightgbm_latent_v18.py. Runtime output /content/ds102_v7_bundle/output_v19_experiment.

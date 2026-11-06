# V18 — LightGBM leaf-wise trên cùng SVD128

Mốc tốt nhất do người dùng báo: SVD128 experimental Kaggle0.77436. Giữ dev riêng vì người dùng chưa rõ quy định gộp dev; không chạy train_final_train_dev.py.

Giả thuyết mới: LightGBM có cách phát triển cây theo từng lá, khác cây đối xứng của CatBoost đang dùng. Cùng ma trận review/title SVD128 và numeric/context có thể tạo ranh giới bổ sung. Không tuyên bố đảm bảo hiệu quả; tài liệu LightGBM cũng lưu ý leaf-wise có thể overfit dữ liệu nhỏ. Tài liệu chính thức: https://lightgbm.readthedocs.io/en/stable/Features.html ; classifier API https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html . Runtime đang có LightGBM4.6.0, licenseMIT.

Một cấu hình cố định trước kết quả:600estimators,lr0.05,31leaves,min_child_samples25,min_split_gain0.02,L1=0.1,L2=10,row/columnsubsample0.85,baggingfrequency1,seed102,2threads,CPU deterministic/force_col_wise. Class weights inverse-sqrt frequency tính từ fitting labels. Không chọn số cây bằng dev/validation. SVD128 và vocabulary fit riêng từ training fold, category/rating levels cũng từ training fold; unknown category ở validation/test chuyển missing. Không pretrained/dữ liệu ngoài.

Trộn20%LightGBM+80%baselineSVD128, giữ weights[1,1.2,1.2,0.9]. Dùng cùng y/fold/basecache, kiểm tra hash/versions. Lưu numeric/category matrices để không tính lại SVD khi resume; NPZ không pickle, có feature fingerprint và prediction checkpoint có tr/va indices+manifestID.

Sàng lọc giống các thử nghiệm mới nhất: OOF+0.001,>=2/3fold dương,worstfold>=-0.004,novel fakeF1 delta>=-0.01. Dừng sớm nếu một fold đã vi phạm worstfloor không thể khắc phục. Bootstrap báo uncertainty, không làm gate significance. Nếu qua CV mới fit fulltrain,dev không giảm quá0.005 mới xuất CSV. CSVbaseline phải khớp SHA bản đã đạt0.77436. Không tự submit Kaggle.

Mọi fit trên hosted Colab; chỉ đọc/ghi source,AST trên Mac. Preflight trên Colab kiểm tra tương thích LightGBM, unknown category và probability shape/finite/row sum. Repeated adaptive OOF/dev selection vẫn có bias; không coi tăng OOF là chắc chắn tăng leaderboard.

Source lightgbm_latent_v18.py và experiment_v18.py; output /content/ds102_v7_bundle/output_v18_experiment. Mô hình/categorical levels/feature matrices lưu trên Colab; chưa phát hành trọng số/dữ liệu ra ngoài.

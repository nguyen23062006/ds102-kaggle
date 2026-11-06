# V26 — tăng khả năng biểu diễn của nhánh CatBoost text

Giữ nguyên từ vựng unigram/bigram, occurrence_lower_bound3, dictionaries40000, tokenizer theo dấu cách, các đặc trưng thống kê và category/rating của nhánh CatBoost text v3. Thay đúng BoW:top_tokens_count700 thành2000. Giữ NaiveBayes/BM25,500cây,depth5,learning_rate0.1,l2=5,SqrtBalanced,seed102; dùng hai luồng trên hosted Colab.

Theo tài liệu CatBoost, top_tokens_count giới hạn số token BoW được tạo đặc trưng; NaiveBayes vàBM25 dùng tính toán online để hạn chế target leakage. Nguồn: https://catboost.ai/docs/en/references/text-processing__feature_calcers . Mọi dictionary/text-calcer của mô hình mới được học trong train của fold; không dùng dữ liệu hoặc mô hình pretrained.

Baseline đang đạt Kaggle0.77436 có hệ số0.35*SVM_probability+0.35*CatText700+0.30*latent128. V26 dùng0.35*SVM_probability+0.175*CatText700+0.175*CatText2000+0.30*latent128; giữ trọng số lớp[1,1.2,1.2,0.9]. Đổi riêng một nửa nhánh CatText thay vì tái cân bằng cả ensemble.

Cache dự đoán từng thành phần colab_components_v5.npz là file người dùng đã cho phép chuyển. Đã chuyển sang runtime hiện tại và kiểm tra SHA256a97e1e591d06cd8634d9709db3303cd106d9118efece504fa2e2f69a266e08cb. Kiểm tra nhãn/folds trùng baseline; tái tạo v3 OOF/dev/test sai khác tối đa1.1921e−7. Source/data/cache/baseline/version được đóng băng trong manifest. Mã nền cũ giữ nguyên.

Một cấu hình cố định, không grid:OOF tăng>=0.001, ít nhất hai trên ba fold tăng, worstfold không giảm quá0.004, novel fakeF1 không giảm quá0.01; bootstrap chỉ báo uncertainty. Khi quaCV mới fit fulltrain; dev không giảm quá0.005 mới xuất submission_v26.csv. Dừng sớm khi không thể qua tiêu chí. Không gộp dev vào train, không tự submit, không train trên Mac. OOF/dev đã được dùng nhiều lần nên selection bias còn tồn tại; chưa bảo đảm tăng leaderboard.

Source mới:text_capacity_v26.py vàexperiment_v26.py. Thư viện CatBoost1.2.10/Apache-2.0. Runtime output:/content/ds102_v7_bundle/output_v26_experiment. Xem report.json và v26_run_state.json để biết trạng thái thực tế; README này mô tả cấu hình, không xác nhận kết quả.

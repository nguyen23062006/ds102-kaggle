# V23 — thêm ngữ cảnh mô tả sản phẩm trong không gian SVD

Nguồn ý tưởng: nghiên cứu metadata ViSpamReviews v2 https://arxiv.org/abs/2405.13292 nêu khả năng dùng mô tả sản phẩm bổ sung review. Nghiên cứu dùng encoder pretrained; ở đây chỉ hiện thực ý tưởng metadata bằng TF-IDF/SVD fit từ chính train. Không dùng pretrained hoặc tải dataset bên ngoài. Split sizes/nhãn gợi ý liên hệ với ViSpamReviews, nhưng không xác minh danh tính Kaggle bằng việc tải dữ liệu/nhãn nguồn. Không so trực tiếp điểm leaderboard với benchmark trong bài báo.

Khoảng trống trong baseline: SVD128 đang biểu diễn review/title; description mới đi qua overlap statistics. Thử bổ sung vector mô tả và các phép so với review, giữ nguyên TF-IDF/vocabulary/IDF/SVD được fit chỉ trên review train của fold. Description chỉ được transform, không fit corpus bổ sung. Bỏ HTML tags, lấy tối đa4000ký tự; deduplicate descriptions trong batch chỉ để tránh transform lặp, không học statistics từ validation/test.

Cấu hình cố định: giữ128dimensions và CatBoost800depth5lr0.06l2=8SqrtBalanced. Thêm first32description coordinates, product với review32coordinates, absolute differences (96features), và4scalars: TF-IDFcosine, latentcosine128D, descriptionlatentnorm, L2distance. Tổng675numeric+category/rating=677columns. Descriptioncosine xử lý theo batch256 để giới hạn memory; các descriptor trùng nhau dùng chung kết quả transform nhưng không dùng số lần lặp làm feature.

Kiểm tra trên Colab:fit_transform(train)≈transform(train.drop(label)), đổi thứ tự dòng chỉ đổi thứ tự outputs, mô tả rỗng không sinh NaN. Giữ cùng seed102,fold/y/data/sourceversionchecks. Trộn cố định0.7v3+0.3description-contextCatBoost,weights[1,1.2,1.2,0.9], so baselineSVD128 cóKaggle0.77436.

Gate như hiện hành:OOF+0.001,>=2/3fold dương,worstfold>=-0.004,novel fakeF1 delta>=-0.01. Dừng sớm khi worstfloor không thể sửa. Bootstrap là uncertainty report,khônggate significance. QuaCV mớifulltrain,dev không giảm quá0.005 mới xuấtCSV; không tựsubmit. Dev giữriêng vì rule chưa rõ; không chạy train+dev. AllML onhostedColab, noMacfit.

Đây là thử nghiệm thích nghi sau nhiều lượt OOF/dev; selectionbias còn tồn tại. Ý tưởng metadata không bảo đảm SVDfeatures có cùng hiệu quả encoder trong paper, cũng không bảo đảm leaderboard tăng. Source description_latent_v23.py,experiment_v23.py. Runtime output /content/ds102_v7_bundle/output_v23_experiment.

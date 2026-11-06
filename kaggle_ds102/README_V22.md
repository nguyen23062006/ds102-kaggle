# V22 — class-aware balanced SVD128 basis

Sau khi LightGBM leaf-wise và các công thức kết hợp không đạt, đổi giả thuyết từ thuật toán cây sang phép chiếu đầu vào. SVD gốc không dùng nhãn và tối ưu tổng bình phương tái tạo trên toàn corpus; lớp0 có nhiều review hơn các lớp thiểu số. Thử một cách reweight training rows để các lớp thiểu số đóng góp nhiều hơn khi học right singular vectors.

Một thay đổi cố định trước kết quả: fit word/char TF-IDF từ train của fold, normalize L2 mỗi row, scale khi fit SVD bằng(N/(4*n_class))**0.25 tính từ đúng training labels. Với review có norm1, covariance contribution của mỗi lớp thay từ tỷ lệ n_class sang xấp xỉ sqrt(n_class), chứ không cân bằng hoàn toàn. Exponent0.25 cố định, không grid.

Quan trọng: row scale chỉ dùng khi fit basis. Sau đó train/validation/dev/test đều được chiếu bằng X_unweighted @ fitted_components.T. Không scale train features theo nhãn khi đưa vào classifier, không cần/đọc nhãn để transform dữ liệu dự đoán. Đã kiểm tra trên Colab với mẫu mất cân bằng:fit_transform(train)≈transform(train.drop(label)); transform không đổi khi nhãn giả bị thay99. Đây là supervised representation fit bên trong training fold, không phải fit từ labels validation/test.

Giữ128dimensions,n_iter5,seed102, từ vựng/IDF train-only, review/title vector relations,62numeric và category/rating như baseline. Giữ CatBoost800depth5lr0.06l2=8,SqrtBalanced,2CPU. Công thức dự đoán cố định0.7v3+0.3CatBoost balanced-projection,weights[1,1.2,1.2,0.9], so với baselineSVD128 đạtKaggle0.77436. Không gộp dev, không pretrained/dữ liệu ngoài, mọifit hostedColab.

Gate giống các lượt trước:OOF+0.001,>=2/3fold dương,worstfold>=-0.004,novel fakeF1 không giảm quá0.01. Prune sớm khi một fold vi phạm worstfloor không thể sửa. Bootstrap report uncertainty,khônggate significance. QuaCV mới fulltrain,dev không giảm quá0.005 mới xuấtCSV; baselinefullprediction phải khớp hash bản0.77436. Không tự submitKaggle.

Repeated/adaptive OOF/dev use vẫn có selection bias; việc reweight có thể ưu tiên template thiểu số và overfit, nên kiểm tra novel fake được giữ. Không bảo đảm tăng leaderboard. Source balanced_svd_v22.py,experiment_v22.py; feature/prediction caches có manifest/tr/va; runtime /content/ds102_v7_bundle/output_v22_experiment.

# V17 — hiệu chỉnh trọng số quyết định cho SVD128

Giả thuyết chốt trước kết quả: trọng số[1,1.2,1.2,0.9] của submissionSVD128 thừa hưởng từ biểu diễn64; thay đổi biểu diễn có thể đổi thang confidence. Dùng lại nguyên hàm selection_v5.fit_weights (grid48cấu hình, penalty0.002) để fit calibrator theo OOF SVD128. Không mở rộng grid hoặc đổi tỷ lệ v3/latent, không huấn luyện lại base model.

Đánh giá calibrator: fit bằng hai OOFfold, áp lên fold còn lại. So trực tiếp với fixedweights hiện tại, không so với rawargmax để tránh thổi mức tăng. Finalweights fit trên toàn trainOOF chỉ dùng cho fullprediction; score fit này ghi rõ không phải validation. Nếu finalweights không đổi thì không xuất bảnCSV trùng bản cũ.

Gate: crossfit delta>=0.001, >=2/3fold dương, worstfold>=-0.004, novel fakeF1 không giảm quá0.01. Báo bootstrap như uncertainty. Nếu qua gate và finalweights khác, kiểm tra dev giảm<=0.005 rồi xuất CSV. Không chọn hệ số theo dev/test; không tự submitKaggle. BaselineCSV phải khớp hashSVD128 đã đạt0.77436.

Đây không phải fully-nested modelCV: các OOFprediction chia sẻ các upstream model fitting, và nhiều lượt trước đã xem OOF/dev. Việc thử lại hiệu chỉnh trên biểu diễn mới tiếp tục tạo selection bias. Không coi điểm CV là bảo đảm leaderboard tăng. Mọi tính toán/fitting calibration trên hostedColab, không pretrained/dữ liệu ngoài.

Source calibrate_svd128_v17.py, dùng cache sẵn được xác minhhash/source/versions/fold/y. Output /content/ds102_v7_bundle/output_v17_experiment.

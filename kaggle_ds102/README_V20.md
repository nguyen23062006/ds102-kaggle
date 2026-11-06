# V20 — LightGBM regularized và class-balanced

Đề xuất sau chẩn đoán V18 fold2: train macroF1=1.0 ở mọi lớp, validation raw macroF1=0.69386; recall fake0.40, brand-only0.30809. Cấu hình trước vừa overfit train vừa bỏ sót nhiều lớp thiểu số trên validation. Đây là nghiên cứu thích nghi, không phải thử nghiệm độc lập mới.

Một cấu hình cố định:400estimators,lr0.04,num_leaves15,max_depth6,min_child_samples60,min_split_gain0.05,L1=0.5,L2=25,row/columnsubsample0.8. Seed102,2CPU,deterministic/force_col_wise. Dùng class weights n/(4*class_count) từ fitting labels thay inverse-sqrt của V18. Các thay đổi cùng kiểm tra giả thuyết regularization+cân bằng lớp, không tách riêng được tác động từng tham số. Không có parameter grid.

Giữ nguyên train-onlySVD128, category/rating levels và numeric577cột. Sao chép feature cache hai fold đã fit từ V18 sang thư mục mới, kiểm tra feature fingerprint; không sửa model/report V18. LightGBM400 được fit lại, không dùng prediction V18 thay cho model mới. Nếu cần fold3/fullfeatures thì fit SVD tương ứng chỉ từ train của partition.

Kết hợp20%newLGBM+80%baselineSVD128, giữ weights[1,1.2,1.2,0.9]. Gate giống V18:OOF+0.001,>=2/3fold dương,worstfold>=-0.004,novel fakeF1 không giảm quá0.01. Prune khi một completedfold vi phạm worstfloor không thể sửa. Bootstrap reportuncertainty,khônggatep-value. Nếu quaCV mới fulltrain,dev không giảm quá0.005 mới xuấtCSV; baselinehash phải khớp CSV cóKaggle0.77436.

Giữ dev riêng; không chạy train+dev vì quy định chưa rõ. Không pretrained/dữ liệu ngoài, mọifit trên hostedColab. LightGBM4.6.0 licenseMIT. Selectionbias do nhiều lần dùngOOF/dev còn tồn tại; không bảo đảm leaderboard tăng. Source lightgbm_latent_v20.py,experiment_v20.py. Output /content/ds102_v7_bundle/output_v20_experiment.

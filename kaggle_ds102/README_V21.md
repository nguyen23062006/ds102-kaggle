# V21 — thay nhánh latent bằng LightGBM regularized/balanced

Phép so sánh thứ hai cho cấu hình V20, đề xuất sau V20fold1 trước kết quả đầy đủ. Công thức cố định0.7v3+0.3LightGBM regularized/balanced, weights[1,1.2,1.2,0.9], so baseline0.7v3+0.3CatBoostSVD128. Không quét tỷ lệ hoặc chọn subset dòng.

Source experiment_v21.py sử dụng lightgbm_latent_v20.py; tái sử dụng prediction V20 khi manifest/source/version/tr/va khớp. Nếu thiếu fold/fullfit thì dùng feature cache hợp lệ để train đúng cùng model. Không sửa báo cáo/source/model V20. Chạy sau V20 để không tranh CPU.

Gate như các lượt trước:OOF+0.001,>=2/3fold dương,worstfold>=-0.004,novel fakeF1 delta>=-0.01; bootstrap là uncertaintykhônggate. Nếu quaCV mớifitfulltrain/devguardrail-0.005 và xuấtCSV, không tựsubmit. No pretrained/outside data; dev riêng; all fits hostedColab. Đây là kiểm tra thích nghi trên OOF đã dùng nhiều lần, không validationđộc lập hay bảo đảm leaderboard tăng.

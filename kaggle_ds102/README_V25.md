# V25 — ExtraTrees trên SVD128

Dùng ExtraTreesClassifier 600 cây, max_features=sqrt, min_samples_leaf=2, bootstrap=False, seed102, hai luồng. Class weights bằng căn bậc hai của N/(4*n_class), chỉ tính từ train của fold. Mô hình dùng đúng575 cột số của LatentFeatures128 và one-hot category/rating, encoder fit trên train và bỏ qua category chưa gặp.

Cache675cột của V23 chỉ được lấy575cột đầu; loại toàn bộ100cột mô tả sản phẩm bổ sung. Kiểm tra equality chính xác với cache độc lập V18 ở hai fold đầu, kiểm tra feature_id của cả ba fold theo chỉ số train/valid và hash dữ liệu; hash toàn bộ cache đầu vào được đóng băng trong manifest V25.

Công thức cố định:0.85*baselineSVD128+0.15*ExtraTrees, trọng số lớp không đổi. Tiêu chí giữ giống V24:OOF+0.001, hai trên ba fold tăng, worstfold>=−0.004, novel fakeF1 không giảm quá0.01; dev guard−0.005 sauCV. Tất cả fit chạy trên hosted Colab.

Kết quả:fold1−0.000391; fold2−0.006227. Mô hình ExtraTrees riêng có macro-F1 lần lượt0.589244 và0.568521. Dừng tự động sau hai fold, exit0, khoảng81.47giây. Không có OOF đủ ba fold, fullfit, đánh giá dev mới hoặc CSV. Không dùng pretrained hoặc dữ liệu ngoài.

Tài liệu nguồn mở: https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html ; thư viện scikit-learn BSD-3-Clause. Dự án chưa được tự công khai hay gán giấy phép mới.

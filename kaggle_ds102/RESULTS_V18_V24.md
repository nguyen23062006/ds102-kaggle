# Kết quả nghiên cứu V18–V24

Mốc tốt nhất được người dùng xác nhận trên Kaggle là **0.77436**, từ `output_v11_experimental/submission_v11_experimental.csv`. Chưa có file nộp mới sau nhóm thử nghiệm này. Các số dưới đây là chênh lệch macro-F1 trên từng fold so với SVD128, không phải điểm leaderboard.

## Những gì đã chạy

- **V18 — LightGBM, trộn 20% vào baseline:** fold1 −0.000857, fold2 −0.020184. Dừng sau hai fold. Mô hình đạt F1 train=1.0 ở fold2 nhưng validation thấp, nhất là lớp2.
- **V19 — thay nhánh latent bằng LightGBM V18:** fold1 −0.003410, fold2 −0.019335. Chỉ dùng lại dự đoán V18, không fit thêm; dừng sau hai fold.
- **V20 — LightGBM regularization mạnh hơn và balanced weighting:** fold1 −0.000122, fold2 −0.007574. Dừng sau hai fold.
- **V21 — thay nhánh latent bằng LightGBM V20:** fold1 −0.001916, fold2 −0.011605. Dùng lại dự đoán V20, không fit thêm; dừng sau hai fold.
- **V22 — SVD128 có trọng số hàng theo lớp khi học cơ sở:** fold1 −0.005485. Dừng sau một fold. Mọi phép transform đều không dùng nhãn; kiểm tra fit/transform đã qua.
- **V23 — thêm quan hệ giữa review và mô tả sản phẩm:** fold1 −0.000779, fold2 −0.002242. Hai fold đều giảm nên không thể đạt tiêu chí ít nhất hai trên ba fold tăng. Dừng chủ động lúc chuẩn bị fold3; mã thoát −15 là do việc dừng có chủ đích. Hai checkpoint đã hoàn tất được giữ lại.
- **V24 — tăng SVD từ 128 lên 192 chiều, giữ nguyên phần còn lại:** fold1 −0.003736, fold2 −0.005335. Tự dừng sau hai fold, mã thoát0, khoảng900giây. Không đủ điều kiện để fit toàn train hoặc tạo CSV.

V23 tăng F1 lớp2 ở fold1 nhưng giảm lớp1; đến fold2 lớp2 cũng giảm. V24 không nối tiếp được lợi ích đã quan sát ở lần tăng64→128. Đây là kết quả của các cấu hình cụ thể, không chứng minh mọi cách dùng description, LightGBM hoặc SVD lớn hơn đều thất bại.

## Phạm vi và kiểm chứng

Mọi fit vectorizer, IDF, SVD và classifier đều chạy trên hosted Google Colab, hai CPU. Không có training trên Mac. Chỉ dùng bốn CSV cuộc thi; dev giữ riêng và chưa dùng để train. Không dùng pretrained model, pretrained embeddings hoặc dữ liệu ngoài.

Giữ nguyên phiên bản baseline, seed102, ba fold nhóm theo comment chuẩn hóa, công thức kết hợp và trọng số lớp. Kiểm tra hash dữ liệu, mã nguồn, thư viện, nhãn và chỉ số fold trước khi so sánh. Mã nguồn V23/V24 trên máy khớp với manifest của lượt thực thi Colab.

Không cấu hình nào hoàn tất ba fold. Do đó **không báo điểm OOF tổng thể, bootstrap hay điểm dev mới** cho V18–V24. Dừng sớm dựa trên tiêu chí đã đặt; không suy diễn điểm trên các hàng chưa có dự đoán. OOF/dev đã được dùng nhiều lần trong dự án nên còn thiên lệch lựa chọn; các lượt này chưa xác nhận được mức tăng leaderboard.

File tốt nhất được kiểm tra lại: đúng3974dòng, hai cột ID/spam_label, ID duy nhất, nhãn0–3, SHA256 `05bffeff984bd4927c8dd3e9bd3613aef6a7abce303e6e1dcb7cfa4a12012059`.

Khi kết thúc, không còn tiến trình thử nghiệm đang chạy trên Colab. Runtime và checkpoint được giữ lại. Báo cáo chi tiết/manifest nằm trong `output_v18_experiment` tới `output_v24_experiment`; trạng thái từng lượt trong `v18_run_state.json` tới `v24_run_state.json`. Bản nén `research_v18_v24_source.zip` chỉ chứa source, notebook nền và tài liệu/kết quả JSON; không chứa CSV dữ liệu, pretrained weights, model binary hoặc token MCP. Các script thử nghiệm dùng cache baseline đã kiểm tra; notebook nền SVD128 có source để fit pipeline từ bốn CSV khi cần tái tạo, không có nghĩa toàn bộ nhóm V18–V24 đã được chạy lại từ đầu trong notebook sạch.

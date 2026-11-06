# V39–V40: thuật toán boosting và từ vựng catalogue

Mốc leaderboard giữ nguyên là V11 experimental **0.77436**. V27 đạt **0.77229**, thấp hơn mốc này; không coi mức tăng OOF nhỏ của V27 là bằng chứng nâng cấp.

Người dùng xác nhận ngày 03/11/2026: chưa cần chốt file ngay, giữ nguyên **không pretrained**. Chỉ fitting trên hosted Colab; Mac dùng để sửa source, kiểm tra cú pháp và lưu báo cáo. Dev không được gộp vào train.

## V39

Đổi CatBoost text700 từ Plain sang Ordered, giữ số cây/độ sâu/từ điển/trọng số lớp. Thay một nửa đóng góp CatBoost text trong baseline. Bài báo về Ordered boosting là động lực cho thử nghiệm; không hàm ý pipeline cũ có rò nhãn validation. Source: `ordered_text_v39.py`, `experiment_v39.py`.

Nguồn: https://catboost.ai/docs/en/references/training-parameters/common và https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html . CPU mặc định Plain; Ordered thường tốn thêm thời gian. Micro-fit multiclass text với Ordered đã chạy thành công trên Colab.

## V40

TF-IDF word/char và SVD32 được học mới từ các mô tả/tên sản phẩm duy nhất trong phần train của từng fold. Không lấy corpus hay trọng số bên ngoài. Chiếu review và catalogue của chính dòng vào không gian này, bổ sung 132 đặc trưng về nội dung/độ tương đồng vào 575 đặc trưng SVD128 ban đầu. Giữ CatBoost800 depth5 và công thức .7v3+.3latent. Source: `catalogue_svd_v40.py`, `experiment_v40.py`.

Khác V23: V23 dùng từ vựng và không gian học từ review để biểu diễn description; V40 học riêng không gian catalogue. Khác V35: không chỉ xoay lại hai vector review/title cũ bằng CCA.

Micro-preflight Colab: 400 dòng, 344 catalogue duy nhất, 132 đặc trưng; đảo thứ tự batch, đổi nhãn, transform dữ liệu có từ mới và lưu/đọc encoder đều qua kiểm tra. Đây chỉ kiểm tra kỹ thuật, chưa chứng minh tăng F1. V40 được xếp sau V39 để hai lượt fit không tranh CPU.

## Chọn ứng viên

Nhóm comment trùng khít theo 3 fold cũ, seed102; chỉ phần fitting được dùng xây từ vựng/encoder/model. Baseline hashes, thứ tự nhãn và feature cache được đối chiếu. Trọng số lớp cố định [1,1.2,1.2,.9]. Các gate thực thi: tăng OOF ít nhất .002, ít nhất 2/3 fold dương, fold xấu nhất không giảm quá .003, fake mới không giảm F1, bootstrap có q05 dương; nếu qua mới fit toàn train và yêu cầu dev tăng ít nhất .001 trước khi xuất CSV. Không thay gate sau khi thấy kết quả. Chuỗi `bootstrap_policy` trong config V39 cũ ghi chỉ báo cáo, nhưng gate thực thi của source vẫn yêu cầu q05>0; giữ source gốc để manifest không đổi trong khi chạy.

OOF và dev đã được dùng nhiều lần; bootstrap chỉ có điều kiện trên các dự đoán và không loại bỏ thiên lệch do chọn mô hình. Không có đảm bảo tăng điểm Kaggle, không tự submit.

V34–V38 đã bị loại, báo cáo đã lưu về thư mục output tương ứng. Bước xem review gốc qua MCP bị xét duyệt tự động chặn; không tìm cách xuất vòng. Thay bằng thống kê tổng hợp trên Colab: 63 fake mới bị bỏ sót có median123 ký tự, 40 fake mới nhận đúng median287 ký tự. Đây là tương quan mô tả, không phải quy tắc gán nhãn. Không sửa nhãn hay gán nhãn tay cho dev/test.

# Chạy thử rồi gửi kết quả để chỉnh tiếp

1. Tải `DS102_Colab_v43_handoff.ipynb` và `DS102_v43_inputs.zip` cùng thư mục này.
2. Dừng phiên thí nghiệm Colab cũ nếu vẫn chạy. MCP đã ngắt ở lúc chuyển sang bàn giao, nên chưa xác nhận các tiến trình cũ đã dừng; notebook mới không điều khiển runtime khác.
3. Mở Google Colab → File/Tệp → Upload notebook/Tải sổ tay lên → chọn notebook. Dùng hosted runtime CPU, không chọn local runtime trên Mac.
4. Chọn Runtime/Thời gian chạy → Run all/Chạy tất cả. Khi ô tải file xuất hiện, chọn `DS102_v43_inputs.zip`. Gói đã chứa bốn CSV và cache dự đoán nền; không cần chọn từng CSV nữa.
5. Giữ tab mở trong lúc chạy. Máy chủ Colab huấn luyện; Mac chỉ hiển thị notebook. Không cần bật GPU cho cấu hình mặc định. Có ba fold, có thể mất vài chục phút hoặc lâu hơn trên 2 CPU.
6. Cuối notebook sẽ tải `feedback_v43.zip`. Gửi ZIP đó lại để phân tích và chỉnh mã. Nếu bị lỗi, ZIP chứa log lỗi; có thể chạy riêng ô cuối để tải báo cáo.
7. Chỉ khi đủ điều kiện đánh giá mới có `submission_v43_candidate.csv`. Nếu bạn thử nộp, gửi thêm tên CSV và điểm Kaggle chính xác. Không nộp ZIP báo cáo hoặc file baseline như một ứng viên mới.

Baseline được kiểm tra phải tái tạo đúng CSV từng đạt **0.77436**, giữ nguyên file cũ. Nhánh thử mặc định là đặc trưng từ/cụm ký tự được chọn riêng cho từng lớp, kết hợp SVD128/CatBoost. Chưa có điểm Kaggle của nhánh này. Cấu hình `catalogue` có sẵn cho lượt sau; giữ `lexical` ở lượt đầu để kết quả dễ đối chiếu.

Không pretrained hoặc dữ liệu ngoài, không gộp dev vào train, không tự submit. Cache dự đoán trong gói là kết quả tự train từ dữ liệu cuộc thi trước đây; không chứa trọng số pretrained bên ngoài. Mã nguồn đầy đủ được nhúng trong notebook và có bản rời `DS102_v43_source.zip`. Không tự xuất bản dự án/dữ liệu.

Notebook và tất cả module đã được kiểm tra cấu trúc/cú pháp; CRC/checksum và định dạng ID/nhãn của baseline đã được kiểm tra. Bộ encoder đã qua micro-preflight Colab trước đó. Chưa chạy lại toàn bộ notebook mới từ đầu đến cuối. Kiểm tra đầu notebook sẽ xác minh phiên bản thư viện, bốn CSV, nhãn/fold của cache và hash CSV baseline trước lượt fit mới.

Chạy lại cùng cấu hình trong runtime còn dữ liệu sẽ sử dụng các fold đã hoàn tất. Đổi mã/cấu hình thì đổi `RUN_TAG43` để tránh dùng nhầm cache. Dữ liệu/checkpoint dưới `/content` có thể mất khi xoá runtime; ZIP báo cáo chỉ chứa metrics/log/manifest, không sao lưu mô hình hoặc OOF.

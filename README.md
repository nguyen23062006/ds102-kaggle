# DS102 Kaggle — dự án làm việc chung

Mã nguồn cho cuộc thi DS102.R11.CNVN: phân loại review thành bốn nhãn, đánh giá bằng F1-macro.

## Kết quả hiện có

- Điểm Kaggle cao nhất do chủ dự án báo: **V11 experimental, 0.77436**.
- Các thí nghiệm mới gồm H1/V60, H3 nested và H2 chưa đạt đủ tiêu chí để thay V11.
- Xem `kaggle_ds102/leaderboard_history.json` và các báo cáo `output_*` được giữ trong repo.
- README trong `kaggle_ds102/` mô tả baseline ban đầu; không phải hướng dẫn cho bản tốt nhất hiện tại.

## Dữ liệu và môi trường

Mỗi thành viên lấy dữ liệu từ cuộc thi bằng tài khoản được cấp quyền. Bốn file là `ds102_train.csv`, `ds102_dev.csv`, `ds102_test.csv`, `ds102_samplesubmission.csv`.

Dữ liệu gốc, submission CSV, cache dự đoán, model weights và ZIP không nằm trong gói mã nguồn này. Cần chia sẻ các cache cần thiết riêng với thành viên được phép truy cập. Các thí nghiệm gần đây kiểm tra SHA256 của dữ liệu và checkpoint; clone repo chưa đủ để chạy lại ngay.

Huấn luyện trên **hosted Google Colab**, không chạy fitting trên Mac của chủ dự án. Không dùng pretrained hoặc dữ liệu ngoài theo yêu cầu hiện tại. Không gộp dev vào train và không tự nộp Kaggle.

`requirements.txt` và `requirements_v3.txt` là các yêu cầu lịch sử của từng phiên bản. Các lượt mới dùng môi trường pinned ghi trong run manifest; không coi hai file requirements đó là môi trường tái lập đầy đủ cho mọi thí nghiệm. Nhiều script dùng đường dẫn `/content/ds102_v53/...` và cần bộ dữ liệu/cache V53 được khôi phục riêng trên Colab.

## Các phần để cùng làm

- `kaggle_ds102/antigravity_review/`: nhận xét và mã do Antigravity đề xuất.
- `kaggle_ds102/claude_review/`: nhận xét và mã đánh giá của Claude.
- `kaggle_ds102/experiment_v*.py`: thí nghiệm theo phiên bản.
- `kaggle_ds102/output_*/`: chỉ có một số báo cáo JSON, không chứa dataset hay model.

## Quy trình cộng tác

1. Clone repository về máy của từng người; mở thư mục clone trong IDE.
2. Trước khi sửa, chạy `git pull` trên main và tạo branch riêng, ví dụ `experiment/intent-v2` hoặc `review/cnn`.
3. Viết thí nghiệm vào thư mục hoặc phiên bản mới, giữ nguyên kết quả cũ. Ghi rõ giả thuyết, cấu hình chốt trước, phép kiểm định và trạng thái đã chạy/chỉ kiểm tra cú pháp.
4. Chỉ commit file liên quan; kiểm tra `git diff --cached` trước khi push.
5. Push branch rồi mở pull request để người còn lại rà soát trước khi merge vào main.

Ví dụ sau khi clone và đang ở main:

```bash
git pull --ff-only
git switch -c experiment/my-idea
# Sửa mã; chạy huấn luyện trên Colab
git add kaggle_ds102/path/to/changed_file.py
git commit -m "Add controlled experiment"
git push -u origin HEAD
```

GitHub lưu file và lịch sử thay đổi; không tự đồng bộ lịch sử chat hay kết nối runtime Colab giữa các thành viên.

## Repository GitHub

Repository riêng tư: https://github.com/nguyen23062006/ds102-kaggle

Chủ repo cần mời thành viên vào Settings → Collaborators → Add people. Sau khi nhận lời mời, thành viên clone bằng:

```bash
git clone https://github.com/nguyen23062006/ds102-kaggle.git
cd ds102-kaggle
```

Đây là bản chia sẻ mã nguồn đã được chọn lọc. Những file thay đổi trong thư mục nghiên cứu gốc chỉ xuất hiện trên GitHub sau khi được đồng bộ sang checkout này, commit và push; GitHub không tự theo dõi mọi thư mục trên máy.

Chủ dự án chưa chọn giấy phép cho mã của nhóm. Việc sử dụng thư viện mã nguồn mở không tự gán giấy phép cho toàn bộ repo; giữ nguyên thông tin tác giả và giấy phép của phụ thuộc.

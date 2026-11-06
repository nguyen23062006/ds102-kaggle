# BÁO CÁO NGHIÊN CỨU VÒNG TIẾP THEO (NEXT ROUND REVIEW)

**Dự án**: Kaggle DS102 – Phân loại bình luận thương mại điện tử (`spam_label` ∈ {0, 1, 2, 3})  
**Mốc Leaderboard cao nhất**: V11 (SVD128 Experimental) = **0.77436**  
**Thời điểm**: 05/11/2026  
**Trạng thái mã nguồn trên Mac**: **Chỉ kiểm tra cú pháp (AST + compile) và test logic tokenization**. Toàn bộ quá trình huấn luyện mô hình được bọc trong `guard()` và chỉ thực thi trên Hosted Google Colab.

---

## 1. Phân tích nguyên nhân H3 Nested thành công trên Inner OOF nhưng thất bại trên Outer Fold 0

### 1.1. Dữ liệu thực nghiệm đã ghi nhận (Evidenced Findings)

Theo báo cáo `output_h3_nested/termination_report.json` và `output_h3_nested/outer0_selected_weights.json` do Codex thực hiện trên Colab:
- **Trên Inner OOF (thuộc outer-train của Fold 0, gồm 9,726 dòng)**:
  - Điểm baseline inner macro-F1: **0.702068**
  - Điểm sau khi chọn trọng số: **0.734212** (mức tăng nội bộ rất lớn: **+0.032144**)
  - Bộ trọng số được thuật toán tối ưu lựa chọn: `[1.0, 2.0, 1.6, 1.1]`
- **Trên Outer Validation Fold 0 (gồm 4,580 dòng)**:
  - Baseline macro-F1 (trọng số cố định `[1.0, 1.2, 1.2, 0.9]`): **0.678225**
  - Candidate macro-F1 (trọng số inner `[1.0, 2.0, 1.6, 1.1]`): **0.669309**
  - Độ lệch (delta): **−0.008917** (vượt ngưỡng cho phép giảm tối đa −0.005 của fold gate)
  - Số dòng bị đổi nhãn: **97 dòng**.
  - Hệ thống dừng sớm theo đúng quy tắc pre-declared worst-fold gate.

### 1.2. Cơ chế thất bại: Phân tích phân phối mẫu, thang xác suất và sự sụp đổ Precision

Qua đối chiếu cấu trúc dữ liệu của Fold 0 (từ `output_v53_experiment/fold0_diagnostics.json`):

1. **Độ lệch phân phối mẫu Lớp 1 bất thường ở Fold 0**:
   - Toàn bộ tập dữ liệu chỉ có **211 mẫu Lớp 1**.
   - Trong Outer Fold 0 (4,580 dòng), Lớp 1 chỉ có vỏn vẹn **39 mẫu** (tỷ lệ 0.85%).
   - Phần Outer-train (9,726 dòng) chứa tới **172 mẫu Lớp 1** (tỷ lệ 1.77%, cao gấp đôi Fold 0).
   - Vì Outer Fold 0 bị "rút cạn" Lớp 1, việc chọn trọng số dựa trên inner OOF (nơi có nhiều Lớp 1 hơn) đã hình thành kỳ vọng sai lệch về xác suất tiên nghiệm.

2. **Hiện tượng Chạm Trần Lưới Tìm Kiếm (Boundary Hitting)**:
   - Nhìn vào bộ trọng số được chọn:
     - $w_1 = 2.0$ (chạm đúng cận trên của lưới $w_1 \in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, \mathbf{2.0}]$)
     - $w_2 = 1.6$ (chạm đúng cận trên của lưới $w_2 \in [0.8, \dots, \mathbf{1.6}]$)
     - $w_3 = 1.1$ (chạm đúng cận trên của lưới $w_3 \in [0.7, \dots, \mathbf{1.1}]$)
   - Trong tối ưu hóa, khi tất cả tham số đều chạm cận trên của không gian tìm kiếm, đó là dấu hiệu chắc chắn của việc thuật toán đang cố gắng "thao túng" (game) chỉ số Macro F1 trên tập huấn luyện nội bộ bằng cách thổi phồng Recall của các lớp thiểu số.

3. **Lệch thang xác suất do quy mô dữ liệu (Data-scale margin shift)**:
   - Các mô hình inner được fit trên 2/3 outer-train (khoảng 6,480 dòng).
   - Mô hình outer được fit trên toàn bộ outer-train (9,726 dòng, gấp 1.5 lần).
   - Như Claude đã chứng minh ở V6 (`CLAUDE_REVIEW.md`), margin của LinearSVC và độ tự tin của cây quyết định CatBoost tỷ lệ thuận với quy mô dữ liệu fit. Trên tập fit nhỏ (inner), xác suất dự đoán cho các lớp thiểu số bị dè dặt (under-confident), đòi hỏi hệ số nhân lớn ($w_1=2.0, w_2=1.6$) để đẩy mẫu qua ngưỡng argmax.
   - Nhưng khi áp hệ số $w_1=2.0, w_2=1.6$ lên mô hình outer (vốn đã sắc nét hơn):
     - Mọi mẫu Lớp 0 có một chút nghi ngờ (chỉ cần $P(\text{Lớp 1}) \ge 0.25$ hoặc $P(\text{Lớp 2}) \ge 0.35$) lập tức bị ép chuyển nhãn sang Lớp 1 hoặc Lớp 2.
     - Lớp 0 có tới **3,398 dòng**. Chỉ cần 2% số mẫu Lớp 0 bị chuyển nhầm sang Lớp 1 sẽ tạo ra ~68 False Positives!
     - Với Lớp 1 chỉ có 39 mẫu thật: thêm vài chục False Positives khiến Precision Lớp 1 sụp đổ (dưới 0.20), kéo tụt $F1_1$ thảm hại và làm hỏng toàn bộ Macro F1.

### 1.3. Phân biệt Bằng chứng Thực tế và Giả thuyết

- **Bằng chứng xác thực (Facts)**:
  - Việc tìm kiếm trọng số unconstrained trên inner OOF đẩy các hệ số lên chạm trần lưới tìm kiếm.
  - Áp dụng bộ trọng số chạm trần này lên outer fold làm thay đổi 97 nhãn và giảm Macro F1 −0.0089.
  - Phân phối Lớp 1 ở Outer Fold 0 (39 mẫu) thấp hơn nhiều so với inner training set (172 mẫu).
- **Giả thuyết khoa học (Hypotheses)**:
  - Nguyên nhân chính là sự bùng nổ False Positives từ Lớp 0 sang Lớp 1 và Lớp 2 do lệch thang xác suất giữa 6.4k dòng (inner) và 9.7k dòng (outer).
- **Nguyên tắc phương pháp luận**:
  - Không suy rộng kết quả của một Outer Fold 0 thành kết luận cho toàn bộ giao thức Nested CV.
  - **Tuyệt đối không tiếp tục tinh chỉnh, dò dẫm lại trọng số trên Outer Validation Fold 0 hoặc tập Dev** để tránh rò rỉ thông tin và vi phạm nguyên tắc khoa học.

---

## 2. Rà soát Giả thuyết H2 (Subword CNN) so với các Thử nghiệm V14/V15

### 2.1. Phân tích nguyên nhân thất bại của V14 và V15

Trong lịch sử dự án, V14 và V15 đã thử nghiệm mô hình mạng nơ-ron tích chập (TextCNN rand - Kim 2014):
- **Cấu hình V14/V15**:
  - Tokenizer: Cấp độ từ (Word-level) dùng regex `(?u)\b\w+\b`, từ điển 30,000 từ, `min_count=2`.
  - Tham số: Nhúng ngẫu nhiên $30,000 \times 64 = 1,920,000$ trọng số nhúng.
  - Đầu vào: Review (160 từ), Title (40 từ), 62 đặc trưng số.
  - Hàm mất mát: `CrossEntropyLoss` với trọng số nghịch đảo căn bậc hai tần suất.
- **Kết quả V14/V15**:
  - V14: Mạng học vẹt và overfit cực nhanh (loss tụt từ 0.13 ở epoch 3 xuống < 0.01 ở epoch 8). Tại Fold 2, delta bị giảm **−0.0085** $\rightarrow$ Dừng sớm.
  - V15: Tăng mạnh regularization (dropout 0.5, embedding dropout 0.2, weight decay 0.05, giảm xuống 5 epoch). Tại Fold 2, delta vẫn giảm **−0.0132** $\rightarrow$ Dừng sớm.
- **Nguyên nhân gốc rễ thất bại của V14/V15**:
  1. *Số lượng tham số quá lớn so với dữ liệu*: 1.92 triệu trọng số nhúng từ chỉ được huấn luyện trên ~9,500 câu ngắn. Hầu hết các từ chỉ xuất hiện 2–5 lần, khiến mạng dễ dàng ghi nhớ từng từ đơn lẻ thay vì học khái quát.
  2. *Bất lực trước tiếng ồn và spam thương mại điện tử*: Ngôn ngữ review Shopee/Tiki chứa nhiều từ viết tắt, kéo dài ký tự (`đẹppppp`, `ráchhhhh`), teencode, và tin rác spam (`ptpppppppppppp`, `Xu :))))`). Tokenizer cấp từ gán toàn bộ những từ này thành `<UNK>` hoặc tạo ra các vector nhúng rời rạc không liên quan.
  3. *CrossEntropy thông thường bị chi phối bởi Lớp 0*: Lớp 0 (73.5%) áp đảo dòng gradient, khiến mạng nhanh chóng hội tụ về việc dự đoán an toàn cho Lớp 0 và bỏ rơi ranh giới khó của Lớp 1 và Lớp 2.

### 2.2. Điểm khác biệt có cơ sở kỹ thuật của H2 Subword CNN

H2 được thiết kế lại hoàn chỉnh để khắc phục triệt để các hạn chế trên:

| Tiêu chí | V14 / V15 TextCNN (Đã thất bại) | H2 Subword CNN (Đề xuất hoàn thiện) |
| :--- | :--- | :--- |
| **Cấp độ phân giải** | Cấp độ Từ (Word unigrams) | Cấp độ Subword / Ký tự (Character 3–5 grams) |
| **Quy mô từ điển & Tham số nhúng** | 30,000 từ $\rightarrow$ **1,920,000 tham số nhúng** | 8,000 n-grams $\rightarrow$ **512,000 tham số nhúng** (giảm gần 4 lần) |
| **Xử lý từ hiếm / Tiếng ồn / Spam** | Bị gán thành `<UNK>` hoặc rời rạc | Phân rã mọi từ thành các n-gram ký tự liên tục; bắt trọn teencode, ký tự kéo dài và template bot |
| **Thứ tự mã hóa chuỗi** | N/A | **Positional Interleaving**: Tại mỗi vị trí ký tự $i$, sinh lần lượt (3-gram, 4-gram, 5-gram). Khi cắt chuỗi ở `max_len=256`, chuỗi vẫn bảo toàn đầy đủ tỷ lệ cả 3 loại n-gram |
| **Xử lý Padding** | Masked pooling cơ bản | **Valid-length Masking**: Tính độ dài chuỗi hợp lệ sau tích chập $(L - K + 1)$, gán giá trị $-10^4$ trước khi max-pooling để triệt tiêu hoàn toàn kích hoạt từ padding |
| **Hàm mất mát** | Standard Weighted Cross-Entropy | **ClassBalancedFocalLoss** ($\gamma=2.0$): Dập tắt gradient từ các mẫu Lớp 0 dễ đoán ($p_t > 0.9$), tập trung gradient vào các ca biên khó của Lớp 1 & Lớp 2 |
| **Lựa chọn Epoch** | Snapshot epoch 10/11/12 (V14) | Snapshot epoch 5/6 trên tổng số 6 epoch cố định; **không dùng outer-validation** |

---

## 3. Hoàn thiện Thí nghiệm H2 Subword CNN có thể chạy thực tế

Toàn bộ mã nguồn hoàn chỉnh đã được xây dựng tại thư mục [`antigravity_review/next_round/`](file:///Users/ngien2306/Documents/ChatGPT/Đồ án DS102/kaggle_ds102/antigravity_review/next_round):
1. [**`h2_subword_cnn.py`**](file:///Users/ngien2306/Documents/ChatGPT/Đồ án DS102/kaggle_ds102/antigravity_review/next_round/h2_subword_cnn.py): Chứa `SubwordTokenizer`, kiến trúc `MaskedSubwordCNN`, hai hàm mất mát (`ClassBalancedFocalLoss` và `WeightedCrossEntropyLoss`), cùng training loop hoàn chỉnh có snapshot averaging và seeding cố định.
2. [**`h2_experiment.py`**](file:///Users/ngien2306/Documents/ChatGPT/Đồ án DS102/kaggle_ds102/antigravity_review/next_round/h2_experiment.py): Driver tích hợp giao thức đánh giá trên cluster folds (`groups.npz`, seed 2053), tái sử dụng cache baseline V53, hỗ trợ ablation có kiểm soát giữa Focal Loss và CrossEntropy, cùng bộ kiểm tra smoke test hoàn chỉnh.
3. [**`test_next_round.py`**](file:///Users/ngien2306/Documents/ChatGPT/Đồ án DS102/kaggle_ds102/antigravity_review/next_round/test_next_round.py): Bộ kiểm tra cục bộ trên Mac (AST, compile, logic tokenization interleaving và guards).

### 3.1. Cấu hình Thử nghiệm Chốt Trước (Pre-declared Config)

- **Kiến trúc**:
  - `vocab_size`: 8,000 (character n-grams 3–5)
  - `embed_dim`: 64
  - `conv_channels`: 64 (kernels: 3, 4, 5)
  - `dense_dim`: 128
  - `dropout`: 0.3, `embed_dropout`: 0.15
- **Tối ưu hóa**:
  - `optimizer`: AdamW (`lr=0.0015`, `weight_decay=0.01`)
  - `batch_size`: 64
  - `epochs`: 6 (snapshot averaging tại epoch 5 và 6)
  - `focal_gamma`: 2.0
  - `seed`: 102 (`torch.use_deterministic_algorithms(True)`)
- **Kết hợp Baseline**:
  $$P_{\text{final}} = 0.85 \times P_{\text{V53\_baseline}} + 0.15 \times P_{\text{subword\_cnn}}$$
  với trọng số lớp chuẩn của baseline: `[1.0, 1.2, 1.2, 0.9]`.
- **Cổng đánh giá (Gates) trên Cluster Folds**:
  - $\Delta \text{Macro pooled} \ge +0.003$
  - Ít nhất 2/3 fold có $\Delta > 0$
  - Fold tệ nhất không giảm quá $-0.005$
  - Novel fake F1 không giảm ($\Delta \text{novel} \ge 0$)
  - Paired cluster bootstrap (2,000 resamples, seed 2053) có $q_{025} > 0$.

### 3.2. Ước tính Chi phí Tính toán (Compute Budget)

- **Môi trường Colab GPU (T4 / V100)**:
  - Huấn luyện 6 epoch trên ~9,500 dòng mất khoảng 15–20 giây mỗi fold.
  - Tổng thời gian cho 3 fold khoảng **1.5 – 2.5 phút**.
- **Môi trường Colab CPU (2 lõi)**:
  - Conv1d với chuỗi độ dài 256 và batch 64 tốn khoảng 45–60 giây mỗi epoch $\rightarrow$ ~5–6 phút mỗi fold.
  - Tổng thời gian cho 3 fold khoảng **18 – 22 phút**.

---

## 4. Đánh giá Khách quan về Bằng chứng Thực nghiệm và Hướng đi Đề xuất

### 4.1. Đánh giá tính khả thi thực tế của H2 Subword CNN

Mặc dù H2 đã được khắc phục toàn diện về mặt kỹ thuật (interleaving tokenization, padding mask, focal loss, giảm tham số), chúng ta cần nhìn nhận thực tế:
- **Giới hạn cố hữu của mạng nơ-ron học từ đầu (No pretraining)**: Với chỉ ~9,500 mẫu huấn luyện trong mỗi fold, việc học một biểu diễn ngữ nghĩa hoàn chỉnh từ con số không luôn gặp bất lợi lớn trước các bộ trích xuất đặc trưng thống kê tần suất (TF-IDF 1-2 gram, character n-gram frequencies) kết hợp với các bộ phân loại lồi (LinearSVC) hoặc cây quyết định (CatBoost).
- **Kinh nghiệm từ V14/V15**: Mạng CNN từ đầu thường có phương sai dự đoán lớn trên các fold khác nhau, dễ kích hoạt điều kiện dừng sớm khi một fold bị sụt giảm.

### 4.2. Hướng đi Thay thế Mạnh mẽ nhất: H1.1 (Refined Intent & Grammar Contrast Features)

Nếu nhìn vào kết quả thực nghiệm mới nhất tại `output_v60_experiment/antigravity_h1_observed_outcome.json`:
- **H1 (Intent Features) là hướng duy nhất từ trước đến nay đạt kết quả dương đồng đều trên cả 3 Cluster Folds**:
  - `cluster_baseline`: 0.724084
  - `cluster_candidate`: 0.727055
  - $\Delta \text{cluster}$: **+0.002971** (sát nút ngưỡng 0.003)
  - Điểm từng fold: **[+0.002388, +0.002212, +0.003993]** (cả 3 fold đều tăng vững chắc!)
  - Paired cluster bootstrap 95% CI: **[+0.000471, +0.006022]** (hoàn toàn dương!)
- Lý do duy nhất V60 chưa được xuất submission là vì kiểm tra dev bị giảm nhẹ (−0.000465, do sửa 6 lỗi nhưng phát sinh 10 lỗi).
- **Đề xuất H1.1**: Thay vì mạo hiểm với mạng nơ-ron từ đầu, việc tinh chỉnh các đặc trưng ngữ pháp của H1 (chính xác hóa ranh giới các từ phủ định/trì hoãn, triệt tiêu 10 lỗi phát sinh trên dev) có xác suất thành công cao hơn nhiều để vượt ngưỡng 0.003 và duy trì tín hiệu dev tích cực.

---

## 5. Bảng Phân định Trạng thái Thực thi Mã nguồn

| Tên File | Vị trí | Trạng thái Thực thi | Môi trường | Mô tả |
| :--- | :--- | :--- | :--- | :--- |
| `h2_subword_cnn.py` | `antigravity_review/next_round/` | **Kiểm tra cú pháp & AST (Exit 0)** | Mac / Thiết kế cho Colab | Tokenizer n-gram xen kẽ, MaskedSubwordCNN, Focal Loss, training loop. |
| `h2_experiment.py` | `antigravity_review/next_round/` | **Kiểm tra cú pháp & AST (Exit 0)** | Mac / Thiết kế cho Colab | Driver thực thi thí nghiệm trên cluster fold và smoke test. |
| `test_next_round.py` | `antigravity_review/next_round/` | **Đã chạy thực tế trên Mac (Exit 0)** | Mac (Python stdlib + NumPy) | Kiểm tra AST, kiểm tra logic xen kẽ n-gram 3-5 của Tokenizer, xác minh guard chặn máy Mac. |

---

## 6. Hướng dẫn Chạy Thử nghiệm trên Colab (Dành cho Codex)

```bash
cd /content/antigravity_review/next_round
PY=/content/ds102_v53/venv/bin/python

# Bước 1: Chạy Smoke test (kiểm tra label-blindness và batch-size invariance)
$PY h2_experiment.py \
    --inputs /content/ds102_v53/inputs \
    --output /content/ds102_h2_experiment/smoke \
    --smoke

# Bước 2: Chạy Thử nghiệm Chính với ClassBalancedFocalLoss
$PY h2_experiment.py \
    --inputs /content/ds102_v53/inputs \
    --output /content/ds102_h2_experiment/run_focal \
    --loss-type focal

# Bước 3: (Tùy chọn ablation) Chạy Thử nghiệm với WeightedCrossEntropyLoss
$PY h2_experiment.py \
    --inputs /content/ds102_v53/inputs \
    --output /content/ds102_h2_experiment/run_ce \
    --loss-type cross_entropy
```

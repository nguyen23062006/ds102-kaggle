"""Đóng gói notebook độc lập từ cùng source đã kiểm tra ở local."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def code_source(name, drop_prefixes=()):
    text = (ROOT/name).read_text(encoding='utf-8')
    text = text.split('\nif __name__ == ', 1)[0]
    return '\n'.join(line for line in text.splitlines() if not line.startswith(drop_prefixes))


def cell(kind, text):
    import hashlib
    result = {'cell_type': kind, 'id': hashlib.sha256(text.encode()).hexdigest()[:12],
              'metadata': {}, 'source': text.splitlines(keepends=True)}
    if kind == 'code':
        compile(text, '<notebook-cell>', 'exec')
        result.update(execution_count=None, outputs=[])
    return result


baseline = code_source('train_submit.py').split('\ndef main(', 1)[0]
features = code_source('improved_features.py', ('from train_submit import ',))
calibration = code_source('select_v2.py').split('\ndef select(', 1)[0]
pipeline = code_source('train_submit_v2.py', ('from train_submit import ', 'from improved_features import ', 'from select_v2 import '))
nb = {
    'nbformat': 4, 'nbformat_minor': 5,
    'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python', 'version': '3.12'}},
    'cells': [
        cell('markdown', '''# DS102 — bản cải thiện v2 (CPU)

Mục tiêu: F1-macro của 4 nhãn spam. Dùng TF-IDF từ/ký tự, đặc trưng review, mức trùng với tên/mô tả sản phẩm, rating, category, num_sold và num_review. Không cần GPU, pretrained model hoặc dữ liệu ngoài.

**Cách dùng:** Import notebook này → Add Input dữ liệu cuộc thi → Run All. Các thư viện cần thiết là numpy, pandas, scipy và scikit-learn (1.5–1.7). Notebook tự chứa code, không cần upload các file `.py`.

Cấu hình mô hình đã được chọn bằng thử nghiệm cross-validation trên train. Notebook chạy lại 3-fold StratifiedGroupKFold cho cấu hình này, kiểm tra calibration bằng các dự đoán OOF, fit toàn bộ train, đánh giá dev rồi tạo `submission_v2.csv`. Comment giống nhau sau chuẩn hóa nằm cùng fold. Mọi vocabulary/IDF/scaler/trọng số mô hình chỉ học từ phần train tương ứng. Không gộp dev vào train.

CV được dùng chọn mô hình nên là ước lượng phục vụ phát triển, không phải một test độc lập. Dev đã được dùng xem lỗi bản cũ; điểm dev không bảo đảm điểm leaderboard. Mỗi lần chạy giữ seed=102. Chưa xác minh được Rules về thư viện của môn học.
'''),
        cell('markdown', '## 1. Đọc dữ liệu, chuẩn hóa và tính metric'),
        cell('code', baseline),
        cell('markdown', '## 2. Đặc trưng văn bản và quan hệ review/sản phẩm'),
        cell('code', features),
        cell('markdown', '## 3. Điều chỉnh ngưỡng trên OOF train'),
        cell('code', calibration),
        cell('markdown', '## 4. Pipeline huấn luyện và kiểm tra submission'),
        cell('code', pipeline),
        cell('markdown', '''## 5. Chạy

Giữ `DATA_DIR = None` trên Kaggle để tự tìm CSV dưới `/kaggle/input`. Nếu notebook gắn nhiều bộ dữ liệu cùng tên file, điền đường dẫn chính xác. Chạy CPU có thể mất vài phút, tùy máy. File cần nộp là **submission_v2.csv** trong Output.
'''),
        cell('code', '''DATA_DIR = None
OUTPUT_DIR = "/kaggle/working" if Path("/kaggle/working").is_dir() else "output_v2"
args = ["--output-dir", OUTPUT_DIR, "--folds", "3"]
if DATA_DIR is not None:
    args += ["--data-dir", DATA_DIR]
result = main_v2(args)
'''),
        cell('code', '''submission = pd.read_csv(Path(OUTPUT_DIR) / "submission_v2.csv")
display(submission.head())
print("Rows:", len(submission))
print("Dev F1-macro:", result["dev"]["macro_f1"])
''')
    ]
}
(ROOT/'ds102_kaggle_v2.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook v2 đã tạo; tất cả code cells hợp lệ về cú pháp.')

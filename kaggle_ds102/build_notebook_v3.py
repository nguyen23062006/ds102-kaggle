"""Notebook v3 tự chứa toàn bộ source cần thiết; không sửa notebook v2."""
import hashlib
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent


def source(name,remove=()):
    s=re.split(r'\nif __name__\s*==', (root/name).read_text(), maxsplit=1)[0]
    return '\n'.join(line for line in s.splitlines() if not line.startswith(remove))


def cell(kind,s):
    c={'cell_type':kind,'id':hashlib.sha256(s.encode()).hexdigest()[:12],'metadata':{},'source':s.splitlines(keepends=True)}
    if kind=='code':
        compile(s,'<cell>','exec')
        c.update(execution_count=None,outputs=[])
    return c


base=source('train_submit.py').split('\ndef main(',1)[0]
features=source('improved_features.py',('from train_submit import ',))
v2=source('train_submit_v2.py',('from train_submit import ','from improved_features import ','from select_v2 import ')).split('\ndef main_v2(',1)[0]
cat=source('catboost_v3.py',('from improved_features import ','from train_submit import '))
v3=source('train_submit_v3.py',('from train_submit import ','from improved_features import ','from train_submit_v2 import ','from catboost_v3 import '))
cells=[
 cell('markdown','''# DS102 — v3: SVM, NB-SVM và CatBoost

Chạy bằng **CPU**, không sử dụng pretrained model hoặc dữ liệu ngoài. Tham số đã được chốt bằng 3-fold cross-validation trên train. Notebook huấn luyện các mô hình trên toàn bộ train, đánh giá dev và tạo `submission_v3.csv`; không chạy lại quá trình tìm tham số. Các kết quả CV tham chiếu trong báo cáo là kết quả thí nghiệm đã chạy trước đó trên đúng checksum dữ liệu.

Import notebook → **Add Input** dữ liệu cuộc thi → **Run All**. Không cần upload các file `.py`. Cần numpy, pandas, scipy, scikit-learn và catboost. Bản local đã kiểm tra với scikit-learn 1.7.2 và catboost 1.2.10. Nếu thiếu CatBoost, cài trong một cell `%pip install catboost` trước khi chạy, hoặc dùng môi trường đã có sẵn thư viện.

Mô hình kết hợp 50% điểm chuẩn hóa của v2 và 50% điểm CatBoost. Không dùng ID làm đặc trưng; không dùng nhãn dev hay test để huấn luyện. Dev được dùng đánh giá trong quá trình phát triển nhiều phiên bản, nên điểm dev không thay thế điểm leaderboard. V2 đạt 0,75946 trên Kaggle theo người dùng; điểm v3 chưa biết cho đến khi nộp.
'''),
 cell('markdown','## 1. Đọc dữ liệu và đặc trưng'),cell('code',base),cell('code',features),
 cell('markdown','## 2. SVM và NB-SVM'),cell('code',v2),
 cell('markdown','## 3. CatBoost và ghép kết quả'),cell('code',cat),cell('code',v3),
 cell('markdown','''## 4. Chạy huấn luyện và xuất CSV

Giữ `DATA_DIR = None` để tự tìm bốn file CSV dưới `/kaggle/input`. Nếu có nhiều bản cùng tên, điền đúng thư mục chứa bộ dữ liệu muốn dùng. Trên máy cá nhân/Colab, điền đường dẫn dữ liệu. CSV để nộp sẽ ở `/kaggle/working/submission_v3.csv` trên Kaggle.
'''),
 cell('code','''DATA_DIR = None
OUTPUT_DIR = "/kaggle/working" if Path("/kaggle/working").is_dir() else "output_v3"
args = ["--output-dir", OUTPUT_DIR]
if DATA_DIR is not None:
    args += ["--data-dir", DATA_DIR]
result = main_v3(args)
'''),
 cell('code','''submission = pd.read_csv(Path(OUTPUT_DIR) / "submission_v3.csv")
display(submission.head())
print("Số dòng:", len(submission))
print("Dev v2:", result["v2_dev_same_run"]["macro_f1"])
print("Dev v3:", result["dev"]["macro_f1"])
''')]
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}},'cells':cells}
(root/'ds102_kaggle_v3.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Notebook v3 created; code cells compile.')

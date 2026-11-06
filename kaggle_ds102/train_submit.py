"""CPU baseline cho DS102: TF-IDF + Linear SVM, chọn theo F1-macro.

Chỉ dùng comment và nhãn train. Dev được đánh giá sau khi chọn tham số
bằng cross-validation trên train; không gộp dev vào train.
"""

import argparse
import html
import json
import platform
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SEED = 102
LABELS = [0, 1, 2, 3]
FILES = {name: f"ds102_{name}.csv" for name in ["train", "dev", "test", "samplesubmission"]}


def normalize_text(value):
    """Giữ dấu, emoji, dấu câu và ký tự lặp vì có thể giúp nhận diện spam."""
    text = html.unescape(unicodedata.normalize("NFC", str(value)))
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    return re.sub(r"\s+", " ", text).strip().lower()


class ReviewStats(BaseEstimator, TransformerMixin):
    """Đặc trưng độ dài, lặp ký tự và tỷ lệ ký tự; không có từ khóa nhãn."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            chars = len(text)
            words = text.split()
            nwords = len(words)
            denom = max(chars, 1)
            word_denom = max(nwords, 1)
            letters = [c for c in text if c.isalpha()]
            longest_run = max((len(m.group()) for m in re.finditer(r"(.)\1*", text)), default=0)
            rows.append([
                np.log1p(chars), np.log1p(nwords), chars / word_denom,
                len(set(words)) / word_denom, len(set(text)) / denom,
                longest_run / denom, np.log1p(longest_run),
                sum(c.isdigit() for c in text) / denom,
                len(letters) / denom,
                sum(not c.isalnum() and not c.isspace() for c in text) / denom,
                sum(ord(c) > 0xFFFF for c in text) / denom,
                text.count("!") / denom, text.count("?") / denom,
                float(bool(re.search(r"https?://|www\.", text))),
            ])
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float32))


def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=1.0, max_features=90000,
            sublinear_tf=True, lowercase=False, dtype=np.float32,
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("char", TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=3, max_features=160000,
            sublinear_tf=True, lowercase=False, dtype=np.float32,
        )),
        ("stats", Pipeline([
            ("extract", ReviewStats()), ("scale", StandardScaler(with_mean=False)),
        ])),
    ], transformer_weights={"word": 1.0, "char": 1.0, "stats": 0.1})


def make_model(config, y):
    weight = config["weight"]
    if weight == "sqrt_balanced":
        counts = pd.Series(y).value_counts()
        weight = {int(k): float(np.sqrt(len(y) / (len(counts) * v))) for k, v in counts.items()}
    elif weight == "none":
        weight = None
    return LinearSVC(C=config["C"], class_weight=weight, dual=True,
                     max_iter=10000, tol=1e-4, random_state=SEED)


def find_data_dir(supplied=None):
    if supplied:
        path = Path(supplied).expanduser().resolve()
        if all((path / f).is_file() for f in FILES.values()):
            return path
        raise FileNotFoundError(f"Thiếu một trong 4 file CSV tại {path}")
    candidates = [Path.cwd(), Path.cwd() / "data", Path.cwd() / "dataset"]
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        candidates += sorted({p.parent for p in kaggle.rglob(FILES["train"])})
    matches = [p for p in candidates if all((p / f).is_file() for f in FILES.values())]
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError("Hãy truyền --data-dir đến thư mục chứa đúng 4 CSV; trên Kaggle dùng Add Input trước.")


def load_data(data_dir):
    data = {name: pd.read_csv(data_dir / file, keep_default_na=False) for name, file in FILES.items()}
    for name in ["train", "dev", "test"]:
        required = {"comment", "spam_label"} if name != "test" else {"comment", "id"}
        if not required.issubset(data[name].columns):
            raise ValueError(f"{name}: thiếu cột {required - set(data[name].columns)}")
        if data[name].empty:
            raise ValueError(f"{name} rỗng")
    for name in ["train", "dev"]:
        values = pd.to_numeric(data[name]["spam_label"], errors="raise")
        if not values.isin(LABELS).all():
            raise ValueError(f"{name}: nhãn phải thuộc {LABELS}")
        data[name]["spam_label"] = values.astype(int)
    if set(data["train"]["spam_label"]) != set(LABELS):
        raise ValueError("Train phải có đủ 4 nhãn")
    sample = data["samplesubmission"]
    if list(sample.columns) != ["ID", "spam_label"]:
        raise ValueError("Sample phải có hai cột ID,spam_label")
    test_ids, sample_ids = data["test"]["id"].astype(str), sample["ID"].astype(str)
    if not test_ids.is_unique or not sample_ids.is_unique or set(test_ids) != set(sample_ids):
        raise ValueError("ID trong test và sample phải duy nhất và trùng tập hợp")
    return data


def metrics(y, pred):
    return {
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "classification_report": classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)
    if args.folds < 2:
        parser.error("--folds phải >= 2")
    data_dir = find_data_dir(args.data_dir)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(data_dir)
    train, dev, test = [data[n] for n in ["train", "dev", "test"]]
    texts = train["comment"].map(normalize_text).to_numpy()
    dev_texts = dev["comment"].map(normalize_text).to_numpy()
    test_texts = test["comment"].map(normalize_text).to_numpy()
    y = train["spam_label"].to_numpy()
    if train.groupby(train["comment"].map(normalize_text))["spam_label"].first().value_counts().min() < args.folds:
        raise ValueError("Không đủ nhóm cho số folds đã chọn")
    print(f"Train={len(train)}, dev={len(dev)}, test={len(test)}", flush=True)
    print(f"Nhãn train: {train.spam_label.value_counts().sort_index().to_dict()}", flush=True)
    configs = [{"C": c, "weight": weight} for weight in ["none", "sqrt_balanced", "balanced"] for c in [1.0, 4.0]]
    oof = np.full((len(configs), len(y)), -1, dtype=np.int64)
    fold_scores = [[] for _ in configs]
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(texts, y, groups=texts), 1):
        print(f"Fold {fold}/{args.folds}: fit TF-IDF chỉ trên train của fold...", flush=True)
        features = make_features()
        X_fit = features.fit_transform(texts[fit_idx])
        X_val = features.transform(texts[val_idx])
        for i, config in enumerate(configs):
            model = make_model(config, y[fit_idx])
            model.fit(X_fit, y[fit_idx])
            pred = model.predict(X_val)
            oof[i, val_idx] = pred
            score = f1_score(y[val_idx], pred, labels=LABELS, average="macro", zero_division=0)
            fold_scores[i].append(float(score))
            print(f"  {config}: F1-macro={score:.5f}", flush=True)
        del X_fit, X_val, features
    cv_scores = [float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)) for p in oof]
    best_idx = int(np.argmax(cv_scores))
    best = configs[best_idx]
    print(f"Chọn bằng OOF train: {best}, F1-macro={cv_scores[best_idx]:.5f}", flush=True)
    print("Huấn luyện cấu hình đã chọn trên toàn bộ train...", flush=True)
    features = make_features()
    X_train = features.fit_transform(texts)
    model = make_model(best, y)
    model.fit(X_train, y)
    dev_pred = model.predict(features.transform(dev_texts))
    dev_y = dev["spam_label"].to_numpy()
    report = metrics(dev_y, dev_pred)
    unique_dev = ~pd.Series(dev_texts).isin(set(texts)).to_numpy()
    print(f"Dev F1-macro={report['macro_f1']:.5f}, accuracy={report['accuracy']:.5f}", flush=True)
    print(classification_report(dev_y, dev_pred, labels=LABELS, digits=4, zero_division=0), flush=True)
    print(f"Dev có {int((~unique_dev).sum())} comment chuẩn hóa xuất hiện trong train.", flush=True)
    test_pred = model.predict(features.transform(test_texts)).astype(int)
    by_id = pd.Series(test_pred, index=test["id"].astype(str))
    submission = data["samplesubmission"].copy()
    submission["spam_label"] = submission["ID"].astype(str).map(by_id)
    if submission["spam_label"].isna().any() or not submission["spam_label"].isin(LABELS).all():
        raise ValueError("Submission có nhãn thiếu hoặc không hợp lệ")
    submission["spam_label"] = submission["spam_label"].astype(int)
    submission_path = output / "submission.csv"
    submission.to_csv(submission_path, index=False)
    saved = pd.read_csv(submission_path)
    pd.testing.assert_frame_equal(saved, submission, check_dtype=False)
    if len(saved) != len(test) or not saved["ID"].equals(data["samplesubmission"]["ID"]):
        raise ValueError("Sai số hàng hoặc thứ tự ID khi lưu")
    detail = pd.DataFrame({"comment": dev["comment"], "true_label": dev_y, "predicted_label": dev_pred,
                           "comment_seen_in_train": ~unique_dev})
    detail.to_csv(output / "dev_predictions.csv", index=False)
    full_report = {
        "seed": SEED, "python": platform.python_version(), "sklearn": sklearn.__version__,
        "rows": {k: len(v) for k, v in data.items()}, "feature_columns": ["comment"],
        "selection": "StratifiedGroupKFold on train; normalized comments as groups; OOF macro F1",
        "fit_data": "train only; dev and test never used to fit TF-IDF, scaler or SVM",
        "candidates": [{**c, "fold_macro_f1": f, "oof_macro_f1": s}
                       for c, f, s in zip(configs, fold_scores, cv_scores)],
        "chosen_config": best, "dev": report,
        "dev_comments_seen_in_train": int((~unique_dev).sum()),
        "dev_without_comment_overlap": metrics(dev_y[unique_dev], dev_pred[unique_dev]) if unique_dev.any() else None,
        "test_prediction_counts": {int(k): int(v) for k, v in submission.spam_label.value_counts().items()},
        "submission_rows_verified": len(saved),
    }
    (output / "metrics.json").write_text(json.dumps(full_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã kiểm tra và lưu {len(saved)} dự đoán: {submission_path}", flush=True)
    return full_report


if __name__ == "__main__":
    main()

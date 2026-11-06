"""Hypothesis 1: Explicit Hedging & Functional Review Intent Feature Extractor.

Addresses the semantic bottleneck between Class 0 (genuine product evaluation)
and Class 2 (logistics-only, unboxing, unopened/un-tested reviews).

Constraint adherence:
- No pretrained models or external weights.
- Only open-source standard libraries (re, numpy).
- Guarded: fitting must occur on hosted Colab.
"""

import os
import platform
import re
import numpy as np


def guard():
    """Ensure training/fitting only occurs in hosted Colab environments."""
    if platform.system() != 'Linux' or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Hosted Google Colab only; local fitting disabled.')


DEFERRAL_PATTERNS = [
    r'chưa\s+(dùng|xài|thử|mở|biết|uống|ăn|test|bóc|trải\s+nghiệm|khui|đập\s+hộp|sử\s+dụng)',
    r'chất\s+lượng\s+thì\s+chưa\s+biết',
    r'dùng\s+lâu\s+(thì\s+)?chưa\s+biết',
    r'thời\s+gian\s+sau\s*(mới|sẽ)?\s*biết',
    r'mua\s+(hộ|giùm|tặng|biếu|cho\s+người\s+thân|cho\s+bạn|cho\s+mẹ|cho\s+ba|cho\s+ck|cho\s+vk)',
    r'sẽ\s+(quay\s+lại|đánh\s+giá\s+sau|review\s+sau|nhận\s+xét\s+sau)',
    r'chưa\s+biết\s+(thế\s+nào|ntn|ra\s+sao|sao|hiệu\s+quả)',
    r'mới\s+nhận\s+(hàng\s+)?chưa',
    r'chưa\s+kịp\s+(dùng|thử|mở|uống|ăn)',
]

LOGISTICS_TERMS = [
    'giao hàng', 'giao', 'shipper', 'vận chuyển', 'đóng gói', 'hộp', 'bọc',
    'chống sốc', 'thùng', 'móp', 'móp méo', 'bể', 'vỡ', 'đúng hẹn', 'nhanh',
    'chậm', 'bao bì', 'túi bóng', 'nguyên vẹn', 'hàng đến', 'ship'
]

PRODUCT_EVAL_TERMS = [
    'chất lượng', 'vải', 'mùi', 'chất liệu', 'mặc', 'êm', 'bền', 'đẹp', 'vừa',
    'mát', 'dày', 'mỏng', 'tiện', 'xịn', 'rách', 'hư', 'hỏng', 'ngon', 'dở',
    'ngọt', 'chua', 'thấm', 'mềm', 'chắc chắn', 'sắc nét', 'hút', 'ấm'
]


class IntentFeatureExtractor:
    """Extract domain-specific intent features distinguishing Class 0 and Class 2."""

    def __init__(self):
        self.deferral_regexes = [re.compile(p, re.IGNORECASE) for p in DEFERRAL_PATTERNS]
        self.logistics_regexes = [re.compile(r'\b' + re.escape(t) + r'\b', re.IGNORECASE) for t in LOGISTICS_TERMS]
        self.eval_regexes = [re.compile(r'\b' + re.escape(t) + r'\b', re.IGNORECASE) for t in PRODUCT_EVAL_TERMS]

    def fit(self, X, y=None):
        # Deterministic rules; no state is learned from labels to avoid leakage
        return self

    def transform(self, df):
        """Transform dataframe into dense feature matrix.
        
        Expected columns: 'comment', 'rating', 'product_name'.
        """
        comments = df['comment'].fillna('').astype(str).tolist()
        ratings = df['rating'].fillna(5).astype(float).to_numpy()

        n = len(comments)
        features = np.zeros((n, 7), dtype=np.float32)

        for i, text in enumerate(comments):
            text_lower = text.lower()

            # 1. Deferral / Un-tested match count
            deferral_count = sum(1 for rgx in self.deferral_regexes if rgx.search(text_lower))
            has_deferral = 1.0 if deferral_count > 0 else 0.0

            # 2. Logistics terms count
            logistics_count = sum(1 for rgx in self.logistics_regexes if rgx.search(text_lower))

            # 3. Direct product evaluation terms count
            eval_count = sum(1 for rgx in self.eval_regexes if rgx.search(text_lower))

            # 4. Logistics-only ratio
            total_terms = logistics_count + eval_count
            logistics_ratio = logistics_count / max(total_terms, 1)

            # 5. Rating vs Deferral discrepancy (e.g. 5 stars rating despite not having used product)
            rating_val = ratings[i] / 5.0
            rating_deferral_interaction = rating_val * has_deferral

            # 6. Logistics only without evaluation flag
            logistics_only_flag = 1.0 if (logistics_count > 0 and eval_count == 0) else 0.0

            features[i, 0] = float(has_deferral)
            features[i, 1] = float(deferral_count)
            features[i, 2] = float(logistics_count)
            features[i, 3] = float(eval_count)
            features[i, 4] = float(logistics_ratio)
            features[i, 5] = float(rating_deferral_interaction)
            features[i, 6] = float(logistics_only_flag)

        return features

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


if __name__ == '__main__':
    # Local syntax and self-test verification without requiring external pandas
    class MockDataFrame(dict):
        def __getitem__(self, key):
            val = super().__getitem__(key)
            class SeriesWrapper(list):
                def fillna(self, default):
                    return SeriesWrapper([v if v is not None else default for v in self])
                def astype(self, t):
                    return SeriesWrapper([t(v) for v in self])
                def tolist(self):
                    return list(self)
                def to_numpy(self):
                    return np.array(self)
            return SeriesWrapper(val)

    test_samples = {
        'comment': [
            'Hàng đóng gói cẩn thận, shipper nhiệt tình. Chưa dùng nên chưa biết chất lượng thế nào.',
            'Máy giặt chạy êm, giặt sạch, vải không bị nhăn. Rất hài lòng.',
            'Mua tặng sinh nhật mẹ chưa mang về nên chưa mở hộp.',
            'Quá tệ, vải mỏng tang, rách chỉ.'
        ],
        'rating': [5, 5, 5, 1],
        'product_name': ['Bút chì', 'Máy giặt', 'Khăn lụa', 'Áo thun']
    }
    df_test = MockDataFrame(test_samples)
    extractor = IntentFeatureExtractor()
    feat = extractor.transform(df_test)
    print("Self-test IntentFeatureExtractor shape:", feat.shape)
    assert feat.shape == (4, 7)
    assert feat[0, 0] == 1.0  # Deferral detected
    assert feat[1, 0] == 0.0  # No deferral
    assert feat[2, 0] == 1.0  # Deferral detected
    print("Self-test passed successfully!")

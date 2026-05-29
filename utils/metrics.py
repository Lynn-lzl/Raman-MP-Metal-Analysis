import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

class MetricsUtil:
    @staticmethod
    def col_letter_to_index(col_letter: str) -> int:
        col_letter = col_letter.strip().upper()
        idx = 0
        for ch in col_letter:
            if not ('A' <= ch <= 'Z'):
                raise ValueError(f"Illegal listed characters: {ch}")
            idx = idx * 26 + (ord(ch) - ord('A') + 1)
        return idx - 1

    @staticmethod
    def compute_metrics(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        sd = np.std(y_true, ddof=1)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rpd = sd / rmse if rmse > 0 else np.inf
        return r2, rmse, rpd

    @staticmethod
    def ravel_pred(pred):
        pred = np.asarray(pred)
        return pred.ravel() if pred.ndim > 1 else pred
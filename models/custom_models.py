import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from utils.logger import Logger
from config import Config

USE_TABPFN = True
TABPFN_DEVICE = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        TABPFN_DEVICE = "cuda"
    from tabpfn import TabPFNRegressor
except Exception:
    USE_TABPFN = False
    TabPFNRegressor = None


class RandomSubspaceRidgeEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=60, subspace_dim=5, alpha=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.subspace_dim = subspace_dim
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        _, n_features = X.shape
        sub_dim = min(int(self.subspace_dim), n_features)
        rng = np.random.RandomState(self.random_state)

        self.models_ = []
        self.feature_indices_ = []
        for _ in range(int(self.n_estimators)):
            idx = rng.choice(n_features, size=sub_dim, replace=False)
            model = Ridge(alpha=self.alpha, random_state=rng.randint(0, 10 ** 9))
            model.fit(X[:, idx], y)
            self.models_.append(model)
            self.feature_indices_.append(idx)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.column_stack(
            [m.predict(X[:, idx]) for m, idx in zip(self.models_, self.feature_indices_)]
        )
        return preds.mean(axis=1)


class MetaModelFactory:
    @staticmethod
    def _make_tabpfn():
        if not (USE_TABPFN and TabPFNRegressor is not None):
            return None
        try:
            return TabPFNRegressor(
                device=TABPFN_DEVICE,
                random_state=Config.RANDOM_STATE
            )
        except TypeError:
            return TabPFNRegressor(
                random_state=Config.RANDOM_STATE
            )

    @staticmethod
    def build_meta():
        tab = MetaModelFactory._make_tabpfn()
        if tab is None:
            Logger.log("[WARN] Meta expects TabPFN, but it is currently unavailable -> fallback Ridge")
            return Pipeline([
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0, random_state=Config.RANDOM_STATE))
            ])

        return Pipeline([
            ("scaler", StandardScaler()),
            ("tabpfn", tab)
        ])
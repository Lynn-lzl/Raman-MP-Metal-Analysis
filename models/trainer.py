import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.base import clone
from models.custom_models import MetaModelFactory
from utils.metrics import MetricsUtil

try:
    from tabpfn import TabPFNRegressor
except Exception:
    TabPFNRegressor = None

class ModelTrainer:
    @staticmethod
    def cross_val_predict_safe(estimator, X, y, cv, method="predict"):
        X = np.asarray(X)
        y = np.asarray(y)

        is_tabpfn = (TabPFNRegressor is not None) and isinstance(estimator, TabPFNRegressor)
        if is_tabpfn:
            pred = np.full(X.shape[0], np.nan, dtype=float)
            for _, (tr, te) in enumerate(cv.split(X, y), start=1):
                try:
                    est = clone(estimator)
                except Exception:
                    est = MetaModelFactory._make_tabpfn()
                est.fit(X[tr], y[tr])
                pred[te] = MetricsUtil.ravel_pred(getattr(est, method)(X[te]))
            return pred

        return MetricsUtil.ravel_pred(cross_val_predict(clone(estimator), X, y, cv=cv, n_jobs=-1, method=method))

    @staticmethod
    def tune_model(estimator, param_grid, X, y, inner_cv, scoring="r2", n_jobs=-1):
        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=inner_cv,
            scoring=scoring,
            n_jobs=n_jobs
        )
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_, gs.best_score_
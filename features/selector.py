import numpy as np
from sklearn.linear_model import MultiTaskLassoCV
from config import Config

class FeatureSelector:
    @staticmethod
    def select_shared_bands_multitask(X_train, Y_train, max_features=None):
        if max_features is None:
            max_features = Config.SHARED_MAX_FEATURES
        mtl = MultiTaskLassoCV(
            alphas=np.logspace(-4, 1, 15),
            cv=Config.INNER_SPLITS,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1,
            max_iter=10000
        )
        mtl.fit(X_train, Y_train)

        coef_abs = np.abs(mtl.coef_)
        importance_all = coef_abs.sum(axis=0)

        nonzero_any = np.any(coef_abs > 1e-6, axis=0)
        nonzero_idx = np.where(nonzero_any)[0]

        if len(nonzero_idx) == 0:
            shared_idx = np.argsort(importance_all)[::-1][:max_features]
        else:
            sorted_nonzero = np.argsort(importance_all[nonzero_idx])[::-1]
            shared_idx = nonzero_idx[sorted_nonzero]
            if len(shared_idx) > max_features:
                shared_idx = shared_idx[:max_features]

        return np.sort(shared_idx), importance_all

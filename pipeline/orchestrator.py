import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.kernel_approximation import RBFSampler
from sklearn.base import clone
from sklearn.metrics import r2_score

from config import Config
from utils.logger import Logger, Timer
from utils.metrics import MetricsUtil
from features.selector import FeatureSelector
from features.builder import FeatureBuilder
from models.custom_models import RandomSubspaceRidgeEnsemble, MetaModelFactory
from models.trainer import ModelTrainer
from pipeline.reporter import ResultReporter


class NestedCVOrchestrator:
    def __init__(self, data_path, output_path, target_metals):
        self.data_path = data_path
        self.output_path = output_path
        self.target_metals = target_metals

    def run(self):
        Logger.log_step("0) Loading Data (Smart Column Recognition)")
        try:
            df = pd.read_csv(self.data_path)
            Logger.log(f"[DATA] path={self.data_path} | raw shape={df.shape}")
        except FileNotFoundError:
            Logger.log(f"[ERROR] File not found, please check the path: {self.data_path}")
            return

        sample_col = df.columns[0]

        for m in self.target_metals:
            if m not in df.columns:
                raise ValueError(f"[ERROR] Target column '{m}' not found in the dataset.")
        Y_full_all = df[self.target_metals].values

        # 3. Dynamically extract spectral features X (pure numeric headers between 400-1800)
        wavelength_cols = []
        for col in df.columns:
            try:
                val = float(col)
                if 400 <= val <= 1800:
                    wavelength_cols.append(col)
            except (ValueError, TypeError):
                continue

        if not wavelength_cols:
            raise ValueError("[ERROR] No valid spectral features found in the 400-1800 range.")

        Logger.log(
            f"[DATA] Auto-detected {len(wavelength_cols)} spectral feature bands "
            f"(from {wavelength_cols[0]} to {wavelength_cols[-1]})"
        )

        X_full_all = df[wavelength_cols].values
        wavelength_cols = pd.Index(wavelength_cols)

        mask = ~np.any(np.isnan(Y_full_all), axis=1)
        X_full = X_full_all[mask]
        Y = Y_full_all[mask]
        sample_ids = df.loc[mask, sample_col].values

        n_samples = X_full.shape[0]
        Logger.log(f"[DATA] Valid samples after filtering NaNs: {n_samples}")

        fold_perf_records = []
        outer_cv = KFold(
            n_splits=Config.OUTER_SPLITS,
            shuffle=True,
            random_state=Config.RANDOM_STATE
        )

        summary_model_names = ["PLS", "KRR", "ElasticNet", "RFRR", "RSRidge", "GlobalStack", "MoE"]
        oof_pred_by_model = {
            model_name: {m: np.full(n_samples, np.nan) for m in self.target_metals}
            for model_name in summary_model_names
        }
        oof_pred_global = oof_pred_by_model["GlobalStack"]
        oof_pred_moe = oof_pred_by_model["MoE"]
        outer_fold_ids = np.full(n_samples, -1, dtype=int)

        for outer_id, (tr_idx, te_idx) in enumerate(outer_cv.split(X_full, Y), start=1):
            Logger.log_step(f"[Outer {outer_id}/{Config.OUTER_SPLITS}] Split")
            outer_fold_ids[te_idx] = outer_id

            X_tr_full, X_te_full = X_full[tr_idx], X_full[te_idx]
            Y_tr, Y_te = Y[tr_idx], Y[te_idx]

            # 只拆 inner_cv：一个用于调参，一个用于生成OOF/训练meta
            inner_tune_cv = KFold(
                n_splits=Config.INNER_SPLITS,
                shuffle=True,
                random_state=Config.RANDOM_STATE + outer_id
            )
            inner_meta_cv = KFold(
                n_splits=Config.INNER_SPLITS,
                shuffle=True,
                random_state=Config.RANDOM_STATE + 1000 + outer_id
            )

            target_feature_limits = getattr(Config, "TARGET_SHARED_MAX_FEATURES", {}) or {}
            target_feature_limits = {
                target: int(max_features)
                for target, max_features in target_feature_limits.items()
            }
            outer_max_features = max(
                [int(Config.SHARED_MAX_FEATURES)]
                + [
                    target_feature_limits[target]
                    for target in self.target_metals
                    if target in target_feature_limits
                ]
            )
            if outer_max_features <= 0:
                raise ValueError("[ERROR] SHARED_MAX_FEATURES must be a positive integer.")

            # Shared Feature Selection
            with Timer() as tm:
                shared_idx_pool, importance_all = FeatureSelector.select_shared_bands_multitask(
                    X_tr_full,
                    Y_tr,
                    max_features=outer_max_features
                )

            Logger.log(
                f"[SharedBands] outer={outer_id} selected pool size={len(shared_idx_pool)} "
                f"max_features={outer_max_features}"
            )

            for j, metal in enumerate(self.target_metals):
                Logger.log_step(f"[Outer {outer_id}] 3) Target Metal = {metal}")
                y_tr, y_te = Y_tr[:, j], Y_te[:, j]

                target_max_features = int(target_feature_limits.get(metal, Config.SHARED_MAX_FEATURES))
                if target_max_features <= 0:
                    raise ValueError(f"[ERROR] Feature limit for target '{metal}' must be positive.")

                ranked_pool = shared_idx_pool[np.argsort(importance_all[shared_idx_pool])[::-1]]
                shared_idx = np.sort(ranked_pool[:target_max_features])

                Logger.log(
                    f"[SharedBands] outer={outer_id} target={metal} "
                    f"target_max_features={target_max_features} selected={len(shared_idx)}"
                )
                _shared_waves, _shared_waves_float = ResultReporter.print_shared_bands(shared_idx, wavelength_cols)
                selected_waves = wavelength_cols[shared_idx]
                waves_float = pd.to_numeric(selected_waves, errors="coerce").values

                zone_feature_indices, zone_desc = FeatureBuilder.build_zones_physical(
                    selected_waves,
                    Config.PHYSICAL_ZONE_EDGES,
                    Config.PHYSICAL_ZONE_NAMES
                )

                X_tr_sel, X_te_sel = X_tr_full[:, shared_idx], X_te_full[:, shared_idx]

                # Global Features Generation
                X_tr_global, _, _, _, cospos_tr = FeatureBuilder.make_global_features_from_selected(
                    X_tr_sel, waves_float
                )
                X_te_global, _, _, _, cospos_te = FeatureBuilder.make_global_features_from_selected(
                    X_te_sel, waves_float
                )

                max_components = max(1, min(15, X_tr_sel.shape[1], len(y_tr) - 1))
                comp_list = list(range(1, max_components + 1))
                alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
                gamma_list = [0.0005, 0.001, 0.005, 0.01, 0.05]
                en_alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
                en_l1_list = [0.1, 0.3, 0.5, 0.7, 0.9]

                base_models = {}

                # 1. PLS
                base_pls = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())])
                best_pls, _, _ = ModelTrainer.tune_model(
                    base_pls,
                    {"pls__n_components": comp_list},
                    X_tr_global, y_tr, inner_tune_cv
                )
                base_models["PLS"] = best_pls

                # 2. KRR
                base_krr = Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel='rbf'))])
                best_krr, _, _ = ModelTrainer.tune_model(
                    base_krr,
                    {"krr__alpha": alpha_list, "krr__gamma": gamma_list},
                    X_tr_global, y_tr, inner_tune_cv
                )
                base_models["KRR"] = best_krr

                # 3. ElasticNet
                base_en = Pipeline([
                    ("scaler", StandardScaler()),
                    ("en", ElasticNet(max_iter=10000, random_state=Config.RANDOM_STATE))
                ])
                best_en, _, _ = ModelTrainer.tune_model(
                    base_en,
                    {"en__alpha": en_alpha_list, "en__l1_ratio": en_l1_list},
                    X_tr_global, y_tr, inner_tune_cv
                )
                base_models["ElasticNet"] = best_en

                # 4. RFRR
                base_rfrr = Pipeline([
                    ("scaler", StandardScaler()),
                    ("rbf", RBFSampler(random_state=Config.RANDOM_STATE)),
                    ("ridge", Ridge(random_state=Config.RANDOM_STATE))
                ])
                best_rfrr, _, _ = ModelTrainer.tune_model(
                    base_rfrr,
                    {
                        "rbf__gamma": [0.001, 0.005, 0.01],
                        "rbf__n_components": [80, 160],
                        "ridge__alpha": [0.01, 0.1, 1.0]
                    },
                    X_tr_global, y_tr, inner_tune_cv
                )
                base_models["RFRR"] = best_rfrr

                # 5. RSRidge
                p_in = int(X_tr_global.shape[1])
                subspace_candidates = sorted({
                    d for d in [2, 3, 5, 8, 10, 15, 20, 30] if d <= p_in
                }) or [max(2, min(5, p_in))]
                base_rsr = Pipeline([
                    ("scaler", StandardScaler()),
                    ("rsr", RandomSubspaceRidgeEnsemble(random_state=Config.RANDOM_STATE))
                ])
                best_rsr, _, _ = ModelTrainer.tune_model(
                    base_rsr,
                    {
                        "rsr__n_estimators": [30, 60, 120],
                        "rsr__subspace_dim": subspace_candidates,
                        "rsr__alpha": [0.1, 1.0]
                    },
                    X_tr_global, y_tr, inner_tune_cv
                )
                base_models["RSRidge"] = best_rsr

                # Base model metrics
                Z_tr_oof, Z_tr_full, Z_te = [], [], []
                for name, est in base_models.items():
                    pred_oof = ModelTrainer.cross_val_predict_safe(
                        est, X_tr_global, y_tr, cv=inner_meta_cv
                    )

                    est_full = clone(est).fit(X_tr_global, y_tr)
                    pred_tr_full = MetricsUtil.ravel_pred(est_full.predict(X_tr_global))
                    r2_tr, rmse_tr, rpd_tr = MetricsUtil.compute_metrics(y_tr, pred_tr_full)
                    fold_perf_records.append({
                        "Fold": outer_id, "Target": metal, "Model": name,
                        "Set": "Train", "R2": r2_tr, "RMSE": rmse_tr, "RPD": rpd_tr
                    })
                    Z_tr_oof.append(pred_oof)
                    Z_tr_full.append(pred_tr_full)

                    pred_te = MetricsUtil.ravel_pred(est_full.predict(X_te_global))
                    r2_te, rmse_te, rpd_te = MetricsUtil.compute_metrics(y_te, pred_te)
                    fold_perf_records.append({
                        "Fold": outer_id, "Target": metal, "Model": name,
                        "Set": "Test", "R2": r2_te, "RMSE": rmse_te, "RPD": rpd_te
                    })
                    oof_pred_by_model[name][metal][te_idx] = pred_te
                    Z_te.append(pred_te)

                Z_tr_oof = np.column_stack(Z_tr_oof)
                Z_tr_full = np.column_stack(Z_tr_full)
                Z_te = np.column_stack(Z_te)

                # Global Stacking
                meta_global = MetaModelFactory.build_meta()
                y_tr_stack_oof = ModelTrainer.cross_val_predict_safe(
                    meta_global, Z_tr_oof, y_tr, cv=inner_meta_cv
                )
                meta_global.fit(Z_tr_oof, y_tr)
                y_tr_stack_full = MetricsUtil.ravel_pred(meta_global.predict(Z_tr_full))
                r2_tr_s, rmse_tr_s, rpd_tr_s = MetricsUtil.compute_metrics(y_tr, y_tr_stack_full)
                fold_perf_records.append({
                    "Fold": outer_id, "Target": metal, "Model": "GlobalStack",
                    "Set": "Train", "R2": r2_tr_s, "RMSE": rmse_tr_s, "RPD": rpd_tr_s
                })

                y_te_stack = MetricsUtil.ravel_pred(meta_global.predict(Z_te))
                r2_te_s, rmse_te_s, rpd_te_s = MetricsUtil.compute_metrics(y_te, y_te_stack)
                fold_perf_records.append({
                    "Fold": outer_id, "Target": metal, "Model": "GlobalStack",
                    "Set": "Test", "R2": r2_te_s, "RMSE": rmse_te_s, "RPD": rpd_te_s
                })
                oof_pred_global[metal][te_idx] = y_te_stack

                # Zone Experts
                zone_tr_oof_preds, zone_tr_full_preds, zone_te_preds, zone_r2_train = [], [], [], []
                for zi, feat_idx in enumerate(zone_feature_indices):
                    X_tr_zone_int, X_te_zone_int = X_tr_sel[:, feat_idx], X_te_sel[:, feat_idx]
                    waves_zone = waves_float[feat_idx]

                    center_z_tr, width_z_tr, isum_z_tr = FeatureBuilder.make_summary_features(
                        X_tr_zone_int, waves_zone
                    )
                    center_z_te, width_z_te, isum_te_z = FeatureBuilder.make_summary_features(
                        X_te_zone_int, waves_zone
                    )

                    parts_tr = [X_tr_zone_int, center_z_tr, width_z_tr, isum_z_tr]
                    parts_te = [X_te_zone_int, center_z_te, width_z_te, isum_te_z]

                    if cospos_tr is not None:
                        parts_tr.append(cospos_tr)
                        parts_te.append(cospos_te)

                    z_local_tr = FeatureBuilder.make_zone_local_cospos_from_zone(X_tr_zone_int, waves_zone)
                    z_local_te = FeatureBuilder.make_zone_local_cospos_from_zone(X_te_zone_int, waves_zone)
                    if z_local_tr is not None:
                        parts_tr.append(z_local_tr)
                        parts_te.append(z_local_te)

                    X_tr_zone = np.concatenate(parts_tr, axis=1)
                    X_te_zone = np.concatenate(parts_te, axis=1)

                    max_comp_z = min(10, max(1, X_tr_zone_int.shape[1]), len(y_tr) - 1)

                    pls_z = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())])
                    best_pls_z, _, _ = ModelTrainer.tune_model(
                        pls_z,
                        {"pls__n_components": list(range(1, max_comp_z + 1))},
                        X_tr_zone, y_tr, inner_tune_cv
                    )

                    krr_z = Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf"))])
                    best_krr_z, _, _ = ModelTrainer.tune_model(
                        krr_z,
                        {"krr__alpha": alpha_list, "krr__gamma": gamma_list},
                        X_tr_zone, y_tr, inner_tune_cv
                    )

                    en_z = Pipeline([
                        ("scaler", StandardScaler()),
                        ("en", ElasticNet(max_iter=10000, random_state=Config.RANDOM_STATE))
                    ])
                    best_en_z, _, _ = ModelTrainer.tune_model(
                        en_z,
                        {"en__alpha": en_alpha_list, "en__l1_ratio": en_l1_list},
                        X_tr_zone, y_tr, inner_tune_cv
                    )

                    Zz_tr = np.column_stack([
                        ModelTrainer.cross_val_predict_safe(best_pls_z, X_tr_zone, y_tr, cv=inner_meta_cv),
                        ModelTrainer.cross_val_predict_safe(best_krr_z, X_tr_zone, y_tr, cv=inner_meta_cv),
                        ModelTrainer.cross_val_predict_safe(best_en_z, X_tr_zone, y_tr, cv=inner_meta_cv)
                    ])

                    pls_z_full = clone(best_pls_z).fit(X_tr_zone, y_tr)
                    krr_z_full = clone(best_krr_z).fit(X_tr_zone, y_tr)
                    en_z_full = clone(best_en_z).fit(X_tr_zone, y_tr)
                    Zz_tr_full = np.column_stack([
                        MetricsUtil.ravel_pred(pls_z_full.predict(X_tr_zone)),
                        MetricsUtil.ravel_pred(krr_z_full.predict(X_tr_zone)),
                        MetricsUtil.ravel_pred(en_z_full.predict(X_tr_zone))
                    ])
                    Zz_te = np.column_stack([
                        MetricsUtil.ravel_pred(pls_z_full.predict(X_te_zone)),
                        MetricsUtil.ravel_pred(krr_z_full.predict(X_te_zone)),
                        MetricsUtil.ravel_pred(en_z_full.predict(X_te_zone))
                    ])

                    meta_zone = MetaModelFactory.build_meta()
                    y_tr_z_oof = ModelTrainer.cross_val_predict_safe(
                        meta_zone, Zz_tr, y_tr, cv=inner_meta_cv
                    )
                    meta_zone.fit(Zz_tr, y_tr)
                    zone_tr_oof_preds.append(y_tr_z_oof)
                    zone_tr_full_preds.append(MetricsUtil.ravel_pred(meta_zone.predict(Zz_tr_full)))
                    zone_te_preds.append(MetricsUtil.ravel_pred(meta_zone.predict(Zz_te)))
                    zone_r2_train.append(r2_score(y_tr, y_tr_z_oof))

                # MoE Processing
                keep_idx = [
                    i for i, r2z in enumerate(zone_r2_train)
                    if r2z > Config.MOE_ZONE_R2_THRESHOLD
                ]

                Logger.log(
                    f"[MoE] outer={outer_id} target={metal} "
                    f"threshold={Config.MOE_ZONE_R2_THRESHOLD} "
                    f"kept_zones={len(keep_idx)} "
                    f"keep_idx={keep_idx} "
                    f"zone_r2_train={[round(x, 4) for x in zone_r2_train]}"
                )

                Z_moe_tr = (
                    np.column_stack([zone_tr_oof_preds[i] for i in keep_idx] + [y_tr_stack_oof])
                    if keep_idx else y_tr_stack_oof.reshape(-1, 1)
                )
                Z_moe_tr_full = (
                    np.column_stack([zone_tr_full_preds[i] for i in keep_idx] + [y_tr_stack_full])
                    if keep_idx else y_tr_stack_full.reshape(-1, 1)
                )
                Z_moe_te = (
                    np.column_stack([zone_te_preds[i] for i in keep_idx] + [y_te_stack])
                    if keep_idx else y_te_stack.reshape(-1, 1)
                )

                moe_meta = MetaModelFactory.build_meta()
                moe_meta.fit(Z_moe_tr, y_tr)
                y_tr_moe_full = MetricsUtil.ravel_pred(moe_meta.predict(Z_moe_tr_full))
                r2_tr_m, rmse_tr_m, rpd_tr_m = MetricsUtil.compute_metrics(y_tr, y_tr_moe_full)
                fold_perf_records.append({
                    "Fold": outer_id, "Target": metal, "Model": "MoE",
                    "Set": "Train", "R2": r2_tr_m, "RMSE": rmse_tr_m, "RPD": rpd_tr_m
                })

                y_te_moe = MetricsUtil.ravel_pred(moe_meta.predict(Z_moe_te))
                r2_te_m, rmse_te_m, rpd_te_m = MetricsUtil.compute_metrics(y_te, y_te_moe)
                fold_perf_records.append({
                    "Fold": outer_id, "Target": metal, "Model": "MoE",
                    "Set": "Test", "R2": r2_te_m, "RMSE": rmse_te_m, "RPD": rpd_te_m
                })

                oof_pred_moe[metal][te_idx] = y_te_moe

        ResultReporter.export_oof_predictions_to_excel(
            self.output_path,
            sample_ids,
            outer_fold_ids,
            self.target_metals,
            Y,
            oof_pred_global,
            oof_pred_moe,
            oof_pred_by_model,
            pd.DataFrame(fold_perf_records)
        )

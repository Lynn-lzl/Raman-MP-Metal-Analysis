import argparse
import json
import os
import platform
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config
from features.builder import FeatureBuilder
from features.selector import FeatureSelector
from models.custom_models import MetaModelFactory, RandomSubspaceRidgeEnsemble
from models.trainer import ModelTrainer
from utils.logger import Logger
from utils.metrics import MetricsUtil


MODEL_FILE_NAME = "spectroscopy_pipeline.joblib"
META_FILE_NAME = "pipeline_meta.json"
SCALER_FILE_NAME = "offline_scaler.joblib"

DEFAULT_TARGET_METALS = ["118Sn (KED)", "209Bi (KED)"]
WAVE_MIN = 400.0
WAVE_MAX = 1800.0
SG_WINDOW = 9
SG_POLY = 2
SG_DERIV = 1


def suppress_warnings() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="^Objective did not converge.*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered in cast")
    warnings.filterwarnings("ignore", message=".*A worker stopped while some jobs were given to the executor.*")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Train deployable full-data models and export artifacts for roman-metal-app."
    )
    parser.add_argument(
        "--preprocessed-csv",
        type=Path,
        default=root / "data" / "Raman_spectroscopy_data_preprocessed.csv",
        help="CSV produced by data_preprocessing/spectral_preprocessing.py.",
    )
    parser.add_argument(
        "--raw-excel",
        type=Path,
        default=root / "data" / "Raman_spectroscopy_data.xlsx",
        help="Raw Excel file used to fit offline_scaler.joblib for website uploads.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=root / "artifacts",
        help="Destination artifacts directory.",
    )
    parser.add_argument(
        "--target-metals",
        nargs="+",
        default=DEFAULT_TARGET_METALS,
        help="Target columns to train and export.",
    )
    parser.add_argument(
        "--shared-max-features",
        type=int,
        default=None,
        help="Number of shared spectral bands for the deployable model. Defaults to max Config target setting.",
    )
    parser.add_argument(
        "--ridge-meta",
        action="store_true",
        help="Force Ridge meta models instead of TabPFN. Useful for smaller/faster artifacts.",
    )
    parser.add_argument(
        "--split-size-mb",
        type=int,
        default=0,
        help="Optional chunk size for spectroscopy_pipeline.joblib.partNNN files. 0 disables splitting.",
    )
    parser.add_argument(
        "--remove-unsplit-model",
        action="store_true",
        help="After splitting, remove spectroscopy_pipeline.joblib so the app rebuilds it from part files.",
    )
    return parser.parse_args()


def numeric_wave_columns(columns: Iterable[Any], wave_min: float = WAVE_MIN, wave_max: float = WAVE_MAX) -> List[Any]:
    wave_cols: List[Any] = []
    for col in columns:
        try:
            value = float(col)
        except (TypeError, ValueError):
            continue
        if wave_min <= value <= wave_max:
            wave_cols.append(col)
    return wave_cols


def snv_transform(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    return (X - mean) / std


def load_training_data(
    csv_path: Path,
    target_metals: Sequence[str],
) -> Tuple[pd.DataFrame, pd.Index, np.ndarray, np.ndarray, np.ndarray]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    for target in target_metals:
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in {csv_path}")

    wavelength_cols = pd.Index(numeric_wave_columns(df.columns))
    if wavelength_cols.empty:
        raise ValueError(f"No numeric wavelength columns in [{WAVE_MIN}, {WAVE_MAX}] were found.")

    Y_all = df[list(target_metals)].to_numpy(dtype=float)
    mask = ~np.any(np.isnan(Y_all), axis=1)
    X = df.loc[mask, wavelength_cols].to_numpy(dtype=float)
    Y = Y_all[mask]

    sample_col = df.columns[0]
    sample_ids = df.loc[mask, sample_col].to_numpy()
    return df, wavelength_cols, X, Y, sample_ids


def fit_offline_scaler(raw_excel_path: Path, expected_waves: Sequence[Any], processed_X: np.ndarray) -> StandardScaler:
    if not raw_excel_path.exists():
        raise FileNotFoundError(f"Raw Excel not found: {raw_excel_path}")

    raw_df = pd.read_excel(raw_excel_path)
    raw_wave_cols = numeric_wave_columns(raw_df.columns)
    if not raw_wave_cols:
        raise ValueError(f"No raw wavelength columns in [{WAVE_MIN}, {WAVE_MAX}] were found.")

    raw_waves = np.asarray([float(c) for c in raw_wave_cols], dtype=float)
    expected_waves_float = np.asarray([float(c) for c in expected_waves], dtype=float)
    if raw_waves.shape != expected_waves_float.shape or not np.allclose(raw_waves, expected_waves_float, atol=1e-6):
        raise ValueError(
            "Raw Excel wavelength columns do not match the preprocessed CSV wavelength columns. "
            "Run data_preprocessing/spectral_preprocessing.py again or check the input files."
        )

    spectra_raw = raw_df[raw_wave_cols].to_numpy(dtype=float)
    spectra_snv = snv_transform(spectra_raw)
    spectra_sg = savgol_filter(
        spectra_snv,
        window_length=SG_WINDOW,
        polyorder=SG_POLY,
        deriv=SG_DERIV,
        axis=1,
        mode="interp",
    )

    scaler = StandardScaler()
    processed_from_raw = scaler.fit_transform(spectra_sg)

    compare_rows = min(processed_from_raw.shape[0], processed_X.shape[0])
    compare_cols = min(processed_from_raw.shape[1], processed_X.shape[1])
    max_abs_diff = float(
        np.max(np.abs(processed_from_raw[:compare_rows, :compare_cols] - processed_X[:compare_rows, :compare_cols]))
    )
    if max_abs_diff > 1e-4:
        Logger.log(
            f"[WARN] Raw-derived preprocessing differs from CSV by max_abs_diff={max_abs_diff:.6g}. "
            "Continuing because the scaler is still required for website uploads."
        )
    else:
        Logger.log(f"[SCALER] Raw Excel preprocessing matches CSV (max_abs_diff={max_abs_diff:.6g}).")

    return scaler


def config_to_dict() -> Dict[str, Any]:
    keys = [
        "RANDOM_STATE",
        "OUTER_SPLITS",
        "INNER_SPLITS",
        "SHARED_MAX_FEATURES",
        "TARGET_SHARED_MAX_FEATURES",
        "PHYSICAL_ZONE_EDGES",
        "PHYSICAL_ZONE_NAMES",
        "MOE_ZONE_R2_THRESHOLD",
        "USE_GLOBAL_COSPOS",
        "USE_ZONE_LOCAL_COSPOS",
        "GLOBAL_COS_OMEGAS",
        "ZONE_COS_OMEGAS",
        "ZONE_LOCAL_COS_MIN_SIZE",
        "ZONE_LOCAL_COS_SELECTED",
    ]
    return {key: getattr(Config, key) for key in keys if hasattr(Config, key)}


def select_deploy_features(X: np.ndarray, Y: np.ndarray, max_features: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_features <= 0:
        raise ValueError("--shared-max-features must be positive.")

    shared_idx_pool, importance_all = FeatureSelector.select_shared_bands_multitask(
        X,
        Y,
        max_features=max_features,
    )
    ranked_pool = shared_idx_pool[np.argsort(importance_all[shared_idx_pool])[::-1]]
    shared_idx = np.sort(ranked_pool[:max_features])
    return shared_idx, importance_all


def tune_global_base_models(X_global: np.ndarray, y: np.ndarray, inner_cv: KFold) -> Dict[str, Any]:
    max_components = max(1, min(15, X_global.shape[1], len(y) - 1))
    comp_list = list(range(1, max_components + 1))
    alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    gamma_list = [0.0005, 0.001, 0.005, 0.01, 0.05]
    en_alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    en_l1_list = [0.1, 0.3, 0.5, 0.7, 0.9]

    base_pls = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())])
    best_pls, _, _ = ModelTrainer.tune_model(
        base_pls,
        {"pls__n_components": comp_list},
        X_global,
        y,
        inner_cv,
    )

    base_krr = Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf"))])
    best_krr, _, _ = ModelTrainer.tune_model(
        base_krr,
        {"krr__alpha": alpha_list, "krr__gamma": gamma_list},
        X_global,
        y,
        inner_cv,
    )

    base_en = Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNet(max_iter=10000, random_state=Config.RANDOM_STATE)),
    ])
    best_en, _, _ = ModelTrainer.tune_model(
        base_en,
        {"en__alpha": en_alpha_list, "en__l1_ratio": en_l1_list},
        X_global,
        y,
        inner_cv,
    )

    base_rfrr = Pipeline([
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(random_state=Config.RANDOM_STATE)),
        ("ridge", Ridge(random_state=Config.RANDOM_STATE)),
    ])
    best_rfrr, _, _ = ModelTrainer.tune_model(
        base_rfrr,
        {
            "rbf__gamma": [0.001, 0.005, 0.01],
            "rbf__n_components": [80, 160],
            "ridge__alpha": [0.01, 0.1, 1.0],
        },
        X_global,
        y,
        inner_cv,
    )

    p_in = int(X_global.shape[1])
    subspace_candidates = sorted({d for d in [2, 3, 5, 8, 10, 15, 20, 30] if d <= p_in}) or [
        max(2, min(5, p_in))
    ]
    base_rsr = Pipeline([
        ("scaler", StandardScaler()),
        ("rsr", RandomSubspaceRidgeEnsemble(random_state=Config.RANDOM_STATE)),
    ])
    best_rsr, _, _ = ModelTrainer.tune_model(
        base_rsr,
        {
            "rsr__n_estimators": [30, 60, 120],
            "rsr__subspace_dim": subspace_candidates,
            "rsr__alpha": [0.1, 1.0],
        },
        X_global,
        y,
        inner_cv,
    )

    return {
        "PLS": best_pls,
        "KRR": best_krr,
        "ElasticNet": best_en,
        "RFRR": best_rfrr,
        "RSRidge": best_rsr,
    }


def zone_feature_matrix(
    X_sel: np.ndarray,
    waves_float: np.ndarray,
    zone_idx: np.ndarray,
    cospos_full: np.ndarray | None,
) -> np.ndarray:
    X_zone_int = X_sel[:, zone_idx]
    waves_zone = waves_float[zone_idx]
    center_z, width_z, isum_z = FeatureBuilder.make_summary_features(X_zone_int, waves_zone)
    parts = [X_zone_int, center_z, width_z, isum_z]

    if cospos_full is not None:
        parts.append(cospos_full)

    z_local = FeatureBuilder.make_zone_local_cospos_from_zone(X_zone_int, waves_zone)
    if z_local is not None:
        parts.append(z_local)

    return np.concatenate(parts, axis=1)


def tune_zone_base_models(X_zone: np.ndarray, y: np.ndarray, raw_zone_width: int, inner_cv: KFold) -> Dict[str, Any]:
    alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    gamma_list = [0.0005, 0.001, 0.005, 0.01, 0.05]
    en_alpha_list = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    en_l1_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    max_comp_z = min(10, max(1, raw_zone_width), len(y) - 1)

    pls_z = Pipeline([("scaler", StandardScaler()), ("pls", PLSRegression())])
    best_pls_z, _, _ = ModelTrainer.tune_model(
        pls_z,
        {"pls__n_components": list(range(1, max_comp_z + 1))},
        X_zone,
        y,
        inner_cv,
    )

    krr_z = Pipeline([("scaler", StandardScaler()), ("krr", KernelRidge(kernel="rbf"))])
    best_krr_z, _, _ = ModelTrainer.tune_model(
        krr_z,
        {"krr__alpha": alpha_list, "krr__gamma": gamma_list},
        X_zone,
        y,
        inner_cv,
    )

    en_z = Pipeline([
        ("scaler", StandardScaler()),
        ("en", ElasticNet(max_iter=10000, random_state=Config.RANDOM_STATE)),
    ])
    best_en_z, _, _ = ModelTrainer.tune_model(
        en_z,
        {"en__alpha": en_alpha_list, "en__l1_ratio": en_l1_list},
        X_zone,
        y,
        inner_cv,
    )

    return {"PLS": best_pls_z, "KRR": best_krr_z, "ElasticNet": best_en_z}


def fit_full_models_from_oof(
    tuned_models: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    inner_meta_cv: KFold,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    oof_preds: List[np.ndarray] = []
    full_preds: List[np.ndarray] = []
    full_models: Dict[str, Any] = {}

    for name, estimator in tuned_models.items():
        pred_oof = ModelTrainer.cross_val_predict_safe(estimator, X, y, cv=inner_meta_cv)
        full_estimator = clone(estimator).fit(X, y)
        pred_full = MetricsUtil.ravel_pred(full_estimator.predict(X))

        oof_preds.append(pred_oof)
        full_preds.append(pred_full)
        full_models[name] = full_estimator

    return full_models, np.column_stack(oof_preds), np.column_stack(full_preds)


def train_target_bundle(
    target_name: str,
    X_sel: np.ndarray,
    waves_float: np.ndarray,
    y: np.ndarray,
    zone_feature_indices: Sequence[np.ndarray],
    zone_desc: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    Logger.log_step(f"Training deployable target: {target_name}")

    inner_tune_cv = KFold(
        n_splits=Config.INNER_SPLITS,
        shuffle=True,
        random_state=Config.RANDOM_STATE,
    )
    inner_meta_cv = KFold(
        n_splits=Config.INNER_SPLITS,
        shuffle=True,
        random_state=Config.RANDOM_STATE + 1000,
    )

    X_global, _, _, _, cospos_full = FeatureBuilder.make_global_features_from_selected(X_sel, waves_float)
    tuned_global = tune_global_base_models(X_global, y, inner_tune_cv)
    base_global, Z_global_oof, Z_global_full = fit_full_models_from_oof(
        tuned_global,
        X_global,
        y,
        inner_meta_cv,
    )

    meta_global_for_oof = MetaModelFactory.build_meta()
    global_stack_oof = ModelTrainer.cross_val_predict_safe(
        meta_global_for_oof,
        Z_global_oof,
        y,
        cv=inner_meta_cv,
    )

    meta_global = MetaModelFactory.build_meta()
    meta_global.fit(Z_global_oof, y)
    global_stack_full = MetricsUtil.ravel_pred(meta_global.predict(Z_global_full))

    base_zones: Dict[str, Dict[str, Any]] = {}
    meta_zones: Dict[str, Any] = {}
    zone_oof_preds: List[np.ndarray] = []
    zone_full_preds: List[np.ndarray] = []
    zone_r2_train: List[float] = []

    for zone_pos, zone_idx in enumerate(zone_feature_indices):
        zone_name = str(zone_desc[zone_pos]["name"]) if zone_pos < len(zone_desc) else f"Zone {zone_pos + 1}"
        Logger.log(f"[ZONE] target={target_name} zone={zone_name} raw_features={len(zone_idx)}")

        X_zone = zone_feature_matrix(X_sel, waves_float, np.asarray(zone_idx, dtype=int), cospos_full)
        tuned_zone = tune_zone_base_models(X_zone, y, raw_zone_width=len(zone_idx), inner_cv=inner_tune_cv)
        zone_base_models, Z_zone_oof, Z_zone_full = fit_full_models_from_oof(
            tuned_zone,
            X_zone,
            y,
            inner_meta_cv,
        )

        meta_zone_for_oof = MetaModelFactory.build_meta()
        zone_stack_oof = ModelTrainer.cross_val_predict_safe(
            meta_zone_for_oof,
            Z_zone_oof,
            y,
            cv=inner_meta_cv,
        )

        meta_zone = MetaModelFactory.build_meta()
        meta_zone.fit(Z_zone_oof, y)
        zone_stack_full = MetricsUtil.ravel_pred(meta_zone.predict(Z_zone_full))

        base_zones[zone_name] = zone_base_models
        meta_zones[zone_name] = meta_zone
        zone_oof_preds.append(zone_stack_oof)
        zone_full_preds.append(zone_stack_full)
        zone_r2_train.append(float(r2_score(y, zone_stack_oof)))

    keep_zone_idx = [
        idx
        for idx, r2_value in enumerate(zone_r2_train)
        if r2_value > Config.MOE_ZONE_R2_THRESHOLD
    ]
    Logger.log(
        f"[MoE] target={target_name} threshold={Config.MOE_ZONE_R2_THRESHOLD} "
        f"keep_idx={keep_zone_idx} zone_r2_train={[round(x, 4) for x in zone_r2_train]}"
    )

    if keep_zone_idx:
        Z_moe_oof = np.column_stack([zone_oof_preds[idx] for idx in keep_zone_idx] + [global_stack_oof])
    else:
        Z_moe_oof = global_stack_oof.reshape(-1, 1)

    moe_meta = MetaModelFactory.build_meta()
    moe_meta.fit(Z_moe_oof, y)

    train_pred = MetricsUtil.ravel_pred(
        moe_meta.predict(
            np.column_stack([zone_full_preds[idx] for idx in keep_zone_idx] + [global_stack_full])
            if keep_zone_idx
            else global_stack_full.reshape(-1, 1)
        )
    )
    train_r2, train_rmse, train_rpd = MetricsUtil.compute_metrics(y, train_pred)
    Logger.log(
        f"[TRAIN] target={target_name} MoE full-data R2={train_r2:.4f} "
        f"RMSE={train_rmse:.4f} RPD={train_rpd:.4f}"
    )

    return {
        "base_global": base_global,
        "meta_global": meta_global,
        "base_zones": base_zones,
        "meta_zones": meta_zones,
        "keep_zone_idx": keep_zone_idx,
        "zone_feature_indices": [np.asarray(idx, dtype=int) for idx in zone_feature_indices],
        "zone_r2_train": zone_r2_train,
        "train_metrics": {"r2": train_r2, "rmse": train_rmse, "rpd": train_rpd},
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        json.dump(jsonable(payload), tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_joblib(path: Path, payload: Any, compress: int = 3) -> None:
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        joblib.dump(payload, tmp_path, compress=compress)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def split_file(path: Path, chunk_size_mb: int, remove_original: bool = False) -> List[Path]:
    if chunk_size_mb <= 0:
        return []

    chunk_size = chunk_size_mb * 1024 * 1024
    for old_part in sorted(path.parent.glob(f"{path.name}.part*")):
        old_part.unlink()

    part_paths: List[Path] = []
    with open(path, "rb") as src:
        part_no = 1
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            part_path = path.parent / f"{path.name}.part{part_no:03d}"
            with open(part_path, "wb") as out:
                out.write(chunk)
            part_paths.append(part_path)
            part_no += 1

    if remove_original:
        path.unlink()

    return part_paths


def deployment_shared_feature_default(target_metals: Sequence[str]) -> int:
    target_limits = getattr(Config, "TARGET_SHARED_MAX_FEATURES", {}) or {}
    limits = [int(Config.SHARED_MAX_FEATURES)]
    limits.extend(int(target_limits[t]) for t in target_metals if t in target_limits)
    return max(limits)


def maybe_force_ridge_meta(use_ridge_meta: bool) -> None:
    if not use_ridge_meta:
        return

    import models.custom_models as custom_models
    import models.trainer as trainer

    custom_models.USE_TABPFN = False
    custom_models.TabPFNRegressor = None
    trainer.TabPFNRegressor = None
    Logger.log("[META] Using Ridge fallback meta models because --ridge-meta was set.")


def prepare_pickle_compat() -> None:
    # Streamlit app.py defines this class at top level for unpickling deployed artifacts.
    RandomSubspaceRidgeEnsemble.__module__ = "__main__"
    globals()["RandomSubspaceRidgeEnsemble"] = RandomSubspaceRidgeEnsemble


def main() -> None:
    args = parse_args()
    suppress_warnings()
    maybe_force_ridge_meta(args.ridge_meta)

    shared_max_features = args.shared_max_features or deployment_shared_feature_default(args.target_metals)
    Config.SHARED_MAX_FEATURES = int(shared_max_features)

    Logger.log(f"[*] Preprocessed CSV: {args.preprocessed_csv}")
    Logger.log(f"[*] Raw Excel: {args.raw_excel}")
    Logger.log(f"[*] Artifact dir: {args.artifact_dir}")
    Logger.log(f"[*] Targets: {', '.join(args.target_metals)}")
    Logger.log(f"[*] Deploy shared features: {shared_max_features}")

    df, wavelength_cols, X_full, Y, sample_ids = load_training_data(args.preprocessed_csv, args.target_metals)
    Logger.log(f"[DATA] Valid samples={X_full.shape[0]} spectral_bands={X_full.shape[1]}")

    processed_X_all = df[wavelength_cols].to_numpy(dtype=float)
    offline_scaler = fit_offline_scaler(args.raw_excel, wavelength_cols, processed_X_all)

    shared_idx, importance_all = select_deploy_features(X_full, Y, shared_max_features)
    selected_waves = wavelength_cols[shared_idx]
    waves_float = pd.to_numeric(selected_waves, errors="coerce").to_numpy(dtype=float)
    zone_feature_indices, zone_desc = FeatureBuilder.build_zones_physical(
        selected_waves,
        Config.PHYSICAL_ZONE_EDGES,
        Config.PHYSICAL_ZONE_NAMES,
    )
    X_sel = X_full[:, shared_idx]

    Logger.log(f"[FEATURES] shared_idx={shared_idx.tolist()}")
    Logger.log(f"[FEATURES] selected_waves={[float(x) for x in waves_float]}")
    Logger.log(f"[FEATURES] zones={[d['name'] for d in zone_desc]}")

    target_models: Dict[str, Any] = {}
    for target_pos, target_name in enumerate(args.target_metals):
        target_models[target_name] = train_target_bundle(
            target_name=target_name,
            X_sel=X_sel,
            waves_float=waves_float,
            y=Y[:, target_pos],
            zone_feature_indices=zone_feature_indices,
            zone_desc=zone_desc,
        )

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_project": str(Path(__file__).resolve().parent),
        "preprocessed_csv": str(args.preprocessed_csv.resolve()),
        "raw_excel": str(args.raw_excel.resolve()),
        "target_metals": list(args.target_metals),
        "n_samples": int(X_full.shape[0]),
        "all_wavelengths": [float(x) for x in pd.to_numeric(wavelength_cols, errors="coerce")],
        "shared_idx": shared_idx.astype(int).tolist(),
        "waves_float": [float(x) for x in waves_float],
        "feature_importance_all": importance_all.astype(float).tolist(),
        "zone_feature_indices": [np.asarray(idx, dtype=int).tolist() for idx in zone_feature_indices],
        "zone_desc": zone_desc,
        "preprocessing": {
            "crop_min": WAVE_MIN,
            "crop_max": WAVE_MAX,
            "snv": True,
            "savgol_window": SG_WINDOW,
            "savgol_polyorder": SG_POLY,
            "savgol_deriv": SG_DERIV,
            "offline_scaler": SCALER_FILE_NAME,
        },
        "config": config_to_dict(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }

    bundle = {
        "models": target_models,
        "metadata": metadata,
        "shared_idx": shared_idx,
        "waves_float": waves_float,
        "zone_feature_indices": zone_feature_indices,
        "zone_desc": zone_desc,
    }

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.artifact_dir / MODEL_FILE_NAME
    meta_path = args.artifact_dir / META_FILE_NAME
    scaler_path = args.artifact_dir / SCALER_FILE_NAME

    prepare_pickle_compat()
    write_joblib(scaler_path, offline_scaler)
    write_json(meta_path, metadata)
    write_joblib(model_path, bundle)

    part_paths = split_file(model_path, args.split_size_mb, remove_original=args.remove_unsplit_model)

    Logger.log_step("Export complete")
    Logger.log(f"[SAVE] scaler: {scaler_path}")
    Logger.log(f"[SAVE] metadata: {meta_path}")
    if model_path.exists():
        Logger.log(f"[SAVE] model bundle: {model_path}")
    if part_paths:
        Logger.log(f"[SAVE] model parts: {len(part_paths)} files, chunk_size_mb={args.split_size_mb}")


if __name__ == "__main__":
    main()

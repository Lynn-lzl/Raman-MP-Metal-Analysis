import numpy as np
import pandas as pd
from utils.logger import Logger
from utils.metrics import MetricsUtil

class ResultReporter:
    @staticmethod
    def print_shared_bands(shared_idx, wavelength_cols):
        shared_waves = wavelength_cols[shared_idx]
        shared_waves_float = pd.to_numeric(pd.Index(shared_waves), errors="coerce").values
        
        order_by_wave = np.argsort(shared_waves_float)
        Logger.log("[SharedBands] ===== Print in ascending order by wavelength =====")
        for k, oi in enumerate(order_by_wave, start=1):
            Logger.log(f"[SharedBands] #{k:02d} idx={shared_idx[oi]:4d} wave={shared_waves_float[oi]:.2f} nm (col='{shared_waves[oi]}')")

        return shared_waves, shared_waves_float

    @staticmethod
    def export_oof_predictions_to_excel(
        output_path,
        sample_ids,
        outer_fold_ids,
        target_metals,
        Y_true_matrix,
        oof_pred_global_dict,
        oof_pred_moe_dict,
        oof_pred_by_model_dict,
        fold_metrics_df
    ):
        sample_ids = np.asarray(sample_ids)
        outer_fold_ids = np.asarray(outer_fold_ids)

        df_g = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})
        df_m = pd.DataFrame({"Sample": sample_ids, "OuterFold": outer_fold_ids.astype(int)})

        for j, metal in enumerate(target_metals):
            y_true = np.asarray(Y_true_matrix[:, j], dtype=float)
            pred_g = np.asarray(oof_pred_global_dict[metal], dtype=float)
            pred_m = np.asarray(oof_pred_moe_dict[metal], dtype=float)

            df_g[f"{metal}_true"] = y_true
            df_g[f"{metal}_pred"] = pred_g
            df_m[f"{metal}_true"] = y_true
            df_m[f"{metal}_pred"] = pred_m

        model_order = {"PLS": 1, "KRR": 2, "ElasticNet": 3, "RFRR": 4, "RSRidge": 5, "GlobalStack": 6, "MoE": 7}
        train_mean = (
            fold_metrics_df[fold_metrics_df["Set"].eq("Train")]
            .groupby(["Target", "Model"])[["R2", "RMSE", "RPD"]]
            .mean()
        )

        summary_rows = []
        for target_idx, metal in enumerate(target_metals):
            y_true = np.asarray(Y_true_matrix[:, target_idx], dtype=float)
            models_for_target = (
                fold_metrics_df.loc[fold_metrics_df["Target"].eq(metal), "Model"]
                .drop_duplicates()
                .tolist()
            )
            models_for_target = sorted(
                models_for_target,
                key=lambda model: model_order.get(model, 99)
            )

            for model in models_for_target:
                row = {"Target": metal, "Model": model}
                if (metal, model) in train_mean.index:
                    train_r2, train_rmse, train_rpd = train_mean.loc[(metal, model)]
                    row.update({
                        "Train R2": train_r2,
                        "Train RMSE": train_rmse,
                        "Train RPD": train_rpd
                    })

                pred = np.asarray(oof_pred_by_model_dict[model][metal], dtype=float)
                if np.any(np.isnan(pred)):
                    missing = int(np.isnan(pred).sum())
                    raise ValueError(
                        f"[ERROR] Missing {missing} OOF predictions for target='{metal}', model='{model}'."
                    )

                oof_r2, oof_rmse, oof_rpd = MetricsUtil.compute_metrics(y_true, pred)
                row.update({
                    "OOF R2": oof_r2,
                    "OOF RMSE": oof_rmse,
                    "OOF RPD": oof_rpd
                })
                summary_rows.append(row)

        df_summary = pd.DataFrame(summary_rows)
        cols_order = ['Target', 'Model', 'Train R2', 'Train RMSE', 'Train RPD', 'OOF R2', 'OOF RMSE', 'OOF RPD']
        df_summary = df_summary[[c for c in cols_order if c in df_summary.columns]]
        df_summary = df_summary.copy()
        df_summary['_sort_idx'] = df_summary['Model'].map(model_order).fillna(99)
        df_summary = df_summary.sort_values(by=['Target', '_sort_idx']).drop(columns=['_sort_idx'])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_g.to_excel(writer, sheet_name="GlobalStack_Predictions", index=False)
            df_m.to_excel(writer, sheet_name="MoE_Predictions", index=False)
            fold_metrics_df.to_excel(writer, sheet_name="Fold_Performance", index=False)
            df_summary.to_excel(writer, sheet_name="Global_Summary", index=False)

        Logger.log(f"[SAVE] OOF predictions and summaries saved to: {output_path}")

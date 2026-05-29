import numpy as np
import pandas as pd
from config import Config

class FeatureBuilder:
    @staticmethod
    def make_global_features_from_selected(X_sel, waves_float):
        waves_float = np.asarray(waves_float, dtype=float)
        X_sel = np.asarray(X_sel, dtype=float)

        weights = np.abs(X_sel)
        intensity_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        center = (weights * waves_float).sum(axis=1, keepdims=True) / intensity_sum

        diff = waves_float[None, :] - center
        var = (weights * diff ** 2).sum(axis=1, keepdims=True) / intensity_sum
        var = np.clip(var, 0.0, None)
        width = np.sqrt(var)

        X_cospos = None
        if Config.USE_GLOBAL_COSPOS and len(Config.GLOBAL_COS_OMEGAS) > 0:
            t = (waves_float - waves_float.min()) / (waves_float.max() - waves_float.min() + 1e-8)
            omegas = np.array(Config.GLOBAL_COS_OMEGAS, dtype=float) * np.pi
            feats = []
            for omega in omegas:
                basis = np.cos(omega * t)
                feat = (weights * basis).sum(axis=1, keepdims=True) / intensity_sum
                feats.append(feat)
            X_cospos = np.concatenate(feats, axis=1)

        parts = [X_sel, center, width, intensity_sum]
        if X_cospos is not None:
            parts.append(X_cospos)

        X_global = np.concatenate(parts, axis=1)
        return X_global, center, width, intensity_sum, X_cospos

    @staticmethod
    def make_summary_features(X_sel, waves_float):
        waves_float = np.asarray(waves_float, dtype=float)
        X_sel = np.asarray(X_sel, dtype=float)

        weights = np.abs(X_sel)
        intensity_sum = weights.sum(axis=1, keepdims=True) + 1e-8
        center = (weights * waves_float).sum(axis=1, keepdims=True) / intensity_sum

        diff = waves_float[None, :] - center
        var = (weights * diff ** 2).sum(axis=1, keepdims=True) / intensity_sum
        var = np.clip(var, 0.0, None)
        width = np.sqrt(var)

        return center, width, intensity_sum

    @staticmethod
    def make_zone_local_cospos_from_zone(X_zone_sel, waves_zone_float):
        if not (Config.USE_ZONE_LOCAL_COSPOS and len(Config.ZONE_COS_OMEGAS) > 0):
            return None
        if X_zone_sel.shape[1] < Config.ZONE_LOCAL_COS_MIN_SIZE:
            return None

        waves_zone_float = np.asarray(waves_zone_float, dtype=float)
        t = (waves_zone_float - waves_zone_float.min()) / (waves_zone_float.max() - waves_zone_float.min() + 1e-8)

        weights = np.abs(np.asarray(X_zone_sel, dtype=float))
        intensity = weights.sum(axis=1, keepdims=True) + 1e-8

        omegas = np.array(Config.ZONE_COS_OMEGAS, dtype=float) * np.pi
        feats = []
        for omega in omegas:
            basis = np.cos(omega * t)
            feat = (weights * basis).sum(axis=1, keepdims=True) / intensity
            feats.append(feat)
        return np.concatenate(feats, axis=1)

    @staticmethod
    def build_zones_physical(selected_waves, edges, zone_names):
        w = pd.to_numeric(pd.Index(selected_waves), errors="coerce").values
        edges = np.asarray(edges, dtype=float)
        n_zones = len(edges) - 1
        
        z_id = np.digitize(w, edges[1:-1], right=False)
        z_id = np.clip(z_id, 0, n_zones - 1)

        zone_feature_indices = []
        zone_desc = []
        for i in range(n_zones):
            idxs = np.where(z_id == i)[0]
            if len(idxs) == 0:
                continue
            zone_feature_indices.append(idxs)
            zone_desc.append({
                "zone_id": i,
                "name": zone_names[i],
                "range": f"[{edges[i]:.0f}, {edges[i + 1]:.0f})" if i < n_zones - 1 else f"[{edges[i]:.0f}, {edges[i + 1]:.0f}]",
                "n_features": int(len(idxs)),
                "min_wave": float(w[idxs].min()),
                "max_wave": float(w[idxs].max()),
            })
        return zone_feature_indices, zone_desc
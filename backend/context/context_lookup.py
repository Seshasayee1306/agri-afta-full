import difflib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _season_from_date_yyyy_mm_dd(sowing_date: str) -> Optional[str]:
    try:
        month = datetime.strptime(sowing_date, "%Y-%m-%d").month
    except Exception:
        return None

    if month in (6, 7, 8, 9):
        return "kharif"
    if month in (10, 11, 12, 1):
        return "rabi"
    return "summer"


def _norm_str(val: Any) -> str:
    return str(val).strip()


def _closest_label(user_value: str, allowed: np.ndarray) -> Tuple[str, str]:
    """
    Returns (normalized_value, match_type).
    match_type: 'exact', 'closest', or 'unknown'
    """
    user_value = _norm_str(user_value)
    if not user_value:
        return "Unknown", "unknown"

    allowed_list = [str(x) for x in allowed if str(x).strip()]
    if not allowed_list:
        return user_value, "unknown"

    for a in allowed_list:
        if a.lower() == user_value.lower():
            return a, "exact"

    match = difflib.get_close_matches(user_value, allowed_list, n=1, cutoff=0.6)
    if match:
        return match[0], "closest"

    return user_value, "unknown"


def _coerce_disease(val: Any) -> str:
    if val is None:
        return "None"
    s = _norm_str(val)
    if not s or s.lower() in ("nan", "none", "no disease"):
        return "None"
    s_cap = s.capitalize()
    if s_cap in ("Mild", "Moderate", "Severe"):
        return s_cap
    return "None"


@dataclass(frozen=True)
class ContextDerived:
    rainfall: float
    ndvi: float
    disease_status: str
    matched_farm_id: Optional[str]
    matched_region: str
    matched_crop_type: str
    matched_soil_type: Optional[str]
    match_notes: Dict[str, Any]


class ContextLookup:
    """
    Deterministic context-derived imputation from Smart_Farming_Crop_Yield_2024.csv.

    It never calls external services. It uses a nearest-neighbor match inside the
    dataset, optionally conditioned on (region, crop_type, soil_type) and season.
    """

    def __init__(
        self,
        dataset_csv_path: str,
        soil_type_csv_path: Optional[str] = None,
    ):
        if not os.path.exists(dataset_csv_path):
            raise FileNotFoundError(f"Context dataset not found: {dataset_csv_path}")

        df = pd.read_csv(dataset_csv_path)

        if soil_type_csv_path and os.path.exists(soil_type_csv_path):
            st = pd.read_csv(soil_type_csv_path)
            if "farm_id" in st.columns and "soil_type" in st.columns and "farm_id" in df.columns:
                df = df.merge(st[["farm_id", "soil_type"]], on="farm_id", how="left")

        required_cols = [
            "region",
            "crop_type",
            "soil_moisture_%",
            "soil_pH",
            "temperature_C",
            "humidity_%",
            "rainfall_mm",
            "NDVI_index",
            "sowing_date",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Context dataset missing columns: {missing}")

        # Keep only rows that can participate in matching + deriving.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(
            subset=[
                "region",
                "crop_type",
                "soil_moisture_%",
                "soil_pH",
                "temperature_C",
                "humidity_%",
                "rainfall_mm",
                "NDVI_index",
                "sowing_date",
            ]
        ).copy()

        df["season"] = pd.to_datetime(df["sowing_date"], errors="coerce").dt.month.map(
            lambda m: "kharif"
            if m in (6, 7, 8, 9)
            else "rabi"
            if m in (10, 11, 12, 1)
            else "summer"
        )

        self.df = df
        self.available_regions = df["region"].dropna().unique()
        self.available_crops = df["crop_type"].dropna().unique()
        self.has_soil_type = "soil_type" in df.columns

        # Global stats for fallback z-score normalization
        self._global_mean = df[["soil_moisture_%", "soil_pH", "temperature_C", "humidity_%"]].mean()
        self._global_std = df[["soil_moisture_%", "soil_pH", "temperature_C", "humidity_%"]].std().replace(0, np.nan)

    def derive(
        self,
        *,
        sowing_date: str,
        region: str,
        crop_type: str,
        soil_type: Optional[str],
        soil_moisture: float,
        ph: float,
        temperature: float,
        humidity: float,
    ) -> ContextDerived:
        matched_region, region_match = _closest_label(region, self.available_regions)
        matched_crop, crop_match = _closest_label(crop_type, self.available_crops)

        candidate = self.df[
            (self.df["region"] == matched_region) & (self.df["crop_type"] == matched_crop)
        ]
        used_group = "region+crop"

        if candidate.shape[0] == 0:
            candidate = self.df.copy()
            used_group = "global"

        user_season = _season_from_date_yyyy_mm_dd(sowing_date)
        if user_season and "season" in candidate.columns:
            same_season = candidate[candidate["season"] == user_season]
            if same_season.shape[0] > 0:
                candidate = same_season
                used_group += "+season"

        matched_soil = None
        soil_match_used = False
        if soil_type and self.has_soil_type:
            soil_type = _norm_str(soil_type)
            if soil_type:
                # case-insensitive exact filter; if empty, ignore.
                soil_mask = candidate["soil_type"].astype(str).str.lower() == soil_type.lower()
                soil_filtered = candidate[soil_mask]
                if soil_filtered.shape[0] > 0:
                    candidate = soil_filtered
                    matched_soil = soil_type
                    soil_match_used = True

        # Z-score normalization within candidate set, fallback to global.
        feat_cols = ["soil_moisture_%", "soil_pH", "temperature_C", "humidity_%"]
        cand_mean = candidate[feat_cols].mean()
        cand_std = candidate[feat_cols].std().replace(0, np.nan)
        mean = cand_mean.fillna(self._global_mean)
        std = cand_std.fillna(self._global_std).fillna(1.0)

        user_vec = np.array([soil_moisture, ph, temperature, humidity], dtype=np.float32)
        user_z = (user_vec - mean.values.astype(np.float32)) / std.values.astype(np.float32)

        cand_mat = candidate[feat_cols].values.astype(np.float32)
        cand_z = (cand_mat - mean.values.astype(np.float32)) / std.values.astype(np.float32)

        # squared euclidean distance
        dists = np.sum((cand_z - user_z) ** 2, axis=1)
        best_idx = int(np.argmin(dists))
        best_row = candidate.iloc[best_idx]

        rainfall = float(best_row["rainfall_mm"])
        ndvi = float(best_row["NDVI_index"])
        disease_status = _coerce_disease(best_row.get("crop_disease_status"))

        matched_farm_id = None
        if "farm_id" in best_row.index:
            matched_farm_id = _norm_str(best_row["farm_id"])

        match_notes: Dict[str, Any] = {
            "grouping_used": used_group,
            "region_match": region_match,
            "crop_match": crop_match,
            "soil_type_used": bool(soil_match_used),
            "user_season": user_season,
            "candidate_rows": int(candidate.shape[0]),
            "best_distance": float(dists[best_idx]),
        }

        return ContextDerived(
            rainfall=rainfall,
            ndvi=ndvi,
            disease_status=disease_status,
            matched_farm_id=matched_farm_id,
            matched_region=matched_region,
            matched_crop_type=matched_crop,
            matched_soil_type=matched_soil,
            match_notes=match_notes,
        )


_LOOKUP_SINGLETON: Optional[ContextLookup] = None


def get_context_lookup() -> ContextLookup:
    global _LOOKUP_SINGLETON
    if _LOOKUP_SINGLETON is not None:
        return _LOOKUP_SINGLETON

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo_root = os.path.dirname(base_dir)

    dataset_csv = os.path.join(repo_root, "dataset", "Smart_Farming_Crop_Yield_2024.csv")
    soil_map_csv = os.path.join(repo_root, "dataset", "smart_farming_soil_type.csv")
    soil_path = soil_map_csv if os.path.exists(soil_map_csv) else None

    _LOOKUP_SINGLETON = ContextLookup(
        dataset_csv_path=dataset_csv,
        soil_type_csv_path=soil_path,
    )
    return _LOOKUP_SINGLETON


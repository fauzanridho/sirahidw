import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "sirah_idw_matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import rasterio
import streamlit as st
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from shapely.geometry import Point
from shapely.ops import unary_union
from streamlit_folium import st_folium


APP_TITLE = "(SIRAH IDW) Sistem Interpolasi Curah Hujan Menggunakan IDW dan LOOCV"
REQUIRED_COLUMNS = ["Tanggal", "Nama_Stasiun", "Longitude", "Latitude", "Curah_Hujan"]
STORE_KEY_COLUMNS = ["Tanggal", "Nama_Stasiun", "Longitude", "Latitude"]
DATA_DIR = Path("data")
MASTER_DATA_PATH = DATA_DIR / "curah_hujan_master.csv"
NODATA_VALUE = -9999.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def clean_column_name(column: str) -> str:
    """Normalize a raw column name into a readable snake-style label."""
    return str(column).strip().replace(" ", "_").replace("-", "_")


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and map common variants to the required schema."""
    alias_map = {
        "tanggal": "Tanggal",
        "date": "Tanggal",
        "nama_stasiun": "Nama_Stasiun",
        "stasiun": "Nama_Stasiun",
        "station": "Nama_Stasiun",
        "nama_station": "Nama_Stasiun",
        "longitude": "Longitude",
        "long": "Longitude",
        "lon": "Longitude",
        "x": "Longitude",
        "latitude": "Latitude",
        "lat": "Latitude",
        "y": "Latitude",
        "curah_hujan": "Curah_Hujan",
        "rainfall": "Curah_Hujan",
        "precipitation": "Curah_Hujan",
        "hujan": "Curah_Hujan",
    }
    renamed = {}
    for column in df.columns:
        cleaned = clean_column_name(column)
        key = cleaned.lower()
        renamed[column] = alias_map.get(key, cleaned)
    return df.rename(columns=renamed)


def validate_required_columns(df: pd.DataFrame) -> List[str]:
    return [column for column in REQUIRED_COLUMNS if column not in df.columns]


def read_csv_auto_delimiter(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    encodings = ["utf-8-sig", "utf-8", "latin1"]
    separators = [None, ",", "\t", ";", "|"]
    best_df = None
    best_score = -1
    last_error = None

    for encoding in encodings:
        for separator in separators:
            try:
                buffer = io.BytesIO(raw_bytes)
                kwargs = {
                    "encoding": encoding,
                    "skipinitialspace": True,
                }
                if separator is None:
                    kwargs.update({"sep": None, "engine": "python"})
                else:
                    kwargs.update({"sep": separator})
                candidate = pd.read_csv(buffer, **kwargs)
                candidate = candidate.dropna(axis=1, how="all")
                canonical = canonicalize_columns(candidate)
                score = len([column for column in REQUIRED_COLUMNS if column in canonical.columns])
                if len(candidate.columns) > 1:
                    score += 1
                if score > best_score:
                    best_df = candidate
                    best_score = score
            except Exception as exc:
                last_error = exc

    if best_df is None:
        raise ValueError(f"Gagal membaca CSV. Detail: {last_error}")

    return best_df


def load_rainfall_file(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        df = read_csv_auto_delimiter(uploaded_file)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau XLSX.")

    df = canonicalize_columns(df)
    missing = validate_required_columns(df)
    if missing:
        raise ValueError(
            "Kolom wajib belum lengkap: "
            + ", ".join(missing)
            + ". Kolom minimal: "
            + ", ".join(REQUIRED_COLUMNS)
            + ". Kolom yang terbaca: "
            + ", ".join(map(str, df.columns))
        )
    return df


def parse_flexible_numeric(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text:
        return np.nan

    text = (
        text.replace("\u00a0", "")
        .replace(" ", "")
        .replace("−", "-")
        .replace("—", "-")
        .replace("–", "-")
    )

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")

    try:
        return float(text)
    except ValueError:
        return np.nan


def parse_flexible_numeric_series(series: pd.Series) -> pd.Series:
    return series.apply(parse_flexible_numeric)


def parse_flexible_dates(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    if pd.api.types.is_datetime64_any_dtype(cleaned):
        return pd.to_datetime(cleaned, errors="coerce")

    numeric_values = pd.to_numeric(cleaned, errors="coerce")
    parsed = pd.to_datetime(cleaned.astype(str).str.strip(), errors="coerce")

    missing = parsed.isna()
    if missing.any():
        parsed_dayfirst = pd.to_datetime(cleaned[missing].astype(str).str.strip(), errors="coerce", dayfirst=True)
        parsed.loc[missing] = parsed_dayfirst

    missing = parsed.isna() & numeric_values.notna()
    if missing.any():
        excel_dates = pd.to_datetime(numeric_values[missing], unit="D", origin="1899-12-30", errors="coerce")
        parsed.loc[missing] = excel_dates

    return parsed


def build_invalid_reason_table(normalized: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    invalid_tanggal = normalized["Tanggal"].isna()
    invalid_stasiun = (
        normalized["Nama_Stasiun"].isna()
        | normalized["Nama_Stasiun"].eq("")
        | normalized["Nama_Stasiun"].str.lower().eq("nan")
    )
    invalid_longitude = normalized["Longitude"].isna()
    invalid_latitude = normalized["Latitude"].isna()

    invalid_mask = invalid_tanggal | invalid_stasiun | invalid_longitude | invalid_latitude
    reason_rows = []
    for label, mask in [
        ("Tanggal tidak valid", invalid_tanggal),
        ("Nama stasiun kosong", invalid_stasiun),
        ("Longitude tidak valid", invalid_longitude),
        ("Latitude tidak valid", invalid_latitude),
    ]:
        count = int(mask.sum())
        if count > 0:
            reason_rows.append({"Alasan": label, "Jumlah_Baris": count})
    return invalid_mask, pd.DataFrame(reason_rows)


def normalize_rainfall_records(df: pd.DataFrame, drop_invalid_index_rows: bool = True) -> Tuple[pd.DataFrame, int]:
    normalized = canonicalize_columns(df.copy())
    missing = validate_required_columns(normalized)
    if missing:
        raise ValueError(
            "Kolom wajib belum lengkap: "
            + ", ".join(missing)
            + ". Kolom minimal: "
            + ", ".join(REQUIRED_COLUMNS)
        )

    normalized["Nama_Stasiun"] = normalized["Nama_Stasiun"].astype(str).str.strip()
    normalized["Longitude"] = parse_flexible_numeric_series(normalized["Longitude"])
    normalized["Latitude"] = parse_flexible_numeric_series(normalized["Latitude"])
    normalized["Curah_Hujan"] = parse_flexible_numeric_series(normalized["Curah_Hujan"])
    normalized["Tanggal"] = parse_flexible_dates(normalized["Tanggal"])

    invalid_index_mask, _ = build_invalid_reason_table(normalized)
    invalid_count = int(invalid_index_mask.sum())
    if drop_invalid_index_rows:
        normalized = normalized.loc[~invalid_index_mask].copy()

    if not normalized.empty:
        normalized["Tanggal"] = normalized["Tanggal"].dt.strftime("%Y-%m-%d")

    ordered_columns = REQUIRED_COLUMNS + [column for column in normalized.columns if column not in REQUIRED_COLUMNS]
    return normalized[ordered_columns].reset_index(drop=True), invalid_count


def load_persistent_rainfall_data() -> pd.DataFrame:
    if not MASTER_DATA_PATH.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    try:
        stored = pd.read_csv(MASTER_DATA_PATH)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    normalized, _ = normalize_rainfall_records(stored, drop_invalid_index_rows=True)
    return normalized


def save_persistent_rainfall_data(df: pd.DataFrame) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    normalized, _ = normalize_rainfall_records(df, drop_invalid_index_rows=True)
    sort_columns = [column for column in ["Tanggal", "Nama_Stasiun", "Longitude", "Latitude"] if column in normalized.columns]
    if sort_columns and not normalized.empty:
        normalized = normalized.sort_values(sort_columns).reset_index(drop=True)
    normalized.to_csv(MASTER_DATA_PATH, index=False)
    return normalized


def append_to_persistent_rainfall_data(new_df: pd.DataFrame, save_mode: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    new_normalized, invalid_count = normalize_rainfall_records(new_df, drop_invalid_index_rows=True)
    if new_normalized.empty:
        checked, _ = normalize_rainfall_records(new_df, drop_invalid_index_rows=False)
        _, reason_table = build_invalid_reason_table(checked)
        reason_text = "; ".join(
            f"{row['Alasan']}: {row['Jumlah_Baris']}"
            for _, row in reason_table.iterrows()
        )
        if not reason_text:
            reason_text = "semua baris kosong atau tidak dapat dinormalisasi"
        raise ValueError(f"Tidak ada baris valid untuk disimpan. Rincian: {reason_text}.")

    existing = load_persistent_rainfall_data()
    previous_count = len(existing)

    if save_mode == "Ganti seluruh data tersimpan":
        combined = new_normalized
    elif save_mode == "Tambahkan semua baris":
        combined = pd.concat([existing, new_normalized], ignore_index=True)
    else:
        combined = pd.concat([existing, new_normalized], ignore_index=True)
        combined = combined.drop_duplicates(subset=STORE_KEY_COLUMNS, keep="last").reset_index(drop=True)

    saved = save_persistent_rainfall_data(combined)
    info = {
        "baris_input_valid": int(len(new_normalized)),
        "baris_input_invalid_dilewati": invalid_count,
        "baris_sebelumnya": previous_count,
        "baris_total": int(len(saved)),
        "baris_bertambah_netto": int(len(saved) - previous_count),
    }
    return saved, info


def manual_input_columns(extra_columns_raw: str = "") -> List[str]:
    stored = load_persistent_rainfall_data()
    existing_columns = list(stored.columns) if not stored.empty else REQUIRED_COLUMNS.copy()
    existing_extra = [column for column in existing_columns if column not in REQUIRED_COLUMNS]
    typed_extra = [clean_column_name(column) for column in extra_columns_raw.split(",") if column.strip()]
    columns = REQUIRED_COLUMNS.copy()
    for column in existing_extra + typed_extra:
        if column and column not in columns:
            columns.append(column)
    return columns


def empty_manual_dataframe(columns: Sequence[str], rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame([{column: None for column in columns} for _ in range(rows)])


def reset_derived_state() -> None:
    for key in [
        "processed_df",
        "preprocess_stats",
        "aggregated_df",
        "period_data",
        "loocv_results",
        "best_params",
        "best_predictions",
        "final_estimates",
        "grid_df",
        "geotiff_bytes",
        "grid_png_bytes",
        "grid_context",
    ]:
        st.session_state[key] = None


def set_active_raw_data(df: pd.DataFrame, label: str) -> None:
    st.session_state.raw_df = df.reset_index(drop=True)
    st.session_state.active_data_label = label
    reset_derived_state()


def ensure_wgs84(gdf: Optional[gpd.GeoDataFrame]) -> Optional[gpd.GeoDataFrame]:
    if gdf is None or gdf.empty:
        return None
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    return gdf.to_crs("EPSG:4326")


def load_boundary_from_zip(uploaded_file) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as archive:
            archive.extractall(tmp_path)
        shp_files = list(tmp_path.rglob("*.shp"))
        if not shp_files:
            raise ValueError("ZIP SHP tidak berisi file .shp.")
        gdf = gpd.read_file(shp_files[0])
    return ensure_wgs84(gdf)


def load_boundary_from_shp_uploads(uploaded_files) -> gpd.GeoDataFrame:
    if not uploaded_files:
        raise ValueError("Tidak ada file SHP yang diupload.")

    if len(uploaded_files) == 1 and Path(uploaded_files[0].name).suffix.lower() == ".zip":
        return load_boundary_from_zip(uploaded_files[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for uploaded_file in uploaded_files:
            safe_name = Path(uploaded_file.name).name
            (tmp_path / safe_name).write_bytes(uploaded_file.getvalue())
        shp_files = list(tmp_path.rglob("*.shp"))
        if not shp_files:
            raise ValueError("Upload komponen SHP harus menyertakan file .shp.")
        gdf = gpd.read_file(shp_files[0])
    return ensure_wgs84(gdf)


def load_boundary_from_geojson(uploaded_file) -> gpd.GeoDataFrame:
    with tempfile.NamedTemporaryFile(suffix=".geojson") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp.flush()
        gdf = gpd.read_file(tmp.name)
    return ensure_wgs84(gdf)


# ---------------------------------------------------------------------------
# Preprocessing and aggregation
# ---------------------------------------------------------------------------
def preprocess_data(df: pd.DataFrame, drop_invalid_index_rows: bool = True) -> Tuple[pd.DataFrame, Dict[str, object]]:
    processed = canonicalize_columns(df.copy())
    processed["Nama_Stasiun"] = processed["Nama_Stasiun"].astype(str).str.strip()
    processed["Longitude"] = pd.to_numeric(processed["Longitude"], errors="coerce")
    processed["Latitude"] = pd.to_numeric(processed["Latitude"], errors="coerce")
    processed["Curah_Hujan"] = pd.to_numeric(processed["Curah_Hujan"], errors="coerce")
    processed["Tanggal"] = pd.to_datetime(processed["Tanggal"], errors="coerce")
    processed["Status_Data"] = np.where(processed["Curah_Hujan"].isna(), "Missing", "Observasi")

    invalid_index_mask = (
        processed["Tanggal"].isna()
        | processed["Nama_Stasiun"].eq("")
        | processed["Nama_Stasiun"].str.lower().eq("nan")
        | processed["Longitude"].isna()
        | processed["Latitude"].isna()
    )

    removed_rows = int(invalid_index_mask.sum()) if drop_invalid_index_rows else 0
    if drop_invalid_index_rows:
        processed = processed.loc[~invalid_index_mask].copy()

    stats = {
        "jumlah_baris": int(len(processed)),
        "baris_index_invalid_dihapus": removed_rows,
        "jumlah_stasiun": int(processed["Nama_Stasiun"].nunique()),
        "tanggal_min": processed["Tanggal"].min(),
        "tanggal_max": processed["Tanggal"].max(),
        "curah_min": processed["Curah_Hujan"].min(skipna=True),
        "curah_max": processed["Curah_Hujan"].max(skipna=True),
        "curah_mean": processed["Curah_Hujan"].mean(skipna=True),
        "curah_total": processed["Curah_Hujan"].sum(skipna=True),
        "curah_missing": int(processed["Curah_Hujan"].isna().sum()),
    }
    return processed, stats


def aggregate_series(series: pd.Series, method: str) -> float:
    method = method.upper()
    if method == "SUM":
        return series.sum(min_count=1)
    if method == "MEAN":
        return series.mean()
    if method == "MAX":
        return series.max()
    raise ValueError(f"Metode agregasi tidak dikenal: {method}")


def aggregate_rainfall(df: pd.DataFrame, scale: str, agg_method: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    working = df.copy()
    working["Tahun"] = working["Tanggal"].dt.year
    working["Bulan"] = working["Tanggal"].dt.month

    if scale == "Harian":
        group_columns = ["Tanggal", "Nama_Stasiun", "Longitude", "Latitude"]
    elif scale == "Bulanan":
        group_columns = ["Nama_Stasiun", "Longitude", "Latitude", "Tahun", "Bulan"]
    elif scale == "Tahunan":
        group_columns = ["Nama_Stasiun", "Longitude", "Latitude", "Tahun"]
    else:
        raise ValueError("Skala data tidak dikenal.")

    aggregated = (
        working.groupby(group_columns, dropna=False)["Curah_Hujan"]
        .apply(lambda value: aggregate_series(value, agg_method))
        .reset_index(name="Curah_Hujan_Agregat")
    )

    if scale == "Harian":
        aggregated["Tahun"] = aggregated["Tanggal"].dt.year
        aggregated["Bulan"] = aggregated["Tanggal"].dt.month
        aggregated["Periode"] = aggregated["Tanggal"].dt.strftime("%Y-%m-%d")
    elif scale == "Bulanan":
        aggregated["Tanggal"] = pd.to_datetime(
            {
                "year": aggregated["Tahun"],
                "month": aggregated["Bulan"],
                "day": 1,
            }
        )
        aggregated["Periode"] = aggregated["Tanggal"].dt.strftime("%Y-%m")
    else:
        aggregated["Bulan"] = np.nan
        aggregated["Tanggal"] = pd.to_datetime(
            {
                "year": aggregated["Tahun"],
                "month": 1,
                "day": 1,
            }
        )
        aggregated["Periode"] = aggregated["Tahun"].astype(str)

    aggregated["Status_Data"] = np.where(aggregated["Curah_Hujan_Agregat"].isna(), "Missing", "Observasi")
    ordered_columns = [
        "Periode",
        "Tanggal",
        "Tahun",
        "Bulan",
        "Nama_Stasiun",
        "Longitude",
        "Latitude",
        "Curah_Hujan_Agregat",
        "Status_Data",
    ]
    return aggregated[ordered_columns].sort_values(["Tanggal", "Nama_Stasiun"]).reset_index(drop=True)


def filter_period_data(aggregated: pd.DataFrame, scale: str, selected_date=None, selected_month: int = None, selected_year: int = None) -> pd.DataFrame:
    if aggregated.empty:
        return aggregated.copy()
    if scale == "Harian" and selected_date is not None:
        selected_ts = pd.Timestamp(selected_date).normalize()
        return aggregated.loc[aggregated["Tanggal"].dt.normalize() == selected_ts].copy()
    if scale == "Bulanan" and selected_month is not None and selected_year is not None:
        return aggregated.loc[(aggregated["Tahun"] == selected_year) & (aggregated["Bulan"] == selected_month)].copy()
    if scale == "Tahunan" and selected_year is not None:
        return aggregated.loc[aggregated["Tahun"] == selected_year].copy()
    return aggregated.copy()


# ---------------------------------------------------------------------------
# IDW and LOOCV
# ---------------------------------------------------------------------------
def get_utm_crs(lons: Sequence[float], lats: Sequence[float]) -> CRS:
    lon = float(np.nanmean(np.asarray(lons, dtype=float)))
    lat = float(np.nanmean(np.asarray(lats, dtype=float)))
    zone = int(math.floor((lon + 180) / 6) + 1)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def transform_coordinate_arrays(
    ref_lons: Sequence[float],
    ref_lats: Sequence[float],
    target_lons: Sequence[float],
    target_lats: Sequence[float],
    distance_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref_lons_arr = np.asarray(ref_lons, dtype=float)
    ref_lats_arr = np.asarray(ref_lats, dtype=float)
    target_lons_arr = np.asarray(target_lons, dtype=float)
    target_lats_arr = np.asarray(target_lats, dtype=float)

    if distance_mode == "UTM otomatis":
        all_lons = np.concatenate([ref_lons_arr, target_lons_arr])
        all_lats = np.concatenate([ref_lats_arr, target_lats_arr])
        transformer = Transformer.from_crs("EPSG:4326", get_utm_crs(all_lons, all_lats), always_xy=True)
        ref_x, ref_y = transformer.transform(ref_lons_arr, ref_lats_arr)
        target_x, target_y = transformer.transform(target_lons_arr, target_lats_arr)
        return np.asarray(ref_x), np.asarray(ref_y), np.asarray(target_x), np.asarray(target_y)

    return ref_lons_arr, ref_lats_arr, target_lons_arr, target_lats_arr


def idw_predict_many(
    ref_lons: Sequence[float],
    ref_lats: Sequence[float],
    values: Sequence[float],
    target_lons: Sequence[float],
    target_lats: Sequence[float],
    power: float,
    k: Optional[int] = None,
    radius: Optional[float] = None,
    distance_mode: str = "UTM otomatis",
    chunk_size: int = 4000,
) -> np.ndarray:
    ref_lons_arr = np.asarray(ref_lons, dtype=float)
    ref_lats_arr = np.asarray(ref_lats, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    target_lons_arr = np.asarray(target_lons, dtype=float)
    target_lats_arr = np.asarray(target_lats, dtype=float)

    valid_ref = np.isfinite(ref_lons_arr) & np.isfinite(ref_lats_arr) & np.isfinite(values_arr)
    ref_lons_arr = ref_lons_arr[valid_ref]
    ref_lats_arr = ref_lats_arr[valid_ref]
    values_arr = values_arr[valid_ref]

    predictions = np.full(len(target_lons_arr), np.nan, dtype=float)
    valid_targets = np.isfinite(target_lons_arr) & np.isfinite(target_lats_arr)
    if len(values_arr) == 0 or not valid_targets.any():
        return predictions

    target_indices = np.where(valid_targets)[0]
    ref_x, ref_y, target_x, target_y = transform_coordinate_arrays(
        ref_lons_arr,
        ref_lats_arr,
        target_lons_arr[valid_targets],
        target_lats_arr[valid_targets],
        distance_mode,
    )
    ref_xy = np.column_stack([ref_x, ref_y])
    target_xy = np.column_stack([target_x, target_y])

    k_eff = None
    if k is not None and k > 0:
        k_eff = min(int(k), len(values_arr))

    radius_value = float(radius) if radius is not None and radius > 0 else None

    for start in range(0, len(target_xy), chunk_size):
        stop = min(start + chunk_size, len(target_xy))
        chunk_xy = target_xy[start:stop]
        distances = np.sqrt(((chunk_xy[:, None, :] - ref_xy[None, :, :]) ** 2).sum(axis=2))
        chunk_predictions = np.full(len(chunk_xy), np.nan, dtype=float)

        exact_mask = distances <= 1e-12
        exact_rows = exact_mask.any(axis=1)
        if exact_rows.any():
            exact_cols = exact_mask[exact_rows].argmax(axis=1)
            chunk_predictions[exact_rows] = values_arr[exact_cols]

        remaining_rows = ~exact_rows
        if remaining_rows.any():
            d_all = distances[remaining_rows]
            if radius_value is not None:
                in_radius = d_all <= radius_value
                no_radius_match = ~in_radius.any(axis=1)
                d_candidates = np.where(in_radius, d_all, np.inf)
                d_candidates[no_radius_match] = d_all[no_radius_match]
            else:
                d_candidates = d_all

            if k_eff is not None:
                neighbor_idx = np.argpartition(d_candidates, k_eff - 1, axis=1)[:, :k_eff]
                selected_distances = np.take_along_axis(d_candidates, neighbor_idx, axis=1)
                selected_values = values_arr[neighbor_idx]
            else:
                selected_distances = d_candidates
                selected_values = np.broadcast_to(values_arr, selected_distances.shape)

            usable = np.isfinite(selected_distances) & (selected_distances > 0)
            weights = np.zeros_like(selected_distances, dtype=float)
            weights[usable] = 1.0 / np.power(selected_distances[usable], power)
            denominator = weights.sum(axis=1)
            numerator = (weights * selected_values).sum(axis=1)
            estimated = np.divide(numerator, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
            chunk_predictions[remaining_rows] = estimated

        predictions[target_indices[start:stop]] = chunk_predictions

    return predictions


def idw_predict(
    ref_lons: Sequence[float],
    ref_lats: Sequence[float],
    values: Sequence[float],
    target_lon: float,
    target_lat: float,
    power: float,
    k: Optional[int] = None,
    radius: Optional[float] = None,
    distance_mode: str = "UTM otomatis",
) -> float:
    return float(
        idw_predict_many(
            ref_lons,
            ref_lats,
            values,
            [target_lon],
            [target_lat],
            power,
            k,
            radius,
            distance_mode,
        )[0]
    )


def parse_number_list(raw_value: str, value_type=float) -> List[float]:
    values = []
    for item in raw_value.replace(";", ",").split(","):
        stripped = item.strip()
        if stripped:
            values.append(value_type(stripped))
    return values


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if not valid.any():
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "Jumlah_Data_Valid": 0}

    actual_valid = actual[valid]
    predicted_valid = predicted[valid]
    errors = predicted_valid - actual_valid
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    positive_actual = actual_valid > 0
    if positive_actual.any():
        mape = float(np.mean(np.abs(errors[positive_actual] / actual_valid[positive_actual])) * 100)
    else:
        mape = np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Jumlah_Data_Valid": int(valid.sum())}


def loocv_for_combo(
    observations: pd.DataFrame,
    power: float,
    k: int,
    radius: Optional[float],
    distance_mode: str,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    actual_values = observations["Curah_Hujan_Agregat"].to_numpy(dtype=float)
    predictions = np.full(len(observations), np.nan, dtype=float)

    for idx in range(len(observations)):
        train = observations.drop(observations.index[idx])
        if train.empty:
            continue
        test = observations.iloc[idx]
        predictions[idx] = idw_predict(
            train["Longitude"],
            train["Latitude"],
            train["Curah_Hujan_Agregat"],
            float(test["Longitude"]),
            float(test["Latitude"]),
            power=power,
            k=k,
            radius=radius,
            distance_mode=distance_mode,
        )

    metrics = calculate_metrics(actual_values, predictions)
    metrics.update({"p": float(power), "k": int(k)})

    prediction_table = observations[
        ["Nama_Stasiun", "Longitude", "Latitude", "Curah_Hujan_Agregat"]
    ].copy()
    prediction_table = prediction_table.rename(columns={"Curah_Hujan_Agregat": "Observasi"})
    prediction_table["Prediksi"] = predictions
    prediction_table["Error"] = prediction_table["Prediksi"] - prediction_table["Observasi"]
    prediction_table["Abs_Error"] = prediction_table["Error"].abs()
    prediction_table["p"] = float(power)
    prediction_table["k"] = int(k)
    return metrics, prediction_table


def run_loocv(
    period_data: pd.DataFrame,
    power_values: Sequence[float],
    k_values: Sequence[int],
    radius: Optional[float],
    metric: str,
    distance_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    observations = period_data.dropna(subset=["Curah_Hujan_Agregat"]).copy()
    observations = observations.reset_index(drop=True)
    if len(observations) < 3:
        raise ValueError("LOOCV membutuhkan minimal 3 stasiun dengan nilai curah hujan valid.")

    rows = []
    prediction_tables = []
    for power in power_values:
        for k in k_values:
            metrics, predictions = loocv_for_combo(observations, power, int(k), radius, distance_mode)
            rows.append(metrics)
            prediction_tables.append(predictions)

    result = pd.DataFrame(rows)
    if result.empty or result[metric].dropna().empty:
        raise ValueError("Tidak ada kombinasi parameter yang menghasilkan prediksi valid.")

    best_row = result.sort_values([metric, "RMSE", "MAE"], ascending=True).iloc[0].to_dict()
    best_predictions = next(
        table for table in prediction_tables if table["p"].iloc[0] == best_row["p"] and table["k"].iloc[0] == best_row["k"]
    )
    return result.sort_values(metric).reset_index(drop=True), best_row, best_predictions.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Grid interpolation and raster export
# ---------------------------------------------------------------------------
def boundary_union(boundary_gdf: Optional[gpd.GeoDataFrame]):
    if boundary_gdf is None or boundary_gdf.empty:
        return None
    return unary_union(boundary_gdf.geometry)


def get_interpolation_bounds(stations: pd.DataFrame, boundary_gdf: Optional[gpd.GeoDataFrame], buffer_ratio: float = 0.05) -> Tuple[float, float, float, float]:
    if boundary_gdf is not None and not boundary_gdf.empty:
        minx, miny, maxx, maxy = boundary_gdf.total_bounds
    else:
        minx = float(stations["Longitude"].min())
        maxx = float(stations["Longitude"].max())
        miny = float(stations["Latitude"].min())
        maxy = float(stations["Latitude"].max())

    dx = max(maxx - minx, 0.01)
    dy = max(maxy - miny, 0.01)
    if boundary_gdf is None or boundary_gdf.empty:
        minx -= dx * buffer_ratio
        maxx += dx * buffer_ratio
        miny -= dy * buffer_ratio
        maxy += dy * buffer_ratio
    return minx, miny, maxx, maxy


def create_grid_interpolation(
    period_data: pd.DataFrame,
    boundary_gdf: Optional[gpd.GeoDataFrame],
    resolution: int,
    power: float,
    k: int,
    radius: Optional[float],
    distance_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    observations = period_data.dropna(subset=["Curah_Hujan_Agregat"]).copy()
    if len(observations) < 2:
        raise ValueError("Grid interpolasi membutuhkan minimal 2 stasiun observasi.")

    minx, miny, maxx, maxy = get_interpolation_bounds(observations, boundary_gdf)
    x_values = np.linspace(minx, maxx, resolution)
    y_values = np.linspace(miny, maxy, resolution)
    lon_grid, lat_grid = np.meshgrid(x_values, y_values)
    flat_lons = lon_grid.ravel()
    flat_lats = lat_grid.ravel()

    grid_values = idw_predict_many(
        observations["Longitude"],
        observations["Latitude"],
        observations["Curah_Hujan_Agregat"],
        flat_lons,
        flat_lats,
        power=power,
        k=k,
        radius=radius,
        distance_mode=distance_mode,
    )

    union_geom = boundary_union(boundary_gdf)
    if union_geom is not None:
        inside = np.array([union_geom.contains(Point(lon, lat)) or union_geom.touches(Point(lon, lat)) for lon, lat in zip(flat_lons, flat_lats)])
        grid_values[~inside] = np.nan

    z_grid = grid_values.reshape(lon_grid.shape)
    grid_df = pd.DataFrame(
        {
            "Longitude": flat_lons,
            "Latitude": flat_lats,
            "Curah_Hujan_IDW": grid_values,
        }
    )
    return lon_grid, lat_grid, z_grid, grid_df


def export_geotiff(lon_grid: np.ndarray, lat_grid: np.ndarray, z_grid: np.ndarray) -> bytes:
    minx = float(np.nanmin(lon_grid))
    maxx = float(np.nanmax(lon_grid))
    miny = float(np.nanmin(lat_grid))
    maxy = float(np.nanmax(lat_grid))
    height, width = z_grid.shape
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    raster_data = np.flipud(z_grid).astype("float32")
    raster_data = np.where(np.isfinite(raster_data), raster_data, NODATA_VALUE).astype("float32")

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": NODATA_VALUE,
    }
    with MemoryFile() as memfile:
        with memfile.open(**profile) as dataset:
            dataset.write(raster_data, 1)
        return memfile.read()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def value_to_color(value: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(value):
        return "#8c8c8c"
    if vmax <= vmin:
        scaled = 0.5
    else:
        scaled = (value - vmin) / (vmax - vmin)
    return px.colors.sample_colorscale("Viridis", [float(np.clip(scaled, 0, 1))])[0]


def make_station_map(period_data: pd.DataFrame, boundary_gdf: Optional[gpd.GeoDataFrame]) -> folium.Map:
    center_lat = float(period_data["Latitude"].mean())
    center_lon = float(period_data["Longitude"].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="CartoDB positron")

    if boundary_gdf is not None and not boundary_gdf.empty:
        folium.GeoJson(
            boundary_gdf,
            name="Batas Wilayah",
            style_function=lambda _: {"fillColor": "#ffffff", "color": "#2b2b2b", "weight": 2, "fillOpacity": 0.05},
        ).add_to(fmap)

    values = period_data["Curah_Hujan_Agregat"]
    vmin = float(values.min(skipna=True)) if values.notna().any() else 0.0
    vmax = float(values.max(skipna=True)) if values.notna().any() else 1.0
    for _, row in period_data.iterrows():
        rainfall = row["Curah_Hujan_Agregat"]
        radius = 6
        if np.isfinite(rainfall) and vmax > vmin:
            radius = 6 + 10 * (float(rainfall) - vmin) / (vmax - vmin)
        rainfall_text = f"{rainfall:.3f}" if np.isfinite(rainfall) else "Missing"
        popup_html = (
            f"<b>{row['Nama_Stasiun']}</b><br>"
            f"Longitude: {row['Longitude']:.6f}<br>"
            f"Latitude: {row['Latitude']:.6f}<br>"
            f"Curah hujan: {rainfall_text}"
        )
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            fill=True,
            fill_opacity=0.8,
            color="#1f2937",
            weight=1,
            fill_color=value_to_color(float(rainfall) if np.isfinite(rainfall) else np.nan, vmin, vmax),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
        border: 1px solid #c7c7c7; padding: 10px 12px; font-size: 12px;">
        <b>Curah hujan</b><br>
        <span style="display:inline-block;width:12px;height:12px;background:#440154;"></span> Rendah<br>
        <span style="display:inline-block;width:12px;height:12px;background:#21918c;"></span> Sedang<br>
        <span style="display:inline-block;width:12px;height:12px;background:#fde725;"></span> Tinggi
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(fmap)
    return fmap


def plot_station_bar(period_data: pd.DataFrame):
    chart_data = period_data.sort_values("Curah_Hujan_Agregat", ascending=False)
    return px.bar(
        chart_data,
        x="Nama_Stasiun",
        y="Curah_Hujan_Agregat",
        color="Curah_Hujan_Agregat",
        color_continuous_scale="Viridis",
        labels={"Curah_Hujan_Agregat": "Curah Hujan"},
        title="Curah hujan per stasiun pada periode terpilih",
    )


def plot_loocv_heatmap(result: pd.DataFrame, metric: str):
    pivot = result.pivot(index="p", columns="k", values=metric)
    fig = px.imshow(
        pivot,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Viridis_r",
        labels={"x": "k", "y": "p", "color": metric},
        title=f"Heatmap parameter p-k berdasarkan {metric}",
    )
    return fig


def plot_grid_contour(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    z_grid: np.ndarray,
    period_data: pd.DataFrame,
    boundary_gdf: Optional[gpd.GeoDataFrame],
    title: str,
):
    fig, ax = plt.subplots(figsize=(10, 7), dpi=120)
    finite_values = z_grid[np.isfinite(z_grid)]
    if finite_values.size:
        contour = ax.contourf(lon_grid, lat_grid, z_grid, levels=20, cmap="viridis")
        fig.colorbar(contour, ax=ax, label="Curah hujan")
    else:
        ax.text(0.5, 0.5, "Tidak ada nilai grid valid", transform=ax.transAxes, ha="center", va="center")

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_gdf.boundary.plot(ax=ax, color="#111827", linewidth=1.2)

    ax.scatter(
        period_data["Longitude"],
        period_data["Latitude"],
        c=period_data["Curah_Hujan_Agregat"],
        cmap="magma",
        edgecolor="white",
        linewidth=0.8,
        s=45,
        label="Stasiun",
    )
    for _, row in period_data.iterrows():
        ax.annotate(row["Nama_Stasiun"], (row["Longitude"], row["Latitude"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Streamlit state and UI helpers
# ---------------------------------------------------------------------------

def initialize_state() -> None:
    defaults = {
        "raw_df": None,
        "active_data_label": None,
        "processed_df": None,
        "preprocess_stats": None,
        "aggregated_df": None,
        "period_data": None,
        "scale": "Harian",
        "agg_method": "SUM",
        "boundary_gdf": None,
        "loocv_results": None,
        "best_params": None,
        "best_predictions": None,
        "final_estimates": None,
        "grid_df": None,
        "geotiff_bytes": None,
        "grid_png_bytes": None,
        "grid_context": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_dataframe(key: str, message: str) -> Optional[pd.DataFrame]:
    df = st.session_state.get(key)
    if df is None or df.empty:
        st.info(message)
        return None
    return df


def show_metric_cards(stats: Dict[str, object]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Jumlah baris", f"{stats['jumlah_baris']:,}")
    col2.metric("Stasiun unik", f"{stats['jumlah_stasiun']:,}")
    col3.metric("Nilai kosong", f"{stats['curah_missing']:,}")
    col4.metric("Baris invalid dihapus", f"{stats['baris_index_invalid_dihapus']:,}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Minimum", f"{stats['curah_min']:.3f}" if pd.notna(stats["curah_min"]) else "-")
    col2.metric("Maksimum", f"{stats['curah_max']:.3f}" if pd.notna(stats["curah_max"]) else "-")
    col3.metric("Rata-rata", f"{stats['curah_mean']:.3f}" if pd.notna(stats["curah_mean"]) else "-")
    col4.metric("Total", f"{stats['curah_total']:.3f}" if pd.notna(stats["curah_total"]) else "-")


def parameter_controls(prefix: str, default_power: float = 2.0, default_k: int = 4) -> Tuple[float, int, Optional[float], str]:
    col1, col2, col3, col4 = st.columns(4)
    power = col1.number_input("Power p", min_value=0.1, max_value=10.0, value=float(default_power), step=0.1, key=f"{prefix}_power")
    k = col2.number_input("Tetangga k", min_value=1, max_value=100, value=max(1, int(default_k)), step=1, key=f"{prefix}_k")
    use_radius = col3.checkbox("Gunakan radius", value=False, key=f"{prefix}_use_radius")
    distance_mode = col4.selectbox("Jarak", ["UTM otomatis", "Derajat langsung"], key=f"{prefix}_distance_mode")
    radius = None
    if use_radius:
        label = "Radius maksimum (meter)" if distance_mode == "UTM otomatis" else "Radius maksimum (derajat)"
        radius = st.number_input(label, min_value=0.0, value=5000.0 if distance_mode == "UTM otomatis" else 0.05, step=100.0 if distance_mode == "UTM otomatis" else 0.01, key=f"{prefix}_radius")
    return float(power), int(k), radius, distance_mode


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_upload_data() -> None:
    st.subheader("Upload Data")
    st.write("Data curah hujan yang diupload atau diinput manual akan disimpan permanen di master data lokal.")

    stored = load_persistent_rainfall_data()
    col1, col2, col3 = st.columns(3)
    col1.metric("Baris data tersimpan", f"{len(stored):,}")
    col2.metric("Stasiun tersimpan", f"{stored['Nama_Stasiun'].nunique():,}" if not stored.empty else "0")
    col3.metric("Lokasi file", str(MASTER_DATA_PATH))

    tab_upload, tab_manual, tab_boundary, tab_manage = st.tabs(
        ["Upload File", "Input Manual", "Batas Wilayah", "Data Tersimpan"]
    )

    with tab_upload:
        st.write("Unggah data curah hujan long format dengan kolom minimal: Tanggal, Nama_Stasiun, Longitude, Latitude, Curah_Hujan.")
        rainfall_file = st.file_uploader("File CSV/XLSX data curah hujan", type=["csv", "xlsx", "xls"])
        save_mode = st.selectbox(
            "Mode penyimpanan file",
            ["Tambahkan dan update duplikat", "Tambahkan semua baris", "Ganti seluruh data tersimpan"],
            help="Duplikat dikenali dari Tanggal, Nama_Stasiun, Longitude, dan Latitude.",
        )
        activate_after_save = st.checkbox("Jadikan master data sebagai dataset aktif setelah disimpan", value=True)

        if rainfall_file is not None:
            try:
                uploaded_df = load_rainfall_file(rainfall_file)
                st.success("Data file berhasil dibaca dan kolom wajib valid.")
                st.dataframe(uploaded_df.head(80), width="stretch")
                checked_df, _ = normalize_rainfall_records(uploaded_df, drop_invalid_index_rows=False)
                invalid_mask, reason_table = build_invalid_reason_table(checked_df)
                valid_rows = int((~invalid_mask).sum())
                st.caption(f"Validasi isi data: {valid_rows:,} baris valid, {int(invalid_mask.sum()):,} baris invalid.")
                if not reason_table.empty:
                    with st.expander("Rincian baris invalid sebelum disimpan"):
                        st.dataframe(reason_table, width="stretch")
                        st.dataframe(checked_df.loc[invalid_mask].head(50), width="stretch")
                if st.button("Simpan file ke data permanen", type="primary"):
                    saved, info = append_to_persistent_rainfall_data(uploaded_df, save_mode)
                    if activate_after_save:
                        set_active_raw_data(saved, "Master data tersimpan")
                    st.success(
                        f"Tersimpan {info['baris_input_valid']:,} baris valid. "
                        f"Total master data sekarang {info['baris_total']:,} baris."
                    )
                    if info["baris_input_invalid_dilewati"] > 0:
                        st.warning(f"{info['baris_input_invalid_dilewati']:,} baris dilewati karena tanggal, stasiun, atau koordinat tidak valid.")
            except Exception as exc:
                st.error(str(exc))

    with tab_manual:
        stored_extra = [column for column in stored.columns if column not in REQUIRED_COLUMNS] if not stored.empty else []
        extra_columns_raw = st.text_input(
            "Kolom tambahan input manual",
            value=", ".join(stored_extra),
            help="Opsional. Isi jika data master punya kolom tambahan di luar kolom wajib.",
        )
        columns = manual_input_columns(extra_columns_raw)
        st.caption("Kolom input mengikuti kolom wajib dan kolom tambahan pada master data.")
        manual_df = st.data_editor(
            empty_manual_dataframe(columns),
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "Tanggal": st.column_config.DateColumn("Tanggal", format="YYYY-MM-DD"),
                "Nama_Stasiun": st.column_config.TextColumn("Nama_Stasiun"),
                "Longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                "Latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                "Curah_Hujan": st.column_config.NumberColumn("Curah_Hujan", format="%.3f"),
            },
            key="manual_rainfall_editor",
        )
        manual_save_mode = st.selectbox(
            "Mode penyimpanan input manual",
            ["Tambahkan dan update duplikat", "Tambahkan semua baris", "Ganti seluruh data tersimpan"],
            key="manual_save_mode",
        )
        if st.button("Simpan input manual ke data permanen", type="primary"):
            try:
                manual_clean = manual_df.dropna(how="all").copy()
                if manual_clean.empty:
                    st.error("Belum ada baris input manual yang diisi.")
                else:
                    saved, info = append_to_persistent_rainfall_data(manual_clean, manual_save_mode)
                    set_active_raw_data(saved, "Master data tersimpan")
                    st.success(
                        f"Input manual tersimpan. Total master data sekarang {info['baris_total']:,} baris."
                    )
                    if info["baris_input_invalid_dilewati"] > 0:
                        st.warning(f"{info['baris_input_invalid_dilewati']:,} baris manual dilewati karena tanggal, stasiun, atau koordinat tidak valid.")
            except Exception as exc:
                st.error(str(exc))

    with tab_boundary:
        col1, col2 = st.columns(2)
        shp_files = col1.file_uploader(
            "SHP/ZIP SHP batas wilayah opsional",
            type=["zip", "shp", "shx", "dbf", "prj", "cpg"],
            accept_multiple_files=True,
        )
        geojson_file = col2.file_uploader("GeoJSON batas wilayah opsional", type=["geojson", "json"])

        boundary = None
        if shp_files:
            try:
                boundary = load_boundary_from_shp_uploads(shp_files)
                st.session_state.boundary_gdf = boundary
                st.success("Batas wilayah dari SHP/ZIP SHP berhasil dibaca.")
            except Exception as exc:
                st.error(f"Gagal membaca SHP/ZIP SHP: {exc}")
        if geojson_file is not None:
            try:
                boundary = load_boundary_from_geojson(geojson_file)
                st.session_state.boundary_gdf = boundary
                st.success("Batas wilayah dari GeoJSON berhasil dibaca.")
            except Exception as exc:
                st.error(f"Gagal membaca GeoJSON: {exc}")

        if st.session_state.boundary_gdf is not None:
            st.write("Preview batas wilayah")
            st.dataframe(st.session_state.boundary_gdf.drop(columns="geometry", errors="ignore").head(20), width="stretch")

    with tab_manage:
        stored = load_persistent_rainfall_data()
        if stored.empty:
            st.info("Belum ada data permanen. Upload file atau isi input manual terlebih dahulu.")
        else:
            col1, col2 = st.columns(2)
            if col1.button("Gunakan seluruh data tersimpan"):
                set_active_raw_data(stored, "Seluruh master data tersimpan")
                st.success("Seluruh master data sudah menjadi dataset aktif.")
            col2.download_button(
                "Download master data CSV",
                dataframe_to_csv_bytes(stored),
                "curah_hujan_master.csv",
                mime="text/csv",
            )
            st.dataframe(stored.tail(200), width="stretch")


def page_select_data() -> None:
    st.subheader("Pilih Data")
    stored = load_persistent_rainfall_data()
    if stored.empty:
        st.info("Belum ada data tersimpan. Tambahkan data lewat menu Upload Data terlebih dahulu.")
        return

    working = stored.copy()
    working["Tanggal"] = pd.to_datetime(working["Tanggal"], errors="coerce")
    working["Longitude"] = pd.to_numeric(working["Longitude"], errors="coerce")
    working["Latitude"] = pd.to_numeric(working["Latitude"], errors="coerce")
    working["Curah_Hujan"] = pd.to_numeric(working["Curah_Hujan"], errors="coerce")

    st.write("Pilih subset data dari master permanen berdasarkan stasiun dan rentang tanggal.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Baris master", f"{len(working):,}")
    col2.metric("Stasiun master", f"{working['Nama_Stasiun'].nunique():,}")
    date_min = working["Tanggal"].min()
    date_max = working["Tanggal"].max()
    col3.metric("Rentang tanggal", f"{date_min.date()} - {date_max.date()}" if pd.notna(date_min) and pd.notna(date_max) else "-")

    stations = sorted(working["Nama_Stasiun"].dropna().unique())
    selected_stations = st.multiselect(
        "Pilih stasiun",
        stations,
        default=stations,
        help="Kosongkan pilihan untuk memakai semua stasiun.",
    )

    date_range = st.date_input(
        "Rentang tanggal",
        value=(date_min.date(), date_max.date()) if pd.notna(date_min) and pd.notna(date_max) else None,
        min_value=date_min.date() if pd.notna(date_min) else None,
        max_value=date_max.date() if pd.notna(date_max) else None,
    )

    start_date = end_date = None
    if isinstance(date_range, (tuple, list)):
        selected_dates = [date_value for date_value in date_range if date_value is not None]
        if len(selected_dates) == 1:
            start_date = end_date = selected_dates[0]
            st.caption("Tanggal akhir belum dipilih. Preview sementara memakai satu tanggal yang dipilih.")
        elif len(selected_dates) >= 2:
            start_date, end_date = selected_dates[0], selected_dates[1]
    elif date_range:
        start_date = end_date = date_range

    if start_date is not None and end_date is not None and start_date > end_date:
        start_date, end_date = end_date, start_date

    filtered = working.copy()
    if selected_stations:
        filtered = filtered.loc[filtered["Nama_Stasiun"].isin(selected_stations)]
    if start_date is not None and end_date is not None:
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        filtered = filtered.loc[(filtered["Tanggal"] >= start_ts) & (filtered["Tanggal"] <= end_ts)]

    col1, col2, col3 = st.columns(3)
    col1.metric("Baris terpilih", f"{len(filtered):,}")
    col2.metric("Stasiun terpilih", f"{filtered['Nama_Stasiun'].nunique():,}")
    col3.metric("Nilai curah kosong", f"{filtered['Curah_Hujan'].isna().sum():,}")

    st.dataframe(filtered.head(500), width="stretch")

    if st.button("Gunakan data terpilih untuk analisis", type="primary"):
        if filtered.empty:
            st.error("Data terpilih kosong. Ubah pilihan stasiun atau rentang tanggal.")
        else:
            output = filtered.copy()
            output["Tanggal"] = output["Tanggal"].dt.strftime("%Y-%m-%d")
            label = f"{len(output):,} baris dari {output['Nama_Stasiun'].nunique():,} stasiun"
            set_active_raw_data(output, label)
            st.success("Data terpilih sudah menjadi dataset aktif. Lanjutkan ke Pra-pemrosesan.")


def page_preprocessing() -> None:
    st.subheader("Pra-pemrosesan Data")
    raw_df = require_dataframe("raw_df", "Pilih data dari master permanen atau tambahkan data lewat Upload Data terlebih dahulu.")
    if raw_df is None:
        stored = load_persistent_rainfall_data()
        if not stored.empty and st.button("Gunakan seluruh data tersimpan"):
            set_active_raw_data(stored, "Seluruh master data tersimpan")
            st.rerun()
        return

    if st.session_state.active_data_label:
        st.caption(f"Dataset aktif: {st.session_state.active_data_label}")

    drop_invalid = st.checkbox("Hapus baris dengan tanggal, nama stasiun, atau koordinat tidak valid", value=True)
    processed, stats = preprocess_data(raw_df, drop_invalid)
    st.session_state.processed_df = processed
    st.session_state.preprocess_stats = stats

    show_metric_cards(stats)
    if pd.notna(stats["tanggal_min"]) and pd.notna(stats["tanggal_max"]):
        st.caption(f"Rentang tanggal: {stats['tanggal_min'].date()} sampai {stats['tanggal_max'].date()}")
    st.dataframe(processed.head(200), width="stretch")


def page_accumulation() -> None:
    st.subheader("Akumulasi Data dan Pemilihan Periode")
    processed = require_dataframe("processed_df", "Lakukan pra-pemrosesan data terlebih dahulu.")
    if processed is None:
        return

    col1, col2 = st.columns(2)
    scale = col1.selectbox("Jenis data", ["Harian", "Bulanan", "Tahunan"], index=["Harian", "Bulanan", "Tahunan"].index(st.session_state.scale))
    agg_method = col2.selectbox("Metode agregasi", ["SUM", "MEAN", "MAX"], index=["SUM", "MEAN", "MAX"].index(st.session_state.agg_method))
    st.session_state.scale = scale
    st.session_state.agg_method = agg_method

    aggregated = aggregate_rainfall(processed, scale, agg_method)
    st.session_state.aggregated_df = aggregated
    st.write(f"Data teragregasi skala {scale.lower()} dengan metode {agg_method}.")

    if aggregated.empty:
        st.warning("Tidak ada data teragregasi.")
        return

    selected_date = None
    selected_month = None
    selected_year = None
    if scale == "Harian":
        available_dates = sorted(aggregated["Tanggal"].dropna().dt.date.unique())
        selected_date = st.selectbox("Pilih tanggal", available_dates)
    elif scale == "Bulanan":
        years = sorted(aggregated["Tahun"].dropna().astype(int).unique())
        months = sorted(aggregated["Bulan"].dropna().astype(int).unique())
        col1, col2 = st.columns(2)
        selected_year = col1.selectbox("Pilih tahun", years)
        selected_month = col2.selectbox("Pilih bulan", months, format_func=lambda month: f"{int(month):02d}")
    else:
        years = sorted(aggregated["Tahun"].dropna().astype(int).unique())
        selected_year = st.selectbox("Pilih tahun", years)

    period_data = filter_period_data(aggregated, scale, selected_date, selected_month, selected_year)
    st.session_state.period_data = period_data.reset_index(drop=True)

    st.metric("Jumlah stasiun pada periode", period_data["Nama_Stasiun"].nunique())
    st.dataframe(
        period_data[["Nama_Stasiun", "Longitude", "Latitude", "Curah_Hujan_Agregat", "Status_Data"]],
        width="stretch",
    )

    if not period_data.empty:
        st.plotly_chart(plot_station_bar(period_data), width="stretch")
        with st.expander("Peta titik stasiun"):
            st_folium(make_station_map(period_data, st.session_state.boundary_gdf), width=900, height=520)


def page_loocv() -> None:
    st.subheader("LOOCV Tuning IDW")
    period_data = require_dataframe("period_data", "Pilih periode analisis pada halaman Akumulasi Data terlebih dahulu.")
    if period_data is None:
        return

    valid_count = int(period_data["Curah_Hujan_Agregat"].notna().sum())
    st.caption(f"Stasiun observasi valid untuk LOOCV: {valid_count}")

    col1, col2 = st.columns(2)
    power_text = col1.text_input("Daftar power p", value="1, 1.5, 2, 2.5, 3")
    k_text = col2.text_input("Daftar k", value="3, 4, 5, 6, 7, 8")
    col1, col2, col3 = st.columns(3)
    metric = col1.selectbox("Metrik utama", ["RMSE", "MAE"])
    distance_mode = col2.selectbox("Mode jarak", ["UTM otomatis", "Derajat langsung"], key="loocv_distance")
    use_radius = col3.checkbox("Gunakan radius opsional", value=False, key="loocv_use_radius")
    radius = None
    if use_radius:
        label = "Radius (meter)" if distance_mode == "UTM otomatis" else "Radius (derajat)"
        radius = st.number_input(label, min_value=0.0, value=5000.0 if distance_mode == "UTM otomatis" else 0.05, step=100.0 if distance_mode == "UTM otomatis" else 0.01)

    if st.button("Jalankan LOOCV", type="primary"):
        try:
            power_values = parse_number_list(power_text, float)
            k_values = [int(value) for value in parse_number_list(k_text, int)]
            if not power_values or not k_values:
                st.error("Daftar p dan k tidak boleh kosong.")
                return
            result, best, predictions = run_loocv(period_data, power_values, k_values, radius, metric, distance_mode)
            st.session_state.loocv_results = result
            st.session_state.best_params = {
                "p": float(best["p"]),
                "k": int(best["k"]),
                "MAE": float(best["MAE"]),
                "RMSE": float(best["RMSE"]),
                "MAPE": float(best["MAPE"]) if pd.notna(best["MAPE"]) else np.nan,
                "metric": metric,
                "radius": radius,
                "distance_mode": distance_mode,
            }
            st.session_state.best_predictions = predictions
            st.success("LOOCV selesai. Parameter terbaik sudah disimpan untuk interpolasi final.")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.loocv_results is not None:
        result = st.session_state.loocv_results
        best = st.session_state.best_params
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("p terbaik", f"{best['p']:.3g}")
        col2.metric("k terbaik", f"{best['k']}")
        col3.metric("MAE", f"{best['MAE']:.4f}")
        col4.metric("RMSE", f"{best['RMSE']:.4f}")
        col5.metric("MAPE", f"{best['MAPE']:.2f}%" if pd.notna(best["MAPE"]) else "-")

        st.dataframe(result, width="stretch")

        col1, col2 = st.columns(2)
        col1.plotly_chart(
            px.line(result.sort_values(["k", "p"]), x="p", y="RMSE", color="k", markers=True, title="RMSE berdasarkan p dan k"),
            width="stretch",
        )
        col2.plotly_chart(
            px.line(result.sort_values(["k", "p"]), x="p", y="MAE", color="k", markers=True, title="MAE berdasarkan p dan k"),
            width="stretch",
        )
        st.plotly_chart(plot_loocv_heatmap(result, "RMSE"), width="stretch")

        if st.session_state.best_predictions is not None:
            predictions = st.session_state.best_predictions
            col1, col2 = st.columns(2)
            scatter_fig = px.scatter(
                predictions,
                x="Observasi",
                y="Prediksi",
                hover_name="Nama_Stasiun",
                title="LOOCV observasi vs prediksi",
            )
            axis_min = float(np.nanmin([predictions["Observasi"].min(), predictions["Prediksi"].min()]))
            axis_max = float(np.nanmax([predictions["Observasi"].max(), predictions["Prediksi"].max()]))
            scatter_fig.add_shape(
                type="line",
                x0=axis_min,
                y0=axis_min,
                x1=axis_max,
                y1=axis_max,
                line={"color": "#374151", "dash": "dash"},
            )
            col1.plotly_chart(scatter_fig, width="stretch")
            col2.plotly_chart(
                px.bar(predictions.sort_values("Abs_Error", ascending=False), x="Nama_Stasiun", y="Error", title="Error per stasiun"),
                width="stretch",
            )


def build_prediction_reference(period_data: pd.DataFrame) -> pd.DataFrame:
    return period_data.dropna(subset=["Curah_Hujan_Agregat"]).copy()


def estimate_targets(
    reference: pd.DataFrame,
    targets: pd.DataFrame,
    power: float,
    k: int,
    radius: Optional[float],
    distance_mode: str,
) -> pd.DataFrame:
    predictions = idw_predict_many(
        reference["Longitude"],
        reference["Latitude"],
        reference["Curah_Hujan_Agregat"],
        targets["Longitude"],
        targets["Latitude"],
        power=power,
        k=k,
        radius=radius,
        distance_mode=distance_mode,
    )
    output = targets.copy()
    output["Curah_Hujan_Estimasi"] = predictions
    return output


def page_final_interpolation() -> None:
    st.subheader("Interpolasi IDW Final")
    period_data = require_dataframe("period_data", "Pilih periode analisis terlebih dahulu.")
    if period_data is None:
        return

    best = st.session_state.best_params or {"p": 2.0, "k": min(4, max(1, len(period_data) - 1)), "radius": None, "distance_mode": "UTM otomatis"}
    if st.session_state.best_params is None:
        st.warning("Parameter LOOCV belum tersedia. Nilai default ditampilkan, tetapi sebaiknya jalankan LOOCV terlebih dahulu.")

    power, k, radius, distance_mode = parameter_controls("final", best["p"], best["k"])
    if best.get("radius") is not None and radius is None:
        st.caption(f"Radius terbaik dari LOOCV: {best['radius']}")

    reference = build_prediction_reference(period_data)
    if len(reference) < 2:
        st.error("Interpolasi final membutuhkan minimal 2 stasiun observasi valid.")
        return

    tab1, tab2, tab3 = st.tabs(["Prediksi titik manual", "Prediksi beberapa titik", "Prediksi data missing"])

    with tab1:
        col1, col2 = st.columns(2)
        target_lon = col1.number_input("Longitude target", value=float(reference["Longitude"].mean()), format="%.6f")
        target_lat = col2.number_input("Latitude target", value=float(reference["Latitude"].mean()), format="%.6f")
        if st.button("Hitung titik manual"):
            estimate = idw_predict(
                reference["Longitude"],
                reference["Latitude"],
                reference["Curah_Hujan_Agregat"],
                target_lon,
                target_lat,
                power=power,
                k=k,
                radius=radius,
                distance_mode=distance_mode,
            )
            manual_result = pd.DataFrame(
                [
                    {
                        "Nama_Lokasi": "Target Manual",
                        "Longitude": target_lon,
                        "Latitude": target_lat,
                        "Curah_Hujan_Asli": np.nan,
                        "Curah_Hujan_Estimasi": estimate,
                        "Status": "Estimasi",
                    }
                ]
            )
            st.session_state.final_estimates = manual_result
            st.metric("Estimasi curah hujan", f"{estimate:.4f}" if np.isfinite(estimate) else "-")
            st.dataframe(manual_result, width="stretch")

    with tab2:
        target_file = st.file_uploader("Upload CSV titik target: Nama_Lokasi, Longitude, Latitude", type=["csv"], key="target_points_csv")
        if target_file is not None:
            try:
                targets = pd.read_csv(target_file)
                targets = canonicalize_columns(targets)
                if "Nama_Lokasi" not in targets.columns:
                    if "Nama_Stasiun" in targets.columns:
                        targets = targets.rename(columns={"Nama_Stasiun": "Nama_Lokasi"})
                    else:
                        targets["Nama_Lokasi"] = [f"Target {idx + 1}" for idx in range(len(targets))]
                missing_columns = [column for column in ["Longitude", "Latitude"] if column not in targets.columns]
                if missing_columns:
                    st.error("Kolom target belum lengkap: " + ", ".join(missing_columns))
                else:
                    targets["Longitude"] = pd.to_numeric(targets["Longitude"], errors="coerce")
                    targets["Latitude"] = pd.to_numeric(targets["Latitude"], errors="coerce")
                    targets = targets.dropna(subset=["Longitude", "Latitude"]).copy()
                    output = estimate_targets(reference, targets[["Nama_Lokasi", "Longitude", "Latitude"]], power, k, radius, distance_mode)
                    output["Curah_Hujan_Asli"] = np.nan
                    output["Status"] = "Estimasi"
                    output = output[["Nama_Lokasi", "Longitude", "Latitude", "Curah_Hujan_Asli", "Curah_Hujan_Estimasi", "Status"]]
                    st.session_state.final_estimates = output
                    st.dataframe(output, width="stretch")
            except Exception as exc:
                st.error(f"Gagal memproses titik target: {exc}")

    with tab3:
        missing_rows = period_data.loc[period_data["Curah_Hujan_Agregat"].isna()].copy()
        st.caption(f"Jumlah data missing pada periode ini: {len(missing_rows)}")
        if st.button("Estimasi data missing"):
            if missing_rows.empty:
                st.info("Tidak ada nilai Curah_Hujan yang kosong pada periode ini.")
            else:
                output = estimate_targets(reference, missing_rows[["Nama_Stasiun", "Longitude", "Latitude"]], power, k, radius, distance_mode)
                output = output.rename(columns={"Nama_Stasiun": "Nama_Lokasi"})
                output["Curah_Hujan_Asli"] = np.nan
                output["Status"] = "Estimasi"
                output = output[["Nama_Lokasi", "Longitude", "Latitude", "Curah_Hujan_Asli", "Curah_Hujan_Estimasi", "Status"]]
                observed_output = reference[["Nama_Stasiun", "Longitude", "Latitude", "Curah_Hujan_Agregat"]].copy()
                observed_output = observed_output.rename(
                    columns={
                        "Nama_Stasiun": "Nama_Lokasi",
                        "Curah_Hujan_Agregat": "Curah_Hujan_Asli",
                    }
                )
                observed_output["Curah_Hujan_Estimasi"] = observed_output["Curah_Hujan_Asli"]
                observed_output["Status"] = "Observasi"
                observed_output = observed_output[["Nama_Lokasi", "Longitude", "Latitude", "Curah_Hujan_Asli", "Curah_Hujan_Estimasi", "Status"]]
                combined_output = pd.concat([observed_output, output], ignore_index=True)
                st.session_state.final_estimates = combined_output
                st.dataframe(combined_output, width="stretch")

    if st.session_state.final_estimates is not None:
        st.download_button(
            "Download CSV hasil estimasi",
            data=dataframe_to_csv_bytes(st.session_state.final_estimates),
            file_name="hasil_estimasi_idw.csv",
            mime="text/csv",
        )


def page_grid_raster() -> None:
    st.subheader("Peta Grid/Raster")
    period_data = require_dataframe("period_data", "Pilih periode analisis terlebih dahulu.")
    if period_data is None:
        return

    best = st.session_state.best_params or {"p": 2.0, "k": min(4, max(1, len(period_data) - 1))}
    if st.session_state.best_params is None:
        st.warning("Parameter LOOCV belum tersedia. Jalankan LOOCV agar p dan k final berbasis validasi silang.")

    st.write("Peta titik stasiun")
    st_folium(make_station_map(period_data, st.session_state.boundary_gdf), width=920, height=460)

    col1, col2, col3 = st.columns(3)
    resolution = col1.select_slider("Resolusi grid", options=[50, 100, 150, 200], value=100)
    power = col2.number_input("Power p grid", min_value=0.1, max_value=10.0, value=float(best["p"]), step=0.1)
    k = col3.number_input("Tetangga k grid", min_value=1, max_value=100, value=int(best["k"]), step=1)
    col1, col2 = st.columns(2)
    distance_mode = col1.selectbox("Mode jarak grid", ["UTM otomatis", "Derajat langsung"], key="grid_distance")
    use_radius = col2.checkbox("Gunakan radius grid", value=False, key="grid_use_radius")
    radius = None
    if use_radius:
        label = "Radius grid (meter)" if distance_mode == "UTM otomatis" else "Radius grid (derajat)"
        radius = st.number_input(label, min_value=0.0, value=5000.0 if distance_mode == "UTM otomatis" else 0.05, step=100.0 if distance_mode == "UTM otomatis" else 0.01)

    if st.button("Buat grid interpolasi dan raster", type="primary"):
        try:
            lon_grid, lat_grid, z_grid, grid_df = create_grid_interpolation(
                period_data,
                st.session_state.boundary_gdf,
                resolution=int(resolution),
                power=float(power),
                k=int(k),
                radius=radius,
                distance_mode=distance_mode,
            )
            period_label = period_data["Periode"].iloc[0] if "Periode" in period_data.columns and not period_data.empty else ""
            title = f"Interpolasi IDW Curah Hujan - {period_label}"
            fig = plot_grid_contour(lon_grid, lat_grid, z_grid, period_data.dropna(subset=["Curah_Hujan_Agregat"]), st.session_state.boundary_gdf, title)
            geotiff_bytes = export_geotiff(lon_grid, lat_grid, z_grid)
            png_bytes = fig_to_png_bytes(fig)

            st.session_state.grid_df = grid_df
            st.session_state.geotiff_bytes = geotiff_bytes
            st.session_state.grid_png_bytes = png_bytes
            st.session_state.grid_context = {
                "lon_grid": lon_grid,
                "lat_grid": lat_grid,
                "z_grid": z_grid,
                "title": title,
            }
            st.success("Grid interpolasi dan raster GeoTIFF berhasil dibuat.")
            st.pyplot(fig)
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.grid_context is not None and st.session_state.grid_df is not None:
        context = st.session_state.grid_context
        fig = plot_grid_contour(
            context["lon_grid"],
            context["lat_grid"],
            context["z_grid"],
            period_data.dropna(subset=["Curah_Hujan_Agregat"]),
            st.session_state.boundary_gdf,
            context["title"],
        )
        st.pyplot(fig)
        col1, col2, col3 = st.columns(3)
        col1.download_button("Download GeoTIFF", st.session_state.geotiff_bytes, "raster_idw.tif", mime="image/tiff")
        col2.download_button("Download PNG peta", st.session_state.grid_png_bytes, "peta_grid_idw.png", mime="image/png")
        col3.download_button("Download CSV grid", dataframe_to_csv_bytes(st.session_state.grid_df), "grid_interpolasi_idw.csv", mime="text/csv")


def page_downloads() -> None:
    st.subheader("Unduh Hasil")
    available = False

    if st.session_state.loocv_results is not None:
        available = True
        st.download_button(
            "Download CSV hasil LOOCV",
            dataframe_to_csv_bytes(st.session_state.loocv_results),
            "hasil_loocv_idw.csv",
            mime="text/csv",
        )

    if st.session_state.best_predictions is not None:
        available = True
        st.download_button(
            "Download CSV observasi vs prediksi LOOCV terbaik",
            dataframe_to_csv_bytes(st.session_state.best_predictions),
            "loocv_observasi_prediksi.csv",
            mime="text/csv",
        )

    if st.session_state.final_estimates is not None:
        available = True
        st.download_button(
            "Download CSV hasil estimasi",
            dataframe_to_csv_bytes(st.session_state.final_estimates),
            "hasil_estimasi_idw.csv",
            mime="text/csv",
        )

    if st.session_state.grid_df is not None:
        available = True
        st.download_button(
            "Download CSV grid interpolasi",
            dataframe_to_csv_bytes(st.session_state.grid_df),
            "grid_interpolasi_idw.csv",
            mime="text/csv",
        )

    if st.session_state.geotiff_bytes is not None:
        available = True
        st.download_button("Download GeoTIFF", st.session_state.geotiff_bytes, "raster_idw.tif", mime="image/tiff")

    if st.session_state.grid_png_bytes is not None:
        available = True
        st.download_button("Download PNG peta", st.session_state.grid_png_bytes, "peta_grid_idw.png", mime="image/png")

    if not available:
        st.info("Belum ada hasil untuk diunduh. Jalankan LOOCV, interpolasi final, atau pembuatan grid/raster terlebih dahulu.")


def main() -> None:
    st.set_page_config(page_title="SIRAH IDW", layout="wide")
    initialize_state()

    st.title(APP_TITLE)
    st.caption("Aplikasi penelitian untuk akumulasi curah hujan, tuning IDW dengan LOOCV, interpolasi final, serta ekspor grid dan raster.")

    menu = st.sidebar.radio(
        "Navigasi",
        [
            "Upload Data",
            "Pilih Data",
            "Pra-pemrosesan",
            "Akumulasi Data",
            "LOOCV Tuning IDW",
            "Interpolasi IDW Final",
            "Peta Grid/Raster",
            "Unduh Hasil",
        ],
    )

    st.sidebar.divider()
    stored_count = len(load_persistent_rainfall_data())
    st.sidebar.info(f"Data tersimpan: {stored_count:,} baris")
    if st.session_state.raw_df is not None:
        st.sidebar.success(f"Dataset aktif: {len(st.session_state.raw_df):,} baris")
    elif st.session_state.active_data_label:
        st.sidebar.success(f"Dataset aktif: {st.session_state.active_data_label}")
    if st.session_state.period_data is not None:
        st.sidebar.info(f"Periode aktif: {st.session_state.period_data['Periode'].iloc[0] if not st.session_state.period_data.empty else '-'}")
    if st.session_state.best_params is not None:
        st.sidebar.success(f"Best IDW: p={st.session_state.best_params['p']:.3g}, k={st.session_state.best_params['k']}")

    if menu == "Upload Data":
        page_upload_data()
    elif menu == "Pilih Data":
        page_select_data()
    elif menu == "Pra-pemrosesan":
        page_preprocessing()
    elif menu == "Akumulasi Data":
        page_accumulation()
    elif menu == "LOOCV Tuning IDW":
        page_loocv()
    elif menu == "Interpolasi IDW Final":
        page_final_interpolation()
    elif menu == "Peta Grid/Raster":
        page_grid_raster()
    elif menu == "Unduh Hasil":
        page_downloads()


if __name__ == "__main__":
    main()

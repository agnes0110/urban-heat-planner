"""BigQuery function tools for the Track 1 Emergency Heatwave Response ADK Agent.

Data Sources in `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate`:
1. Operational Heat Offset, Vulnerability & Tri-Agency Dispatch Tools:
   - BigQuery ML Model: `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`
     (evaluated via `ML.EXPLAIN_PREDICT` over `v_tract_heat_features`)
2. Historic Data Tools (used EXCLUSIVELY by `historic_heat_data_agent`):
   - `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations`
   - `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps`
"""

from __future__ import annotations

import math
from typing import Any

import requests
from google.cloud import bigquery

from .config import (
    BQML_OFFSET_MODEL,
    BQML_OFFSET_MODEL_NAME,
    DALLAS_DEFAULT_LAT,
    DALLAS_DEFAULT_LON,
    DANGEROUS_THRESHOLD_TEMP,
    DATASET_ID,
    FEATURES_TABLE,
    FEATURES_TABLE_NAME,
    PROJECT_ID,
    TABLE_NAME,
    UVI_TABLE,
)

# Historic tables (ONLY these 2 tables are used by the Historic Heat Data Agent)
HEAT_OBSERVATIONS_TABLE_NAME = "heat_observations"
STATION_DAILY_TEMPS_TABLE_NAME = "station_daily_temps"
HEAT_OBSERVATIONS_TABLE = f"`{PROJECT_ID}.{DATASET_ID}.{HEAT_OBSERVATIONS_TABLE_NAME}`"
STATION_DAILY_TEMPS_TABLE = f"`{PROJECT_ID}.{DATASET_ID}.{STATION_DAILY_TEMPS_TABLE_NAME}`"

# Defensive SQL filter for clean ACS rows (excludes sentinel -666666666 / 666668692 rows)
CLEAN_ROW_FILTER = (
    "median_age > 0 "
    "AND universal_vulnerability_index >= 0 "
    "AND building_age_years BETWEEN 0 AND 250 "
    "AND surrounding_3km_bldg_age BETWEEN 0 AND 250"
)

_bq_client: bigquery.Client | None = None


def _get_bq_client() -> bigquery.Client:
    """Return a cached BigQuery client initialized for the target project.
    Automatically uses `gcloud auth print-access-token` in Cloud Shell to avoid
    GCE metadata server RefreshError.
    """
    global _bq_client
    if _bq_client is None:
        import os
        import shutil
        import subprocess

        if os.environ.get("CLOUD_SHELL") == "true" and shutil.which("gcloud"):
            try:
                from google.oauth2.credentials import Credentials

                tok = subprocess.check_output(
                    ["gcloud", "auth", "print-access-token"], text=True, timeout=5
                ).strip()
                if tok:
                    _bq_client = bigquery.Client(
                        project=PROJECT_ID, credentials=Credentials(token=tok)
                    )
                    return _bq_client
            except Exception:
                pass
        _bq_client = bigquery.Client(project=PROJECT_ID)
    return _bq_client


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 points."""
    radius_km = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _build_tri_agency_action_plan(
    uvi_score: float,
    no_vehicle_rate: float,
    no_cars: int,
    average_age: float,
    heat_wave_risk_flag: str,
    urban_density_status: str,
    greenery_status: str,
    nearby_parks: int,
    capacity_status: str,
    cooling_center_type: str,
    energy_grid_status: str,
    observed_avg_temp: float,
) -> dict[str, Any]:
    """Generate the 3-pronged Tri-Agency Action Plan (Emergency, Hospital, Power Grid)
    and decompose the Universal Vulnerability Index (UVI) pillars.
    """
    # 1. Emergency Services Dispatch
    if no_vehicle_rate >= 0.12 and average_age >= 34.0:
        emergency_unit = (
            "Door-to-Door Wellness Check Team + Mobile Mist/Hydration Van"
        )
        emergency_rationale = (
            f"High carless isolation ({no_vehicle_rate * 100:.1f}% without vehicles, "
            f"{no_cars:,} households) combined with older residents (avg age {average_age:.1f}) "
            f"and existing {cooling_center_type} at '{capacity_status}'."
        )
    elif no_vehicle_rate >= 0.12 or no_cars >= 250:
        emergency_unit = (
            "Mobile Hydration Truck & Transit-Corridor Cooling Canopy"
        )
        emergency_rationale = (
            f"Concentrated transit-dependent population ({no_cars:,} carless households, "
            f"{no_vehicle_rate * 100:.1f}% no-vehicle rate) where local {cooling_center_type} "
            f"is '{capacity_status}'."
        )
    elif average_age >= 36.0:
        emergency_unit = (
            "Extended-Hours Public Library Cooling Center + Senior Paratransit Shuttle"
        )
        emergency_rationale = (
            f"Elevated older adult population (avg age {average_age:.1f}) requiring accessible "
            f"cooling relief and shuttle transport as local {cooling_center_type} is '{capacity_status}'."
        )
    else:
        emergency_unit = (
            "Pop-Up Hydration Station & Extended Public Library Cooling Hours"
        )
        emergency_rationale = (
            f"High thermal exposure ({observed_avg_temp:.2f}°F observed avg, '{heat_wave_risk_flag}' "
            f"heat wave risk) with local {cooling_center_type} currently '{capacity_status}'."
        )

    # 2. Hospital & Clinical Staff Standby (CMS Medicare Hospital Access Pillar)
    if uvi_score >= 100.0:
        hospital_alert_level = "CRITICAL SURGE STANDBY (Tier 1 Hospital Desert Risk)"
        hospital_action = (
            f"Alert nearest regional Emergency Departments and EMS medic units to go on immediate "
            f"heat-stroke & dehydration surge standby. UVI={uvi_score:.1f} indicates severe "
            f"CMS Medicare hospital-access gap paired with {no_cars:,} carless households "
            f"(avg age {average_age:.1f}) who cannot self-transport to clinical care."
        )
    elif uvi_score >= 60.0 or average_age >= 36.0:
        hospital_alert_level = "ELEVATED CLINICAL STANDBY (Tier 2 Triage Alert)"
        hospital_action = (
            f"Place receiving hospital triage nurses and mobile paramedic units on heat-exhaustion "
            f"standby (UVI={uvi_score:.1f}, avg age {average_age:.1f}, {no_cars:,} carless households). "
            f"Pre-stage IV hydration supplies for elderly arrivals."
        )
    else:
        hospital_alert_level = "STANDARD HEAT ADVISORY MONITORING (Tier 3)"
        hospital_action = (
            f"Maintain standard ED heat-illness surveillance (UVI={uvi_score:.1f}) and coordinate "
            f"non-emergency clinical transport with 311 cooling shuttles."
        )

    # 3. Electricity & Power Grid Team (NOAA CDD + OSM Building Density Pillar)
    is_high_heat_trap = (
        heat_wave_risk_flag == "Highly Likely"
        or urban_density_status == "Moderate Density"
        or greenery_status == "Urban Heat Island - Higher Risk"
    )
    if heat_wave_risk_flag == "Highly Likely" or observed_avg_temp >= 105.8:
        grid_alert_level = "HIGH FEEDER LOAD & TRANSFORMER WATCH (Priority 1)"
        grid_action = (
            f"Dispatch utility line crews to monitor local distribution transformers and pre-stage "
            f"mobile backup generators. Observed avg temp of {observed_avg_temp:.2f}°F "
            f"('{heat_wave_risk_flag}' heatwave flag), NOAA Cooling Degree Days status "
            f"('{energy_grid_status}'), and OpenStreetMap 1km building density "
            f"('{urban_density_status}', '{greenery_status}') create severe peak AC demand "
            f"and nighttime transformer overheating risk."
        )
    elif is_high_heat_trap:
        grid_alert_level = "PREVENTIVE AC LOAD & OUTAGE STANDBY (Priority 2)"
        grid_action = (
            f"Suspend non-emergency maintenance switching and monitor feeder load "
            f"(NOAA status: '{energy_grid_status}', OSM 1km density: '{urban_density_status}', "
            f"greenery: '{greenery_status}', {observed_avg_temp:.2f}°F). Ensure rapid restoration "
            f"priority if brownouts occur where {no_cars:,} households lack vehicles."
        )
    else:
        grid_alert_level = "ROUTINE GRID TELEMETRY WATCH (Priority 3)"
        grid_action = (
            f"Monitor feeder telemetry under '{energy_grid_status}' conditions "
            f"({observed_avg_temp:.2f}°F, '{urban_density_status}') and keep backup power "
            f"available for local {cooling_center_type}."
        )

    # UVI 3-Pillar Breakdown
    uvi_pillars = {
        "health_access_cms_medicare": (
            f"UVI Score {uvi_score:.2f} incorporates CMS Medicare Zip-Code hospital proximity "
            f"penalties with {no_cars:,} carless households ({no_vehicle_rate * 100:.1f}%) "
            f"and average age {average_age:.1f}."
        ),
        "energy_stress_noaa_cdd": (
            f"NOAA 2023 Cooling Degree Days (CDD) grid status: '{energy_grid_status}', "
            f"observed avg temperature {observed_avg_temp:.2f}°F, heatwave flag: '{heat_wave_risk_flag}'."
        ),
        "urban_heat_island_osm_1km": (
            f"OpenStreetMap 1km spatial join (`ST_DWithin`): '{urban_density_status}', "
            f"greenery status: '{greenery_status}' ({nearby_parks} nearby parks), driving concrete "
            f"albedo heat retention."
        ),
    }

    return {
        "recommended_intervention": emergency_unit,
        "intervention_rationale": emergency_rationale,
        "tri_agency_dispatch": {
            "emergency_services_dispatch": {
                "unit": emergency_unit,
                "rationale": emergency_rationale,
                "cooling_center_status": f"{cooling_center_type} ({capacity_status})",
            },
            "hospital_staff_standby": {
                "alert_level": hospital_alert_level,
                "action": hospital_action,
                "directive": hospital_action,
                "reasoning": f"UVI={uvi_score:.1f} with {no_cars:,} carless households ({no_vehicle_rate * 100:.1f}%) and median age {average_age:.1f}.",
            },
            "electricity_grid_response": {
                "alert_level": grid_alert_level,
                "action": grid_action,
                "directive": grid_action,
                "reasoning": f"NOAA CDD grid status '{energy_grid_status}', heatwave flag '{heat_wave_risk_flag}', and urban density '{urban_density_status}' ({observed_avg_temp:.1f}°F).",
            },
        },
        "uvi_pillars": uvi_pillars,
    }


def _format_bqml_attributions(raw_attributions: Any) -> list[dict[str, Any]]:
    """Convert BigQuery ML `top_feature_attributions` STRUCT array into clean dicts."""
    if not raw_attributions:
        return []
    formatted: list[dict[str, Any]] = []
    for item in raw_attributions:
        feat = item.get("feature") if isinstance(item, dict) else getattr(item, "feature", None)
        attr = item.get("attribution") if isinstance(item, dict) else getattr(item, "attribution", 0.0)
        formatted.append(
            {
                "feature": str(feat),
                "attribution": round(float(attr or 0.0), 4),
            }
        )
    return formatted


def predict_point_heat_offset(
    latitude: float,
    longitude: float,
    forecast_high_f: float = DANGEROUS_THRESHOLD_TEMP,
) -> dict[str, Any]:
    """Evaluate BigQuery ML `ML.EXPLAIN_PREDICT` spatial heat offset predictions and
    per-feature attributions from `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`,
    Universal Vulnerability Index (UVI), and the Tri-Agency response plan for any
    coordinate in Dallas.

    Checks the Dallas bounding box first (refusing out-of-bounds cities), joins the
    coordinate to the nearest census tract via `ST_DISTANCE`, runs BigQuery ML
    `ML.EXPLAIN_PREDICT` on `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`
    (`STRUCT(5 AS top_k_features)`), explains the 3 UVI pillars (CMS Medicare Health
    Access, NOAA CDD Energy Stress, OpenStreetMap 1km Building Density), and outputs
    concrete actions for Emergency Services, Hospital Staff Standby, and Electricity
    Grid Teams.

    Args:
        latitude: WGS84 latitude in decimal degrees (e.g. 32.755).
        longitude: WGS84 longitude in decimal degrees (e.g. -96.808).
        forecast_high_f: Regional forecast daytime high in °F (default 102.0).

    Returns:
        Dictionary containing `ML.EXPLAIN_PREDICT` `predicted_heat_offset_f` and
        `top_feature_attributions` from `heat_offset_model`, nearest tract UVI score,
        3-pillar drivers, demographics, and Tri-Agency operational recommendations.
    """
    client = _get_bq_client()

    # Step 1: Check bounding box across Dallas tracts scored by heat_offset_model
    bounds_sql = f"""
    SELECT
      'Dallas' AS city,
      COUNT(*) AS num_tracts,
      AVG(city_avg_temp_f) AS city_avg_temp_f,
      AVG(universal_vulnerability_index) AS city_avg_uvi,
      MIN(tract_lat) AS min_lat,
      MAX(tract_lat) AS max_lat,
      MIN(tract_lon) AS min_lon,
      MAX(tract_lon) AS max_lon
    FROM {FEATURES_TABLE}
    WHERE {CLEAN_ROW_FILTER}
    """
    bounds = list(client.query(bounds_sql).result())[0]

    margin = 0.08
    if not (
        (bounds.min_lat - margin) <= latitude <= (bounds.max_lat + margin)
        and (bounds.min_lon - margin) <= longitude <= (bounds.max_lon + margin)
    ):
        return {
            "status": "out_of_bounds",
            "requested_coordinates": {"latitude": latitude, "longitude": longitude},
            "loaded_city": bounds.city,
            "valid_bounding_box": {
                "min_lat": round(float(bounds.min_lat), 4),
                "max_lat": round(float(bounds.max_lat), 4),
                "min_lon": round(float(bounds.min_lon), 4),
                "max_lon": round(float(bounds.max_lon), 4),
            },
            "message": (
                f"Coordinates ({latitude:.4f}, {longitude:.4f}) fall outside the loaded "
                f"{bounds.city} coverage area for `heat_offset_model` "
                f"(lat [{bounds.min_lat:.4f}, {bounds.max_lat:.4f}], "
                f"lon [{bounds.min_lon:.4f}, {bounds.max_lon:.4f}]). "
                "The `heat_offset_model` is trained exclusively on Dallas, TX census tracts "
                "and refuses to extrapolate to out-of-bounds cities."
            ),
        }

    # Step 2: Query nearest tract and run ML.EXPLAIN_PREDICT on heat_offset_model
    point_sql = f"""
    WITH city_stats AS (
      SELECT
        AVG(city_avg_temp_f) AS city_avg_temp_f,
        AVG(universal_vulnerability_index) AS city_avg_uvi,
        AVG(median_age) AS city_avg_age,
        SAFE_DIVIDE(SUM(no_cars), NULLIF(SUM(households), 0)) AS city_no_vehicle_rate
      FROM {FEATURES_TABLE}
      WHERE {CLEAN_ROW_FILTER}
    ),
    ranked AS (
      SELECT
        f.*,
        'Dallas' AS city,
        c.city_avg_uvi,
        c.city_avg_age,
        c.city_no_vehicle_rate,
        ST_DISTANCE(
          ST_GEOGPOINT(@lon, @lat),
          ST_GEOGPOINT(f.tract_lon, f.tract_lat)
        ) / 1000.0 AS distance_to_tract_km,
        PERCENT_RANK() OVER (ORDER BY f.universal_vulnerability_index ASC) AS uvi_percentile
      FROM {FEATURES_TABLE} f
      CROSS JOIN city_stats c
      WHERE {CLEAN_ROW_FILTER}
    ),
    nearest_tract AS (
      SELECT *
      FROM ranked
      ORDER BY distance_to_tract_km ASC
      LIMIT 1
    )
    SELECT *
    FROM ML.EXPLAIN_PREDICT(
      MODEL {BQML_OFFSET_MODEL},
      TABLE nearest_tract,
      STRUCT(5 AS top_k_features)
    )
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("lat", "FLOAT64", float(latitude)),
            bigquery.ScalarQueryParameter("lon", "FLOAT64", float(longitude)),
        ]
    )
    row = list(client.query(point_sql, job_config=job_config).result())[0]

    bqml_offset_f = float(row.predicted_heat_offset_f or 0.0)
    observed_offset_f = (
        float(row.heat_offset_f) if row.heat_offset_f is not None else bqml_offset_f
    )
    offset_f = bqml_offset_f
    city_avg_temp_f = float(row.city_avg_temp_f or 105.32)
    observed_avg_temp = (
        float(row.avg_temp_f)
        if row.avg_temp_f is not None
        else round(city_avg_temp_f + bqml_offset_f, 2)
    )
    expected_local_temp_f = float(forecast_high_f) + offset_f
    uvi_score = float(row.universal_vulnerability_index)
    no_vehicle_rate = float(row.no_vehicle_rate or 0.0)
    no_cars = int(row.no_cars or 0)
    households = int(row.households or 0)
    avg_age = float(row.median_age or 0.0)
    total_pop = int(row.total_pop or 0)
    seniors_65_plus = int(row.seniors_65_plus or 0)
    senior_share = float(row.senior_share or 0.0)
    poverty_rate = float(row.poverty_rate or 0.0)
    building_age_years = float(row.building_age_years or 0.0)
    housing_density_sqkm = float(row.housing_density_sqkm or 0.0)

    offset_attributions = _format_bqml_attributions(row.top_feature_attributions)

    tri_plan = _build_tri_agency_action_plan(
        uvi_score=uvi_score,
        no_vehicle_rate=no_vehicle_rate,
        no_cars=no_cars,
        average_age=avg_age,
        heat_wave_risk_flag=str(row.heat_wave_risk_flag),
        urban_density_status=str(row.urban_density_status),
        greenery_status=str(row.greenery_status),
        nearby_parks=int(row.nearby_parks or 0),
        capacity_status=str(row.capacity_status),
        cooling_center_type=str(row.cooling_center_type),
        energy_grid_status=str(row.energy_grid_status),
        observed_avg_temp=observed_avg_temp,
    )

    uvi_pillar_drivers = [
        {
            "pillar": "Health Access (CMS Medicare Hospital Proxy & Carless Isolation)",
            "status": (
                f"UVI={uvi_score:.2f} (Top {(1.0 - float(row.uvi_percentile)) * 100:.0f}% risk in Dallas; "
                f"city avg {float(row.city_avg_uvi):.2f})"
            ),
            "detail": (
                f"{no_cars:,} carless households ({no_vehicle_rate * 100:.1f}% vs. city avg "
                f"{float(row.city_no_vehicle_rate) * 100:.1f}%), {seniors_65_plus:,} seniors 65+, "
                f"and median age {avg_age:.1f} with zero/limited nearby CMS Medicare hospitals."
            ),
        },
        {
            "pillar": "Urban Heat Island (OpenStreetMap 1km Building Density & Albedo)",
            "status": f"{row.urban_density_status} | {row.greenery_status}",
            "detail": (
                f"{int(row.nearby_parks)} nearby parks within 1km (`ST_DWithin`); "
                f"`heat_offset_model` predicted heat offset {bqml_offset_f:+.2f}°F "
                f"({'measured' if row.is_measured else 'BQML model-extrapolated'} tract, "
                f"effective temp {observed_avg_temp:.2f}°F vs. Dallas mean {city_avg_temp_f:.2f}°F)."
            ),
        },
        {
            "pillar": "Energy Stress (NOAA 2023 Cooling Degree Days & Grid Load)",
            "status": f"{row.energy_grid_status} | Heatwave Flag: {row.heat_wave_risk_flag}",
            "detail": (
                f"Local cooling center ({row.cooling_center_type}) is currently '{row.capacity_status}', "
                f"amplifying indoor AC reliance across {building_age_years:.0f}-year-old housing stock."
            ),
        },
    ]

    offset_attr_str = ", ".join(
        f"{a['feature']} ({a['attribution']:+.3f}°F)" for a in offset_attributions[:5]
    )

    return {
        "status": "ok",
        "source_model": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "bqml_models_used": [
            f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        ],
        "city": row.city,
        "requested_coordinates": {"latitude": latitude, "longitude": longitude},
        "nearest_tract": {
            "geo_id": row.geo_id,
            "is_measured_in_heat_watch": bool(row.is_measured),
            "tract_centroid": {
                "latitude": round(float(row.tract_lat), 5),
                "longitude": round(float(row.tract_lon), 5),
            },
            "distance_to_tract_km": round(float(row.distance_to_tract_km), 2),
        },
        "universal_vulnerability_index": round(uvi_score, 2),
        "risk_level": row.risk_level,
        "city_avg_uvi": round(float(row.city_avg_uvi), 2),
        "observed_avg_temp_f": round(observed_avg_temp, 2),
        "city_baseline_avg_temp_f": round(city_avg_temp_f, 2),
        "predicted_heat_offset_f": round(bqml_offset_f, 3),
        "bqml_predicted_heat_offset_f": round(bqml_offset_f, 3),
        "observed_heat_offset_f": round(observed_offset_f, 3) if row.is_measured else None,
        "forecast_high_f": round(float(forecast_high_f), 1),
        "expected_local_temp_f": round(expected_local_temp_f, 1),
        "bqml_expected_local_temp_f": round(expected_local_temp_f, 1),
        "heat_wave_risk_flag": row.heat_wave_risk_flag,
        "urban_density_status": row.urban_density_status,
        "greenery_status": row.greenery_status,
        "nearby_parks": int(row.nearby_parks or 0),
        "energy_grid_status": row.energy_grid_status,
        "cooling_center": {
            "type": row.cooling_center_type,
            "capacity_status": row.capacity_status,
        },
        "bqml_explain_predict": {
            "heat_offset_model": {
                "model_id": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
                "sql_function": "ML.EXPLAIN_PREDICT",
                "predicted_heat_offset_f": round(bqml_offset_f, 4),
                "observed_heat_offset_f": round(observed_offset_f, 4) if row.is_measured else None,
                "is_measured": bool(row.is_measured),
                "baseline_prediction_value": round(float(row.baseline_prediction_value or 0.0), 4),
                "prediction_value": round(float(row.prediction_value or 0.0), 4),
                "top_feature_attributions": offset_attributions,
            },
        },
        "top_feature_attributions": offset_attributions,
        "heat_offset_top_feature_attributions": offset_attributions,
        "uvi_pillar_drivers": uvi_pillar_drivers,
        "attribution_summary": (
            f"BQML `ML.EXPLAIN_PREDICT` (`{BQML_OFFSET_MODEL_NAME}`): "
            f"predicted_heat_offset={bqml_offset_f:+.3f}°F "
            f"(top feature attributions: {offset_attr_str}) | "
            f"UVI {uvi_score:.2f} ({row.risk_level}) | "
            f"Expected Local Temp {expected_local_temp_f:.1f}°F"
        ),
        "tract_demographics": {
            "total_pop": total_pop,
            "households": households,
            "no_cars": no_cars,
            "no_vehicle_rate_pct": round(no_vehicle_rate * 100.0, 1),
            "poverty_rate_pct": round(poverty_rate * 100.0, 1),
            "seniors_65_plus": seniors_65_plus,
            "senior_share_pct": round(senior_share * 100.0, 1),
            "average_age": round(avg_age, 1),
            "building_age_years": round(building_age_years, 1),
            "housing_density_sqkm": round(housing_density_sqkm, 1),
        },
        **tri_plan,
    }


def recommend_emergency_deployments(
    forecast_high_f: float = DANGEROUS_THRESHOLD_TEMP,
    num_units: int = 4,
    min_spacing_km: float = 1.2,
    priority_mode: str = "balanced",
) -> dict[str, Any]:
    """Recommend spatially de-clustered emergency deployments across Dallas using
    BigQuery ML `ML.EXPLAIN_PREDICT` on `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`.

    Delivers coordinated operational orders for all 3 response teams:
    1. Emergency Services Dispatch (Mobile Cooling Units, Hydration Trucks, Wellness Checks)
    2. Hospital & Clinical Staff Standby (CMS Medicare Hospital-Access Surge Alerts)
    3. Electricity & Power Grid Team (NOAA CDD & OSM 1km Density Transformer Watch)

    Args:
        forecast_high_f: Forecast daytime high temperature in °F (default 102.0).
        num_units: Number of mobile response units/teams to site (default 4, max 10).
        min_spacing_km: Minimum Haversine distance in km between deployed units (default 1.2).
        priority_mode: Ranking mode:
            - 'balanced' (default): Combines `universal_vulnerability_index` + `predicted_heat_offset_f` from `heat_offset_model`
            - 'vulnerability_heavy': Ranks primarily by `universal_vulnerability_index` & `no_cars`
            - 'heat_only': Ranks strictly by `predicted_heat_offset_f` from `heat_offset_model` (for equity contrast)

    Returns:
        Dictionary with spatially de-clustered tract deployments, BigQuery ML
        `ML.EXPLAIN_PREDICT` (`heat_offset_model`) feature attributions, and Tri-Agency dispatch plans.
    """
    client = _get_bq_client()
    num_units = max(1, min(int(num_units), 10))
    min_spacing_km = max(0.0, float(min_spacing_km))

    mode = (priority_mode or "balanced").strip().lower()
    if mode == "heat_only":
        order_clause = "predicted_heat_offset_f DESC, universal_vulnerability_index DESC"
    elif mode == "vulnerability_heavy":
        order_clause = "universal_vulnerability_index DESC, no_cars DESC, predicted_heat_offset_f DESC"
    else:
        mode = "balanced"
        order_clause = "combined_priority_score DESC, universal_vulnerability_index DESC"

    query = f"""
    WITH clean_features AS (
      SELECT *
      FROM {FEATURES_TABLE}
      WHERE {CLEAN_ROW_FILTER}
    ),
    explained AS (
      SELECT *
      FROM ML.EXPLAIN_PREDICT(
        MODEL {BQML_OFFSET_MODEL},
        TABLE clean_features,
        STRUCT(5 AS top_k_features)
      )
    )
    SELECT
      *,
      'Dallas' AS city,
      ROUND(
        COALESCE(avg_temp_f, city_avg_temp_f + predicted_heat_offset_f),
        2
      ) AS effective_temp_f,
      ROUND(
        universal_vulnerability_index
        + 25.0 * predicted_heat_offset_f
        + CASE WHEN heat_wave_risk_flag = 'Highly Likely' THEN 15.0 ELSE 0.0 END
        + CASE WHEN capacity_status = 'Over Capacity' THEN 10.0 ELSE 0.0 END,
        2
      ) AS combined_priority_score
    FROM explained
    ORDER BY {order_clause}
    LIMIT 60
    """
    candidates = list(client.query(query).result())

    selected: list[dict[str, Any]] = []
    skipped_adjacent: list[str] = []

    for row in candidates:
        if len(selected) >= num_units:
            break
        lat = float(row.tract_lat)
        lon = float(row.tract_lon)

        too_close = False
        for existing in selected:
            dist = _haversine_km(
                lat,
                lon,
                existing["coordinates"]["latitude"],
                existing["coordinates"]["longitude"],
            )
            if dist < min_spacing_km:
                too_close = True
                skipped_adjacent.append(
                    f"{row.geo_id} ({dist:.2f} km from {existing['geo_id']})"
                )
                break
        if too_close:
            continue

        offset_f = float(row.predicted_heat_offset_f or 0.0)
        uvi_score = float(row.universal_vulnerability_index)
        no_vehicle_rate = float(row.no_vehicle_rate or 0.0)
        no_cars = int(row.no_cars or 0)
        avg_age = float(row.median_age or 0.0)
        effective_temp_f = float(row.effective_temp_f or 105.32)
        bqml_attributions = _format_bqml_attributions(row.top_feature_attributions)

        tri_plan = _build_tri_agency_action_plan(
            uvi_score=uvi_score,
            no_vehicle_rate=no_vehicle_rate,
            no_cars=no_cars,
            average_age=avg_age,
            heat_wave_risk_flag=str(row.heat_wave_risk_flag),
            urban_density_status=str(row.urban_density_status),
            greenery_status=str(row.greenery_status),
            nearby_parks=int(row.nearby_parks or 0),
            capacity_status=str(row.capacity_status),
            cooling_center_type=str(row.cooling_center_type),
            energy_grid_status=str(row.energy_grid_status),
            observed_avg_temp=effective_temp_f,
        )

        selected.append(
            {
                "unit_number": len(selected) + 1,
                "geo_id": row.geo_id,
                "is_measured_in_heat_watch": bool(row.is_measured),
                "coordinates": {
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                },
                "universal_vulnerability_index": round(uvi_score, 2),
                "risk_level": row.risk_level,
                "combined_priority_score": round(float(row.combined_priority_score), 2),
                "observed_avg_temp_f": round(effective_temp_f, 2),
                "predicted_heat_offset_f": round(offset_f, 3),
                "expected_local_temp_f": round(float(forecast_high_f) + offset_f, 1),
                "heat_wave_risk_flag": row.heat_wave_risk_flag,
                "urban_density_status": row.urban_density_status,
                "greenery_status": row.greenery_status,
                "nearby_parks": int(row.nearby_parks or 0),
                "energy_grid_status": row.energy_grid_status,
                "cooling_center": {
                    "type": row.cooling_center_type,
                    "capacity_status": row.capacity_status,
                },
                "bqml_explain_predict": {
                    "model_id": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
                    "sql_function": "ML.EXPLAIN_PREDICT",
                    "predicted_heat_offset_f": round(offset_f, 4),
                    "baseline_prediction_value": round(float(row.baseline_prediction_value or 0.0), 4),
                    "prediction_value": round(float(row.prediction_value or 0.0), 4),
                    "top_feature_attributions": bqml_attributions,
                },
                "top_feature_attributions": bqml_attributions,
                "demographics": {
                    "total_pop": int(row.total_pop or 0),
                    "households": int(row.households or 0),
                    "no_cars": no_cars,
                    "no_vehicle_rate_pct": round(no_vehicle_rate * 100.0, 1),
                    "poverty_rate_pct": round(float(row.poverty_rate or 0.0) * 100.0, 1),
                    "seniors_65_plus": int(row.seniors_65_plus or 0),
                    "average_age": round(avg_age, 1),
                    "building_age_years": round(float(row.building_age_years or 0.0), 1),
                },
                **tri_plan,
            }
        )

    return {
        "status": "ok",
        "source_model": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "bqml_model_used": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "city": "Dallas",
        "forecast_high_f": round(float(forecast_high_f), 1),
        "priority_mode": mode,
        "min_spacing_km": round(min_spacing_km, 2),
        "units_requested": num_units,
        "units_deployed": len(selected),
        "de_clustered_adjacent_tracts_skipped": skipped_adjacent[:8],
        "deployments": selected,
    }


def compare_deployment_sites(
    lat_a: float,
    lon_a: float,
    label_a: str,
    lat_b: float,
    lon_b: float,
    label_b: str,
    forecast_high_f: float = DANGEROUS_THRESHOLD_TEMP,
) -> dict[str, Any]:
    """Compare two candidate deployment locations side-by-side using BigQuery ML
    `ML.EXPLAIN_PREDICT` on `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`.

    Evaluates the tradeoff between `heat_offset_model` predicted thermal offset
    (`predicted_heat_offset_f` & `top_feature_attributions`), Universal Vulnerability
    Index (CMS Medicare hospital access + carless/elderly isolation), OpenStreetMap 1km
    building density heat trap, and NOAA Cooling Degree Days power grid stress.

    Args:
        lat_a: Latitude of Site A.
        lon_a: Longitude of Site A.
        label_a: Human-readable name for Site A (e.g. 'Downtown Commercial Core').
        lat_b: Latitude of Site B.
        lon_b: Longitude of Site B.
        label_b: Human-readable name for Site B (e.g. 'South Dallas Residential').
        forecast_high_f: Forecast daytime high in °F (default 102.0).

    Returns:
        Side-by-side comparison and Tri-Agency operational prioritization verdict.
    """
    site_a = predict_point_heat_offset(lat_a, lon_a, forecast_high_f=forecast_high_f)
    site_b = predict_point_heat_offset(lat_b, lon_b, forecast_high_f=forecast_high_f)

    if site_a.get("status") != "ok" or site_b.get("status") != "ok":
        return {
            "status": "error",
            "message": "One or both locations fall outside the Dallas coverage area in `heat_offset_model`.",
            "site_a": {"label": label_a, "result": site_a},
            "site_b": {"label": label_b, "result": site_b},
        }

    uvi_diff = round(
        site_a["universal_vulnerability_index"] - site_b["universal_vulnerability_index"],
        2,
    )
    offset_diff_f = round(
        site_a["predicted_heat_offset_f"] - site_b["predicted_heat_offset_f"],
        3,
    )
    heat_diff_f = round(
        site_a["observed_avg_temp_f"] - site_b["observed_avg_temp_f"],
        2,
    )
    no_veh_diff_pct = round(
        site_a["tract_demographics"]["no_vehicle_rate_pct"]
        - site_b["tract_demographics"]["no_vehicle_rate_pct"],
        1,
    )
    age_diff = round(
        site_a["tract_demographics"]["average_age"]
        - site_b["tract_demographics"]["average_age"],
        1,
    )

    score_a = site_a["universal_vulnerability_index"] + 25.0 * site_a["predicted_heat_offset_f"]
    score_b = site_b["universal_vulnerability_index"] + 25.0 * site_b["predicted_heat_offset_f"]
    if score_a >= score_b:
        priority_winner = label_a
        winner_site = site_a
        runner_up_label = label_b
        runner_up_site = site_b
    else:
        priority_winner = label_b
        winner_site = site_b
        runner_up_label = label_a
        runner_up_site = site_a

    return {
        "status": "ok",
        "source_model": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "forecast_high_f": forecast_high_f,
        "site_a": {"label": label_a, **site_a},
        "site_b": {"label": label_b, **site_b},
        "deltas_a_minus_b": {
            "universal_vulnerability_index_diff": uvi_diff,
            "predicted_heat_offset_diff_f": offset_diff_f,
            "observed_avg_temp_diff_f": heat_diff_f,
            "no_vehicle_rate_diff_pct_pts": no_veh_diff_pct,
            "average_age_diff_years": age_diff,
        },
        "recommended_priority_site": priority_winner,
        "tradeoff_verdict": (
            f"Prioritize {priority_winner} (Tract {winner_site['nearest_tract']['geo_id']}, "
            f"UVI={winner_site['universal_vulnerability_index']:.2f}, "
            f"`heat_offset_model` offset={winner_site['predicted_heat_offset_f']:+.2f}°F, "
            f"no-vehicle rate={winner_site['tract_demographics']['no_vehicle_rate_pct']:.1f}%, "
            f"avg age={winner_site['tract_demographics']['average_age']:.1f}) over "
            f"{runner_up_label} (Tract {runner_up_site['nearest_tract']['geo_id']}, "
            f"UVI={runner_up_site['universal_vulnerability_index']:.2f}, "
            f"`heat_offset_model` offset={runner_up_site['predicted_heat_offset_f']:+.2f}°F, "
            f"no-vehicle rate={runner_up_site['tract_demographics']['no_vehicle_rate_pct']:.1f}%). "
            f"Recommended unit at {priority_winner}: {winner_site['recommended_intervention']}."
        ),
    }


def get_city_uvi_and_infrastructure_summary(
    threshold_f: float = DANGEROUS_THRESHOLD_TEMP,
) -> dict[str, Any]:
    """Summarize city-wide BigQuery ML `heat_offset_model` (`ML.EXPLAIN_PREDICT`) thermal
    offsets, Universal Vulnerability Index (UVI), NOAA 2023 Cooling Degree Days (CDD)
    energy stress, OpenStreetMap 1km urban density heat traps, and Public Library cooling
    center capacity across all 549 clean Dallas census tracts.

    Args:
        threshold_f: Effective temperature threshold in °F to count extreme heat tracts
            (default 102.0°F).

    Returns:
        Dictionary summarizing `heat_offset_model` predictions across measured and
        unmeasured tracts, UVI exposure, heatwave risk flag breakdown, OSM building
        density/greenery breakdown, NOAA grid status, and cooling capacity.
    """
    client = _get_bq_client()
    summary_sql = f"""
    WITH clean_features AS (
      SELECT *
      FROM {FEATURES_TABLE}
      WHERE {CLEAN_ROW_FILTER}
    ),
    explained AS (
      SELECT
        *,
        COALESCE(avg_temp_f, city_avg_temp_f + predicted_heat_offset_f) AS effective_temp_f
      FROM ML.EXPLAIN_PREDICT(
        MODEL {BQML_OFFSET_MODEL},
        TABLE clean_features,
        STRUCT(5 AS top_k_features)
      )
    )
    SELECT
      'Dallas' AS city,
      COUNT(*) AS total_valid_tracts,
      COUNTIF(is_measured) AS measured_heat_watch_tracts,
      COUNTIF(NOT is_measured) AS bqml_extrapolated_unmeasured_tracts,
      ROUND(MIN(predicted_heat_offset_f), 3) AS min_predicted_offset_f,
      ROUND(AVG(predicted_heat_offset_f), 3) AS mean_predicted_offset_f,
      ROUND(MAX(predicted_heat_offset_f), 3) AS max_predicted_offset_f,
      ROUND(MIN(effective_temp_f), 2) AS min_effective_temp_f,
      ROUND(AVG(effective_temp_f), 2) AS mean_effective_temp_f,
      ROUND(MAX(effective_temp_f), 2) AS max_effective_temp_f,
      COUNTIF(effective_temp_f >= @threshold_f) AS tracts_above_threshold,
      COUNTIF(heat_wave_risk_flag = 'Highly Likely') AS highly_likely_heatwave_tracts,
      COUNTIF(heat_wave_risk_flag = 'Moderate') AS moderate_heatwave_tracts,
      COUNTIF(greenery_status = 'Urban Heat Island - Higher Risk') AS uhi_higher_risk_tracts,
      COUNTIF(greenery_status != 'Urban Heat Island - Higher Risk') AS green_area_tracts,
      COUNTIF(urban_density_status = 'Moderate Density') AS moderate_density_tracts,
      COUNTIF(urban_density_status = 'Low Density / Open Space') AS low_density_tracts,
      COUNTIF(capacity_status = 'Over Capacity') AS over_capacity_cooling_tracts,
      COUNTIF(capacity_status != 'Over Capacity') AS under_capacity_cooling_tracts,
      ANY_VALUE(energy_grid_status) AS energy_grid_status_baseline,
      ROUND(AVG(universal_vulnerability_index), 2) AS mean_uvi,
      ROUND(MAX(universal_vulnerability_index), 2) AS max_uvi
    FROM explained
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("threshold_f", "FLOAT64", float(threshold_f)),
        ]
    )
    row = list(client.query(summary_sql, job_config=job_config).result())[0]

    hotspots_sql = f"""
    WITH clean_features AS (
      SELECT *
      FROM {FEATURES_TABLE}
      WHERE {CLEAN_ROW_FILTER}
    ),
    explained AS (
      SELECT *
      FROM ML.EXPLAIN_PREDICT(
        MODEL {BQML_OFFSET_MODEL},
        TABLE clean_features,
        STRUCT(5 AS top_k_features)
      )
    )
    SELECT
      geo_id,
      is_measured,
      ROUND(predicted_heat_offset_f, 3) AS predicted_heat_offset_f,
      ROUND(COALESCE(avg_temp_f, city_avg_temp_f + predicted_heat_offset_f), 2) AS effective_temp_f,
      ROUND(universal_vulnerability_index, 2) AS universal_vulnerability_index,
      risk_level,
      heat_wave_risk_flag,
      urban_density_status,
      greenery_status,
      nearby_parks,
      capacity_status,
      energy_grid_status,
      CAST(no_cars AS INT64) AS no_cars,
      ROUND(median_age, 1) AS average_age,
      top_feature_attributions
    FROM explained
    ORDER BY
      universal_vulnerability_index + 25.0 * predicted_heat_offset_f DESC
    LIMIT 5
    """
    hotspots = []
    for r in client.query(hotspots_sql).result():
        d = dict(r.items())
        d["top_feature_attributions"] = _format_bqml_attributions(d.get("top_feature_attributions"))
        hotspots.append(d)

    return {
        "status": "ok",
        "source_model": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "city": row.city,
        "threshold_f": float(threshold_f),
        "total_valid_tracts": int(row.total_valid_tracts),
        "measured_heat_watch_tracts": int(row.measured_heat_watch_tracts),
        "bqml_extrapolated_unmeasured_tracts": int(row.bqml_extrapolated_unmeasured_tracts),
        "heat_offset_model_distribution_f": {
            "min_predicted_offset_f": float(row.min_predicted_offset_f),
            "mean_predicted_offset_f": float(row.mean_predicted_offset_f),
            "max_predicted_offset_f": float(row.max_predicted_offset_f),
        },
        "temperature_distribution_f": {
            "min_observed_temp_f": float(row.min_effective_temp_f),
            "mean_observed_temp_f": float(row.mean_effective_temp_f),
            "max_observed_temp_f": float(row.max_effective_temp_f),
            "tracts_at_or_above_threshold": int(row.tracts_above_threshold),
        },
        "noaa_heatwave_and_energy_stress": {
            "highly_likely_heatwave_tracts": int(row.highly_likely_heatwave_tracts),
            "moderate_heatwave_tracts": int(row.moderate_heatwave_tracts),
            "energy_grid_status": row.energy_grid_status_baseline,
            "note": (
                "Derived from BigQuery ML `heat_offset_model` (`ML.EXPLAIN_PREDICT`) "
                "and NOAA 2023 Cooling Degree Days (CDD)."
            ),
        },
        "osm_1km_urban_heat_island_breakdown": {
            "uhi_higher_risk_tracts": int(row.uhi_higher_risk_tracts),
            "green_area_tracts": int(row.green_area_tracts),
            "moderate_building_density_tracts": int(row.moderate_density_tracts),
            "low_density_open_space_tracts": int(row.low_density_tracts),
        },
        "cooling_center_capacity_summary": {
            "over_capacity_tracts": int(row.over_capacity_cooling_tracts),
            "under_capacity_tracts": int(row.under_capacity_cooling_tracts),
            "over_capacity_share_pct": round(
                100.0 * int(row.over_capacity_cooling_tracts) / max(1, int(row.total_valid_tracts)),
                1,
            ),
        },
        "uvi_summary": {
            "mean_uvi": float(row.mean_uvi),
            "max_uvi": float(row.max_uvi),
        },
        "top_thermal_and_heatwave_hotspots": hotspots,
    }


def get_station_daily_temp_history(
    threshold_f: float = 100.0,
    station_filter: str = "",
    start_year: int = 2019,
    end_year: int = 2025,
) -> dict[str, Any]:
    """Query 7-year historical weather station daily maximum temperatures, threshold
    exceedance frequencies, annual trends (2019–2025), and per-station breakdowns
    using EXCLUSIVELY `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps`.

    Does NOT use web search or any other table.

    Args:
        threshold_f: Daily maximum temperature threshold in °F to count extreme heat days
            (default 100.0°F; common values: 95.0, 100.0, 102.0, 105.0).
        station_filter: Optional station ID or station name substring (e.g. 'DAL-FTW',
            'DALLAS FAA', 'FERRIS', 'ARLINGTON', 'USW00003927'). Leave empty for all 16 stations.
        start_year: Start year inclusive (default 2019; dataset spans 2019–2025).
        end_year: End year inclusive (default 2025; dataset spans 2019–2025).

    Returns:
        Dictionary with 7-year overall station summary, year-by-year frequency and trend
        table (2019–2025), early-vs-recent trend comparison, per-station breakdown across
        weather stations, and top 10 hottest station-days on record.
    """
    client = _get_bq_client()
    start_year = max(2019, min(int(start_year), 2025))
    end_year = max(start_year, min(int(end_year), 2025))
    station_filter_clean = (station_filter or "").strip().upper()

    where_station = ""
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("threshold_f", "FLOAT64", float(threshold_f)),
        bigquery.ScalarQueryParameter("start_year", "INT64", start_year),
        bigquery.ScalarQueryParameter("end_year", "INT64", end_year),
    ]
    if station_filter_clean:
        where_station = (
            "AND (UPPER(station_id) LIKE @station_pat OR UPPER(station_name) LIKE @station_pat)"
        )
        params.append(
            bigquery.ScalarQueryParameter("station_pat", "STRING", f"%{station_filter_clean}%")
        )

    job_config = bigquery.QueryJobConfig(query_parameters=params)

    overall_sql = f"""
    SELECT
      COUNT(*) AS total_station_days,
      COUNT(DISTINCT station_id) AS distinct_stations,
      CAST(MIN(observation_date) AS STRING) AS min_observation_date,
      CAST(MAX(observation_date) AS STRING) AS max_observation_date,
      COUNT(DISTINCT EXTRACT(YEAR FROM observation_date)) AS years_covered,
      ROUND(MIN(max_temp_f), 1) AS min_daily_max_f,
      ROUND(AVG(max_temp_f), 2) AS mean_daily_max_f,
      ROUND(MAX(max_temp_f), 1) AS peak_daily_max_f,
      COUNTIF(max_temp_f >= @threshold_f) AS station_days_at_or_above_threshold,
      COUNTIF(max_temp_f >= 95.0) AS station_days_ge_95f,
      COUNTIF(max_temp_f >= 100.0) AS station_days_ge_100f,
      COUNTIF(max_temp_f >= 102.0) AS station_days_ge_102f,
      COUNTIF(max_temp_f >= 105.0) AS station_days_ge_105f
    FROM {STATION_DAILY_TEMPS_TABLE}
    WHERE EXTRACT(YEAR FROM observation_date) BETWEEN @start_year AND @end_year
      {where_station}
    """
    overall_row = list(client.query(overall_sql, job_config=job_config).result())[0]

    annual_sql = f"""
    SELECT
      EXTRACT(YEAR FROM observation_date) AS year,
      COUNT(DISTINCT station_id) AS active_stations,
      COUNT(*) AS total_station_days,
      ROUND(AVG(max_temp_f), 2) AS avg_daily_max_f,
      ROUND(MAX(max_temp_f), 1) AS peak_max_temp_f,
      COUNTIF(max_temp_f >= @threshold_f) AS station_days_ge_threshold,
      ROUND(
        SAFE_DIVIDE(COUNTIF(max_temp_f >= @threshold_f), COUNT(DISTINCT station_id)),
        1
      ) AS avg_days_ge_threshold_per_station,
      ROUND(
        SAFE_DIVIDE(COUNTIF(max_temp_f >= 95.0), COUNT(DISTINCT station_id)),
        1
      ) AS avg_days_ge_95f_per_station,
      ROUND(
        SAFE_DIVIDE(COUNTIF(max_temp_f >= 100.0), COUNT(DISTINCT station_id)),
        1
      ) AS avg_days_ge_100f_per_station,
      ROUND(
        SAFE_DIVIDE(COUNTIF(max_temp_f >= 102.0), COUNT(DISTINCT station_id)),
        1
      ) AS avg_days_ge_102f_per_station,
      ROUND(
        SAFE_DIVIDE(COUNTIF(max_temp_f >= 105.0), COUNT(DISTINCT station_id)),
        1
      ) AS avg_days_ge_105f_per_station
    FROM {STATION_DAILY_TEMPS_TABLE}
    WHERE EXTRACT(YEAR FROM observation_date) BETWEEN @start_year AND @end_year
      {where_station}
    GROUP BY year
    ORDER BY year ASC
    """
    annual_trends = [dict(r.items()) for r in client.query(annual_sql, job_config=job_config).result()]

    station_sql = f"""
    SELECT
      station_id,
      station_name,
      ROUND(AVG(latitude), 5) AS latitude,
      ROUND(AVG(longitude), 5) AS longitude,
      COUNT(*) AS total_days_recorded,
      COUNT(DISTINCT EXTRACT(YEAR FROM observation_date)) AS years_active,
      CAST(MIN(observation_date) AS STRING) AS first_observation_date,
      CAST(MAX(observation_date) AS STRING) AS last_observation_date,
      ROUND(MIN(max_temp_f), 1) AS min_max_temp_f,
      ROUND(AVG(max_temp_f), 2) AS avg_daily_max_f,
      ROUND(MAX(max_temp_f), 1) AS peak_max_temp_f,
      COUNTIF(max_temp_f >= @threshold_f) AS total_days_ge_threshold,
      ROUND(
        SAFE_DIVIDE(COUNTIF(max_temp_f >= @threshold_f), COUNT(*) / 365.25),
        1
      ) AS annualized_days_ge_threshold_per_year,
      COUNTIF(max_temp_f >= 95.0) AS total_days_ge_95f,
      COUNTIF(max_temp_f >= 100.0) AS total_days_ge_100f,
      COUNTIF(max_temp_f >= 102.0) AS total_days_ge_102f,
      COUNTIF(max_temp_f >= 105.0) AS total_days_ge_105f
    FROM {STATION_DAILY_TEMPS_TABLE}
    WHERE EXTRACT(YEAR FROM observation_date) BETWEEN @start_year AND @end_year
      {where_station}
    GROUP BY station_id, station_name
    ORDER BY total_days_ge_threshold DESC, peak_max_temp_f DESC
    """
    stations = [dict(r.items()) for r in client.query(station_sql, job_config=job_config).result()]

    hottest_days_sql = f"""
    SELECT
      CAST(observation_date AS STRING) AS observation_date,
      station_id,
      station_name,
      ROUND(max_temp_f, 1) AS max_temp_f
    FROM {STATION_DAILY_TEMPS_TABLE}
    WHERE EXTRACT(YEAR FROM observation_date) BETWEEN @start_year AND @end_year
      {where_station}
    ORDER BY max_temp_f DESC, observation_date DESC
    LIMIT 10
    """
    hottest_days = [
        dict(r.items()) for r in client.query(hottest_days_sql, job_config=job_config).result()
    ]

    # Compute early (2019-2021) vs recent (2022-2025) trend comparison from annual_trends
    early_years = [y for y in annual_trends if 2019 <= int(y["year"]) <= 2021]
    recent_years = [y for y in annual_trends if 2022 <= int(y["year"]) <= 2025]
    early_avg_days = (
        round(
            sum(float(y["avg_days_ge_threshold_per_station"] or 0.0) for y in early_years)
            / len(early_years),
            1,
        )
        if early_years
        else None
    )
    recent_avg_days = (
        round(
            sum(float(y["avg_days_ge_threshold_per_station"] or 0.0) for y in recent_years)
            / len(recent_years),
            1,
        )
        if recent_years
        else None
    )

    return {
        "status": "ok",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{STATION_DAILY_TEMPS_TABLE_NAME}",
        "strict_data_policy": (
            "100% grounded in `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps` only. "
            "No external web search or invented data."
        ),
        "threshold_f": float(threshold_f),
        "station_filter": station_filter_clean or "ALL_STATIONS",
        "year_range": {"start_year": start_year, "end_year": end_year},
        "overall_7yr_summary": dict(overall_row.items()),
        "period_trend_comparison": {
            "early_period_2019_2021_avg_days_ge_threshold_per_station": early_avg_days,
            "recent_period_2022_2025_avg_days_ge_threshold_per_station": recent_avg_days,
            "peak_heat_years": "2022 and 2023 recorded the highest peak temperature (111.9°F) and highest frequency of >=100°F days per station (40.6 days in 2022; 43.7 days in 2023).",
        },
        "annual_breakdown_by_year": annual_trends,
        "station_breakdown": stations,
        "top_10_hottest_station_days": hottest_days,
    }


def get_historic_heat_observations(
    threshold_f: float = 105.0,
    top_n: int = 10,
    min_lat: float | None = None,
    max_lat: float | None = None,
    min_lon: float | None = None,
    max_lon: float | None = None,
) -> dict[str, Any]:
    """Query historic street-level Heat Watch traverse observations in Dallas using
    EXCLUSIVELY `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations`.

    Does NOT use web search or any other table. Explicitly reports city-wide campaign
    statistics, temperature distribution (`temp_f`), heat offset vs. the Dallas campaign
    mean (`105.32°F`), hottest and coolest traverse coordinates, and the exact NULL
    status of `observed_at`, `heat_index_f`, and `humidity_pct`.

    Args:
        threshold_f: Temperature threshold in °F to count extreme street-level readings
            (default 105.0°F).
        top_n: Number of hottest and coolest street-level points to return (default 10, max 25).
        min_lat: Optional minimum latitude bounding-box filter.
        max_lat: Optional maximum latitude bounding-box filter.
        min_lon: Optional minimum longitude bounding-box filter.
        max_lon: Optional maximum longitude bounding-box filter.

    Returns:
        Dictionary with campaign summary, heat offset range, threshold counts, null column
        audit, and top N hottest/coolest street-level readings.
    """
    client = _get_bq_client()
    top_n = max(1, min(int(top_n), 25))

    bbox_clauses: list[str] = []
    params: list[bigquery.ScalarQueryParameter] = [
        bigquery.ScalarQueryParameter("threshold_f", "FLOAT64", float(threshold_f)),
        bigquery.ScalarQueryParameter("top_n", "INT64", top_n),
    ]
    if min_lat is not None:
        bbox_clauses.append("latitude >= @min_lat")
        params.append(bigquery.ScalarQueryParameter("min_lat", "FLOAT64", float(min_lat)))
    if max_lat is not None:
        bbox_clauses.append("latitude <= @max_lat")
        params.append(bigquery.ScalarQueryParameter("max_lat", "FLOAT64", float(max_lat)))
    if min_lon is not None:
        bbox_clauses.append("longitude >= @min_lon")
        params.append(bigquery.ScalarQueryParameter("min_lon", "FLOAT64", float(min_lon)))
    if max_lon is not None:
        bbox_clauses.append("longitude <= @max_lon")
        params.append(bigquery.ScalarQueryParameter("max_lon", "FLOAT64", float(max_lon)))

    where_bbox = ("WHERE " + " AND ".join(bbox_clauses)) if bbox_clauses else ""
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    summary_sql = f"""
    WITH city_baseline AS (
      SELECT AVG(temp_f) AS city_mean_temp_f
      FROM {HEAT_OBSERVATIONS_TABLE}
    )
    SELECT
      ANY_VALUE(h.city) AS city,
      COUNT(*) AS total_readings,
      ROUND(MIN(h.latitude), 5) AS min_latitude,
      ROUND(MAX(h.latitude), 5) AS max_latitude,
      ROUND(MIN(h.longitude), 5) AS min_longitude,
      ROUND(MAX(h.longitude), 5) AS max_longitude,
      ROUND(MIN(h.temp_f), 2) AS min_temp_f,
      ROUND(AVG(h.temp_f), 2) AS mean_temp_f,
      ROUND(ANY_VALUE(c.city_mean_temp_f), 2) AS citywide_campaign_mean_temp_f,
      ROUND(STDDEV(h.temp_f), 2) AS stddev_temp_f,
      ROUND(APPROX_QUANTILES(h.temp_f, 100)[OFFSET(50)], 2) AS median_temp_f,
      ROUND(APPROX_QUANTILES(h.temp_f, 100)[OFFSET(90)], 2) AS p90_temp_f,
      ROUND(APPROX_QUANTILES(h.temp_f, 100)[OFFSET(95)], 2) AS p95_temp_f,
      ROUND(MAX(h.temp_f), 2) AS max_temp_f,
      ROUND(MIN(h.temp_f - c.city_mean_temp_f), 2) AS min_heat_offset_vs_city_mean_f,
      ROUND(MAX(h.temp_f - c.city_mean_temp_f), 2) AS max_heat_offset_vs_city_mean_f,
      COUNTIF(h.temp_f >= @threshold_f) AS readings_at_or_above_threshold,
      ROUND(SAFE_DIVIDE(COUNTIF(h.temp_f >= @threshold_f), COUNT(*)) * 100.0, 1) AS pct_at_or_above_threshold,
      COUNTIF(h.temp_f >= 105.0) AS readings_ge_105f,
      COUNTIF(h.temp_f >= 106.0) AS readings_ge_106f,
      COUNTIF(h.temp_f >= 107.0) AS readings_ge_107f,
      COUNTIF(h.temp_f >= 108.0) AS readings_ge_108f,
      COUNTIF(h.temp_f >= 109.0) AS readings_ge_109f,
      COUNTIF(h.observed_at IS NOT NULL) AS non_null_observed_at_count,
      COUNTIF(h.heat_index_f IS NOT NULL) AS non_null_heat_index_f_count,
      COUNTIF(h.humidity_pct IS NOT NULL) AS non_null_humidity_pct_count
    FROM {HEAT_OBSERVATIONS_TABLE} h
    CROSS JOIN city_baseline c
    {where_bbox}
    """
    summary_row = list(client.query(summary_sql, job_config=job_config).result())[0]

    hottest_sql = f"""
    WITH city_baseline AS (
      SELECT AVG(temp_f) AS city_mean_temp_f
      FROM {HEAT_OBSERVATIONS_TABLE}
    )
    SELECT
      h.city,
      ROUND(h.latitude, 5) AS latitude,
      ROUND(h.longitude, 5) AS longitude,
      ROUND(h.temp_f, 2) AS temp_f,
      ROUND(h.temp_f - c.city_mean_temp_f, 2) AS heat_offset_vs_city_mean_f
    FROM {HEAT_OBSERVATIONS_TABLE} h
    CROSS JOIN city_baseline c
    {where_bbox}
    ORDER BY h.temp_f DESC, h.latitude ASC
    LIMIT @top_n
    """
    hottest_points = [
        dict(r.items()) for r in client.query(hottest_sql, job_config=job_config).result()
    ]

    coolest_sql = f"""
    WITH city_baseline AS (
      SELECT AVG(temp_f) AS city_mean_temp_f
      FROM {HEAT_OBSERVATIONS_TABLE}
    )
    SELECT
      h.city,
      ROUND(h.latitude, 5) AS latitude,
      ROUND(h.longitude, 5) AS longitude,
      ROUND(h.temp_f, 2) AS temp_f,
      ROUND(h.temp_f - c.city_mean_temp_f, 2) AS heat_offset_vs_city_mean_f
    FROM {HEAT_OBSERVATIONS_TABLE} h
    CROSS JOIN city_baseline c
    {where_bbox}
    ORDER BY h.temp_f ASC, h.latitude ASC
    LIMIT @top_n
    """
    coolest_points = [
        dict(r.items()) for r in client.query(coolest_sql, job_config=job_config).result()
    ]

    return {
        "status": "ok",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{HEAT_OBSERVATIONS_TABLE_NAME}",
        "strict_data_policy": (
            "100% grounded in `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations` only. "
            "No external web search or invented data."
        ),
        "threshold_f": float(threshold_f),
        "campaign_summary": dict(summary_row.items()),
        "null_columns_notice": (
            "In `heat_observations` for Dallas, `observed_at`, `heat_index_f`, and `humidity_pct` "
            "have 0 non-null rows (all 19,491 readings sit inside the same single-day Heat Watch "
            "traverse campaign window without individual clock timestamps). Do not invent timestamps, "
            "humidity, or heat index values."
        ),
        "top_hottest_traverse_points": hottest_points,
        "top_coolest_traverse_points": coolest_points,
    }


def get_historic_location_profile(
    latitude: float,
    longitude: float,
    radius_km: float = 1.5,
    threshold_f: float = 100.0,
) -> dict[str, Any]:
    """Retrieve the complete historic temperature profile for a specific coordinate
    (`latitude`, `longitude`) using EXCLUSIVELY the 2 historic BigQuery tables:
      1. `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations` (street-level readings
         within `radius_km` plus nearest point reading and heat offset vs. city mean)
      2. `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps` (nearest weather
         station and its 7-year daily temperature frequency and year-by-year trend 2019–2025).

    Args:
        latitude: WGS84 latitude in decimal degrees (e.g. 32.7767).
        longitude: WGS84 longitude in decimal degrees (e.g. -96.7970).
        radius_km: Search radius in km around the coordinate for street-level
            `heat_observations` readings (default 1.5 km).
        threshold_f: Daily maximum temperature threshold in °F for 7-year station
            frequency analysis (default 100.0°F).

    Returns:
        Dictionary with local `heat_observations` readings/offsets and nearest weather
        station 7-year history from `station_daily_temps`.
    """
    client = _get_bq_client()
    radius_km = max(0.1, min(float(radius_km), 20.0))

    params = [
        bigquery.ScalarQueryParameter("lat", "FLOAT64", float(latitude)),
        bigquery.ScalarQueryParameter("lon", "FLOAT64", float(longitude)),
        bigquery.ScalarQueryParameter("radius_m", "FLOAT64", float(radius_km) * 1000.0),
        bigquery.ScalarQueryParameter("threshold_f", "FLOAT64", float(threshold_f)),
    ]
    job_config = bigquery.QueryJobConfig(query_parameters=params)

    # 1. Street-level heat_observations within radius_km + nearest observation
    ho_radius_sql = f"""
    WITH city_baseline AS (
      SELECT AVG(temp_f) AS city_mean_temp_f
      FROM {HEAT_OBSERVATIONS_TABLE}
    ),
    with_dist AS (
      SELECT
        h.city,
        h.latitude,
        h.longitude,
        h.temp_f,
        h.temp_f - c.city_mean_temp_f AS heat_offset_f,
        c.city_mean_temp_f,
        ST_DISTANCE(
          ST_GEOGPOINT(@lon, @lat),
          ST_GEOGPOINT(h.longitude, h.latitude)
        ) / 1000.0 AS distance_km
      FROM {HEAT_OBSERVATIONS_TABLE} h
      CROSS JOIN city_baseline c
    )
    SELECT
      COUNTIF(distance_km <= @radius_m / 1000.0) AS readings_within_radius,
      ROUND(MIN(IF(distance_km <= @radius_m / 1000.0, temp_f, NULL)), 2) AS radius_min_temp_f,
      ROUND(AVG(IF(distance_km <= @radius_m / 1000.0, temp_f, NULL)), 2) AS radius_mean_temp_f,
      ROUND(MAX(IF(distance_km <= @radius_m / 1000.0, temp_f, NULL)), 2) AS radius_max_temp_f,
      ROUND(AVG(IF(distance_km <= @radius_m / 1000.0, heat_offset_f, NULL)), 2) AS radius_mean_offset_f,
      ROUND(MAX(IF(distance_km <= @radius_m / 1000.0, heat_offset_f, NULL)), 2) AS radius_max_offset_f,
      ROUND(ANY_VALUE(city_mean_temp_f), 2) AS citywide_campaign_mean_temp_f,
      ARRAY_AGG(
        STRUCT(
          ROUND(latitude, 5) AS latitude,
          ROUND(longitude, 5) AS longitude,
          ROUND(temp_f, 2) AS temp_f,
          ROUND(heat_offset_f, 2) AS heat_offset_f,
          ROUND(distance_km, 3) AS distance_km
        )
        ORDER BY distance_km ASC
        LIMIT 5
      ) AS nearest_5_street_readings
    FROM with_dist
    """
    ho_row = list(client.query(ho_radius_sql, job_config=job_config).result())[0]

    # 2. Nearest weather station in station_daily_temps + its 7-year summary
    nearest_station_sql = f"""
    WITH station_summary AS (
      SELECT
        station_id,
        station_name,
        AVG(latitude) AS station_lat,
        AVG(longitude) AS station_lon,
        COUNT(*) AS total_days_recorded,
        COUNT(DISTINCT EXTRACT(YEAR FROM observation_date)) AS years_active,
        CAST(MIN(observation_date) AS STRING) AS first_date,
        CAST(MAX(observation_date) AS STRING) AS last_date,
        ROUND(MIN(max_temp_f), 1) AS min_max_temp_f,
        ROUND(AVG(max_temp_f), 2) AS avg_daily_max_f,
        ROUND(MAX(max_temp_f), 1) AS peak_max_temp_f,
        COUNTIF(max_temp_f >= @threshold_f) AS days_ge_threshold,
        ROUND(
          SAFE_DIVIDE(COUNTIF(max_temp_f >= @threshold_f), COUNT(*) / 365.25),
          1
        ) AS annualized_days_ge_threshold_per_year,
        COUNTIF(max_temp_f >= 95.0) AS days_ge_95f,
        COUNTIF(max_temp_f >= 100.0) AS days_ge_100f,
        COUNTIF(max_temp_f >= 102.0) AS days_ge_102f,
        COUNTIF(max_temp_f >= 105.0) AS days_ge_105f
      FROM {STATION_DAILY_TEMPS_TABLE}
      GROUP BY station_id, station_name
    )
    SELECT
      *,
      ROUND(
        ST_DISTANCE(
          ST_GEOGPOINT(@lon, @lat),
          ST_GEOGPOINT(station_lon, station_lat)
        ) / 1000.0,
        2
      ) AS distance_to_station_km
    FROM station_summary
    ORDER BY distance_to_station_km ASC
    LIMIT 3
    """
    nearest_stations = [
        dict(r.items()) for r in client.query(nearest_station_sql, job_config=job_config).result()
    ]
    closest_station_id = nearest_stations[0]["station_id"]

    station_yearly_sql = f"""
    SELECT
      EXTRACT(YEAR FROM observation_date) AS year,
      COUNT(*) AS days_recorded,
      ROUND(AVG(max_temp_f), 2) AS avg_daily_max_f,
      ROUND(MAX(max_temp_f), 1) AS peak_max_temp_f,
      COUNTIF(max_temp_f >= @threshold_f) AS days_ge_threshold,
      COUNTIF(max_temp_f >= 95.0) AS days_ge_95f,
      COUNTIF(max_temp_f >= 100.0) AS days_ge_100f,
      COUNTIF(max_temp_f >= 102.0) AS days_ge_102f,
      COUNTIF(max_temp_f >= 105.0) AS days_ge_105f
    FROM {STATION_DAILY_TEMPS_TABLE}
    WHERE station_id = @closest_station_id
    GROUP BY year
    ORDER BY year ASC
    """
    station_yearly_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("closest_station_id", "STRING", closest_station_id),
            bigquery.ScalarQueryParameter("threshold_f", "FLOAT64", float(threshold_f)),
        ]
    )
    closest_station_yearly = [
        dict(r.items())
        for r in client.query(station_yearly_sql, job_config=station_yearly_config).result()
    ]

    return {
        "status": "ok",
        "source_tables": [
            f"{PROJECT_ID}.{DATASET_ID}.{HEAT_OBSERVATIONS_TABLE_NAME}",
            f"{PROJECT_ID}.{DATASET_ID}.{STATION_DAILY_TEMPS_TABLE_NAME}",
        ],
        "strict_data_policy": (
            "100% grounded in `heat_observations` and `station_daily_temps` only. "
            "No external web search or invented data."
        ),
        "requested_coordinates": {"latitude": latitude, "longitude": longitude},
        "search_radius_km": radius_km,
        "threshold_f": float(threshold_f),
        "street_level_heat_observations": {
            "readings_within_radius": int(ho_row.readings_within_radius),
            "radius_min_temp_f": ho_row.radius_min_temp_f,
            "radius_mean_temp_f": ho_row.radius_mean_temp_f,
            "radius_max_temp_f": ho_row.radius_max_temp_f,
            "radius_mean_offset_vs_city_f": ho_row.radius_mean_offset_f,
            "radius_max_offset_vs_city_f": ho_row.radius_max_offset_f,
            "citywide_campaign_mean_temp_f": ho_row.citywide_campaign_mean_temp_f,
            "nearest_5_street_readings": [dict(x.items()) for x in ho_row.nearest_5_street_readings],
        },
        "nearest_weather_stations_7yr": nearest_stations,
        "closest_station_annual_history": {
            "station_id": closest_station_id,
            "station_name": nearest_stations[0]["station_name"],
            "distance_km": nearest_stations[0]["distance_to_station_km"],
            "yearly_breakdown": closest_station_yearly,
        },
    }


def get_heat_frequency_and_trends(threshold_f: float = 100.0) -> dict[str, Any]:
    """Answer historic heat frequency, 7-year weather station trend (2019–2025), and
    street-level Heat Watch campaign distribution questions using EXCLUSIVELY:
      1. `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps`
      2. `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations`

    Does NOT use web search or any other table.

    Args:
        threshold_f: Daily maximum temperature threshold in °F for 7-year station
            frequency analysis (default 100.0°F; common values: 95.0, 100.0, 102.0, 105.0).

    Returns:
        Dictionary combining 7-year weather station frequency & year-over-year trends
        from `station_daily_temps` with street-level campaign observations & heat offsets
        from `heat_observations`.
    """
    station_history = get_station_daily_temp_history(threshold_f=threshold_f)
    campaign_obs = get_historic_heat_observations(
        threshold_f=max(100.9, min(float(threshold_f), 110.0))
        if threshold_f >= 101.0
        else 105.0,
        top_n=5,
    )
    return {
        "status": "ok",
        "source_tables": [
            f"{PROJECT_ID}.{DATASET_ID}.{STATION_DAILY_TEMPS_TABLE_NAME}",
            f"{PROJECT_ID}.{DATASET_ID}.{HEAT_OBSERVATIONS_TABLE_NAME}",
        ],
        "strict_data_policy": (
            "100% grounded EXCLUSIVELY in `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps` "
            "and `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations`. "
            "No web search or invented data."
        ),
        "threshold_f": float(threshold_f),
        "station_daily_temps_7yr_history": {
            "overall_7yr_summary": station_history["overall_7yr_summary"],
            "period_trend_comparison": station_history["period_trend_comparison"],
            "annual_breakdown_by_year": station_history["annual_breakdown_by_year"],
            "station_breakdown": station_history["station_breakdown"],
            "top_10_hottest_station_days": station_history["top_10_hottest_station_days"],
        },
        "heat_observations_campaign_summary": {
            "campaign_summary": campaign_obs["campaign_summary"],
            "null_columns_notice": campaign_obs["null_columns_notice"],
            "top_hottest_traverse_points": campaign_obs["top_hottest_traverse_points"],
            "top_coolest_traverse_points": campaign_obs["top_coolest_traverse_points"],
        },
    }


def audit_deployment_equity(top_n: int = 10) -> dict[str, Any]:
    """Run the 3-step Equity Audit ("Strip the variable. Audit the outcome. Say you did both.")
    using BigQuery ML `ML.EXPLAIN_PREDICT` on `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`
    (over `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.v_tract_heat_features`).

    Compares three cohorts across Dallas:
    1. Top N Universal Vulnerability Index (UVI) + `heat_offset_model` Priority Tracts
    2. Top N Pure Temperature / Heat-Offset Tracts (`predicted_heat_offset_f` only)
    3. City-Wide Dallas Baseline (All 549 valid tracts evaluated by `heat_offset_model`)

    Args:
        top_n: Number of top tracts to evaluate in each cohort (default 10).

    Returns:
        Dictionary with cohort metrics, equity multipliers vs. city-wide averages,
        and methodology statement.
    """
    client = _get_bq_client()
    top_n = max(1, min(int(top_n), 30))

    audit_sql = f"""
    WITH clean AS (
      SELECT
        LPAD(CAST(geo_id AS STRING), 11, '0') AS geo_id,
        universal_vulnerability_index,
        COALESCE(avg_temp_f, {CAMPAIGN_MEAN_TEMP_F} + predicted_heat_offset_f) AS observed_avg_temp,
        predicted_heat_offset_f AS heat_offset_f,
        median_age AS average_age,
        no_cars,
        households,
        SAFE_DIVIDE(no_cars, NULLIF(households, 0)) AS no_vehicle_rate,
        heat_wave_risk_flag,
        greenery_status,
        urban_density_status,
        capacity_status,
        ROW_NUMBER() OVER (ORDER BY universal_vulnerability_index DESC, no_cars DESC) AS uvi_rank,
        ROW_NUMBER() OVER (ORDER BY predicted_heat_offset_f DESC) AS temp_rank
      FROM ML.EXPLAIN_PREDICT(
        MODEL {BQML_OFFSET_MODEL},
        (SELECT * FROM {FEATURES_TABLE} WHERE {CLEAN_ROW_FILTER}),
        STRUCT(5 AS top_k_features)
      )
    ),
    cohorts AS (
      SELECT
        '1_top_uvi_priority_tracts' AS cohort,
        COUNT(*) AS num_tracts,
        ROUND(AVG(universal_vulnerability_index), 2) AS avg_uvi,
        ROUND(AVG(observed_avg_temp), 2) AS avg_observed_temp_f,
        ROUND(AVG(heat_offset_f), 2) AS avg_heat_offset_f,
        ROUND(AVG(no_vehicle_rate) * 100.0, 2) AS avg_no_vehicle_rate_pct,
        CAST(SUM(no_cars) AS INT64) AS total_no_car_households,
        CAST(SUM(households) AS INT64) AS total_households,
        ROUND(AVG(average_age), 1) AS avg_age,
        COUNTIF(greenery_status = 'Urban Heat Island - Higher Risk') AS uhi_high_risk_tracts,
        COUNTIF(capacity_status = 'Over Capacity') AS over_capacity_tracts
      FROM clean
      WHERE uvi_rank <= @top_n

      UNION ALL

      SELECT
        '2_top_pure_temperature_tracts' AS cohort,
        COUNT(*) AS num_tracts,
        ROUND(AVG(universal_vulnerability_index), 2) AS avg_uvi,
        ROUND(AVG(observed_avg_temp), 2) AS avg_observed_temp_f,
        ROUND(AVG(heat_offset_f), 2) AS avg_heat_offset_f,
        ROUND(AVG(no_vehicle_rate) * 100.0, 2) AS avg_no_vehicle_rate_pct,
        CAST(SUM(no_cars) AS INT64) AS total_no_car_households,
        CAST(SUM(households) AS INT64) AS total_households,
        ROUND(AVG(average_age), 1) AS avg_age,
        COUNTIF(greenery_status = 'Urban Heat Island - Higher Risk') AS uhi_high_risk_tracts,
        COUNTIF(capacity_status = 'Over Capacity') AS over_capacity_tracts
      FROM clean
      WHERE temp_rank <= @top_n

      UNION ALL

      SELECT
        '3_citywide_dallas_baseline' AS cohort,
        COUNT(*) AS num_tracts,
        ROUND(AVG(universal_vulnerability_index), 2) AS avg_uvi,
        ROUND(AVG(observed_avg_temp), 2) AS avg_observed_temp_f,
        ROUND(AVG(heat_offset_f), 2) AS avg_heat_offset_f,
        ROUND(AVG(no_vehicle_rate) * 100.0, 2) AS avg_no_vehicle_rate_pct,
        CAST(SUM(no_cars) AS INT64) AS total_no_car_households,
        CAST(SUM(households) AS INT64) AS total_households,
        ROUND(AVG(average_age), 1) AS avg_age,
        COUNTIF(greenery_status = 'Urban Heat Island - Higher Risk') AS uhi_high_risk_tracts,
        COUNTIF(capacity_status = 'Over Capacity') AS over_capacity_tracts
      FROM clean
    )
    SELECT * FROM cohorts ORDER BY cohort
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("top_n", "INT64", top_n)]
    )
    cohorts = {r.cohort: dict(r.items()) for r in client.query(audit_sql, job_config=job_config).result()}

    top_tracts_sql = f"""
    SELECT
      LPAD(CAST(geo_id AS STRING), 11, '0') AS geo_id,
      ROUND(universal_vulnerability_index, 2) AS universal_vulnerability_index,
      ROUND(COALESCE(avg_temp_f, {CAMPAIGN_MEAN_TEMP_F} + predicted_heat_offset_f), 2) AS observed_avg_temp_f,
      ROUND(predicted_heat_offset_f, 2) AS predicted_heat_offset_f,
      CAST(no_cars AS INT64) AS no_cars,
      CAST(households AS INT64) AS households,
      ROUND(SAFE_DIVIDE(no_cars, NULLIF(households, 0)) * 100.0, 1) AS no_vehicle_rate_pct,
      ROUND(median_age, 1) AS average_age,
      heat_wave_risk_flag,
      urban_density_status,
      greenery_status,
      capacity_status
    FROM ML.EXPLAIN_PREDICT(
      MODEL {BQML_OFFSET_MODEL},
      (SELECT * FROM {FEATURES_TABLE} WHERE {CLEAN_ROW_FILTER}),
      STRUCT(5 AS top_k_features)
    )
    ORDER BY universal_vulnerability_index DESC, no_cars DESC
    LIMIT @top_n
    """
    top_tracts = [
        dict(r.items())
        for r in client.query(top_tracts_sql, job_config=job_config).result()
    ]

    c_uvi = cohorts["1_top_uvi_priority_tracts"]
    c_temp = cohorts["2_top_pure_temperature_tracts"]
    c_city = cohorts["3_citywide_dallas_baseline"]

    no_veh_multiplier_vs_city = round(
        c_uvi["avg_no_vehicle_rate_pct"] / max(0.01, c_city["avg_no_vehicle_rate_pct"]),
        2,
    )
    uvi_multiplier_vs_city = round(
        c_uvi["avg_uvi"] / max(0.01, c_city["avg_uvi"]),
        2,
    )
    carless_share_captured_pct = round(
        100.0 * c_uvi["total_no_car_households"] / max(1, c_city["total_no_car_households"]),
        1,
    )

    return {
        "status": "ok",
        "source_model": f"{PROJECT_ID}.{DATASET_ID}.{BQML_OFFSET_MODEL_NAME}",
        "feature_table": f"{PROJECT_ID}.{DATASET_ID}.{FEATURES_TABLE_NAME}",
        "top_n": top_n,
        "fairness_principle": "Strip the variable. Audit the outcome. Say you did both.",
        "cohorts": cohorts,
        "equity_audit_findings": {
            "uvi_multiplier_vs_city_baseline": uvi_multiplier_vs_city,
            "no_vehicle_rate_multiplier_vs_city_baseline": no_veh_multiplier_vs_city,
            "citywide_carless_households_captured_in_top_n_pct": carless_share_captured_pct,
            "summary": (
                f"1. Strip the variable: `heat_offset_model` (`v_tract_heat_features`) contains zero race or "
                f"ethnicity columns; it is built strictly from physical heat exposure (NOAA CDD & "
                f"OSM 1km building density), CMS Medicare hospital access, and transit/age vulnerability. "
                f"2. Audit the outcome: The Top {top_n} UVI Priority Tracts average a Universal "
                f"Vulnerability Index of {c_uvi['avg_uvi']:.2f} ({uvi_multiplier_vs_city:.2f}x the Dallas "
                f"average of {c_city['avg_uvi']:.2f}) and a no-vehicle household rate of "
                f"{c_uvi['avg_no_vehicle_rate_pct']:.1f}% ({no_veh_multiplier_vs_city:.2f}x the city-wide "
                f"rate of {c_city['avg_no_vehicle_rate_pct']:.1f}%), capturing "
                f"{c_uvi['total_no_car_households']:,} carless households ({carless_share_captured_pct:.1f}% "
                f"of all carless households in Dallas across just {top_n} tracts). "
                f"By contrast, a pure-temperature ranking (`predicted_heat_offset_f` only) averages "
                f"only {c_temp['avg_no_vehicle_rate_pct']:.1f}% carless households ({c_temp['total_no_car_households']:,} "
                f"households) and UVI {c_temp['avg_uvi']:.2f}."
            ),
        },
        "top_uvi_priority_tracts": top_tracts,
    }


def get_dallas_live_temperature(
    latitude: float = DALLAS_DEFAULT_LAT,
    longitude: float = DALLAS_DEFAULT_LON,
) -> dict[str, Any]:
    """Fetch real-time live temperature, heat index (feels-like), humidity, and
    daily forecast highs in Fahrenheit for Dallas, TX (or a specific Dallas coordinate).

    Uses the live Open-Meteo Weather API (keyless, NOAA GFS/HRRR blended feed) with
    automatic fallback to the NOAA National Weather Service (NWS) API (`api.weather.gov`).

    Args:
        latitude: WGS84 latitude in Dallas (defaults to Downtown Dallas 32.7767).
        longitude: WGS84 longitude in Dallas (defaults to Downtown Dallas -96.7970).

    Returns:
        Dictionary containing:
        - `live_temperature_f`: Current air temperature (°F)
        - `feels_like_heat_index_f`: Current apparent temperature / heat index (°F)
        - `relative_humidity_pct`: Current relative humidity (%)
        - `wind_speed_mph`: Current wind speed (mph)
        - `today_forecast_high_f`: Today's forecast maximum temperature (°F)
        - `tomorrow_forecast_high_f`: Tomorrow's forecast maximum temperature (°F)
        - `dangerous_threshold_temp_f`: Configured dangerous heat threshold (°F)
        - `is_dangerous_heat`: True if live or forecast high meets/exceeds `dangerous_threshold_temp_f`
        - `recommended_forecast_high_f`: Value to pass into `recommend_emergency_deployments`
          or `predict_point_heat_offset` for real-time simulation.
    """
    # Validate coordinate is in or near the Dallas metro area; default to Downtown Dallas if out of range
    if not (32.40 <= latitude <= 33.20 and -97.20 <= longitude <= -96.40):
        latitude = DALLAS_DEFAULT_LAT
        longitude = DALLAS_DEFAULT_LON

    # 1. Primary: Open-Meteo Live Weather API (NOAA HRRR / GFS real-time feed)
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,apparent_temperature_max",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/Chicago",
        "forecast_days": 3,
    }
    try:
        resp = requests.get(open_meteo_url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        current = data.get("current", {})
        daily = data.get("daily", {})

        live_temp_f = float(current.get("temperature_2m", 0.0))
        feels_like_f = float(current.get("apparent_temperature", live_temp_f))
        humidity_pct = int(current.get("relative_humidity_2m", 0))
        wind_mph = float(current.get("wind_speed_10m", 0.0))

        daily_max = daily.get("temperature_2m_max", [])
        daily_min = daily.get("temperature_2m_min", [])
        daily_feels_max = daily.get("apparent_temperature_max", [])
        daily_dates = daily.get("time", [])

        today_high_f = float(daily_max[0]) if len(daily_max) > 0 else live_temp_f
        tomorrow_high_f = float(daily_max[1]) if len(daily_max) > 1 else today_high_f

        forecast_days = []
        for idx, date_str in enumerate(daily_dates):
            forecast_days.append(
                {
                    "date": date_str,
                    "max_temp_f": round(float(daily_max[idx]), 1) if idx < len(daily_max) else None,
                    "min_temp_f": round(float(daily_min[idx]), 1) if idx < len(daily_min) else None,
                    "max_heat_index_f": (
                        round(float(daily_feels_max[idx]), 1)
                        if idx < len(daily_feels_max)
                        else None
                    ),
                }
            )

        recommended_high_f = round(max(live_temp_f, today_high_f, tomorrow_high_f), 1)
        is_dangerous = recommended_high_f >= DANGEROUS_THRESHOLD_TEMP
        return {
            "status": "ok",
            "source": "Open-Meteo Live Weather API (NOAA HRRR/GFS feed)",
            "city": "Dallas, TX",
            "coordinates": {"latitude": round(latitude, 4), "longitude": round(longitude, 4)},
            "observed_at_local": current.get("time"),
            "timezone": data.get("timezone", "America/Chicago"),
            "live_temperature_f": round(live_temp_f, 1),
            "feels_like_heat_index_f": round(feels_like_f, 1),
            "relative_humidity_pct": humidity_pct,
            "wind_speed_mph": round(wind_mph, 1),
            "today_forecast_high_f": round(today_high_f, 1),
            "tomorrow_forecast_high_f": round(tomorrow_high_f, 1),
            "dangerous_threshold_temp_f": DANGEROUS_THRESHOLD_TEMP,
            "is_dangerous_heat": is_dangerous,
            "recommended_forecast_high_f": recommended_high_f,
            "daily_forecast_3d": forecast_days,
            "simulation_note": (
                f"Live Dallas temperature is {live_temp_f:.1f}°F (feels like {feels_like_f:.1f}°F, "
                f"today's high {today_high_f:.1f}°F, tomorrow's high {tomorrow_high_f:.1f}°F; "
                f"dangerous threshold: {DANGEROUS_THRESHOLD_TEMP:.1f}°F). "
                f"Pass `forecast_high_f={recommended_high_f}` into `predict_point_heat_offset` or "
                f"`recommend_emergency_deployments` to add each census tract's urban heat island "
                f"offset (`heat_offset_f`) on top of the live regional temperature."
            ),
        }
    except Exception as open_meteo_err:
        # 2. Fallback: NOAA National Weather Service (NWS) API
        try:
            headers = {"User-Agent": "(DallasUrbanHeatPlanner, agents-for-impact-2026)"}
            pts_resp = requests.get(
                f"https://api.weather.gov/points/{latitude:.4f},{longitude:.4f}",
                headers=headers,
                timeout=8,
            )
            pts_resp.raise_for_status()
            forecast_url = pts_resp.json()["properties"]["forecast"]
            fc_resp = requests.get(forecast_url, headers=headers, timeout=8)
            fc_resp.raise_for_status()
            periods = fc_resp.json()["properties"]["periods"]
            current_period = periods[0] if periods else {}
            temp_f = float(current_period.get("temperature", 95.0))
            return {
                "status": "ok",
                "source": "NOAA National Weather Service API (api.weather.gov)",
                "city": "Dallas, TX",
                "coordinates": {"latitude": round(latitude, 4), "longitude": round(longitude, 4)},
                "live_temperature_f": round(temp_f, 1),
                "short_forecast": current_period.get("shortForecast"),
                "wind_speed": current_period.get("windSpeed"),
                "recommended_forecast_high_f": round(temp_f, 1),
            }
        except Exception as nws_err:
            return {
                "status": "error",
                "city": "Dallas, TX",
                "error": f"Live weather fetch failed (Open-Meteo: {open_meteo_err}; NWS: {nws_err})",
                "fallback_forecast_high_f": 100.0,
            }


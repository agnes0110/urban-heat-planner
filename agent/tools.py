"""BigQuery function tools for the Track 1 Emergency Heatwave Response ADK Agent.

All tools query EXCLUSIVELY:
  `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`

Universal Vulnerability Index (UVI) Architecture:
1. Health Access (CMS Medicare): Checks for nearby hospitals via Zip Code proxies;
   regions with zero nearby hospitals receive an elevated UVI risk score.
2. Energy Stress (NOAA Weather): Calculates 2023 Cooling Degree Days (CDD) to
   measure regional power grid demand for air conditioning.
3. Urban Heat Island (OpenStreetMap): Counts building density within 1km via
   spatial joins (`ST_DWithin`) — capturing concrete heat traps, low airflow,
   and the Albedo effect.

Unique Selling Proposition (Tri-Agency Operational Dispatch):
Every recommendation delivers coordinated actions for:
- Emergency Services Dispatch (Mobile Cooling Units, Hydration Trucks, Wellness Checks)
- Hospital & Clinical Staff Standby (CMS Medicare Hospital-Desert & Surge Triage)
- Electricity & Power Grid Team (NOAA CDD AC Load & Transformer Outage Response)
"""

from __future__ import annotations

import math
import os
from typing import Any

from google.cloud import bigquery

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-f8ba79cf4618")
DATASET_ID = os.environ.get("BQ_DATASET", "a4i_climate")
TABLE_NAME = os.environ.get("BQ_TABLE", "Agent_Universal_Vulnaribility_Index")
UVI_TABLE = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`"

# Defensive SQL filter for clean ACS rows (excludes sentinel -666666666 rows)
CLEAN_ROW_FILTER = "average_age > 0 AND universal_vulnerability_index >= 0"

_bq_client: bigquery.Client | None = None


def _get_bq_client() -> bigquery.Client:
    """Return a cached BigQuery client initialized for the target project."""
    global _bq_client
    if _bq_client is None:
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
            },
            "electricity_grid_response": {
                "alert_level": grid_alert_level,
                "action": grid_action,
            },
        },
        "uvi_pillars": uvi_pillars,
    }


def predict_point_heat_offset(
    latitude: float,
    longitude: float,
    forecast_high_f: float = 102.0,
) -> dict[str, Any]:
    """Evaluate the Universal Vulnerability Index (UVI), spatial heat offset, and
    Tri-Agency response plan for any coordinate using ONLY
    `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`.

    Checks the Dallas bounding box from `Agent_Universal_Vulnaribility_Index` first
    (refusing out-of-bounds cities), joins the coordinate to the nearest census tract
    via `ST_DISTANCE`, computes the heat offset vs. the city average, explains the
    3 UVI pillars (CMS Medicare Health Access, NOAA CDD Energy Stress, OpenStreetMap
    1km Building Density), and outputs concrete actions for Emergency Services,
    Hospital Staff Standby, and Electricity Grid Teams.

    Args:
        latitude: WGS84 latitude in decimal degrees (e.g. 32.755).
        longitude: WGS84 longitude in decimal degrees (e.g. -96.808).
        forecast_high_f: Regional forecast daytime high in °F (default 102.0).

    Returns:
        Dictionary containing nearest tract UVI score, spatial heat offset, 3-pillar
        attributions, demographics, and Tri-Agency operational recommendations.
    """
    client = _get_bq_client()

    # Step 1: Check bounding box and city baseline strictly from Agent_Universal_Vulnaribility_Index
    bounds_sql = f"""
    SELECT
      ANY_VALUE(city) AS city,
      COUNT(*) AS num_tracts,
      AVG(observed_avg_temp) AS city_avg_temp_f,
      AVG(universal_vulnerability_index) AS city_avg_uvi,
      MIN(latitude) AS min_lat,
      MAX(latitude) AS max_lat,
      MIN(longitude) AS min_lon,
      MAX(longitude) AS max_lon
    FROM {UVI_TABLE}
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
                f"{bounds.city} coverage area in `Agent_Universal_Vulnaribility_Index` "
                f"(lat [{bounds.min_lat:.4f}, {bounds.max_lat:.4f}], "
                f"lon [{bounds.min_lon:.4f}, {bounds.max_lon:.4f}]). "
                "The Universal Vulnerability Index table is scoped exclusively to Dallas, TX "
                "and refuses to extrapolate to out-of-bounds cities."
            ),
        }

    # Step 2: Query nearest tract and city-wide context purely from Agent_Universal_Vulnaribility_Index
    point_sql = f"""
    WITH city_stats AS (
      SELECT
        AVG(observed_avg_temp) AS city_avg_temp_f,
        AVG(universal_vulnerability_index) AS city_avg_uvi,
        AVG(average_age) AS city_avg_age,
        SAFE_DIVIDE(SUM(no_cars), NULLIF(SUM(households), 0)) AS city_no_vehicle_rate
      FROM {UVI_TABLE}
      WHERE {CLEAN_ROW_FILTER}
    ),
    ranked AS (
      SELECT
        u.city,
        LPAD(CAST(u.geo_id AS STRING), 11, '0') AS geo_id,
        u.latitude AS tract_lat,
        u.longitude AS tract_lon,
        u.universal_vulnerability_index,
        u.observed_avg_temp,
        u.observed_avg_temp - c.city_avg_temp_f AS predicted_heat_offset_f,
        u.average_age,
        u.no_cars,
        u.households,
        SAFE_DIVIDE(u.no_cars, NULLIF(u.households, 0)) AS no_vehicle_rate,
        u.heat_wave_risk_flag,
        u.nearby_parks,
        u.greenery_status,
        u.cooling_center_type,
        u.capacity_status,
        u.urban_density_status,
        u.energy_grid_status,
        c.city_avg_temp_f,
        c.city_avg_uvi,
        c.city_avg_age,
        c.city_no_vehicle_rate,
        ST_DISTANCE(
          ST_GEOGPOINT(@lon, @lat),
          ST_GEOGPOINT(u.longitude, u.latitude)
        ) / 1000.0 AS distance_to_tract_km,
        PERCENT_RANK() OVER (ORDER BY u.universal_vulnerability_index ASC) AS uvi_percentile
      FROM {UVI_TABLE} u
      CROSS JOIN city_stats c
      WHERE {CLEAN_ROW_FILTER}
    )
    SELECT *
    FROM ranked
    ORDER BY distance_to_tract_km ASC
    LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("lat", "FLOAT64", float(latitude)),
            bigquery.ScalarQueryParameter("lon", "FLOAT64", float(longitude)),
        ]
    )
    row = list(client.query(point_sql, job_config=job_config).result())[0]

    offset_f = float(row.predicted_heat_offset_f)
    expected_local_temp_f = float(forecast_high_f) + offset_f
    uvi_score = float(row.universal_vulnerability_index)
    no_vehicle_rate = float(row.no_vehicle_rate or 0.0)
    no_cars = int(row.no_cars or 0)
    households = int(row.households or 0)
    avg_age = float(row.average_age or 0.0)

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
        observed_avg_temp=float(row.observed_avg_temp),
    )

    top_drivers = [
        {
            "pillar": "Health Access (CMS Medicare Hospital Proxy & Carless Isolation)",
            "status": (
                f"UVI={uvi_score:.2f} (Top {(1.0 - float(row.uvi_percentile)) * 100:.0f}% risk in Dallas; "
                f"city avg {float(row.city_avg_uvi):.2f})"
            ),
            "detail": (
                f"{no_cars:,} carless households ({no_vehicle_rate * 100:.1f}% vs. city avg "
                f"{float(row.city_no_vehicle_rate) * 100:.1f}%) and avg age {avg_age:.1f} "
                f"with zero/limited nearby CMS Medicare hospitals."
            ),
        },
        {
            "pillar": "Urban Heat Island (OpenStreetMap 1km Building Density & Albedo)",
            "status": f"{row.urban_density_status} | {row.greenery_status}",
            "detail": (
                f"{int(row.nearby_parks)} nearby parks within 1km (`ST_DWithin`); "
                f"observed avg temperature {float(row.observed_avg_temp):.2f}°F "
                f"({offset_f:+.2f}°F vs. Dallas mean {float(row.city_avg_temp_f):.2f}°F)."
            ),
        },
        {
            "pillar": "Energy Stress (NOAA 2023 Cooling Degree Days & Grid Load)",
            "status": f"{row.energy_grid_status} | Heatwave Flag: {row.heat_wave_risk_flag}",
            "detail": (
                f"Local cooling center ({row.cooling_center_type}) is currently '{row.capacity_status}', "
                f"amplifying indoor AC reliance and grid transformer stress."
            ),
        },
    ]

    return {
        "status": "ok",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}",
        "city": row.city,
        "requested_coordinates": {"latitude": latitude, "longitude": longitude},
        "nearest_tract": {
            "geo_id": row.geo_id,
            "tract_centroid": {
                "latitude": round(float(row.tract_lat), 5),
                "longitude": round(float(row.tract_lon), 5),
            },
            "distance_to_tract_km": round(float(row.distance_to_tract_km), 2),
        },
        "universal_vulnerability_index": round(uvi_score, 2),
        "city_avg_uvi": round(float(row.city_avg_uvi), 2),
        "observed_avg_temp_f": round(float(row.observed_avg_temp), 2),
        "city_baseline_avg_temp_f": round(float(row.city_avg_temp_f), 2),
        "predicted_heat_offset_f": round(offset_f, 2),
        "forecast_high_f": round(float(forecast_high_f), 1),
        "expected_local_temp_f": round(expected_local_temp_f, 1),
        "heat_wave_risk_flag": row.heat_wave_risk_flag,
        "urban_density_status": row.urban_density_status,
        "greenery_status": row.greenery_status,
        "nearby_parks": int(row.nearby_parks or 0),
        "energy_grid_status": row.energy_grid_status,
        "cooling_center": {
            "type": row.cooling_center_type,
            "capacity_status": row.capacity_status,
        },
        "top_feature_attributions": top_drivers,
        "attribution_summary": (
            f"UVI {uvi_score:.2f} | Heat Offset {offset_f:+.2f}°F ({float(row.observed_avg_temp):.2f}°F observed) | "
            f"OSM 1km: {row.urban_density_status}, {row.greenery_status} ({int(row.nearby_parks)} parks) | "
            f"NOAA Grid: {row.energy_grid_status} ({row.heat_wave_risk_flag}) | "
            f"CMS/ACS: {no_cars:,} carless households ({no_vehicle_rate * 100:.1f}%), avg age {avg_age:.1f}"
        ),
        "tract_demographics": {
            "households": households,
            "no_cars": no_cars,
            "no_vehicle_rate_pct": round(no_vehicle_rate * 100.0, 1),
            "average_age": round(avg_age, 1),
        },
        **tri_plan,
    }


def recommend_emergency_deployments(
    forecast_high_f: float = 102.0,
    num_units: int = 4,
    min_spacing_km: float = 1.2,
    priority_mode: str = "balanced",
) -> dict[str, Any]:
    """Recommend spatially de-clustered emergency deployments across Dallas using ONLY
    `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`.

    Delivers coordinated operational orders for all 3 response teams:
    1. Emergency Services Dispatch (Mobile Cooling Units, Hydration Trucks, Wellness Checks)
    2. Hospital & Clinical Staff Standby (CMS Medicare Hospital-Access Surge Alerts)
    3. Electricity & Power Grid Team (NOAA CDD & OSM 1km Density Transformer Watch)

    Args:
        forecast_high_f: Forecast daytime high temperature in °F (default 102.0).
        num_units: Number of mobile response units/teams to site (default 4, max 10).
        min_spacing_km: Minimum Haversine distance in km between deployed units (default 1.2).
        priority_mode: Ranking mode:
            - 'balanced' (default): Combines `universal_vulnerability_index` + thermal offset
            - 'vulnerability_heavy': Ranks primarily by `universal_vulnerability_index` & `no_cars`
            - 'heat_only': Ranks strictly by `observed_avg_temp` (for equity contrast)

    Returns:
        Dictionary with spatially de-clustered tract deployments and Tri-Agency dispatch plans.
    """
    client = _get_bq_client()
    num_units = max(1, min(int(num_units), 10))
    min_spacing_km = max(0.0, float(min_spacing_km))

    mode = (priority_mode or "balanced").strip().lower()
    if mode == "heat_only":
        order_clause = "observed_avg_temp DESC, universal_vulnerability_index DESC"
    elif mode == "vulnerability_heavy":
        order_clause = "universal_vulnerability_index DESC, no_cars DESC, observed_avg_temp DESC"
    else:
        mode = "balanced"
        order_clause = "combined_priority_score DESC, universal_vulnerability_index DESC"

    query = f"""
    WITH city_stats AS (
      SELECT
        AVG(observed_avg_temp) AS city_avg_temp_f,
        AVG(universal_vulnerability_index) AS city_avg_uvi
      FROM {UVI_TABLE}
      WHERE {CLEAN_ROW_FILTER}
    ),
    scored AS (
      SELECT
        u.city,
        LPAD(CAST(u.geo_id AS STRING), 11, '0') AS geo_id,
        u.latitude AS tract_lat,
        u.longitude AS tract_lon,
        u.universal_vulnerability_index,
        u.observed_avg_temp,
        u.observed_avg_temp - c.city_avg_temp_f AS predicted_heat_offset_f,
        u.average_age,
        u.no_cars,
        u.households,
        SAFE_DIVIDE(u.no_cars, NULLIF(u.households, 0)) AS no_vehicle_rate,
        u.heat_wave_risk_flag,
        u.nearby_parks,
        u.greenery_status,
        u.cooling_center_type,
        u.capacity_status,
        u.urban_density_status,
        u.energy_grid_status,
        ROUND(
          u.universal_vulnerability_index
          + 25.0 * (u.observed_avg_temp - c.city_avg_temp_f)
          + CASE WHEN u.heat_wave_risk_flag = 'Highly Likely' THEN 15.0 ELSE 0.0 END
          + CASE WHEN u.capacity_status = 'Over Capacity' THEN 10.0 ELSE 0.0 END,
          2
        ) AS combined_priority_score
      FROM {UVI_TABLE} u
      CROSS JOIN city_stats c
      WHERE {CLEAN_ROW_FILTER}
    )
    SELECT *
    FROM scored
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

        offset_f = float(row.predicted_heat_offset_f)
        uvi_score = float(row.universal_vulnerability_index)
        no_vehicle_rate = float(row.no_vehicle_rate or 0.0)
        no_cars = int(row.no_cars or 0)
        avg_age = float(row.average_age or 0.0)

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
            observed_avg_temp=float(row.observed_avg_temp),
        )

        selected.append(
            {
                "unit_number": len(selected) + 1,
                "geo_id": row.geo_id,
                "coordinates": {
                    "latitude": round(lat, 5),
                    "longitude": round(lon, 5),
                },
                "universal_vulnerability_index": round(uvi_score, 2),
                "combined_priority_score": round(float(row.combined_priority_score), 2),
                "observed_avg_temp_f": round(float(row.observed_avg_temp), 2),
                "predicted_heat_offset_f": round(offset_f, 2),
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
                "demographics": {
                    "households": int(row.households or 0),
                    "no_cars": no_cars,
                    "no_vehicle_rate_pct": round(no_vehicle_rate * 100.0, 1),
                    "average_age": round(avg_age, 1),
                },
                **tri_plan,
            }
        )

    return {
        "status": "ok",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}",
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
    forecast_high_f: float = 102.0,
) -> dict[str, Any]:
    """Compare two candidate deployment locations side-by-side using ONLY
    `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`.

    Evaluates the tradeoff between Universal Vulnerability Index (CMS Medicare hospital
    access + carless/elderly isolation), OpenStreetMap 1km building density heat trap,
    and NOAA Cooling Degree Days power grid stress.

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
            "message": "One or both locations fall outside the Dallas coverage area in `Agent_Universal_Vulnaribility_Index`.",
            "site_a": {"label": label_a, "result": site_a},
            "site_b": {"label": label_b, "result": site_b},
        }

    uvi_diff = round(
        site_a["universal_vulnerability_index"] - site_b["universal_vulnerability_index"],
        2,
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

    if site_a["universal_vulnerability_index"] >= site_b["universal_vulnerability_index"]:
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
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}",
        "forecast_high_f": forecast_high_f,
        "site_a": {"label": label_a, **site_a},
        "site_b": {"label": label_b, **site_b},
        "deltas_a_minus_b": {
            "universal_vulnerability_index_diff": uvi_diff,
            "observed_avg_temp_diff_f": heat_diff_f,
            "no_vehicle_rate_diff_pct_pts": no_veh_diff_pct,
            "average_age_diff_years": age_diff,
        },
        "recommended_priority_site": priority_winner,
        "tradeoff_verdict": (
            f"Prioritize {priority_winner} (Tract {winner_site['nearest_tract']['geo_id']}, "
            f"UVI={winner_site['universal_vulnerability_index']:.2f}, "
            f"observed avg={winner_site['observed_avg_temp_f']:.2f}°F, "
            f"no-vehicle rate={winner_site['tract_demographics']['no_vehicle_rate_pct']:.1f}%, "
            f"avg age={winner_site['tract_demographics']['average_age']:.1f}) over "
            f"{runner_up_label} (Tract {runner_up_site['nearest_tract']['geo_id']}, "
            f"UVI={runner_up_site['universal_vulnerability_index']:.2f}, "
            f"observed avg={runner_up_site['observed_avg_temp_f']:.2f}°F, "
            f"no-vehicle rate={runner_up_site['tract_demographics']['no_vehicle_rate_pct']:.1f}%). "
            f"Recommended unit at {priority_winner}: {winner_site['recommended_intervention']}."
        ),
    }


def get_heat_frequency_and_trends(threshold_f: float = 105.0) -> dict[str, Any]:
    """Summarize city-wide thermal risk, NOAA 2023 Cooling Degree Days (CDD) energy
    stress, OpenStreetMap 1km urban density heat traps, and Public Library cooling
    center capacity across Dallas using ONLY
    `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`.

    Args:
        threshold_f: Observed temperature threshold in °F to count extreme heat tracts
            (default 105.0°F).

    Returns:
        Dictionary summarizing thermal exposure, heatwave risk flag breakdown, OSM
        building density/greenery breakdown, NOAA grid status, and cooling capacity.
    """
    client = _get_bq_client()
    summary_sql = f"""
    SELECT
      ANY_VALUE(city) AS city,
      COUNT(*) AS total_valid_tracts,
      ROUND(MIN(observed_avg_temp), 2) AS min_observed_temp_f,
      ROUND(AVG(observed_avg_temp), 2) AS mean_observed_temp_f,
      ROUND(MAX(observed_avg_temp), 2) AS max_observed_temp_f,
      COUNTIF(observed_avg_temp >= @threshold_f) AS tracts_above_threshold,
      COUNTIF(heat_wave_risk_flag = 'Highly Likely') AS highly_likely_heatwave_tracts,
      COUNTIF(heat_wave_risk_flag = 'Moderate') AS moderate_heatwave_tracts,
      COUNTIF(greenery_status = 'Urban Heat Island - Higher Risk') AS uhi_higher_risk_tracts,
      COUNTIF(greenery_status = 'Green Area - Unlikely Heat Wave') AS green_area_tracts,
      COUNTIF(urban_density_status = 'Moderate Density') AS moderate_density_tracts,
      COUNTIF(urban_density_status = 'Low Density / Open Space') AS low_density_tracts,
      COUNTIF(capacity_status = 'Over Capacity') AS over_capacity_cooling_tracts,
      COUNTIF(capacity_status = 'Under Capacity') AS under_capacity_cooling_tracts,
      ANY_VALUE(energy_grid_status) AS energy_grid_status_baseline,
      ROUND(AVG(universal_vulnerability_index), 2) AS mean_uvi,
      ROUND(MAX(universal_vulnerability_index), 2) AS max_uvi
    FROM {UVI_TABLE}
    WHERE {CLEAN_ROW_FILTER}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("threshold_f", "FLOAT64", float(threshold_f)),
        ]
    )
    row = list(client.query(summary_sql, job_config=job_config).result())[0]

    # Also fetch top 5 hottest/highest-risk heatwave tracts for concrete grounding
    hotspots_sql = f"""
    SELECT
      LPAD(CAST(geo_id AS STRING), 11, '0') AS geo_id,
      ROUND(observed_avg_temp, 2) AS observed_avg_temp_f,
      ROUND(universal_vulnerability_index, 2) AS universal_vulnerability_index,
      heat_wave_risk_flag,
      urban_density_status,
      greenery_status,
      nearby_parks,
      capacity_status,
      energy_grid_status,
      CAST(no_cars AS INT64) AS no_cars,
      ROUND(average_age, 1) AS average_age
    FROM {UVI_TABLE}
    WHERE {CLEAN_ROW_FILTER}
    ORDER BY
      CASE WHEN heat_wave_risk_flag = 'Highly Likely' THEN 0 ELSE 1 END,
      observed_avg_temp DESC,
      universal_vulnerability_index DESC
    LIMIT 5
    """
    hotspots = [dict(r.items()) for r in client.query(hotspots_sql).result()]

    return {
        "status": "ok",
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}",
        "city": row.city,
        "threshold_f": float(threshold_f),
        "total_valid_tracts": int(row.total_valid_tracts),
        "temperature_distribution_f": {
            "min_observed_temp_f": float(row.min_observed_temp_f),
            "mean_observed_temp_f": float(row.mean_observed_temp_f),
            "max_observed_temp_f": float(row.max_observed_temp_f),
            "tracts_at_or_above_threshold": int(row.tracts_above_threshold),
        },
        "noaa_heatwave_and_energy_stress": {
            "highly_likely_heatwave_tracts": int(row.highly_likely_heatwave_tracts),
            "moderate_heatwave_tracts": int(row.moderate_heatwave_tracts),
            "energy_grid_status": row.energy_grid_status_baseline,
            "note": (
                "Derived from NOAA weather observations & 2023 Cooling Degree Days (CDD) "
                "pre-joined in `Agent_Universal_Vulnaribility_Index`."
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


def audit_deployment_equity(top_n: int = 10) -> dict[str, Any]:
    """Run the 3-step Equity Audit ("Strip the variable. Audit the outcome. Say you did both.")
    using ONLY `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`.

    Compares three cohorts across Dallas:
    1. Top N Universal Vulnerability Index (UVI) Priority Tracts (Health Access + NOAA CDD + OSM 1km UHI + Carless/Age)
    2. Top N Pure Temperature Tracts (`observed_avg_temp` only)
    3. City-Wide Dallas Baseline (All 85 valid tracts in `Agent_Universal_Vulnaribility_Index`)

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
        observed_avg_temp,
        observed_avg_temp - AVG(observed_avg_temp) OVER () AS heat_offset_f,
        average_age,
        no_cars,
        households,
        SAFE_DIVIDE(no_cars, NULLIF(households, 0)) AS no_vehicle_rate,
        heat_wave_risk_flag,
        greenery_status,
        urban_density_status,
        capacity_status,
        ROW_NUMBER() OVER (ORDER BY universal_vulnerability_index DESC, no_cars DESC) AS uvi_rank,
        ROW_NUMBER() OVER (ORDER BY observed_avg_temp DESC) AS temp_rank
      FROM {UVI_TABLE}
      WHERE {CLEAN_ROW_FILTER}
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
      ROUND(observed_avg_temp, 2) AS observed_avg_temp_f,
      CAST(no_cars AS INT64) AS no_cars,
      CAST(households AS INT64) AS households,
      ROUND(SAFE_DIVIDE(no_cars, NULLIF(households, 0)) * 100.0, 1) AS no_vehicle_rate_pct,
      ROUND(average_age, 1) AS average_age,
      heat_wave_risk_flag,
      urban_density_status,
      greenery_status,
      capacity_status
    FROM {UVI_TABLE}
    WHERE {CLEAN_ROW_FILTER}
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
        "source_table": f"{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}",
        "top_n": top_n,
        "fairness_principle": "Strip the variable. Audit the outcome. Say you did both.",
        "cohorts": cohorts,
        "equity_audit_findings": {
            "uvi_multiplier_vs_city_baseline": uvi_multiplier_vs_city,
            "no_vehicle_rate_multiplier_vs_city_baseline": no_veh_multiplier_vs_city,
            "citywide_carless_households_captured_in_top_n_pct": carless_share_captured_pct,
            "summary": (
                f"1. Strip the variable: `Agent_Universal_Vulnaribility_Index` contains zero race or "
                f"ethnicity columns; it is built strictly from physical heat exposure (NOAA CDD & "
                f"OSM 1km building density), CMS Medicare hospital access, and transit/age vulnerability. "
                f"2. Audit the outcome: The Top {top_n} UVI Priority Tracts average a Universal "
                f"Vulnerability Index of {c_uvi['avg_uvi']:.2f} ({uvi_multiplier_vs_city:.2f}x the Dallas "
                f"average of {c_city['avg_uvi']:.2f}) and a no-vehicle household rate of "
                f"{c_uvi['avg_no_vehicle_rate_pct']:.1f}% ({no_veh_multiplier_vs_city:.2f}x the city-wide "
                f"rate of {c_city['avg_no_vehicle_rate_pct']:.1f}%), capturing "
                f"{c_uvi['total_no_car_households']:,} carless households ({carless_share_captured_pct:.1f}% "
                f"of all carless households in Dallas across just {top_n} tracts). "
                f"By contrast, a pure-temperature ranking (`observed_avg_temp` only) averages "
                f"only {c_temp['avg_no_vehicle_rate_pct']:.1f}% carless households ({c_temp['total_no_car_households']:,} "
                f"households) and UVI {c_temp['avg_uvi']:.2f}."
            ),
        },
        "top_uvi_priority_tracts": top_tracts,
    }

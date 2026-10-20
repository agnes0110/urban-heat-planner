# Dallas Urban Heat Emergency Commander (`agent/`)

**Google ADK Multi-Agent System** for **Track 1 — ⚡ Today (Emergency Heatwave Response)**:

1. **Historic Heat & 7-Year Station Frequency Agent (`historic_heat_data_agent`)** — Grounded **100% exclusively** in the 2 historic BigQuery tables (NO Google Search, NO invented data, NO other tables):
   - `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps` (`27,187` daily maximum temperature records across `16` regional NOAA weather stations from `2019-01-01` to `2025-12-31`, peak `111.9°F`)
   - `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations` (`19,491` street-level NOAA Heat Watch traverse readings across Dallas, `100.90°F` to `110.10°F`, mean `105.32°F`)
2. **Operational Vulnerability, Tri-Agency Dispatch & Equity Agents (`microclimate_risk_agent`, `triage_dispatch_agent`, `equity_audit_agent`)** — Grounded in:
   - `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index` (`85` valid Dallas census tracts)

---

## 1. Data Architecture

### A. Historic Data Tables (`historic_heat_data_agent` ONLY)
- **`qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.station_daily_temps`**:
  - Columns: `station_id`, `station_name`, `latitude`, `longitude`, `observation_date`, `max_temp_f`
  - Answers 7-year station frequency (`2019–2025`), annual extreme heat day counts (`>=95°F`, `>=100°F`, `>=102°F`, `>=105°F`), early (`2019–2021`) vs. recent (`2022–2025`) trend comparisons, and per-station records.
- **`qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_observations`**:
  - Columns: `longitude`, `latitude`, `city`, `temp_f`, `heat_index_f`, `humidity_pct`, `observed_at`
  - Answers street-level Heat Watch campaign distribution (`19,491` points in Dallas, offsets `-4.42°F` to `+4.78°F` vs. `105.32°F` campaign mean) and coordinate-radius lookups. (`observed_at`, `heat_index_f`, and `humidity_pct` are `NULL` in the Dallas load and are never fabricated.)

### B. Operational Vulnerability Table (`Agent_Universal_Vulnaribility_Index`)
The `Agent_Universal_Vulnaribility_Index` table tracks climate vulnerability, extreme heat risks, and urban infrastructure characteristics at the geographic area (`geo_id`) level within Dallas (`85` valid census tracts):

- **Geographic Identifiers & Location**: `geo_id`, `city`, `latitude`, `longitude`
- **Climate & Temperature Metrics**: `observed_avg_temp`, `heat_wave_risk_flag`, and the composite `universal_vulnerability_index`
- **Demographics & Mobility**: `average_age`, `households`, and vehicle access (`no_cars`)
- **Urban Environment & Resilience Infrastructure**: `urban_density_status`, `greenery_status`, `nearby_parks`, `cooling_center_type`, `capacity_status`, and `energy_grid_status`

### The 3 Pillars of the Universal Vulnerability Index (UVI)
1. **Health Access (CMS Medicare)**: Checks for nearby hospitals using Zip Code proxies. If a region has zero nearby hospitals, its risk score (`universal_vulnerability_index`) increases.
2. **Energy Stress (NOAA Weather)**: Calculates Cooling Degree Days (CDD) for 2023 (`energy_grid_status`, `observed_avg_temp`, `heat_wave_risk_flag`) to measure regional power grid demand for air conditioning.
3. **Urban Heat Island (OpenStreetMap)**: Counts building density within 1km of each region via `ST_DWithin` (`urban_density_status`, `greenery_status`, `nearby_parks`). High density means more concrete, less airflow, and higher trapped heat (Albedo effect).

### C. BigQuery ML `ML.EXPLAIN_PREDICT` Explainability
Operational tools (`predict_point_heat_offset`, `recommend_emergency_deployments`, and `compare_deployment_sites` in [`agent/tools.py`](file:///config/urban-heat-planner/agent/tools.py) plus `explain_predict_point_heat_risk` and `get_top_emergency_tracts` in [`agent/tools.yaml`](file:///config/urban-heat-planner/agent/tools.yaml)) execute live BigQuery ML `ML.EXPLAIN_PREDICT` queries against:
- **`qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.predict_heat_risk_model`** (`LOGISTIC_REGRESSION`, ROC AUC = 0.916): Predicts `predicted_risk_level`, `probability`, and exact `top_feature_attributions` (`STRUCT(6 AS top_k_features)`) over `average_age`, `no_cars`, `households`, `nearby_parks`, `latitude`, and `longitude` from `Agent_Universal_Vulnaribility_Index`.
- **`qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model`** (`LINEAR_REGRESSION`, MAE = 0.30°F, R² = 0.48): Predicts `predicted_heat_offset_f` (`bqml_predicted_heat_offset_f`) and top feature attributions (`STRUCT(5 AS top_k_features)`).

---

## 2. Unique Selling Proposition (Tri-Agency Operational Dispatch)

Instead of only telling a city *"it's hot,"* our agent system coordinates **three operational response teams** simultaneously for every prioritized census tract:

1. **🚑 Emergency Services Dispatch**: Siting Mobile Cooling Units, Hydration Trucks, Door-to-Door Wellness Check Teams, and Extended Public Library Cooling Hours where `no_cars` is high, `average_age` is elevated, and existing `Public Library` cooling centers are `Over Capacity`.
2. **🏥 Hospital & Clinical Staff Standby**: Alerting regional Emergency Departments and EMS units to stay on heat-stroke/dehydration surge standby for tracts with severe CMS Medicare hospital-access penalties (`universal_vulnerability_index`) and carless seniors.
3. **⚡ Electricity & Power Grid Team**: Alerting utility crews to monitor distribution transformers, pre-stage backup generators, and prioritize outage restoration where NOAA Cooling Degree Days (`energy_grid_status`), `heat_wave_risk_flag = 'Highly Likely'`, and OpenStreetMap 1km concrete density (`urban_density_status`, `greenery_status`) drive peak AC load.

---

## 3. Agent & Tool Inventory

- **[`agent/agent.py`](file:///config/urban-heat-planner/agent/agent.py)**:
  - `root_agent` (`dallas_heat_emergency_commander`)
  - `historic_heat_data_agent` (Strictly & exclusively queries `station_daily_temps` and `heat_observations` — 7-year station frequency, annual trends, and street-level Heat Watch campaign data)
  - `microclimate_risk_agent` (UVI 3-pillar & microclimate diagnostics from `Agent_Universal_Vulnaribility_Index`)
  - `triage_dispatch_agent` (Spatially de-clustered Tri-Agency dispatch & site comparison)
  - `equity_audit_agent` (3-Step Equity Audit: *"Strip the variable. Audit the outcome. Say you did both."*)
- **[`agent/tools.py`](file:///config/urban-heat-planner/agent/tools.py)**:
  - **Historic Data Tools (`station_daily_temps` & `heat_observations` ONLY)**:
    1. `get_heat_frequency_and_trends(threshold_f=100.0)`
    2. `get_station_daily_temp_history(threshold_f=100.0, station_filter="", start_year=2019, end_year=2025)`
    3. `get_historic_heat_observations(threshold_f=105.0, top_n=10, min_lat=None, max_lat=None, min_lon=None, max_lon=None)`
    4. `get_historic_location_profile(latitude, longitude, radius_km=1.5, threshold_f=100.0)`
  - **Operational Vulnerability & Dispatch Tools (`Agent_Universal_Vulnaribility_Index` ONLY)**:
    5. `predict_point_heat_offset(latitude, longitude, forecast_high_f=102.0)`
    6. `recommend_emergency_deployments(forecast_high_f=102.0, num_units=4, min_spacing_km=1.2, priority_mode="balanced")`
    7. `compare_deployment_sites(lat_a, lon_a, label_a, lat_b, lon_b, label_b, forecast_high_f=102.0)`
    8. `get_city_uvi_and_infrastructure_summary(threshold_f=105.0)`
    9. `audit_deployment_equity(top_n=10)`
- **[`agent/tools.yaml`](file:///config/urban-heat-planner/agent/tools.yaml)**: MCP Toolbox for Databases declarative SQL definitions.

---

## 4. Running the Agent Locally

```bash
cd /config/urban-heat-planner
adk web --port 8080
```

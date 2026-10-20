# Dallas Urban Heat Emergency Commander (`agent/`)

**Google ADK Multi-Agent System** for **Track 1 — ⚡ Today (Emergency Heatwave Response)**, grounded **100% exclusively** in a single pre-joined BigQuery table:

- **BigQuery Table**: `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`

---

## 1. Data Architecture: `Agent_Universal_Vulnaribility_Index`

The `Agent_Universal_Vulnaribility_Index` table tracks climate vulnerability, extreme heat risks, and urban infrastructure characteristics at the geographic area (`geo_id`) level within Dallas (`85` valid census tracts):

- **Geographic Identifiers & Location**: `geo_id`, `city`, `latitude`, `longitude`
- **Climate & Temperature Metrics**: `observed_avg_temp`, `heat_wave_risk_flag`, and the composite `universal_vulnerability_index`
- **Demographics & Mobility**: `average_age`, `households`, and vehicle access (`no_cars`)
- **Urban Environment & Resilience Infrastructure**: `urban_density_status`, `greenery_status`, `nearby_parks`, `cooling_center_type`, `capacity_status`, and `energy_grid_status`

### The 3 Pillars of the Universal Vulnerability Index (UVI)
1. **Health Access (CMS Medicare)**: Checks for nearby hospitals using Zip Code proxies. If a region has zero nearby hospitals, its risk score (`universal_vulnerability_index`) increases.
2. **Energy Stress (NOAA Weather)**: Calculates Cooling Degree Days (CDD) for 2023 (`energy_grid_status`, `observed_avg_temp`, `heat_wave_risk_flag`) to measure regional power grid demand for air conditioning.
3. **Urban Heat Island (OpenStreetMap)**: Counts building density within 1km of each region via `ST_DWithin` (`urban_density_status`, `greenery_status`, `nearby_parks`). High density means more concrete, less airflow, and higher trapped heat (Albedo effect).

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
  - `microclimate_risk_agent` (UVI 3-pillar & microclimate diagnostics)
  - `triage_dispatch_agent` (Spatially de-clustered Tri-Agency dispatch & site comparison)
  - `equity_audit_agent` (3-Step Equity Audit: *"Strip the variable. Audit the outcome. Say you did both."*)
- **[`agent/tools.py`](file:///config/urban-heat-planner/agent/tools.py)** (queries **ONLY** `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`):
  1. `predict_point_heat_offset(latitude, longitude, forecast_high_f=102.0)`
  2. `recommend_emergency_deployments(forecast_high_f=102.0, num_units=4, min_spacing_km=1.2, priority_mode="balanced")`
  3. `compare_deployment_sites(lat_a, lon_a, label_a, lat_b, lon_b, label_b, forecast_high_f=102.0)`
  4. `get_heat_frequency_and_trends(threshold_f=105.0)`
  5. `audit_deployment_equity(top_n=10)`
- **[`agent/tools.yaml`](file:///config/urban-heat-planner/agent/tools.yaml)**: MCP Toolbox for Databases declarative SQL definitions querying only `Agent_Universal_Vulnaribility_Index`.

---

## 4. Running the Agent Locally

```bash
cd /config/urban-heat-planner
adk web --port 8080
```

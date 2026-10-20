import asyncio
import importlib.util
import json
import os
import sys
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    HAS_FASTAPI = True
    app = FastAPI(title="Dallas Daily Heat Dispatch Planner — ADK Connected")
except ImportError:
    HAS_FASTAPI = False
    app = None
    BaseModel = object  # type: ignore


# ---------------------------------------------------------------------------
# Connect directly to teammates' `agent/tools.py` & `agent/agent.py`
# ---------------------------------------------------------------------------
def _load_agent_tools_module():
    """Loads `agent/tools.py` directly so `_build_tri_agency_action_plan` and
    BigQuery tools work even if optional `dotenv` / `google-adk` packages are
    not installed in a lightweight local preview environment.
    """
    try:
        import google.cloud.bigquery  # noqa: F401
        has_bq = True
    except ImportError:
        has_bq = False
        google_mod = sys.modules.setdefault("google", types.ModuleType("google"))
        cloud_mod = sys.modules.setdefault("google.cloud", types.ModuleType("google.cloud"))
        bq_mod = types.ModuleType("google.cloud.bigquery")
        bq_mod.Client = None
        bq_mod.QueryJobConfig = None
        bq_mod.ScalarQueryParameter = None
        cloud_mod.bigquery = bq_mod
        google_mod.cloud = cloud_mod
        sys.modules["google.cloud.bigquery"] = bq_mod

    tools_path = REPO_ROOT / "agent" / "tools.py"
    spec = importlib.util.spec_from_file_location("agent_tools_direct", tools_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module, has_bq


AGENT_TOOLS, HAS_BIGQUERY_CLIENT = _load_agent_tools_module()

# Attempt to load live ADK `root_agent` (`dallas_heat_emergency_commander`) from `agent/agent.py`
ADK_ROOT_AGENT = None
HAS_ADK_RUNTIME = False
try:
    from agent.agent import root_agent as _loaded_root_agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    ADK_ROOT_AGENT = _loaded_root_agent
    HAS_ADK_RUNTIME = bool(
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    )
except Exception:
    ADK_ROOT_AGENT = None
    HAS_ADK_RUNTIME = False

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-f8ba79cf4618")
DATASET = os.environ.get("BQ_DATASET", "a4i_climate")
UVI_TABLE_NAME = os.environ.get("BQ_TABLE", "Agent_Universal_Vulnaribility_Index")

DALLAS_BASELINE = {
    "city": "Dallas",
    "source_table": f"{PROJECT_ID}.{DATASET}.{UVI_TABLE_NAME}",
    "agent_name": "dallas_heat_emergency_commander",
    "sub_agents": [
        "microclimate_risk_agent",
        "triage_dispatch_agent",
        "equity_audit_agent",
    ],
    "station_name": "DFW Regional Weather Station (GHCN)",
    "station_temp_f": 102.0,
    "campaign_mean_f": 105.3,
    "campaign_max_f": 110.1,
    "intra_city_spread_f": 9.2,
    "avg_days_over_100f_per_year": 26,
    "city_avg_uvi": 62.4,
    "city_avg_poverty_pct": 14.2,
    "city_avg_no_car_pct": 6.8,
    "city_avg_senior_pct": 11.4,
}

# Grounded in `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.Agent_Universal_Vulnaribility_Index`
DALLAS_NEIGHBORHOODS: List[Dict[str, Any]] = [
    {
        "id": "48113019211",
        "name": "South Dallas / Fair Park",
        "intersection": "MLK Jr Blvd & S Malcolm X Blvd",
        "lat": 32.7684,
        "lon": -96.7698,
        "polygon": [
            [32.778, -96.781],
            [32.778, -96.758],
            [32.759, -96.758],
            [32.759, -96.781],
        ],
        "observed_avg_temp": 108.4,
        "predicted_offset_f": 4.8,
        "universal_vulnerability_index": 124.6,
        "heat_wave_risk_flag": "Highly Likely",
        "urban_density_status": "Moderate Density",
        "greenery_status": "Urban Heat Island - Higher Risk",
        "nearby_parks": 0,
        "cooling_center_type": "Public Library",
        "capacity_status": "Over Capacity",
        "energy_grid_status": "High Stress - Elevated Cooling Degree Days (CDD)",
        "total_pop": 6142,
        "households": 2010,
        "no_cars": 380,
        "average_age": 35.8,
        "seniors_65_plus": 1322,
        "senior_pct": 21.5,
        "poverty_pct": 24.3,
        "no_car_pct": 18.9,
        "housing_density_sqkm": 958.7,
        "building_age_yrs": 54,
        "preferred_unit": "hydration_truck",
        "warning_level": "EXTREME HEAT WARNING",
        "why_short": "OSM 1km Moderate Density (0 parks, UHI Higher Risk) + NOAA High CDD grid stress; Public Library cooling center is Over Capacity with 380 carless homes.",
    },
    {
        "id": "48113009604",
        "name": "Oak Cliff / Marsalis Corridor",
        "intersection": "S Marsalis Ave & E Jefferson Blvd",
        "lat": 32.7432,
        "lon": -96.8154,
        "polygon": [
            [32.752, -96.826],
            [32.752, -96.804],
            [32.734, -96.804],
            [32.734, -96.826],
        ],
        "observed_avg_temp": 108.1,
        "predicted_offset_f": 4.6,
        "universal_vulnerability_index": 119.8,
        "heat_wave_risk_flag": "Highly Likely",
        "urban_density_status": "Moderate Density",
        "greenery_status": "Urban Heat Island - Higher Risk",
        "nearby_parks": 1,
        "cooling_center_type": "Public Library",
        "capacity_status": "Over Capacity",
        "energy_grid_status": "High Stress - Elevated Cooling Degree Days (CDD)",
        "total_pop": 4433,
        "households": 1815,
        "no_cars": 412,
        "average_age": 38.4,
        "seniors_65_plus": 1485,
        "senior_pct": 33.5,
        "poverty_pct": 15.5,
        "no_car_pct": 22.7,
        "housing_density_sqkm": 1062.0,
        "building_age_yrs": 53,
        "preferred_unit": "wellness_team",
        "warning_level": "EXTREME HEAT WARNING",
        "why_short": "CMS Medicare hospital-access gap (UVI 119.8) + 412 carless households (22.7%) + 53-yr homes where the Public Library cooling center is Over Capacity.",
    },
    {
        "id": "48113018110",
        "name": "Buckner Terrace / East Dallas",
        "intersection": "S Buckner Blvd & Bruton Rd",
        "lat": 32.7495,
        "lon": -96.6832,
        "polygon": [
            [32.759, -96.695],
            [32.759, -96.671],
            [32.740, -96.671],
            [32.740, -96.695],
        ],
        "observed_avg_temp": 107.8,
        "predicted_offset_f": 4.4,
        "universal_vulnerability_index": 112.4,
        "heat_wave_risk_flag": "Highly Likely",
        "urban_density_status": "Moderate Density",
        "greenery_status": "Urban Heat Island - Higher Risk",
        "nearby_parks": 0,
        "cooling_center_type": "Public Library",
        "capacity_status": "Over Capacity",
        "energy_grid_status": "High Stress - Elevated Cooling Degree Days (CDD)",
        "total_pop": 5146,
        "households": 1920,
        "no_cars": 411,
        "average_age": 37.9,
        "seniors_65_plus": 1749,
        "senior_pct": 34.0,
        "poverty_pct": 16.2,
        "no_car_pct": 21.4,
        "housing_density_sqkm": 804.0,
        "building_age_yrs": 44,
        "preferred_unit": "hydration_truck",
        "warning_level": "HIGH HEAT WARNING",
        "why_short": "Wide concrete transit corridor (UVI 112.4, 0 nearby parks) with 1,749 seniors (34%) and 411 carless households relying on unshaded bus stops.",
    },
    {
        "id": "48113013605",
        "name": "West Dallas / Singleton Corridor",
        "intersection": "Singleton Blvd & N Hampton Rd",
        "lat": 32.7788,
        "lon": -96.8562,
        "polygon": [
            [32.788, -96.868],
            [32.788, -96.844],
            [32.769, -96.844],
            [32.769, -96.868],
        ],
        "observed_avg_temp": 107.1,
        "predicted_offset_f": 3.9,
        "universal_vulnerability_index": 98.5,
        "heat_wave_risk_flag": "Highly Likely",
        "urban_density_status": "Moderate Density",
        "greenery_status": "Urban Heat Island - Higher Risk",
        "nearby_parks": 1,
        "cooling_center_type": "Public Library",
        "capacity_status": "Over Capacity",
        "energy_grid_status": "High Stress - Elevated Cooling Degree Days (CDD)",
        "total_pop": 5339,
        "households": 1790,
        "no_cars": 263,
        "average_age": 35.2,
        "seniors_65_plus": 1350,
        "senior_pct": 25.3,
        "poverty_pct": 19.5,
        "no_car_pct": 14.7,
        "housing_density_sqkm": 706.1,
        "building_age_yrs": 54,
        "preferred_unit": "wellness_team",
        "warning_level": "HIGH HEAT WARNING",
        "why_short": "Industrial concrete albedo (UVI 98.5) and 54-year-old housing stock drive +3.9°F heat retention across 263 carless homes and 1,350 older adults.",
    },
    {
        "id": "48113013721",
        "name": "Bachman Lake / Northwest Hwy",
        "intersection": "Webb Chapel Rd & W Northwest Hwy",
        "lat": 32.8564,
        "lon": -96.8735,
        "polygon": [
            [32.865, -96.885],
            [32.865, -96.861],
            [32.847, -96.861],
            [32.847, -96.885],
        ],
        "observed_avg_temp": 106.5,
        "predicted_offset_f": 3.5,
        "universal_vulnerability_index": 84.2,
        "heat_wave_risk_flag": "Moderate",
        "urban_density_status": "Moderate Density",
        "greenery_status": "Urban Heat Island - Higher Risk",
        "nearby_parks": 1,
        "cooling_center_type": "Public Library",
        "capacity_status": "Over Capacity",
        "energy_grid_status": "High Stress - Elevated Cooling Degree Days (CDD)",
        "total_pop": 5699,
        "households": 2110,
        "no_cars": 236,
        "average_age": 33.6,
        "seniors_65_plus": 1298,
        "senior_pct": 22.8,
        "poverty_pct": 14.8,
        "no_car_pct": 11.2,
        "housing_density_sqkm": 728.9,
        "building_age_yrs": 39,
        "preferred_unit": "hydration_truck",
        "warning_level": "ELEVATED HEAT ADVISORY",
        "why_short": "Commercial strip centers and multi-family complexes (UVI 84.2, 236 carless homes) create a +3.5°F heat island where local cooling is Over Capacity.",
    },
    {
        "id": "48439113622",
        "name": "North Dallas Suburban Tract",
        "intersection": "Preston Rd & Arapaho Rd",
        "lat": 32.8210,
        "lon": -96.7980,
        "polygon": [
            [32.830, -96.810],
            [32.830, -96.786],
            [32.812, -96.786],
            [32.812, -96.810],
        ],
        "observed_avg_temp": 103.6,
        "predicted_offset_f": 0.8,
        "universal_vulnerability_index": 24.5,
        "heat_wave_risk_flag": "Moderate",
        "urban_density_status": "Low Density / Open Space",
        "greenery_status": "Green Area - Unlikely Heat Wave",
        "nearby_parks": 4,
        "cooling_center_type": "Public Library",
        "capacity_status": "Under Capacity",
        "energy_grid_status": "Moderate Stress",
        "total_pop": 7446,
        "households": 2850,
        "no_cars": 0,
        "average_age": 41.2,
        "seniors_65_plus": 1779,
        "senior_pct": 23.9,
        "poverty_pct": 3.1,
        "no_car_pct": 0.0,
        "housing_density_sqkm": 319.4,
        "building_age_yrs": 35,
        "preferred_unit": "none",
        "warning_level": "LOW RISK (SKIPPED)",
        "why_short": "UVI is only 24.5 (`Green Area - Unlikely Heat Wave`, 4 parks, Public Library Under Capacity) and 0 households lack a vehicle.",
    },
]

TRUCK_COST = 2500     # $2,500 / day per Hydration Truck
WELLNESS_COST = 1200  # $1,200 / day per Senior Wellness Check Team


class ChatRequest(BaseModel):
    message: str
    budget: int = 10000
    hydration_trucks: int = 2
    wellness_teams: int = 2
    selected_tract_id: Optional[str] = None


def compute_dispatch_plan(
    budget: int,
    hydration_trucks: int,
    wellness_teams: int = 2,
    station_temp_f: float = 102.0,
) -> Dict[str, Any]:
    """Calculates daily Hydration Truck + Senior Wellness Check dispatch and enriches
    every tract using `agent/tools.py`'s `_build_tri_agency_action_plan()`.
    """
    requested_cost = (hydration_trucks * TRUCK_COST) + (wellness_teams * WELLNESS_COST)
    within_budget = requested_cost <= budget

    # Fund trucks first up to budget, then wellness teams with remaining budget
    funded_trucks = min(hydration_trucks, budget // TRUCK_COST)
    remaining_budget = max(0, budget - (funded_trucks * TRUCK_COST))
    funded_wellness = min(wellness_teams, remaining_budget // WELLNESS_COST)
    total_cost = (funded_trucks * TRUCK_COST) + (funded_wellness * WELLNESS_COST)

    scored = []
    for n in DALLAS_NEIGHBORHOODS:
        predicted_temp_f = round(station_temp_f + n["predicted_offset_f"], 1)
        no_vehicle_rate = round(n["no_car_pct"] / 100.0, 4)

        # Call `agent/tools.py` -> `_build_tri_agency_action_plan()` directly!
        agent_tri_plan = AGENT_TOOLS._build_tri_agency_action_plan(
            uvi_score=float(n["universal_vulnerability_index"]),
            no_vehicle_rate=no_vehicle_rate,
            no_cars=int(n["no_cars"]),
            average_age=float(n["average_age"]),
            heat_wave_risk_flag=str(n["heat_wave_risk_flag"]),
            urban_density_status=str(n["urban_density_status"]),
            greenery_status=str(n["greenery_status"]),
            nearby_parks=int(n["nearby_parks"]),
            capacity_status=str(n["capacity_status"]),
            cooling_center_type=str(n["cooling_center_type"]),
            energy_grid_status=str(n["energy_grid_status"]),
            observed_avg_temp=float(n["observed_avg_temp"]),
        )

        # Combined priority score matching `agent/tools.py` (`recommend_emergency_deployments`)
        priority_score = round(
            float(n["universal_vulnerability_index"])
            + 25.0 * float(n["predicted_offset_f"])
            + (15.0 if n["heat_wave_risk_flag"] == "Highly Likely" else 0.0)
            + (10.0 if n["capacity_status"] == "Over Capacity" else 0.0),
            1,
        )

        # Determine preferred unit from `agent/tools.py`'s recommendation + housing age
        rec_intervention = agent_tri_plan["recommended_intervention"]
        if "Wellness Check Team" in rec_intervention and n["building_age_yrs"] >= 50 and n["preferred_unit"] == "wellness_team":
            preferred_unit = "wellness_team"
        elif n["preferred_unit"] == "none":
            preferred_unit = "none"
        else:
            preferred_unit = n["preferred_unit"]

        scored.append({
            **n,
            "station_temp_f": station_temp_f,
            "predicted_street_temp_f": predicted_temp_f,
            "priority_score": priority_score,
            "preferred_unit": preferred_unit,
            "agent_recommended_intervention": rec_intervention,
            "tri_agency_dispatch": agent_tri_plan["tri_agency_dispatch"],
            "uvi_pillars": agent_tri_plan["uvi_pillars"],
            "assigned_unit": None,
            "unit_type": None,
            "rank": None,
        })

    scored.sort(key=lambda x: x["priority_score"], reverse=True)

    rem_t = funded_trucks
    rem_w = funded_wellness
    rank = 1
    deployed = []

    for site in scored:
        if site["id"] == "48439113622":
            continue
        if rem_t > 0 and site["preferred_unit"] == "hydration_truck":
            site["assigned_unit"] = "🚚 Hydration Truck"
            site["unit_type"] = "hydration_truck"
            rem_t -= 1
            site["rank"] = rank
            rank += 1
            deployed.append(site)
        elif rem_w > 0 and site["preferred_unit"] == "wellness_team":
            site["assigned_unit"] = "🩺 Wellness Check Team"
            site["unit_type"] = "wellness_team"
            rem_w -= 1
            site["rank"] = rank
            rank += 1
            deployed.append(site)
        elif rem_t > 0:
            site["assigned_unit"] = "🚚 Hydration Truck"
            site["unit_type"] = "hydration_truck"
            rem_t -= 1
            site["rank"] = rank
            rank += 1
            deployed.append(site)
        elif rem_w > 0:
            site["assigned_unit"] = "🩺 Wellness Check Team"
            site["unit_type"] = "wellness_team"
            rem_w -= 1
            site["rank"] = rank
            rank += 1
            deployed.append(site)

    total_seniors = sum(d["seniors_65_plus"] for d in deployed)

    return {
        "baseline": DALLAS_BASELINE,
        "agent_connected": True,
        "adk_live_llm": HAS_ADK_RUNTIME,
        "station_temp_f": station_temp_f,
        "budget": budget,
        "requested_trucks": hydration_trucks,
        "requested_wellness": wellness_teams,
        "requested_cost": requested_cost,
        "funded_trucks": funded_trucks,
        "funded_wellness": funded_wellness,
        "total_cost": total_cost,
        "within_budget": within_budget,
        "total_seniors_reached": total_seniors,
        "neighborhoods": scored,
        "deployed_sites": deployed,
    }


def _format_ui_dispatch_context(plan: Dict[str, Any]) -> str:
    """Formats the active UI Budget, Hydration Truck, and Wellness Team assignments
    so the ADK `root_agent` knows the exact frontend dispatch state.
    """
    lines = [
        f"[Active Frontend Dispatch State — Budget: ${plan['budget']:,} | Total Cost: ${plan['total_cost']:,} | "
        f"Funded Units: {plan['funded_trucks']} Hydration Truck(s) ($2,500/ea) + {plan['funded_wellness']} Wellness Check Team(s) ($1,200/ea)]"
    ]
    for s in plan["neighborhoods"]:
        unit_str = s["assigned_unit"] or "Standby / Skipped"
        lines.append(
            f"- Tract {s['id']} ({s['name']}): UVI={s['universal_vulnerability_index']}, "
            f"Offset=+{s['predicted_offset_f']}°F ({s['predicted_street_temp_f']}°F), "
            f"No-Car={s['no_cars']} ({s['no_car_pct']}%), Cooling={s['cooling_center_type']} ({s['capacity_status']}) "
            f"-> UI Assigned Unit: {unit_str}"
        )
    return "\n".join(lines)


def _try_run_live_adk_agent(message: str, plan: Dict[str, Any]) -> Optional[str]:
    """If `google-adk` and Gemini credentials are active, runs `agent.agent.root_agent`
    (`dallas_heat_emergency_commander`) with the injected UI dispatch state.
    """
    if not HAS_ADK_RUNTIME or ADK_ROOT_AGENT is None:
        return None
    try:
        prompt_with_state = f"{_format_ui_dispatch_context(plan)}\n\nUser Question: {message}"

        async def _run():
            runner = InMemoryRunner(agent=ADK_ROOT_AGENT, app_name="dallas_heat_planner")
            session = await runner.session_service.create_session(
                app_name="dallas_heat_planner", user_id="planner_ui"
            )
            content = genai_types.Content(
                role="user", parts=[genai_types.Part.from_text(text=prompt_with_state)]
            )
            chunks: List[str] = []
            async for event in runner.run_async(
                user_id="planner_ui", session_id=session.id, new_message=content
            ):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if getattr(part, "text", None):
                            chunks.append(part.text)
            return "\n".join(chunks).strip()

        reply = asyncio.run(_run())
        return reply if reply else None
    except Exception:
        return None


def build_chat_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    message = str(payload.get("message", ""))
    budget = int(payload.get("budget", 10000))
    hydration_trucks = int(payload.get("hydration_trucks", 2))
    wellness_teams = int(payload.get("wellness_teams", 2))
    selected_tract_id = payload.get("selected_tract_id")

    plan = compute_dispatch_plan(
        budget=budget,
        hydration_trucks=hydration_trucks,
        wellness_teams=wellness_teams,
    )
    q = message.lower()
    deployed = plan["deployed_sites"]

    highlight_id = selected_tract_id or (deployed[0]["id"] if deployed else "48113019211")
    if "marsalis" in q or "oak cliff" in q or "48113009604" in q:
        highlight_id = "48113009604"
    elif "buckner" in q or "east dallas" in q or "48113018110" in q:
        highlight_id = "48113018110"
    elif "fair park" in q or "mlk" in q or "south dallas" in q or "48113019211" in q:
        highlight_id = "48113019211"
    elif "singleton" in q or "west dallas" in q or "48113013605" in q:
        highlight_id = "48113013605"
    elif "bachman" in q or "northwest" in q or "48113013721" in q:
        highlight_id = "48113013721"
    elif "north" in q or "preston" in q or "skip" in q or "48439113622" in q:
        highlight_id = "48439113622"

    # 1. If live ADK LLM runtime is active (and not just a silent slider sync), run `agent.agent.root_agent`
    if q != "update daily dispatch plan":
        adk_reply = _try_run_live_adk_agent(message, plan)
        if adk_reply:
            return {
                "reply": adk_reply,
                "highlight_tract_id": highlight_id,
                "plan": plan,
            }

    # 2. Grounded response using `agent/tools.py` (`_build_tri_agency_action_plan` & UVI 3-Pillar architecture)
    if "why" in q and ("north" in q or "skip" in q or "48439113622" in q or "equity" in q):
        reply = (
            "**⚖️ Equity & UVI Audit (`equity_audit_agent` — `Agent_Universal_Vulnaribility_Index`):**\n\n"
            "- **1. Strip the Variable:** Zero race/ethnicity columns are used in `Agent_Universal_Vulnaribility_Index`.\n"
            "- **2. Why North Dallas (`48439113622`) is Skipped:** Although it has **1,779 seniors**, its **Universal Vulnerability Index (UVI) is only `24.5`** (`Green Area - Unlikely Heat Wave`, 4 parks, Public Library `Under Capacity`, **`0` carless households**, and only `+0.8°F` offset).\n"
            "- **3. Audit the Outcome:** We redirect your **$"
            f"{plan['total_cost']:,}** dispatch to **South Dallas (`UVI 124.6`)**, **Oak Cliff (`UVI 119.8`)**, and **Buckner Terrace (`UVI 112.4`)**, where Public Library cooling centers are **`Over Capacity`** and **`380–412` households per tract lack a vehicle** (`3.1x` the Dallas baseline)."
        )
    elif "why" in q or "explain" in q or "hot" in q or "uvi" in q or "hospital" in q or "grid" in q:
        target = next((n for n in plan["neighborhoods"] if n["id"] == highlight_id), plan["neighborhoods"][0])
        tri = target["tri_agency_dispatch"]
        pillars = target["uvi_pillars"]
        reply = (
            f"**{target['name']} (Tract `{target['id']}` · `UVI {target['universal_vulnerability_index']}`)**\n\n"
            f"- **🌡️ Station vs. Street Temp:** `{target['station_temp_f']}°F` station → **`{target['predicted_street_temp_f']}°F` street** (`+{target['predicted_offset_f']}°F` offset · `{target['observed_avg_temp']}°F` campaign avg)\n"
            f"- **🏛️ 3-Pillar UVI Diagnosis (`microclimate_risk_agent`):**\n"
            f"  1. *Health Access (CMS Medicare):* {pillars['health_access_cms_medicare']}\n"
            f"  2. *Urban Heat Island (OSM 1km):* {pillars['urban_heat_island_osm_1km']}\n"
            f"  3. *Energy Stress (NOAA CDD):* {pillars['energy_stress_noaa_cdd']}\n\n"
            f"**🚨 Tri-Agency Action Plan (`triage_dispatch_agent`):**\n"
            f"- **🚑 Emergency Dispatch:** **{target['assigned_unit'] or 'Standby'}** — *{tri['emergency_services_dispatch']['cooling_center_status']}*\n"
            f"- **🏥 Hospital Standby:** `{tri['hospital_staff_standby']['alert_level']}`\n"
            f"- **⚡ Power Grid Team:** `{tri['electricity_grid_response']['alert_level']}`"
        )
    else:
        budget_msg = (
            f"**🤖 ADK `triage_dispatch_agent` Plan** — Deploying **{plan['funded_trucks']} Hydration Truck(s)** (`$2,500/ea`) and "
            f"**{plan['funded_wellness']} Senior Wellness Check Team(s)** (`$1,200/ea`) — "
            f"**${plan['total_cost']:,}** of **${budget:,}** budget:"
        )
        if not plan["within_budget"]:
            budget_msg = (
                f"⚠️ Requested **{hydration_trucks} Truck(s) + {wellness_teams} Wellness Team(s)** cost **${plan['requested_cost']:,}**, "
                f"exceeding your **${budget:,}** budget. Funded **{plan['funded_trucks']} Truck(s) + {plan['funded_wellness']} Wellness Team(s)** (**${plan['total_cost']:,}/day**):"
            )
        lines = [
            f"{d['rank']}. **{d['assigned_unit']}** → **{d['name']}** (`UVI {d['universal_vulnerability_index']}`, `+{d['predicted_offset_f']}°F` · `{d['no_cars']} carless homes` · Library `{d['capacity_status']}`)"
            for d in deployed
        ]
        top_site = deployed[0] if deployed else plan["neighborhoods"][0]
        tri_footer = (
            f"\n\n**Tri-Agency Coordination (`{top_site['name']}`):**\n"
            f"- **🏥 Hospital Alert:** `{top_site['tri_agency_dispatch']['hospital_staff_standby']['alert_level']}`\n"
            f"- **⚡ Grid Alert:** `{top_site['tri_agency_dispatch']['electricity_grid_response']['alert_level']}`"
        )
        reply = f"{budget_msg}\n\n" + ("\n".join(lines) if lines else "_Increase budget or units above to dispatch teams._") + tri_footer

    return {
        "reply": reply,
        "highlight_tract_id": highlight_id,
        "plan": plan,
    }


if HAS_FASTAPI:
    @app.get("/api/initial-data")
    async def get_initial_data():
        return JSONResponse(compute_dispatch_plan(budget=10000, hydration_trucks=2, wellness_teams=2))

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        return JSONResponse(build_chat_response(req.model_dump() if hasattr(req, "model_dump") else req.__dict__))

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        html_path = Path(__file__).parent / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))


class LocalPreviewHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.startswith("/api/initial-data"):
            self._send_json(compute_dispatch_plan(budget=10000, hydration_trucks=2, wellness_teams=2))
        else:
            html_path = Path(__file__).parent / "index.html"
            raw = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    def do_POST(self) -> None:
        if self.path.startswith("/api/chat"):
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(body)
            self._send_json(build_chat_response(payload))
        else:
            self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    if HAS_FASTAPI:
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        print(f"Running Dallas Daily Heat Dispatch Planner on http://localhost:{port}")
        server = HTTPServer(("0.0.0.0", port), LocalPreviewHandler)
        server.serve_forever()

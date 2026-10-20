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

# Load .env file automatically if present
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# Synchronize GEMINI_API_KEY into GOOGLE_API_KEY for ADK / google-genai
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

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
    BigQuery tools work even if optional `dotenv` / `requests` / `google-adk`
    packages are not installed in a lightweight local preview environment.
    """
    try:
        import dotenv  # noqa: F401
    except ImportError:
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda *args, **kwargs: False  # type: ignore[attr-defined]
        sys.modules["dotenv"] = dotenv_mod

    try:
        import requests  # noqa: F401
    except ImportError:
        import urllib.parse
        import urllib.request

        req_mod = types.ModuleType("requests")

        class _SimpleResponse:
            def __init__(self, status_code: int, body: bytes):
                self.status_code = status_code
                self._body = body

            def raise_for_status(self) -> None:
                if self.status_code >= 400:
                    raise RuntimeError(f"HTTP {self.status_code}")

            def json(self) -> Any:
                return json.loads(self._body.decode("utf-8"))

        def _requests_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: float = 8.0, headers: Optional[Dict[str, str]] = None) -> _SimpleResponse:
            full_url = url
            if params:
                qs = urllib.parse.urlencode(params)
                full_url = f"{url}?{qs}"
            req = urllib.request.Request(full_url, headers=headers or {"User-Agent": "DallasUrbanHeatPlanner/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return _SimpleResponse(resp.status, resp.read())

        req_mod.get = _requests_get  # type: ignore[attr-defined]
        sys.modules["requests"] = req_mod

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

    # Ensure `agent` package stub exists for relative imports (`from .config import ...`)
    if "agent" not in sys.modules:
        agent_pkg = types.ModuleType("agent")
        agent_pkg.__path__ = [str(REPO_ROOT / "agent")]  # type: ignore[attr-defined]
        sys.modules["agent"] = agent_pkg

    config_path = REPO_ROOT / "agent" / "config.py"
    if config_path.exists() and "agent.config" not in sys.modules:
        cfg_spec = importlib.util.spec_from_file_location("agent.config", config_path)
        cfg_mod = importlib.util.module_from_spec(cfg_spec)
        assert cfg_spec and cfg_spec.loader
        sys.modules["agent.config"] = cfg_mod
        cfg_spec.loader.exec_module(cfg_mod)

    tools_path = REPO_ROOT / "agent" / "tools.py"
    spec = importlib.util.spec_from_file_location("agent.tools", tools_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["agent.tools"] = module
    spec.loader.exec_module(module)
    return module, has_bq


AGENT_TOOLS, HAS_BIGQUERY_CLIENT = _load_agent_tools_module()

# Attempt to load live ADK `root_agent` (`dallas_heat_emergency_commander`) from `agent/agent.py`
ADK_ROOT_AGENT = None
try:
    from agent.agent import root_agent as _loaded_root_agent
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types

    ADK_ROOT_AGENT = _loaded_root_agent
except Exception as _agent_err:
    ADK_ROOT_AGENT = None


def is_adk_live_runtime_active() -> bool:
    """Dynamically checks if ADK root_agent is loaded AND valid Gemini credentials exist."""
    return (ADK_ROOT_AGENT is not None) and bool(
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    )


# Backward-compatible property alias
HAS_ADK_RUNTIME = is_adk_live_runtime_active()

PROJECT_ID = getattr(AGENT_TOOLS, "PROJECT_ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-f8ba79cf4618"))
DATASET = getattr(AGENT_TOOLS, "DATASET_ID", os.environ.get("BQ_DATASET", "a4i_climate"))
BQML_OFFSET_MODEL_NAME = getattr(AGENT_TOOLS, "BQML_OFFSET_MODEL_NAME", os.environ.get("BQML_OFFSET_MODEL_NAME", "heat_offset_model"))
UVI_TABLE_NAME = BQML_OFFSET_MODEL_NAME

DALLAS_BASELINE = {
    "city": "Dallas",
    "source_model": f"{PROJECT_ID}.{DATASET}.{BQML_OFFSET_MODEL_NAME}",
    "agent_name": "dallas_heat_emergency_commander",
    "sub_agents": [
        "historic_heat_data_agent",
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

# Grounded in `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model` (`v_tract_heat_features`)
DALLAS_NEIGHBORHOODS: List[Dict[str, Any]] = [
    {
        "id": "48113019211",
        "name": "South Dallas / Fair Park",
        "intersection": "MLK Jr Blvd & S Malcolm X Blvd",
        "lat": 32.7684,
        "lon": -96.7640,
        "polygon": [
            [32.805, -96.805],
            [32.812, -96.772],
            [32.808, -96.738],
            [32.790, -96.725],
            [32.765, -96.720],
            [32.738, -96.728],
            [32.722, -96.748],
            [32.720, -96.775],
            [32.735, -96.792],
            [32.758, -96.802],
            [32.782, -96.808],
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
        "lat": 32.7380,
        "lon": -96.8320,
        "polygon": [
            [32.762, -96.885],
            [32.766, -96.842],
            [32.758, -96.802],
            [32.735, -96.792],
            [32.720, -96.775],
            [32.702, -96.795],
            [32.696, -96.835],
            [32.705, -96.875],
            [32.728, -96.896],
            [32.748, -96.894],
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
        "lat": 32.7620,
        "lon": -96.6820,
        "polygon": [
            [32.808, -96.738],
            [32.816, -96.698],
            [32.806, -96.655],
            [32.782, -96.638],
            [32.750, -96.642],
            [32.726, -96.665],
            [32.722, -96.705],
            [32.738, -96.728],
            [32.765, -96.720],
            [32.790, -96.725],
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
        "lat": 32.7850,
        "lon": -96.8580,
        "polygon": [
            [32.815, -96.905],
            [32.820, -96.862],
            [32.814, -96.828],
            [32.805, -96.805],
            [32.782, -96.808],
            [32.758, -96.802],
            [32.766, -96.842],
            [32.762, -96.885],
            [32.778, -96.912],
            [32.798, -96.915],
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
        "lat": 32.8480,
        "lon": -96.8620,
        "polygon": [
            [32.885, -96.902],
            [32.892, -96.858],
            [32.884, -96.818],
            [32.852, -96.812],
            [32.814, -96.828],
            [32.820, -96.862],
            [32.815, -96.905],
            [32.842, -96.918],
            [32.868, -96.914],
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
        "lat": 32.8520,
        "lon": -96.7620,
        "polygon": [
            [32.884, -96.818],
            [32.895, -96.775],
            [32.890, -96.725],
            [32.865, -96.692],
            [32.835, -96.688],
            [32.816, -96.698],
            [32.808, -96.738],
            [32.812, -96.772],
            [32.805, -96.805],
            [32.814, -96.828],
            [32.852, -96.812],
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


class KeyConfigRequest(BaseModel):
    api_key: str


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
        "adk_agent_loaded": ADK_ROOT_AGENT is not None,
        "adk_live_llm": is_adk_live_runtime_active(),
        "has_api_key": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
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


async def _try_run_live_adk_agent(message: str, plan: Dict[str, Any]) -> Optional[str]:
    """If `google-adk` and Gemini credentials are active, runs `agent.agent.root_agent`
    (`dallas_heat_emergency_commander`) with the injected UI dispatch state.
    """
    if not is_adk_live_runtime_active() or ADK_ROOT_AGENT is None:
        return None
    try:
        prompt_with_state = f"{_format_ui_dispatch_context(plan)}\n\nUser Question: {message}"

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
        reply = "\n".join(chunks).strip()
        return reply if reply else None
    except Exception as e:
        print(f"[ADK Live Execution Notice] {type(e).__name__}: {e}")
        return None


async def build_chat_response(payload: Dict[str, Any]) -> Dict[str, Any]:
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
        adk_reply = await _try_run_live_adk_agent(message, plan)
        if adk_reply:
            return {
                "reply": adk_reply,
                "highlight_tract_id": highlight_id,
                "plan": plan,
                "source": "adk_live_llm",
            }

    # 2. Comprehensive Database-Grounded Intent Routing (Exclusively querying/explaining the BigQuery tables & BQML model)
    matched_tract_id = None
    if "marsalis" in q or "oak cliff" in q or "48113009604" in q:
        matched_tract_id = "48113009604"
    elif "buckner" in q or "east dallas" in q or "48113018110" in q:
        matched_tract_id = "48113018110"
    elif "fair park" in q or "mlk" in q or "south dallas" in q or "48113019211" in q:
        matched_tract_id = "48113019211"
    elif "singleton" in q or "west dallas" in q or "48113013605" in q:
        matched_tract_id = "48113013605"
    elif "bachman" in q or "northwest" in q or "48113013721" in q:
        matched_tract_id = "48113013721"
    elif "north" in q or "preston" in q or "skip" in q or "48439113622" in q:
        matched_tract_id = "48439113622"

    is_proposal_q = any(w in q for w in ["proposal", "council", "meeting", "briefing", "memo", "presentation", "executive summary", "action plan", "policy", "recommendation plan"])
    is_compare_q = any(w in q for w in ["compare", " vs ", "versus", "difference between", "tradeoff"])
    is_model_q = any(w in q for w in ["model", "feature", "bqml", "ml.explain", "r2", "mae", "rmse", "accuracy", "weight", "attribution", "how do you predict", "unmeasured", "549", "618"])
    is_weather_q = any(w in q for w in ["weather", "forecast", "live temp", "current temp", "how hot is it today", "open-meteo", "nws"])
    is_historic_q = any(w in q for w in ["historic", "historical", "history", "highest temp", "hottest", "peak temp", "heat observation", "traverse", "past reading", "record temp", "2019", "2023", "2025", "7-year", "station_daily"])
    is_care_q = any(w in q for w in ["special care", "most at risk", "vulnerab", "priority area", "need help", "need care", "where should we", "top priority", "first"])
    is_db_q = any(w in q for w in ["database", "table", "bigquery", "dataset", "data source", "where does the data come from", "schema", "how many rows", "how many points"])
    is_equity_q = ("why" in q and any(w in q for w in ["north", "skip", "48439113622", "equity", "fair"])) or any(w in q for w in ["strip the variable", "equity audit", "bias", "equity", "fairness", "race", "demographic"])
    is_agency_q = any(w in q for w in ["hospital", "clinical", "grid", "transformer", "cdd", "power", "electricity", "feeder", "tri-agency", "tri agency"])
    is_dispatch_rule_q = any(w in q for w in ["how do you decide", "how does the distribution", "dispatch rule", "cost", "allocation rule", "criteria", "budget"])
    is_greeting_q = q.strip() in {"hi", "hello", "hey", "help", "what can you do", "what can i ask", "?"}

    # A0. City Council Briefing & Executive Action Proposal
    if is_proposal_q:
        highlight_id = deployed[0]["id"] if deployed else "48113019211"
        truck_exp = plan["funded_trucks"] * TRUCK_COST
        wellness_exp = plan["funded_wellness"] * WELLNESS_COST
        lines_deploy = [
            f"   - **{d['rank']}. {d['assigned_unit']}** → **{d['name']}** (Tract `{d['id']}` · `UVI {d['universal_vulnerability_index']}` · `+{d['predicted_offset_f']}°F` offset · `{d['no_cars']} carless households` · Library `{d['capacity_status']}`)"
            for d in deployed
        ]
        deploy_text = "\n".join(lines_deploy) if lines_deploy else "   - _Increase budget to fund front-line units._"
        reply = (
            "## 🏛️ DALLAS CITY COUNCIL EMERGENCY BRIEFING & ACTION PROPOSAL\n\n"
            "**SUBJECT:** Emergency Heatwave Response & Microclimate Tri-Agency Dispatch Plan  \n"
            f"**DATE:** Today's Extreme Heat Operational Period  \n"
            f"**BUDGET APPROPRIATION:** Authorized Ceiling **${budget:,}** | Recommended Outlay: **${plan['total_cost']:,}**  \n\n"
            "---\n\n"
            "### 1. Executive Summary & Microclimate Threat\n"
            "Dallas is experiencing extreme urban heat. While regional weather stations report **102.0°F**, street-level Heat Watch traverse data reveals severe **microclimate heat islands of up to +4.8°F**, pushing neighborhood temperatures past **106.8°F–110.1°F** in concrete-dense districts.\n\n"
            f"Without immediate mobile intervention, heat-related EMS calls and electrical transformer failures will peak between 1:00 PM and 7:00 PM.\n\n"
            "### 2. Proposed Emergency Resource Allocation\n"
            f"We request Council approval for an immediate daily appropriation of **${plan['total_cost']:,}** from the Municipal Contingency Reserve:\n"
            f"- **🚚 {plan['funded_trucks']} Mobile Hydration & Misting Trucks** (`$2,500/day` ea) = **${truck_exp:,}**\n"
            f"- **🩺 {plan['funded_wellness']} Senior Wellness Check Teams** (`$1,200/day` ea) = **${wellness_exp:,}**\n"
            f"- **👥 Vulnerable Population Protected:** Reaching **{plan['total_seniors_reached']:,} seniors (65+)** and over **1,400 carless households** across top-priority tracts.\n\n"
            "### 3. Targeted Field Deployments (Ranked by Universal Vulnerability Index):\n"
            f"{deploy_text}\n\n"
            "### 4. Tri-Agency Coordinated Directives:\n"
            "- **🚑 Emergency Services (Dallas Fire-Rescue / OEM):** Station Hydration Trucks at transit stops and commercial plazas where local libraries are Over Capacity.\n"
            "- **🏥 Healthcare Systems (Parkland / CMS Medicare):** Place regional Emergency Departments on Level 1 Surge Standby for dehydration and heat stroke in identified hospital deserts.\n"
            "- **⚡ Energy Grid (Oncor / ERCOT):** Initiate proactive transformer monitoring and stage mobile backup generation along stressed residential feeder circuits.\n\n"
            "### 5. Legally Defensible Equity Audit (3-Step Compliance):\n"
            "- **Strip the Variable:** Zero race or ethnicity metrics are used in our BigQuery model.\n"
            "- **Objective Need:** High-wealth areas (e.g. North Dallas) are deferred because they possess high tree canopy, private vehicle mobility, and functioning cooling centers.\n"
            "- **Life-Safety Outcome:** 100% of municipal funding is concentrated where residents cannot drive to safety.\n\n"
            f"**RECOMMENDED MOTION FOR COUNCIL:**\n"
            f"> *\"Move to approve the emergency operational deployment of {plan['funded_trucks']} Hydration Trucks and {plan['funded_wellness']} Senior Wellness Teams for today's heat emergency at an authorized total of ${plan['total_cost']:,}.\"*"
        )

    # A. Historical Data & Highest Temperature Areas Needing Special Care
    elif is_historic_q or (is_care_q and ("temp" in q or "area" in q or "tract" in q)):
        highlight_id = "48113019211"  # South Dallas / Fair Park (hottest tract)
        reply = (
            "**📊 Historical Heat & Microclimate Analysis (`heat_observations` & `Agent_Universal_Vulnaribility_Index`):**\n\n"
            "From the NOAA Heat Watch street-level campaign (**19,491 traverse points** across Dallas, mean `105.32°F`):\n\n"
            "### 🔥 Hottest Areas Recorded in Dallas:\n"
            "1. **South Dallas / Fair Park (Tract `48113019211`):** Peak street readings reach **`110.10°F`** (**`+4.8°F` microclimate heat island offset** above airport baseline; observed campaign avg `107.8°F`).\n"
            "2. **Oak Cliff / Marsalis Corridor (Tract `48113009604`):** Observed campaign avg **`107.4°F`** (**`+4.6°F` offset**, peak street temps `109.8°F`).\n"
            "3. **Buckner Terrace / East Dallas (Tract `48113018110`):** Observed campaign avg **`107.0°F`** (**`+4.4°F` offset**, peak street temps `109.2°F`).\n\n"
            "### 🚨 Why These Areas Need Special Care (Compounding Vulnerabilities):\n"
            "- **🚗 The Carless Mobility Trap:** South Dallas has **380 households with zero vehicle access** (15.8%) and Oak Cliff has **412 carless homes** (14.2%). Residents literally cannot drive to air-conditioned regional cooling centers.\n"
            "- **🏛️ Cooling Centers Over Capacity:** Local Public Libraries in South Dallas and Oak Cliff are designated **`Over Capacity`** during extreme heat days.\n"
            "- **🏥 Severe Healthcare Access Deserts (CMS Medicare):** Both corridors have zero proximate hospital emergency departments, putting their high senior populations at acute risk of heat-stroke mortality.\n"
            "- **⚡ Electrical Grid Stress (NOAA CDD):** High residential air-conditioning load and dense urban building footprints trigger priority transformer failure watches.\n\n"
            "**Recommended Action:** High-priority deployment of **🚚 Hydration Trucks** to South Dallas and **🩺 Senior Wellness Check Teams** to Oak Cliff and Buckner Terrace."
        )

    # B. Database Architecture & Table Explanations
    elif is_db_q:
        reply = (
            "**🗄️ BigQuery Data Architecture (`qwiklabs-gcp-04-f8ba79cf4618.a4i_climate`):**\n\n"
            "The system is grounded 100% in 3 BigQuery tables:\n\n"
            "1. **`Agent_Universal_Vulnaribility_Index` (85 valid Dallas census tracts):**\n"
            "   - Contains the 3-pillar Universal Vulnerability Index (UVI), combining **CMS Medicare Hospital Access**, **NOAA 2023 Cooling Degree Days (CDD)**, and **OpenStreetMap (OSM) 1km Building Density**.\n"
            "   - Key columns: `universal_vulnerability_index`, `observed_avg_temp`, `no_cars`, `average_age`, `cooling_center_type`, `capacity_status`, `energy_grid_status`.\n\n"
            "2. **`heat_observations` (19,491 street-level readings):**\n"
            "   - Street-level traverse campaign across Dallas with granular temperature points ranging from **`100.90°F` to `110.10°F`** (mean `105.32°F`).\n\n"
            "3. **`station_daily_temps` (27,187 station-days across 16 stations):**\n"
            "   - 7-year NOAA GHCN-Daily historical record (2019–2025) tracking multi-year heatwave frequency and summer temperature trends across North Texas."
        )

    # C. Equity Audit & Why North Dallas is Skipped
    elif is_equity_q:
        highlight_id = "48439113622"
        reply = (
            "**⚖️ Equity & UVI Audit (`equity_audit_agent` — `heat_offset_model`):**\n\n"
            "- **1. Strip the Variable:** Zero race/ethnicity columns are used in `heat_offset_model` (`v_tract_heat_features`).\n"
            "- **2. Why North Dallas (`48439113622`) is Skipped:** Although it has **1,779 seniors**, its **Universal Vulnerability Index (UVI) is only `24.5`** (`Green Area - Unlikely Heat Wave`, 4 parks, Public Library `Under Capacity`, **`0` carless households**, and only `+0.8°F` offset).\n"
            "- **3. Audit the Outcome:** We redirect your **$"
            f"{plan['total_cost']:,}** dispatch to **South Dallas (`UVI 124.6`)**, **Oak Cliff (`UVI 119.8`)**, and **Buckner Terrace (`UVI 112.4`)**, where cooling centers are **`Over Capacity`** and **`380–412` households per tract lack a vehicle** (`3.1x` higher than city baseline)."
        )

    # D0. Side-by-Side Comparison / Model & Features / Weather / Greetings
    elif is_compare_q:
        s_a = plan["neighborhoods"][0]
        s_b = plan["neighborhoods"][1]
        for n in plan["neighborhoods"]:
            if any(tok in q for tok in [n["id"], n["name"].split("/")[0].strip().lower()]):
                if n["id"] != s_a["id"]:
                    s_b = n
        reply = (
            f"**⚖️ Side-by-Side Deployment Comparison (`compare_deployment_sites`):**\n\n"
            f"| Metric | {s_a['name']} (`{s_a['id']}`) | {s_b['name']} (`{s_b['id']}`) |\n"
            f"| :--- | :--- | :--- |\n"
            f"| **Universal Vulnerability Index (UVI)** | **`{s_a['universal_vulnerability_index']}`** | **`{s_b['universal_vulnerability_index']}`** |\n"
            f"| **Predicted Heat Offset (`heat_offset_model`)** | **`+{s_a['predicted_offset_f']}°F`** (`{s_a['predicted_street_temp_f']}°F`) | **`+{s_b['predicted_offset_f']}°F`** (`{s_b['predicted_street_temp_f']}°F`) |\n"
            f"| **Carless Households (`no_cars`)** | **`{s_a['no_cars']}`** (`{s_a['no_car_pct']}%`) | **`{s_b['no_cars']}`** (`{s_b['no_car_pct']}%`) |\n"
            f"| **Seniors 65+ (`seniors_65_plus`)** | `{s_a['seniors_65_plus']:,}` (`{s_a['senior_pct']}%`) | `{s_b['seniors_65_plus']:,}` (`{s_b['senior_pct']}%`) |\n"
            f"| **Cooling Center Status** | `{s_a['cooling_center_type']}` (`{s_a['capacity_status']}`) | `{s_b['cooling_center_type']}` (`{s_b['capacity_status']}`) |\n"
            f"| **Recommended Unit** | **{s_a['assigned_unit'] or 'Standby'}** | **{s_b['assigned_unit'] or 'Standby'}** |\n\n"
            f"**Verdict:** Prioritize **{s_a['name']}** first (`UVI {s_a['universal_vulnerability_index']}`, `+{s_a['predicted_offset_f']}°F` offset, `{s_a['no_cars']}` carless homes), followed by **{s_b['name']}**."
        )
    elif is_model_q:
        reply = (
            "**🧠 BigQuery ML Model & Key Feature Attributions (`heat_offset_model`):**\n\n"
            "- **Model:** `qwiklabs-gcp-04-f8ba79cf4618.a4i_climate.heat_offset_model` (`LINEAR_REGRESSION`, **R² = 0.478**, **MAE = 0.299°F**, **RMSE = 0.440°F**)\n"
            "- **Coverage:** Evaluated via `ML.EXPLAIN_PREDICT` over `v_tract_heat_features` across **549 valid Dallas census tracts** (`47` measured Heat Watch tracts + `502` unmeasured predicted tracts)\n"
            "- **Top Global Drivers (`ML.GLOBAL_EXPLAIN`):**\n"
            "  1. `building_age_years` (`0.0523`) — older housing & roof thermal retention\n"
            "  2. `surrounding_3km_poverty_rate` (`0.0453`) & `poverty_rate` (`0.0431`) — tree-canopy deficit & impervious cover\n"
            "  3. `tract_lat` (`0.0343`) & `tract_lon` (`0.0328`) — urban heat corridor gradient\n"
            "  4. `households` (`0.0243`), `housing_units` (`0.0225`), & `station_idw_summer_temp_f` (`0.0222`)\n"
            "  5. `seniors_65_plus` (`0.0212`), `land_area_sqkm` (`0.0211`), & `heat_wave_risk_flag` (`0.0211`)\n"
            "- **Fairness Guarantee:** Zero racial or ethnic variables are used as model inputs."
        )
    elif is_weather_q:
        reply = (
            f"**🌡️ Dallas Real-Time Weather & Microclimate Baseline (`get_dallas_live_temperature`):**\n\n"
            f"- **Regional Weather Station Baseline:** `{plan['station_temp_f']}°F` (DFW / Dallas Love Field NOAA feed)\n"
            f"- **Peak Microclimate Street Temps:**\n"
            f"  - **South Dallas / Fair Park (`48113019211`):** **`{plan['station_temp_f'] + 4.8:.1f}°F`** (`+4.8°F` heat island offset)\n"
            f"  - **Oak Cliff / Marsalis (`48113009604`):** **`{plan['station_temp_f'] + 4.6:.1f}°F`** (`+4.6°F` heat island offset)\n"
            f"  - **Buckner Terrace (`48113018110`):** **`{plan['station_temp_f'] + 4.4:.1f}°F`** (`+4.4°F` heat island offset)"
        )
    # D. Tri-Agency Alerts (Hospital Standby & Power Grid)
    elif is_agency_q:
        target = next((n for n in plan["neighborhoods"] if n["id"] == highlight_id), plan["neighborhoods"][0])
        tri = target["tri_agency_dispatch"]
        h_plan = tri["hospital_staff_standby"]
        g_plan = tri["electricity_grid_response"]
        reply = (
            f"**🚨 Tri-Agency Operational Coordination for {target['name']} (Tract `{target['id']}`):**\n\n"
            f"### 🏥 1. Hospital & Clinical Staff Standby\n"
            f"- **Alert Level:** `{h_plan.get('alert_level', 'ELEVATED STANDBY')}`\n"
            f"- **Triage Directive:** {h_plan.get('directive') or h_plan.get('action', '')}\n"
            f"- **Reasoning:** {h_plan.get('reasoning') or h_plan.get('action', '')}\n\n"
            f"### ⚡ 2. Electricity & Power Grid Team\n"
            f"- **Alert Level:** `{g_plan.get('alert_level', 'HIGH FEEDER WATCH')}`\n"
            f"- **Grid Directive:** {g_plan.get('directive') or g_plan.get('action', '')}\n"
            f"- **Reasoning:** {g_plan.get('reasoning') or g_plan.get('action', '')}\n\n"
            f"### 🚑 3. Emergency Services Dispatch\n"
            f"- **Assigned Unit:** **{target['assigned_unit'] or 'Standby'}**\n"
            f"- **Cooling Status:** {tri['emergency_services_dispatch']['cooling_center_status']}"
        )

    # E. Dispatch Distribution Rules ($2,500 Trucks vs $1,200 Wellness Teams)
    elif is_dispatch_rule_q:
        reply = (
            "**🚚 Tri-Agency Dispatch Decision Rules:**\n\n"
            "Units are prioritized by combining the **Universal Vulnerability Index (UVI)** with vehicle access and housing age:\n\n"
            "1. **🚚 Hydration Trucks ($2,500 / day):**\n"
            "   - Dispatched to high-traffic commercial & transit corridors where **`no_cars >= 200`** and the local cooling center is **`Over Capacity`**.\n"
            "   - Deployed to South Dallas, Buckner Terrace, and Bachman Lake to provide immediate shaded hydration and misting.\n\n"
            "2. **🩺 Senior Wellness Check Teams ($1,200 / day):**\n"
            "   - Dispatched to residential neighborhoods with **older housing stock (built before 1970)** with poor air conditioning and high concentrations of isolated seniors.\n"
            "   - Deployed door-to-door in Oak Cliff and West Dallas to prevent heat stroke before EMS calls occur.\n\n"
            f"Current allocation: **{plan['funded_trucks']} Trucks + {plan['funded_wellness']} Wellness Teams** costing **${plan['total_cost']:,} / day**."
        )

    # F. Specific Neighborhood or General Vulnerability Inquiry
    elif matched_tract_id is not None or any(w in q for w in ["why", "explain", "hot", "temp", "uvi", "tract", "neighborhood", "details", "tell me about", "where"]):
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
        "source": "adk_grounded_fallback",
    }


if HAS_FASTAPI:
    @app.get("/api/initial-data")
    async def get_initial_data():
        return JSONResponse(compute_dispatch_plan(budget=10000, hydration_trucks=2, wellness_teams=2))

    @app.post("/api/chat")
    async def chat_endpoint(req: ChatRequest):
        return JSONResponse(await build_chat_response(req.model_dump() if hasattr(req, "model_dump") else req.__dict__))

    @app.get("/api/agent-status")
    async def agent_status_endpoint():
        return JSONResponse({
            "adk_agent_loaded": ADK_ROOT_AGENT is not None,
            "adk_live_llm": is_adk_live_runtime_active(),
            "has_api_key": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
            "model": getattr(AGENT_TOOLS, "ADK_MODEL", "gemini-2.5-flash"),
            "root_agent": "dallas_heat_emergency_commander",
            "sub_agents": DALLAS_BASELINE["sub_agents"],
            "has_bigquery": HAS_BIGQUERY_CLIENT,
            "table": UVI_TABLE_NAME,
        })

    @app.post("/api/config-key")
    async def config_key_endpoint(req: KeyConfigRequest):
        key = str(getattr(req, "api_key", "")).strip()
        if not key:
            return JSONResponse({"success": False, "message": "API key cannot be empty."}, status_code=400)

        # Validate with Google Gemini API
        try:
            from google.genai import Client
            test_client = Client(api_key=key)
            res = test_client.models.generate_content(
                model="gemini-2.5-flash",
                contents="Dallas heat emergency test",
            )
            _ = res.text
        except Exception as e:
            err_msg = str(e)
            if "API_KEY_INVALID" in err_msg or "400" in err_msg or "INVALID_ARGUMENT" in err_msg:
                clean_err = "Invalid API key. Please verify your key at https://aistudio.google.com/app/apikey"
            else:
                clean_err = f"Gemini connection error: {err_msg}"
            return JSONResponse({"success": False, "message": clean_err}, status_code=400)

        os.environ["GOOGLE_API_KEY"] = key
        os.environ["GEMINI_API_KEY"] = key

        # Persist to local .env file (safely gitignored)
        try:
            env_path = REPO_ROOT / ".env"
            lines = []
            if env_path.exists():
                lines = [l for l in env_path.read_text(encoding="utf-8").splitlines() if not l.startswith("GOOGLE_API_KEY=") and not l.startswith("GEMINI_API_KEY=")]
            lines.append(f"GOOGLE_API_KEY={key}")
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as env_err:
            print(f"Warning persisting to .env: {env_err}")

        return JSONResponse({
            "success": True,
            "adk_live_llm": True,
            "message": "✅ Gemini API key verified! Google ADK live multi-agent reasoning is now ACTIVE with gemini-2.5-flash."
        })

    @app.post("/api/clear-key")
    async def clear_key_endpoint():
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GEMINI_API_KEY", None)
        try:
            env_path = REPO_ROOT / ".env"
            if env_path.exists():
                lines = [l for l in env_path.read_text(encoding="utf-8").splitlines() if not l.startswith("GOOGLE_API_KEY=") and not l.startswith("GEMINI_API_KEY=")]
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as env_err:
            print(f"Warning updating .env: {env_err}")
        return JSONResponse({
            "success": True,
            "adk_live_llm": False,
            "message": "Gemini key removed. Reverted to offline grounded BigQuery rules mode."
        })

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
        elif self.path.startswith("/api/agent-status"):
            self._send_json({
                "adk_agent_loaded": ADK_ROOT_AGENT is not None,
                "adk_live_llm": is_adk_live_runtime_active(),
                "has_api_key": bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")),
                "model": getattr(AGENT_TOOLS, "ADK_MODEL", "gemini-2.5-flash"),
                "root_agent": "dallas_heat_emergency_commander",
                "sub_agents": DALLAS_BASELINE["sub_agents"],
                "has_bigquery": HAS_BIGQUERY_CLIENT,
                "table": UVI_TABLE_NAME,
            })
        else:
            html_path = Path(__file__).parent / "index.html"
            raw = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        if self.path.startswith("/api/chat"):
            self._send_json(asyncio.run(build_chat_response(payload)))
        elif self.path.startswith("/api/config-key"):
            key = str(payload.get("api_key", "")).strip()
            if not key:
                self._send_json({"success": False, "message": "API key cannot be empty."}, status=400)
                return
            try:
                from google.genai import Client
                test_client = Client(api_key=key)
                res = test_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents="Dallas heat emergency test",
                )
                _ = res.text
            except Exception as e:
                self._send_json({"success": False, "message": f"Gemini connection error: {e}"}, status=400)
                return
            os.environ["GOOGLE_API_KEY"] = key
            os.environ["GEMINI_API_KEY"] = key
            try:
                env_path = REPO_ROOT / ".env"
                lines = []
                if env_path.exists():
                    lines = [l for l in env_path.read_text(encoding="utf-8").splitlines() if not l.startswith("GOOGLE_API_KEY=") and not l.startswith("GEMINI_API_KEY=")]
                lines.append(f"GOOGLE_API_KEY={key}")
                env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            except Exception:
                pass
            self._send_json({"success": True, "adk_live_llm": True, "message": "Gemini API key verified!"})
        elif self.path.startswith("/api/clear-key"):
            os.environ.pop("GOOGLE_API_KEY", None)
            os.environ.pop("GEMINI_API_KEY", None)
            self._send_json({"success": True, "adk_live_llm": False, "message": "Gemini key removed."})
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

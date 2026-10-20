import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel

    HAS_FASTAPI = True
    app = FastAPI(title="Dallas Daily Heat Dispatch Planner")
except ImportError:
    HAS_FASTAPI = False
    app = None
    BaseModel = object  # type: ignore

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-f8ba79cf4618")
DATASET = os.environ.get("BQ_DATASET", "a4i_climate")

DALLAS_BASELINE = {
    "city": "Dallas",
    "station_name": "DFW Regional Weather Station (GHCN)",
    "station_temp_f": 102.0,
    "campaign_mean_f": 105.3,
    "campaign_max_f": 110.1,
    "intra_city_spread_f": 9.2,
    "avg_days_over_100f_per_year": 26,
    "city_avg_poverty_pct": 14.2,
    "city_avg_no_car_pct": 6.8,
    "city_avg_senior_pct": 11.4,
}

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
        "predicted_offset_f": 4.8,
        "total_pop": 6142,
        "seniors_65_plus": 1322,
        "senior_pct": 21.5,
        "poverty_pct": 24.3,
        "no_car_pct": 18.9,
        "housing_density_sqkm": 958.7,
        "building_age_yrs": 54,
        "preferred_unit": "hydration_truck",
        "warning_level": "EXTREME HEAT WARNING",
        "why_short": "Dense unshaded parking lots (959 units/km²) and 54-year-old buildings trap afternoon heat where 18.9% of homes have no car.",
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
        "predicted_offset_f": 4.6,
        "total_pop": 4433,
        "seniors_65_plus": 1485,
        "senior_pct": 33.5,
        "poverty_pct": 15.5,
        "no_car_pct": 22.7,
        "housing_density_sqkm": 1062.0,
        "building_age_yrs": 53,
        "preferred_unit": "wellness_team",
        "warning_level": "EXTREME HEAT WARNING",
        "why_short": "High multi-family density (1,062 units/km²) and 53-year-old structures retain heat in a tract with 33.5% seniors and 22.7% carless homes.",
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
        "predicted_offset_f": 4.4,
        "total_pop": 5146,
        "seniors_65_plus": 1749,
        "senior_pct": 34.0,
        "poverty_pct": 16.2,
        "no_car_pct": 21.4,
        "housing_density_sqkm": 804.0,
        "building_age_yrs": 44,
        "preferred_unit": "hydration_truck",
        "warning_level": "HIGH HEAT WARNING",
        "why_short": "Wide asphalt transit corridor with zero tree/water buffer and 1,749 seniors (34% of residents) relying on unshaded bus stops.",
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
        "predicted_offset_f": 3.9,
        "total_pop": 5339,
        "seniors_65_plus": 1350,
        "senior_pct": 25.3,
        "poverty_pct": 19.5,
        "no_car_pct": 14.7,
        "housing_density_sqkm": 706.1,
        "building_age_yrs": 54,
        "preferred_unit": "wellness_team",
        "warning_level": "HIGH HEAT WARNING",
        "why_short": "Industrial pavement and 54-year-old housing stock drive +3.9°F afternoon heat retention across 1,350 older adults.",
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
        "predicted_offset_f": 3.5,
        "total_pop": 5699,
        "seniors_65_plus": 1298,
        "senior_pct": 22.8,
        "poverty_pct": 14.8,
        "no_car_pct": 11.2,
        "housing_density_sqkm": 728.9,
        "building_age_yrs": 39,
        "preferred_unit": "hydration_truck",
        "warning_level": "ELEVATED HEAT ADVISORY",
        "why_short": "Dense commercial strip centers and apartment complexes (729 units/km²) create a +3.5°F heat island along pedestrian transit stops.",
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
        "predicted_offset_f": 0.8,
        "total_pop": 7446,
        "seniors_65_plus": 1779,
        "senior_pct": 23.9,
        "poverty_pct": 3.1,
        "no_car_pct": 0.0,
        "housing_density_sqkm": 319.4,
        "building_age_yrs": 35,
        "preferred_unit": "none",
        "warning_level": "LOW RISK (SKIPPED)",
        "why_short": "Only +0.8°F above station due to low housing density (319/km²) and tree canopy; 100% of households own vehicles and poverty is 3.1%.",
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
    """Calculates daily Hydration Truck + Senior Wellness Check dispatch from the sliders within the user's budget."""
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
        priority_score = round(
            (n["predicted_offset_f"] * 12.0)
            + (n["no_car_pct"] * 1.4)
            + (n["senior_pct"] * 0.9)
            + (n["poverty_pct"] * 1.1),
            1,
        )
        scored.append({
            **n,
            "station_temp_f": station_temp_f,
            "predicted_street_temp_f": predicted_temp_f,
            "priority_score": priority_score,
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
    elif "north" in q or "preston" in q or "skip" in q or "48439113622" in q:
        highlight_id = "48439113622"

    if "why" in q and ("north" in q or "skip" in q or "48439113622" in q):
        reply = (
            "**Why North Dallas (`48439113622`) is skipped:**\n\n"
            "Although it has **1,779 seniors**, our BQML model predicts only a **+0.8°F offset** (low housing density & mature canopy), "
            "and Census ACS shows **0.0% of households lack a car** (3.1% poverty). We redirect your daily budget to **Oak Cliff** and **Buckner Terrace**, where over **21% of households have no vehicle** and heat offsets exceed **+4.4°F**."
        )
    elif "why" in q or "explain" in q or "hot" in q:
        target = next((n for n in plan["neighborhoods"] if n["id"] == highlight_id), plan["neighborhoods"][0])
        reply = (
            f"**{target['name']} (`{target['intersection']}`)**\n\n"
            f"- **Station vs. Street Temp:** `{target['station_temp_f']}°F` station → **`{target['predicted_street_temp_f']}°F` predicted** (`+{target['predicted_offset_f']}°F`)\n"
            f"- **Why it's hotter:** {target['why_short']}\n"
            f"- **Daily Action:** {target['assigned_unit'] or 'Standby (adjust sliders to cover)'}"
        )
    else:
        budget_msg = (
            f"Deploying **{plan['funded_trucks']} Hydration Truck(s)** (`$2,500/ea`) and "
            f"**{plan['funded_wellness']} Senior Wellness Check Team(s)** (`$1,200/ea`) — "
            f"total **${plan['total_cost']:,}/day** within your **${budget:,}** budget:"
        )
        if not plan["within_budget"]:
            budget_msg = (
                f"⚠️ Requested **{hydration_trucks} Truck(s) + {wellness_teams} Wellness Team(s)** cost **${plan['requested_cost']:,}**, "
                f"exceeding your **${budget:,}** budget. Funded **{plan['funded_trucks']} Truck(s) + {plan['funded_wellness']} Wellness Team(s)** (**${plan['total_cost']:,}/day**):"
            )
        lines = [
            f"{d['rank']}. **{d['assigned_unit']}** → **{d['name']}** (`{d['predicted_street_temp_f']}°F`, `+{d['predicted_offset_f']}°F` offset · {d['seniors_65_plus']:,} seniors)"
            for d in deployed
        ]
        reply = f"{budget_msg}\n\n" + ("\n".join(lines) if lines else "_Increase sliders to dispatch units._")

    return {
        "reply": reply,
        "highlight_tract_id": highlight_id,
        "plan": plan,
    }


if HAS_FASTAPI:
    @app.get("/api/initial-data")
    async def get_initial_data():
        return JSONResponse(compute_dispatch_plan(budget=10000, hydration_trucks=2))

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
            self._send_json(compute_dispatch_plan(budget=10000, hydration_trucks=2))
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"🚀 Dallas Daily Heat Dispatch Planner running at http://localhost:{port}")
    server = HTTPServer(("127.0.0.1", port), LocalPreviewHandler)
    server.serve_forever()

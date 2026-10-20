"""Centralized configuration for the Dallas Urban Heat Emergency Commander ADK Agent."""

from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Core Project, Dataset, & Temperature Threshold Settings
# ---------------------------------------------------------------------------
project_id: str = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-f8ba79cf4618")
dataset: str = os.environ.get("BQ_DATASET", "a4i_climate")
dangerous_threshold_temp: float = float(
    os.environ.get("DANGEROUS_THRESHOLD_TEMP", "102.0")
)

# Uppercase aliases for convenience across modules
PROJECT_ID: str = project_id
DATASET: str = dataset
DATASET_ID: str = dataset
DANGEROUS_THRESHOLD_TEMP: float = dangerous_threshold_temp

# ---------------------------------------------------------------------------
# BigQuery Table Names & Fully-Qualified Identifiers
# ---------------------------------------------------------------------------
TABLE_NAME: str = os.environ.get("BQ_TABLE", "Agent_Universal_Vulnaribility_Index")
HEAT_OBSERVATIONS_TABLE_NAME: str = os.environ.get(
    "BQ_HEAT_OBSERVATIONS_TABLE", "heat_observations"
)
STATION_DAILY_TEMPS_TABLE_NAME: str = os.environ.get(
    "BQ_STATION_TEMPS_TABLE", "station_daily_temps"
)

UVI_TABLE: str = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_NAME}`"
HEAT_OBSERVATIONS_TABLE: str = (
    f"`{PROJECT_ID}.{DATASET_ID}.{HEAT_OBSERVATIONS_TABLE_NAME}`"
)
STATION_DAILY_TEMPS_TABLE: str = (
    f"`{PROJECT_ID}.{DATASET_ID}.{STATION_DAILY_TEMPS_TABLE_NAME}`"
)

# ---------------------------------------------------------------------------
# Model & Dallas Geography Defaults
# ---------------------------------------------------------------------------
ADK_MODEL: str = os.environ.get("ADK_MODEL", "gemini-2.5-flash")
DALLAS_DEFAULT_LAT: float = 32.7767
DALLAS_DEFAULT_LON: float = -96.7970


@dataclass(frozen=True)
class AgentConfig:
    """Typed configuration container for the Urban Heat ADK Agent."""

    project_id: str = PROJECT_ID
    dataset: str = DATASET
    dangerous_threshold_temp: float = DANGEROUS_THRESHOLD_TEMP
    table_name: str = TABLE_NAME
    uvi_table: str = UVI_TABLE
    heat_observations_table: str = HEAT_OBSERVATIONS_TABLE
    station_daily_temps_table: str = STATION_DAILY_TEMPS_TABLE
    adk_model: str = ADK_MODEL
    dallas_default_lat: float = DALLAS_DEFAULT_LAT
    dallas_default_lon: float = DALLAS_DEFAULT_LON


CONFIG = AgentConfig()


"""Managed MCP Server integrations (BigQuery & Google Maps) and custom tools for the Urban Heat Planner agent."""

import os
from typing import Optional

import google.auth
import google.auth.transport.requests
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "qwiklabs-gcp-04-f8ba79cf4618")
DATASET_ID = os.environ.get("BQ_DATASET", "a4i_climate")
FULL_DATASET_ID = f"{PROJECT_ID}.{DATASET_ID}"

# --- BigQuery Managed MCP Server ---
BIGQUERY_MCP_URL = "https://bigquery.googleapis.com/mcp"
BIGQUERY_SCOPES = ["https://www.googleapis.com/auth/bigquery"]

# Since project (`qwiklabs-gcp-04-f8ba79cf4618`) and dataset (`a4i_climate`) are fixed,
# we only need table discovery, schema inspection, and SQL execution from the MCP server.
RECOMMENDED_BQ_MCP_TOOLS = [
    "execute_sql",       # Run analytical & historical SQL queries against a4i_climate
    "get_table_info",    # Inspect table schemas (columns, types, descriptions)
    "list_table_ids",    # List tables in qwiklabs-gcp-04-f8ba79cf4618.a4i_climate
]

# --- Google Maps Managed MCP Server (Maps Grounding Lite) ---
MAPS_MCP_URL = "https://mapstools.googleapis.com/mcp"
MAPS_SCOPES = [
    "https://www.googleapis.com/auth/maps-platform.mapstools",
    "https://www.googleapis.com/auth/cloud-platform",
]
RECOMMENDED_MAPS_MCP_TOOLS = [
    "search_places",     # Geocode street names/intersections & find libraries, parks, cooling centers
    "compute_routes",    # Calculate real walking distance/time from vulnerable tracts to cooling centers
]


def _get_bq_auth_headers(
    readonly_context: Optional[ReadonlyContext] = None,
) -> dict[str, str]:
    """Fetches and refreshes Application Default Credentials for the BigQuery MCP server."""
    credentials, default_project = google.auth.default(scopes=BIGQUERY_SCOPES)
    if not credentials.valid:
        credentials.refresh(google.auth.transport.requests.Request())

    project_id = PROJECT_ID or default_project or ""
    return {
        "Authorization": f"Bearer {credentials.token}",
        "x-goog-user-project": project_id,
    }


def _get_maps_auth_headers(
    readonly_context: Optional[ReadonlyContext] = None,
) -> dict[str, str]:
    """Builds authentication headers for Google Maps MCP server (API key or ADC OAuth)."""
    api_key = os.environ.get("MAPS_API_KEY") or os.environ.get("GOOGLE_MAPS_API_KEY")
    if api_key:
        return {"X-Goog-Api-Key": api_key}

    credentials, default_project = google.auth.default(scopes=MAPS_SCOPES)
    if not credentials.valid:
        credentials.refresh(google.auth.transport.requests.Request())

    project_id = PROJECT_ID or default_project or ""
    return {
        "Authorization": f"Bearer {credentials.token}",
        "x-goog-user-project": project_id,
    }


def get_bigquery_mcp_toolset(
    tool_filter: Optional[list[str]] = None,
) -> McpToolset:
    """Creates an ADK McpToolset connected to Google's managed BigQuery MCP server.

    Args:
        tool_filter: Optional list of MCP tool names to expose. Defaults to
            RECOMMENDED_BQ_MCP_TOOLS.
    """
    initial_headers = _get_bq_auth_headers()
    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=BIGQUERY_MCP_URL,
            headers=initial_headers,
            timeout=30.0,
            sse_read_timeout=300.0,
        ),
        header_provider=_get_bq_auth_headers,
        tool_filter=tool_filter or RECOMMENDED_BQ_MCP_TOOLS,
    )


def get_maps_mcp_toolset(
    tool_filter: Optional[list[str]] = None,
) -> McpToolset:
    """Creates an ADK McpToolset connected to Google's managed Maps MCP server.

    Supports both `MAPS_API_KEY` / `GOOGLE_MAPS_API_KEY` environment variables and
    Application Default Credentials (OAuth Bearer token + `x-goog-user-project`).

    Args:
        tool_filter: Optional list of Maps MCP tool names to expose. Defaults to
            RECOMMENDED_MAPS_MCP_TOOLS.
    """
    initial_headers = _get_maps_auth_headers()
    return McpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=MAPS_MCP_URL,
            headers=initial_headers,
            timeout=30.0,
            sse_read_timeout=300.0,
        ),
        header_provider=_get_maps_auth_headers,
        tool_filter=tool_filter or RECOMMENDED_MAPS_MCP_TOOLS,
    )



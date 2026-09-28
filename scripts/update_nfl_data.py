from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# CONFIGURATION
# ============================================================

SEASON = 2026

OUTPUT_DIR = Path("data")
PLAYER_OUTPUT = OUTPUT_DIR / "players.json"
STATUS_OUTPUT = OUTPUT_DIR / "data-status.json"

PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    f"stats_player/stats_player_week_{SEASON}.csv"
)

ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE"}

REQUEST_TIMEOUT = 60

HEADERS = {
    "User-Agent": "nfl-player-dashboard/1.0"
}


# ============================================================
# ERROR HANDLING
# ============================================================

def fail(message: str) -> None:
    print()
    print("=" * 60, file=sys.stderr)
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    sys.exit(1)


# ============================================================
# DOWNLOAD
# ============================================================

def download_csv(url: str) -> pd.DataFrame:

    print()
    print("Downloading nflverse data:")
    print(url)

    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        fail(
            f"Unable to download nflverse data: {exc}"
        )

    content_type = (
        response.headers
        .get("content-type", "")
        .lower()
    )

    size = len(response.content)

    print()
    print(f"HTTP status: {response.status_code}")
    print(
        f"Content type: "
        f"{content_type or 'unknown'}"
    )
    print(f"Downloaded bytes: {size:,}")
    print(f"Final URL: {response.url}")

    beginning = response.content[:500].lower()

    if (
        b"<html" in beginning
        or b"<!doctype html" in beginning
    ):
        fail(
            "The nflverse URL returned HTML "
            "instead of CSV data."
        )

    if size < 1000:
        fail(
            f"The downloaded dataset is "
            f"unexpectedly small ({size:,} bytes)."
        )

    try:
        dataframe = pd.read_csv(
            io.BytesIO(response.content)
        )

    except Exception as exc:
        fail(
            f"Unable to parse downloaded CSV: {exc}"
        )

    print()
    print(
        f"CSV successfully parsed: "
        f"{len(dataframe):,} rows x "
        f"{len(dataframe.columns):,} columns"
    )

    return dataframe


# ============================================================
# NORMALIZE SOURCE SCHEMA
# ============================================================

def normalize_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    nflverse occasionally changes field naming.

    The dashboard will always use 'recent_team'
    internally regardless of the source field name.
    """

    df = df.copy()

    print()
    print("Checking nflverse schema...")

    # --------------------------------------------------------
    # TEAM
    # --------------------------------------------------------

    team_candidates = [
        "recent_team",
        "team",
        "posteam",
    ]

    team_column = None

    for candidate in team_candidates:

        if candidate in df.columns:

            team_column = candidate
            break

    if team_column is None:

        print()
        print(
            "Available columns:"
        )

        print(
            ", ".join(
                sorted(df.columns)
            )
        )

        fail(
            "No recognized team column found. "
            "Expected one of: "
            "recent_team, team, posteam"
        )

    print(
        f"Team column detected: {team_column}"
    )

    if team_column != "recent_team":

        df["recent_team"] = (
            df[team_column]
        )

        print(
            f"Normalized '{team_column}' "
            "to 'recent_team'."
        )

    # --------------------------------------------------------
    # PLAYER NAME
    # --------------------------------------------------------

    if "player_display_name" not in df.columns:

        player_name_candidates = [
            "player_name",
            "name",
        ]

        player_name_column = None

        for candidate in player_name_candidates:

            if candidate in df.columns:

                player_name_column = candidate
                break

        if player_name_column is None:

            fail(
                "No recognized player-name "
                "column found."
            )

        df["player_display_name"] = (
            df[player_name_column]
        )

        print(
            f"Normalized '{player_name_column}' "
            "to 'player_display_name'."
        )

    print(
        "Schema normalization complete."
    )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_player_data(
    df: pd.DataFrame,
) -> None:

    print()
    print(
        "Validating nflverse player data..."
    )

    required_columns = {
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "season",
        "week",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        fail(
            "Dataset is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if df.empty:

        fail(
            "Player dataset contains zero rows."
        )

    numeric_season = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    season_df = df[
        numeric_season == SEASON
    ].copy()

    if season_df.empty:

        fail(
            f"No {SEASON} player statistics "
            "were found."
        )

    if len(season_df) < 100:

        fail(
            f"Only {len(season_df):,} "
            f"{SEASON} player-week records "
            "were found."
        )

    positions = set(
        season_df["position"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    missing_positions = (
        ALLOWED_POSITIONS
        - positions
    )

    if missing_positions:

        fail(
            "Expected fantasy positions "
            "are missing: "
            + ", ".join(
                sorted(missing_positions)
            )
        )

    weeks = pd.to_numeric(
        season_df["week"],
        errors="coerce",
    ).dropna()

    if weeks.empty:

        fail(
            "No valid NFL week numbers "
            "were found."
        )

    latest_week = int(
        weeks.max()
    )

    if not 1 <= latest_week <= 22:

        fail(
            f"Unexpected latest NFL week: "
            f"{latest_week}"
        )

    teams = (
        season_df["recent_team"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(teams) < 20:

        fail(
            f"Only {len(teams)} teams "
            "were detected."
        )

    print()
    print("Validation PASSED")

    print(
        f"{SEASON} player-week rows: "
        f"{len(season_df):,}"
    )

    print(
        f"Latest week found: "
        f"{latest_week}"
    )

    print(
        f"Teams found: "
        f"{len(teams)}"
    )

    print(
        "Fantasy positions confirmed: "
        + ", ".join(
            sorted(ALLOWED_POSITIONS)
        )
    )


# ============================================================
# JSON CLEANING
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    return value


# ============================================================
# BUILD PLAYER DATASET
# ============================================================

def build_player_dataset(
    df: pd.DataFrame,
):

    print()
    print(
        "Building QB/RB/WR/TE player dataset..."
    )

    df = df.copy()

    df["position"] = (
        df["position"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    numeric_season = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    df = df[
        (numeric_season == SEASON)
        &
        (
            df["position"]
            .isin(ALLOWED_POSITIONS)
        )
    ].copy()

    df["week"] = pd.to_numeric(
        df["week"],
        errors="coerce",
    )

    df = df[
        df["week"].notna()
    ].copy()

    df["week"] = (
        df["week"]
        .astype(int)
    )

    df = df.sort_values(
        [
            "player_display_name",
            "week",
        ]
    )

    records = []

    for _, row in df.iterrows():

        record = {
            column: clean_value(value)
            for column, value
            in row.items()
        }

        records.append(record)

    print(
        f"Normalized records: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# BUILD STATUS FILE
# ============================================================

def build_status(
    df: pd.DataFrame,
    records,
):

    fantasy_df = df.copy()

    fantasy_df["position"] = (
        fantasy_df["position"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    numeric_season = pd.to_numeric(
        fantasy_df["season"],
        errors="coerce",
    )

    fantasy_df = fantasy_df[
        (numeric_season == SEASON)
        &
        (
            fantasy_df["position"]
            .isin(ALLOWED_POSITIONS)
        )
    ].copy()

    weeks = pd.to_numeric(
        fantasy_df["week"],
        errors="coerce",
    ).dropna()

    latest_week = int(
        weeks.max()
    )

    teams = sorted(
        fantasy_df["recent_team"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    players = (
        fantasy_df[
            [
                "player_id",
                "player_display_name",
                "position",
            ]
        ]
        .drop_duplicates()
    )

    position_counts = (
        players["position"]
        .value_counts()
        .to_dict()
    )

    return {

        "status": "validated",

        "source": "nflverse",

        "source_dataset":
            "Player Summary Stats",

        "season": SEASON,

        "latest_week":
            latest_week,

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "player_week_records":
            len(records),

        "unique_players":
            int(len(players)),

        "teams_found":
            len(teams),

        "teams":
            teams,

        "position_counts": {
            str(key): int(value)
            for key, value
            in position_counts.items()
        },

        "license":
            "CC BY 4.0",

        "attribution": (
            "Player statistics provided "
            "by nflverse. "
            "Source: nflverse-data "
            "Player Summary Stats."
        ),
    }


# ============================================================
# WRITE JSON SAFELY
# ============================================================

def write_json(
    path: Path,
    data,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    try:

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                data,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )

        temporary.replace(path)

    except Exception as exc:

        if temporary.exists():
            temporary.unlink()

        fail(
            f"Unable to write "
            f"{path}: {exc}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 60)

    print(
        f"NFL PLAYER DASHBOARD — "
        f"{SEASON} DATA UPDATE"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    df = download_csv(
        PLAYER_STATS_URL
    )

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    df = normalize_schema(
        df
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    validate_player_data(
        df
    )

    # --------------------------------------------------------
    # BUILD
    # --------------------------------------------------------

    records = (
        build_player_dataset(
            df
        )
    )

    if not records:

        fail(
            "Normalized player dataset "
            "contains zero records."
        )

    status = build_status(
        df,
        records,
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    print()
    print(
        "Writing validated dashboard data..."
    )

    write_json(
        PLAYER_OUTPUT,
        records,
    )

    write_json(
        STATUS_OUTPUT,
        status,
    )

    # --------------------------------------------------------
    # FINAL VERIFICATION REPORT
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SUCCESS")
    print("=" * 60)

    print(
        f"Player-week records: "
        f"{status['player_week_records']:,}"
    )

    print(
        f"Unique fantasy players: "
        f"{status['unique_players']:,}"
    )

    print(
        f"Teams represented: "
        f"{status['teams_found']}"
    )

    print(
        f"Latest NFL week: "
        f"{status['latest_week']}"
    )

    print()
    print("Players by position:")

    for position in [
        "QB",
        "RB",
        "WR",
        "TE",
    ]:

        count = (
            status[
                "position_counts"
            ].get(
                position,
                0,
            )
        )

        print(
            f"  {position}: {count:,}"
        )

    print()
    print(
        f"Player data: "
        f"{PLAYER_OUTPUT}"
    )

    print(
        f"Validation status: "
        f"{STATUS_OUTPUT}"
    )

    print()
    print(
        "Source: nflverse"
    )

    print(
        "License: CC BY 4.0"
    )


if __name__ == "__main__":
    main()

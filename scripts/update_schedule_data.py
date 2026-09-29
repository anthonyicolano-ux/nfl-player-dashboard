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
SCHEDULE_OUTPUT = OUTPUT_DIR / "schedule.json"

# nflverse game/schedule data
SCHEDULE_URL = (
    "https://github.com/nflverse/nfldata/"
    "raw/master/data/games.csv"
)

REQUEST_TIMEOUT = 90

HEADERS = {
    "User-Agent": "nfl-player-dashboard/2.0"
}

REGULAR_SEASON_TYPES = {
    "REG"
}

TEAM_ALIASES = {
    "JAX": "JAC",
    "WSH": "WAS",
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
}


# ============================================================
# ERROR HANDLING
# ============================================================

def fail(message: str) -> None:

    print()
    print(
        "=" * 72,
        file=sys.stderr,
    )

    print(
        f"VALIDATION FAILED: {message}",
        file=sys.stderr,
    )

    print(
        "=" * 72,
        file=sys.stderr,
    )

    sys.exit(1)


# ============================================================
# HELPERS
# ============================================================

def normalize_team(value) -> str:

    if pd.isna(value):
        return ""

    team = (
        str(value)
        .upper()
        .strip()
    )

    return TEAM_ALIASES.get(
        team,
        team,
    )


def clean_value(value):

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    return value


# ============================================================
# DOWNLOAD SCHEDULE
# ============================================================

def download_schedule() -> pd.DataFrame:

    print("=" * 72)
    print("NFL SCHEDULE DATA UPDATE")
    print("=" * 72)

    print()
    print(
        "Downloading nflverse "
        "schedule data:"
    )

    print(
        SCHEDULE_URL
    )

    try:

        response = requests.get(
            SCHEDULE_URL,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        fail(
            "Unable to download "
            f"schedule data: {exc}"
        )

    print(
        f"HTTP status: "
        f"{response.status_code}"
    )

    print(
        f"Downloaded bytes: "
        f"{len(response.content):,}"
    )

    beginning = (
        response.content[:500]
        .lower()
    )

    if (
        b"<html" in beginning
        or
        b"<!doctype html" in beginning
    ):

        fail(
            "Schedule download returned "
            "HTML instead of CSV."
        )

    try:

        df = pd.read_csv(
            io.BytesIO(
                response.content
            )
        )

    except Exception as exc:

        fail(
            "Unable to parse "
            f"schedule CSV: {exc}"
        )

    print(
        f"Parsed: "
        f"{len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    return df


# ============================================================
# VALIDATE SOURCE SCHEMA
# ============================================================

def validate_source_schema(
    df: pd.DataFrame,
) -> None:

    required = {
        "season",
        "week",
        "game_type",
        "gameday",
        "away_team",
        "home_team",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        fail(
            "Schedule dataset is missing "
            "required columns: "
            + ", ".join(
                sorted(missing)
            )
        )


# ============================================================
# BUILD TEAM-LEVEL SCHEDULE
# ============================================================

def build_schedule(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_source_schema(
        df
    )

    work = df.copy()

    work["season"] = (
        pd.to_numeric(
            work["season"],
            errors="coerce",
        )
    )

    work["week"] = (
        pd.to_numeric(
            work["week"],
            errors="coerce",
        )
    )

    work["game_type"] = (
        work["game_type"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    work = work[
        (work["season"] == SEASON)
        &
        (
            work["game_type"]
            .isin(
                REGULAR_SEASON_TYPES
            )
        )
        &
        work["week"].notna()
    ].copy()

    if work.empty:

        fail(
            f"No {SEASON} regular-season "
            "schedule records were found."
        )

    work["week"] = (
        work["week"]
        .astype(int)
    )

    work["home_team"] = (
        work["home_team"]
        .apply(normalize_team)
    )

    work["away_team"] = (
        work["away_team"]
        .apply(normalize_team)
    )

    if (
        (work["home_team"] == "")
        |
        (work["away_team"] == "")
    ).any():

        fail(
            "Schedule contains missing "
            "home or away teams."
        )

    # --------------------------------------------------------
    # HOME TEAM RECORDS
    # --------------------------------------------------------

    home = pd.DataFrame(
        {
            "season":
                work["season"]
                .astype(int),

            "week":
                work["week"],

            "team":
                work["home_team"],

            "opponent":
                work["away_team"],

            "home_away":
                "HOME",

            "game_date":
                work["gameday"],

            "game_type":
                work["game_type"],
        }
    )

    # --------------------------------------------------------
    # AWAY TEAM RECORDS
    # --------------------------------------------------------

    away = pd.DataFrame(
        {
            "season":
                work["season"]
                .astype(int),

            "week":
                work["week"],

            "team":
                work["away_team"],

            "opponent":
                work["home_team"],

            "home_away":
                "AWAY",

            "game_date":
                work["gameday"],

            "game_type":
                work["game_type"],
        }
    )

    schedule = pd.concat(
        [
            home,
            away,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # CLEAN DATES
    # --------------------------------------------------------

    schedule["game_date"] = (
        pd.to_datetime(
            schedule["game_date"],
            errors="coerce",
        )
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    if (
        schedule["game_date"]
        .isna()
        .any()
    ):

        fail(
            "At least one schedule record "
            "has an invalid game date."
        )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    schedule = (
        schedule
        .sort_values(
            [
                "week",
                "team",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return schedule


# ============================================================
# VALIDATION
# ============================================================

def validate_schedule(
    schedule: pd.DataFrame,
) -> None:

    print()
    print("=" * 72)
    print("VALIDATING NFL SCHEDULE")
    print("=" * 72)

    teams = set(
        schedule["team"]
        .dropna()
        .unique()
    )

    opponents = set(
        schedule["opponent"]
        .dropna()
        .unique()
    )

    weeks = sorted(
        schedule["week"]
        .dropna()
        .unique()
    )

    duplicate_count = int(
        schedule.duplicated(
            subset=[
                "team",
                "week",
            ]
        )
        .sum()
    )

    print(
        f"Teams represented: "
        f"{len(teams)}/32"
    )

    print(
        f"Opponent teams represented: "
        f"{len(opponents)}/32"
    )

    print(
        f"Schedule records: "
        f"{len(schedule):,}"
    )

    print(
        f"Regular-season weeks: "
        f"{min(weeks)}-{max(weeks)}"
    )

    print(
        f"Duplicate team/week records: "
        f"{duplicate_count}"
    )

    if len(teams) != 32:

        fail(
            "Schedule does not contain "
            "all 32 NFL teams."
        )

    if len(opponents) != 32:

        fail(
            "Schedule opponent data does "
            "not contain all 32 NFL teams."
        )

    if duplicate_count:

        fail(
            "Schedule contains "
            f"{duplicate_count} duplicate "
            "team/week records."
        )

    if min(weeks) != 1:

        fail(
            "Regular-season schedule "
            "does not begin with Week 1."
        )

    if max(weeks) < 18:

        fail(
            "Regular-season schedule "
            "does not contain Week 18."
        )

    invalid_home_away = set(
        schedule[
            ~schedule["home_away"]
            .isin(
                [
                    "HOME",
                    "AWAY",
                ]
            )
        ]["home_away"]
    )

    if invalid_home_away:

        fail(
            "Schedule contains invalid "
            "home/away values."
        )

    # Every schedule record should have
    # team != opponent.

    if (
        schedule["team"]
        ==
        schedule["opponent"]
    ).any():

        fail(
            "Schedule contains a team "
            "playing itself."
        )

    print(
        "SCHEDULE VALIDATION PASSED"
    )


# ============================================================
# JSON OUTPUT
# ============================================================

def dataframe_records(
    df: pd.DataFrame,
) -> list[dict]:

    records = []

    for _, row in df.iterrows():

        record = {}

        for column, value in row.items():

            record[column] = (
                clean_value(
                    value
                )
            )

        records.append(
            record
        )

    return records


def write_json(
    path: Path,
    data,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

        handle.write("\n")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    raw = (
        download_schedule()
    )

    schedule = (
        build_schedule(
            raw
        )
    )

    validate_schedule(
        schedule
    )

    records = (
        dataframe_records(
            schedule
        )
    )

    write_json(
        SCHEDULE_OUTPUT,
        records,
    )

    print()
    print("=" * 72)
    print(
        "NFL SCHEDULE UPDATE COMPLETE"
    )
    print("=" * 72)

    print(
        f"Season: {SEASON}"
    )

    print(
        f"Records: "
        f"{len(records):,}"
    )

    print(
        f"Output: "
        f"{SCHEDULE_OUTPUT}"
    )

    print(
        "Generated UTC: "
        + datetime.now(
            timezone.utc
        ).isoformat()
    )

    print(
        "FINAL VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()

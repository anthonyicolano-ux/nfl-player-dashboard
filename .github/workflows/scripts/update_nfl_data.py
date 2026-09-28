from __future__ import annotations

import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests


SEASON = 2026

OUTPUT_DIR = Path("data")
PLAYER_OUTPUT = OUTPUT_DIR / "players.json"
STATUS_OUTPUT = OUTPUT_DIR / "data-status.json"

# Official nflverse release asset.
PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    f"stats_player/stats_player_week_{SEASON}.csv"
)

ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE"}

REQUEST_TIMEOUT = 60

HEADERS = {
    "User-Agent": "nfl-player-dashboard/1.0"
}


def fail(message: str) -> None:
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    sys.exit(1)


def download_csv(url: str) -> pd.DataFrame:
    print(f"Downloading: {url}")

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    # Basic protection against accidentally receiving an HTML error page.
    beginning = response.content[:200].lower()

    if b"<html" in beginning or b"<!doctype html" in beginning:
        fail("Source returned HTML instead of CSV data.")

    if len(response.content) < 1000:
        fail("Downloaded player dataset is unexpectedly small.")

    print(
        f"Downloaded {len(response.content):,} bytes "
        f"(content-type: {content_type or 'unknown'})"
    )

    try:
        return pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:
        fail(f"Unable to parse nflverse CSV: {exc}")


def validate_player_data(df: pd.DataFrame) -> None:
    required_columns = {
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "season",
        "week",
    }

    missing = required_columns - set(df.columns)

    if missing:
        fail(
            "Player dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    if df.empty:
        fail("Player dataset contains zero rows.")

    seasons = set(
        pd.to_numeric(df["season"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )

    if SEASON not in seasons:
        fail(f"{SEASON} is not present in the player dataset.")

    season_df = df[
        pd.to_numeric(df["season"], errors="coerce") == SEASON
    ].copy()

    if len(season_df) < 100:
        fail(
            f"Only {len(season_df)} {SEASON} player-week rows were found."
        )

    positions = set(
        season_df["position"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
    )

    missing_positions = ALLOWED_POSITIONS - positions

    if missing_positions:
        fail(
            "Expected fantasy positions are missing: "
            + ", ".join(sorted(missing_positions))
        )

    weeks = pd.to_numeric(
        season_df["week"], errors="coerce"
    ).dropna()

    if weeks.empty:
        fail("No valid NFL weeks were found.")

    latest_week = int(weeks.max())

    if not 1 <= latest_week <= 22:
        fail(f"Unexpected latest NFL week: {latest_week}")

    print("Player data validation passed.")
    print(f"{SEASON} rows: {len(season_df):,}")
    print(f"Latest week: {latest_week}")
    print(f"Positions: {sorted(positions)}")


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    return value


def build_player_dataset(df: pd.DataFrame):
    df = df.copy()

    df["position"] = (
        df["position"]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    df = df[
        (pd.to_numeric(df["season"], errors="coerce") == SEASON)
        & (df["position"].isin(ALLOWED_POSITIONS))
    ].copy()

    df["week"] = pd.to_numeric(
        df["week"], errors="coerce"
    )

    df = df[df["week"].notna()].copy()
    df["week"] = df["week"].astype(int)

    df = df.sort_values(
        ["player_display_name", "week"]
    )

    records = []

    for _, row in df.iterrows():
        record = {
            column: clean_value(value)
            for column, value in row.items()
        }

        records.append(record)

    return records


def build_status(df: pd.DataFrame, records):
    fantasy_df = df[
        (pd.to_numeric(df["season"], errors="coerce") == SEASON)
        & (
            df["position"]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(ALLOWED_POSITIONS)
        )
    ].copy()

    weeks = pd.to_numeric(
        fantasy_df["week"], errors="coerce"
    ).dropna()

    latest_week = int(weeks.max())

    teams = sorted(
        fantasy_df["recent_team"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    players = (
        fantasy_df[
            ["player_id", "player_display_name", "position"]
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
        "source_dataset": "Player Summary Stats",
        "season": SEASON,
        "latest_week": latest_week,
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "player_week_records": len(records),
        "unique_players": int(len(players)),
        "teams_found": len(teams),
        "teams": teams,
        "position_counts": {
            str(key): int(value)
            for key, value in position_counts.items()
        },
        "license": "CC BY 4.0",
        "attribution": (
            "Player statistics provided by nflverse. "
            "See nflverse-data for source and licensing details."
        ),
    }


def write_json(path: Path, data) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

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


def main() -> None:
    print(
        f"Starting nflverse {SEASON} player-data update."
    )

    df = download_csv(PLAYER_STATS_URL)

    validate_player_data(df)

    records = build_player_dataset(df)

    if not records:
        fail("Normalized player dataset contains zero records.")

    status = build_status(df, records)

    write_json(
        PLAYER_OUTPUT,
        records,
    )

    write_json(
        STATUS_OUTPUT,
        status,
    )

    print()
    print("SUCCESS")
    print(
        f"Player-week records: "
        f"{status['player_week_records']:,}"
    )
    print(
        f"Unique players: "
        f"{status['unique_players']:,}"
    )
    print(
        f"Teams: {status['teams_found']}"
    )
    print(
        f"Latest week: "
        f"{status['latest_week']}"
    )
    print(
        f"Output: {PLAYER_OUTPUT}"
    )
    print(
        f"Status: {STATUS_OUTPUT}"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import io
import json
import re
import sys
import unicodedata
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

SNAP_COUNTS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    f"snap_counts/snap_counts_{SEASON}.csv"
)

ALLOWED_POSITIONS = {"QB", "RB", "WR", "TE"}

REQUEST_TIMEOUT = 90

HEADERS = {
    "User-Agent": "nfl-player-dashboard/1.1"
}

# We deliberately use a conservative threshold.
# Some fantasy players may legitimately have no offensive snaps.
MIN_SNAP_MATCH_RATE = 0.70


# ============================================================
# ERROR HANDLING
# ============================================================

def fail(message: str) -> None:
    print()
    print("=" * 70, file=sys.stderr)
    print(f"VALIDATION FAILED: {message}", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    sys.exit(1)


# ============================================================
# DOWNLOAD
# ============================================================

def download_csv(url: str, label: str) -> pd.DataFrame:
    print()
    print("=" * 70)
    print(f"DOWNLOADING: {label}")
    print("=" * 70)
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
        fail(f"Unable to download {label}: {exc}")

    size = len(response.content)

    print(f"HTTP status: {response.status_code}")
    print(f"Downloaded bytes: {size:,}")
    print(
        "Content type:",
        response.headers.get("content-type", "unknown"),
    )

    beginning = response.content[:500].lower()

    if (
        b"<html" in beginning
        or b"<!doctype html" in beginning
    ):
        fail(f"{label} returned HTML instead of CSV.")

    if size < 1000:
        fail(
            f"{label} download is unexpectedly small "
            f"({size:,} bytes)."
        )

    try:
        df = pd.read_csv(io.BytesIO(response.content))
    except Exception as exc:
        fail(f"Unable to parse {label} CSV: {exc}")

    print(
        f"Parsed successfully: "
        f"{len(df):,} rows x {len(df.columns):,} columns"
    )

    return df


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def normalize_name(value) -> str:
    """
    Conservative name normalization for joining nflverse
    player stats to PFR snap-count records.
    """

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        c for c in value
        if not unicodedata.combining(c)
    )

    value = value.lower().strip()

    # Remove common punctuation.
    value = value.replace("’", "'")
    value = re.sub(r"[.'`-]", "", value)

    # Remove common suffixes.
    value = re.sub(
        r"\b(jr|sr|ii|iii|iv|v)\b",
        "",
        value,
    )

    # Keep letters/numbers only.
    value = re.sub(r"[^a-z0-9]", "", value)

    return value


def normalize_team(value) -> str:
    if pd.isna(value):
        return ""

    team = str(value).upper().strip()

    aliases = {
        "JAX": "JAC",
        "WSH": "WAS",
        "OAK": "LV",
        "SD": "LAC",
        "STL": "LA",
        "LAR": "LA",
    }

    return aliases.get(team, team)


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        value = value.item()

    return value


# ============================================================
# PLAYER STATS SCHEMA
# ============================================================

def normalize_player_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    print()
    print("Normalizing player-stat schema...")

    team_candidates = [
        "recent_team",
        "team",
        "posteam",
    ]

    team_column = next(
        (
            column
            for column in team_candidates
            if column in df.columns
        ),
        None,
    )

    if team_column is None:
        fail(
            "No recognized team column in player stats. "
            "Expected recent_team, team, or posteam."
        )

    if team_column != "recent_team":
        df["recent_team"] = df[team_column]

    if "player_display_name" not in df.columns:

        name_candidates = [
            "player_name",
            "name",
        ]

        name_column = next(
            (
                column
                for column in name_candidates
                if column in df.columns
            ),
            None,
        )

        if name_column is None:
            fail(
                "No recognized player-name column "
                "in player stats."
            )

        df["player_display_name"] = df[name_column]

    df["recent_team"] = (
        df["recent_team"]
        .apply(normalize_team)
    )

    df["position"] = (
        df["position"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["_join_name"] = (
        df["player_display_name"]
        .apply(normalize_name)
    )

    return df


# ============================================================
# PLAYER VALIDATION
# ============================================================

def validate_player_data(
    df: pd.DataFrame,
) -> None:

    required = {
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "season",
        "week",
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            "Player dataset is missing: "
            + ", ".join(sorted(missing))
        )

    numeric_season = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    season_df = df[
        numeric_season == SEASON
    ].copy()

    if len(season_df) < 100:
        fail(
            f"Only {len(season_df):,} "
            f"{SEASON} player-stat rows found."
        )

    positions = set(
        season_df["position"]
        .dropna()
        .unique()
    )

    missing_positions = (
        ALLOWED_POSITIONS - positions
    )

    if missing_positions:
        fail(
            "Missing fantasy positions: "
            + ", ".join(
                sorted(missing_positions)
            )
        )

    teams = set(
        season_df["recent_team"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(teams) != 32:
        fail(
            f"Expected 32 teams but found "
            f"{len(teams)}."
        )

    weeks = pd.to_numeric(
        season_df["week"],
        errors="coerce",
    ).dropna()

    if weeks.empty:
        fail("Player data contains no valid weeks.")

    latest_week = int(weeks.max())

    if not 1 <= latest_week <= 22:
        fail(
            f"Unexpected player-stat week: "
            f"{latest_week}"
        )

    print()
    print("PLAYER DATA VALIDATION PASSED")
    print(f"Teams: {len(teams)}/32")
    print(f"Latest week: {latest_week}")


# ============================================================
# SNAP SCHEMA / VALIDATION
# ============================================================

def normalize_snap_schema(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    print()
    print("Normalizing snap-count schema...")

    required = {
        "season",
        "week",
        "player",
        "position",
        "team",
        "opponent",
        "offense_snaps",
        "offense_pct",
    }

    missing = required - set(df.columns)

    if missing:
        fail(
            "Snap-count dataset is missing: "
            + ", ".join(sorted(missing))
        )

    df["team"] = (
        df["team"]
        .apply(normalize_team)
    )

    df["opponent"] = (
        df["opponent"]
        .apply(normalize_team)
    )

    df["position"] = (
        df["position"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["_join_name"] = (
        df["player"]
        .apply(normalize_name)
    )

    df["week"] = pd.to_numeric(
        df["week"],
        errors="coerce",
    )

    df["season"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    df["offense_snaps"] = pd.to_numeric(
        df["offense_snaps"],
        errors="coerce",
    )

    df["offense_pct"] = pd.to_numeric(
        df["offense_pct"],
        errors="coerce",
    )

    return df


def validate_snap_data(
    df: pd.DataFrame,
) -> None:

    season_df = df[
        df["season"] == SEASON
    ].copy()

    if len(season_df) < 500:
        fail(
            f"Only {len(season_df):,} "
            f"{SEASON} snap records found."
        )

    weeks = (
        season_df["week"]
        .dropna()
    )

    if weeks.empty:
        fail("Snap data contains no valid weeks.")

    latest_week = int(weeks.max())

    teams = set(
        season_df["team"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(teams) < 30:
        fail(
            f"Only {len(teams)} teams found "
            "in snap-count data."
        )

    print()
    print("SNAP DATA VALIDATION PASSED")
    print(
        f"Snap records: "
        f"{len(season_df):,}"
    )
    print(
        f"Snap teams: "
        f"{len(teams)}"
    )
    print(
        f"Latest snap week: "
        f"{latest_week}"
    )


# ============================================================
# BUILD FANTASY PLAYER DATA
# ============================================================

def build_player_dataset(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

    df["week"] = df["week"].astype(int)

    return df


# ============================================================
# JOIN SNAP COUNTS
# ============================================================

def join_snap_counts(
    players: pd.DataFrame,
    snaps: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("JOINING SNAP COUNTS")
    print("=" * 70)

    snap_df = snaps[
        (snaps["season"] == SEASON)
        &
        (
            snaps["position"]
            .isin(ALLOWED_POSITIONS)
        )
    ].copy()

    snap_df = snap_df[
        snap_df["week"].notna()
    ].copy()

    snap_df["week"] = (
        snap_df["week"]
        .astype(int)
    )

    # Keep the fields we currently need.
    snap_keep = [
        "_join_name",
        "team",
        "week",
        "opponent",
        "offense_snaps",
        "offense_pct",
    ]

    # Defensive duplicate handling.
    snap_df = (
        snap_df[snap_keep]
        .sort_values(
            [
                "_join_name",
                "team",
                "week",
                "offense_snaps",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        )
        .drop_duplicates(
            subset=[
                "_join_name",
                "team",
                "week",
            ],
            keep="first",
        )
    )

    joined = players.merge(
        snap_df,
        how="left",
        left_on=[
            "_join_name",
            "recent_team",
            "week",
        ],
        right_on=[
            "_join_name",
            "team",
            "week",
        ],
        validate="many_to_one",
    )

    joined.drop(
        columns=["team"],
        inplace=True,
        errors="ignore",
    )

    total = len(joined)

    matched = int(
        joined["offense_snaps"]
        .notna()
        .sum()
    )

    match_rate = (
        matched / total
        if total
        else 0
    )

    print(
        f"Player-week records: {total:,}"
    )
    print(
        f"Snap matches: {matched:,}"
    )
    print(
        f"Snap match rate: "
        f"{match_rate:.1%}"
    )

    # Print unmatched examples for debugging,
    # but do not expose any credentials or secrets.
    unmatched = (
        joined[
            joined["offense_snaps"].isna()
        ][
            [
                "player_display_name",
                "position",
                "recent_team",
                "week",
            ]
        ]
        .head(20)
    )

    if not unmatched.empty:

        print()
        print(
            "Sample records without snap match:"
        )

        for _, row in unmatched.iterrows():
            print(
                f"  {row['player_display_name']} | "
                f"{row['position']} | "
                f"{row['recent_team']} | "
                f"Week {row['week']}"
            )

    if match_rate < MIN_SNAP_MATCH_RATE:
        fail(
            "Snap-count join coverage is too low: "
            f"{match_rate:.1%}. "
            "No output will be published."
        )

    return joined, matched, match_rate


# ============================================================
# DERIVED METRICS
# ============================================================

def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    result = (
        numerator
        / denominator.replace(0, pd.NA)
    )

    return result


def add_derived_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    # Touches
    if (
        "carries" in df.columns
        and "receptions" in df.columns
    ):
        df["touches"] = (
            pd.to_numeric(
                df["carries"],
                errors="coerce",
            ).fillna(0)
            +
            pd.to_numeric(
                df["receptions"],
                errors="coerce",
            ).fillna(0)
        )

    # Opportunities
    if (
        "carries" in df.columns
        and "targets" in df.columns
    ):
        df["opportunities"] = (
            pd.to_numeric(
                df["carries"],
                errors="coerce",
            ).fillna(0)
            +
            pd.to_numeric(
                df["targets"],
                errors="coerce",
            ).fillna(0)
        )

    if "targets" in df.columns:
        df["targets_per_snap"] = safe_divide(
            df["targets"],
            df["offense_snaps"],
        )

    if "carries" in df.columns:
        df["carries_per_snap"] = safe_divide(
            df["carries"],
            df["offense_snaps"],
        )

    if "fantasy_points" in df.columns:

        df["fantasy_points_per_snap"] = (
            safe_divide(
                df["fantasy_points"],
                df["offense_snaps"],
            )
        )

        df["fantasy_points_per_100_snaps"] = (
            df["fantasy_points_per_snap"]
            * 100
        )

    if "fantasy_points_ppr" in df.columns:

        df["ppr_points_per_snap"] = (
            safe_divide(
                df["fantasy_points_ppr"],
                df["offense_snaps"],
            )
        )

        df["ppr_points_per_100_snaps"] = (
            df["ppr_points_per_snap"]
            * 100
        )

    if (
        "receiving_yards" in df.columns
        and "targets" in df.columns
    ):
        df["receiving_yards_per_target"] = (
            safe_divide(
                df["receiving_yards"],
                df["targets"],
            )
        )

    if (
        "receptions" in df.columns
        and "targets" in df.columns
    ):
        df["catch_rate"] = (
            safe_divide(
                df["receptions"],
                df["targets"],
            )
        )

    if (
        "rushing_yards" in df.columns
        and "carries" in df.columns
    ):
        df["rushing_yards_per_carry_calc"] = (
            safe_divide(
                df["rushing_yards"],
                df["carries"],
            )
        )

    return df


# ============================================================
# JSON OUTPUT
# ============================================================

def dataframe_records(
    df: pd.DataFrame,
):

    # Internal join helper should never appear
    # in the dashboard.
    df = df.drop(
        columns=["_join_name"],
        errors="ignore",
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

    return records


def write_json(
    path: Path,
    data,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
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
            f"Unable to write {path}: {exc}"
        )


# ============================================================
# STATUS FILE
# ============================================================

def build_status(
    df: pd.DataFrame,
    snap_records: int,
    snap_match_rate: float,
):

    weeks = pd.to_numeric(
        df["week"],
        errors="coerce",
    ).dropna()

    latest_week = int(
        weeks.max()
    )

    teams = sorted(
        df["recent_team"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    players = (
        df[
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
        "season": SEASON,
        "latest_week": latest_week,
        "generated_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "player_week_records": int(len(df)),
        "unique_players": int(len(players)),
        "teams_found": int(len(teams)),
        "teams": teams,
        "position_counts": {
            str(k): int(v)
            for k, v
            in position_counts.items()
        },
        "sources": {
            "player_stats": {
                "status": "loaded",
                "provider": "nflverse",
                "dataset": "Player Summary Stats",
            },
            "snap_counts": {
                "status": "loaded",
                "provider": "nflverse",
                "dataset": "PFR Snap Counts",
                "matched_player_week_records":
                    int(snap_records),
                "match_rate":
                    round(
                        float(snap_match_rate),
                        4,
                    ),
            },
        },
        "license": "CC BY 4.0",
        "attribution": (
            "NFL data provided through nflverse. "
            "See nflverse-data and the applicable "
            "dataset documentation for attribution "
            "and licensing information."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print(
        f"NFL PLAYER DASHBOARD — "
        f"{SEASON} PLAYER + SNAP UPDATE"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # PLAYER STATS
    # --------------------------------------------------------

    player_raw = download_csv(
        PLAYER_STATS_URL,
        "nflverse Player Summary Stats",
    )

    player_raw = normalize_player_schema(
        player_raw
    )

    validate_player_data(
        player_raw
    )

    players = build_player_dataset(
        player_raw
    )

    # --------------------------------------------------------
    # SNAP COUNTS
    # --------------------------------------------------------

    snap_raw = download_csv(
        SNAP_COUNTS_URL,
        "nflverse Snap Counts",
    )

    snap_raw = normalize_snap_schema(
        snap_raw
    )

    validate_snap_data(
        snap_raw
    )

    # --------------------------------------------------------
    # JOIN
    # --------------------------------------------------------

    joined, snap_matches, snap_match_rate = (
        join_snap_counts(
            players,
            snap_raw,
        )
    )

    # --------------------------------------------------------
    # DERIVED METRICS
    # --------------------------------------------------------

    joined = add_derived_metrics(
        joined
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    teams = set(
        joined["recent_team"]
        .dropna()
        .astype(str)
        .unique()
    )

    if len(teams) != 32:
        fail(
            f"Final dataset has "
            f"{len(teams)} teams instead of 32."
        )

    records = dataframe_records(
        joined
    )

    if not records:
        fail(
            "Final player dataset contains "
            "zero records."
        )

    status = build_status(
        joined,
        snap_matches,
        snap_match_rate,
    )

    # --------------------------------------------------------
    # WRITE
    # --------------------------------------------------------

    print()
    print("Writing validated output...")

    write_json(
        PLAYER_OUTPUT,
        records,
    )

    write_json(
        STATUS_OUTPUT,
        status,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print()
    print("PLAYER DATA")
    print(
        f"Player-week records: "
        f"{status['player_week_records']:,}"
    )
    print(
        f"Unique players: "
        f"{status['unique_players']:,}"
    )
    print(
        f"Teams: "
        f"{status['teams_found']}/32"
    )
    print(
        f"Latest week: "
        f"{status['latest_week']}"
    )

    print()
    print("PLAYERS BY POSITION")

    for position in [
        "QB",
        "RB",
        "WR",
        "TE",
    ]:

        print(
            f"{position}: "
            f"{status['position_counts'].get(position, 0):,}"
        )

    print()
    print("SNAP DATA")
    print("Downloaded: YES")
    print(
        f"Matched player-week records: "
        f"{snap_matches:,}"
    )
    print(
        f"Snap match rate: "
        f"{snap_match_rate:.1%}"
    )

    print()
    print("OUTPUT")
    print(
        f"Player data: "
        f"{PLAYER_OUTPUT}"
    )
    print(
        f"Status: "
        f"{STATUS_OUTPUT}"
    )

    print()
    print("VALIDATION: PASSED")


if __name__ == "__main__":
    main()

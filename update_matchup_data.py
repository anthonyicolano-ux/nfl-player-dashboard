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
MATCHUP_OUTPUT = OUTPUT_DIR / "matchups.json"

BASE_RELEASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download"
)

PLAYER_STATS_URL = (
    f"{BASE_RELEASE_URL}/stats_player/"
    f"stats_player_week_{SEASON}.csv"
)

ALLOWED_POSITIONS = {
    "QB",
    "RB",
    "WR",
    "TE",
}

REQUEST_TIMEOUT = 90

HEADERS = {
    "User-Agent": "nfl-player-dashboard/2.0"
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

    return numerator.div(
        denominator.where(
            denominator != 0
        )
    )


# ============================================================
# DOWNLOAD PLAYER DATA
# ============================================================

def download_player_stats() -> pd.DataFrame:

    print("=" * 72)
    print("NFL MATCHUP DATA UPDATE")
    print("=" * 72)

    print()
    print(
        "Downloading nflverse "
        "player-week data:"
    )

    print(
        PLAYER_STATS_URL
    )

    try:

        response = requests.get(
            PLAYER_STATS_URL,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

    except requests.RequestException as exc:

        fail(
            "Unable to download "
            f"player data: {exc}"
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
            "Player data download "
            "returned HTML instead of CSV."
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
            f"player CSV: {exc}"
        )

    print(
        f"Parsed: "
        f"{len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    return df


# ============================================================
# NORMALIZE PLAYER DATA
# ============================================================

def normalize_player_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    team_candidates = [
        "recent_team",
        "team",
        "posteam",
    ]

    team_column = next(
        (
            column
            for column
            in team_candidates
            if column in df.columns
        ),
        None,
    )

    if team_column is None:

        fail(
            "No recognized offensive "
            "team column in player data."
        )

    if team_column != "recent_team":

        df["recent_team"] = (
            df[team_column]
        )

    required = {
        "recent_team",
        "position",
        "season",
        "week",
        "fantasy_points_ppr",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        fail(
            "Player dataset is missing "
            "required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    # nflverse schemas have used different
    # opponent column names over time.
    opponent_candidates = [
        "opponent_team",
        "opponent",
        "opp",
    ]

    opponent_column = next(
        (
            column
            for column
            in opponent_candidates
            if column in df.columns
        ),
        None,
    )

    if opponent_column is None:

        fail(
            "Player dataset does not "
            "contain an opponent column. "
            "Checked: opponent_team, "
            "opponent, opp."
        )

    df["opponent"] = (
        df[opponent_column]
        .apply(normalize_team)
    )

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

    df["season"] = (
        pd.to_numeric(
            df["season"],
            errors="coerce",
        )
    )

    df["week"] = (
        pd.to_numeric(
            df["week"],
            errors="coerce",
        )
    )

    df = df[
        (df["season"] == SEASON)
        &
        (
            df["position"]
            .isin(
                ALLOWED_POSITIONS
            )
        )
        &
        (
            df["week"]
            .notna()
        )
        &
        (
            df["recent_team"] != ""
        )
        &
        (
            df["opponent"] != ""
        )
    ].copy()

    df["week"] = (
        df["week"]
        .astype(int)
    )

    numeric_columns = [
        "fantasy_points_ppr",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ]

    for column in numeric_columns:

        if column not in df.columns:
            df[column] = 0

        df[column] = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .fillna(0)
        )

    return df


# ============================================================
# COMPLETE DEFENSE / POSITION GRID
# ============================================================

def complete_grid(
    df: pd.DataFrame,
) -> pd.DataFrame:

    defenses = sorted(
        df["opponent"]
        .dropna()
        .unique()
    )

    if len(defenses) != 32:

        fail(
            "Expected 32 defenses "
            "in opponent data; "
            f"found {len(defenses)}."
        )

    grid = (
        pd.MultiIndex
        .from_product(
            [
                defenses,
                sorted(
                    ALLOWED_POSITIONS
                ),
            ],
            names=[
                "defense",
                "position",
            ],
        )
        .to_frame(
            index=False
        )
    )

    return grid


# ============================================================
# BUILD MATCHUP DATA
# ============================================================

def build_matchups(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 72)
    print(
        "BUILDING DEFENSE VS "
        "POSITION MATCHUPS"
    )
    print("=" * 72)

    latest_week = int(
        df["week"].max()
    )

    recent_start = max(
        1,
        latest_week - 2,
    )

    # --------------------------------------------------------
    # GAME COUNTS
    #
    # One defense can face many fantasy players
    # during one NFL game. Therefore player rows
    # cannot be used as the game denominator.
    # --------------------------------------------------------

    game_keys = (
        df[
            [
                "opponent",
                "recent_team",
                "week",
            ]
        ]
        .drop_duplicates()
        .rename(
            columns={
                "opponent":
                    "defense",

                "recent_team":
                    "offense_team",
            }
        )
    )

    defense_games = (
        game_keys
        .groupby(
            "defense",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "games"
            }
        )
    )

    recent_game_keys = (
        game_keys[
            game_keys["week"]
            .between(
                recent_start,
                latest_week,
            )
        ]
    )

    recent_games = (
        recent_game_keys
        .groupby(
            "defense",
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "size":
                    "last3_games"
            }
        )
    )

    # --------------------------------------------------------
    # SEASON TOTALS
    # --------------------------------------------------------

    work = (
        df.rename(
            columns={
                "opponent":
                    "defense"
            }
        )
        .copy()
    )

    aggregation = {

        "fantasy_points_ppr":
            "sum",

        "passing_yards":
            "sum",

        "passing_tds":
            "sum",

        "interceptions":
            "sum",

        "rushing_yards":
            "sum",

        "rushing_tds":
            "sum",

        "targets":
            "sum",

        "receptions":
            "sum",

        "receiving_yards":
            "sum",

        "receiving_tds":
            "sum",

    }

    season = (
        work
        .groupby(
            [
                "defense",
                "position",
            ],
            as_index=False,
        )
        .agg(
            aggregation
        )
        .rename(
            columns={

                "fantasy_points_ppr":
                    "fantasy_points_ppr_allowed",

                "passing_yards":
                    "passing_yards_allowed",

                "passing_tds":
                    "passing_tds_allowed",

                "rushing_yards":
                    "rushing_yards_allowed",

                "rushing_tds":
                    "rushing_tds_allowed",

                "targets":
                    "targets_allowed",

                "receptions":
                    "receptions_allowed",

                "receiving_yards":
                    "receiving_yards_allowed",

                "receiving_tds":
                    "receiving_tds_allowed",

            }
        )
    )

    # --------------------------------------------------------
    # RECENT / LAST THREE WEEKS
    # --------------------------------------------------------

    recent = work[
        work["week"]
        .between(
            recent_start,
            latest_week,
        )
    ]

    recent_totals = (
        recent
        .groupby(
            [
                "defense",
                "position",
            ],
            as_index=False,
        )
        ["fantasy_points_ppr"]
        .sum()
        .rename(
            columns={
                "fantasy_points_ppr":
                    "last3_fantasy_points_ppr_allowed"
            }
        )
    )

    # --------------------------------------------------------
    # BUILD ALL 128 COMBINATIONS
    # --------------------------------------------------------

    result = complete_grid(
        df
    )

    result = (
        result.merge(
            season,
            on=[
                "defense",
                "position",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    result = (
        result.merge(
            defense_games,
            on="defense",
            how="left",
            validate="many_to_one",
        )
    )

    result = (
        result.merge(
            recent_totals,
            on=[
                "defense",
                "position",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    result = (
        result.merge(
            recent_games,
            on="defense",
            how="left",
            validate="many_to_one",
        )
    )

    # --------------------------------------------------------
    # CLEAN NUMERIC VALUES
    # --------------------------------------------------------

    total_columns = [

        "fantasy_points_ppr_allowed",

        "passing_yards_allowed",
        "passing_tds_allowed",
        "interceptions",

        "rushing_yards_allowed",
        "rushing_tds_allowed",

        "targets_allowed",
        "receptions_allowed",
        "receiving_yards_allowed",
        "receiving_tds_allowed",

        "last3_fantasy_points_ppr_allowed",

    ]

    for column in total_columns:

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .fillna(0)
        )

    result["games"] = (
        pd.to_numeric(
            result["games"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    result["last3_games"] = (
        pd.to_numeric(
            result["last3_games"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # PER GAME CALCULATIONS
    # --------------------------------------------------------

    result[
        "ppr_allowed_per_game"
    ] = (
        safe_divide(
            result[
                "fantasy_points_ppr_allowed"
            ],
            result["games"],
        )
        .fillna(0)
    )

    result[
        "last3_ppr_allowed_per_game"
    ] = (
        safe_divide(
            result[
                "last3_fantasy_points_ppr_allowed"
            ],
            result[
                "last3_games"
            ],
        )
        .fillna(0)
    )

    # --------------------------------------------------------
    # POSITION RANKINGS
    #
    # Rank 1 = MOST PPR points allowed per game.
    #
    # Therefore:
    # lower rank number = more fantasy production
    # allowed to that position.
    # --------------------------------------------------------

    result[
        "season_rank"
    ] = (
        result
        .groupby(
            "position"
        )[
            "ppr_allowed_per_game"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    result[
        "last3_rank"
    ] = (
        result
        .groupby(
            "position"
        )[
            "last3_ppr_allowed_per_game"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype("Int64")
    )

    # --------------------------------------------------------
    # METADATA
    # --------------------------------------------------------

    result["season"] = SEASON

    result[
        "through_week"
    ] = latest_week

    result[
        "recent_start_week"
    ] = recent_start

    # --------------------------------------------------------
    # SORT OUTPUT
    # --------------------------------------------------------

    position_order = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 4,
    }

    result[
        "_position_order"
    ] = (
        result["position"]
        .map(
            position_order
        )
    )

    result = (
        result
        .sort_values(
            [
                "defense",
                "_position_order",
            ]
        )
        .drop(
            columns=[
                "_position_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    validate_matchups(
        result,
        latest_week,
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_matchups(
    df: pd.DataFrame,
    latest_week: int,
) -> None:

    defenses = set(
        df["defense"]
        .dropna()
        .unique()
    )

    positions = set(
        df["position"]
        .dropna()
        .unique()
    )

    duplicates = int(
        df.duplicated(
            subset=[
                "defense",
                "position",
            ]
        )
        .sum()
    )

    print(
        f"Defenses represented: "
        f"{len(defenses)}/32"
    )

    print(
        "Positions represented: "
        + ", ".join(
            sorted(
                positions
            )
        )
    )

    print(
        f"Defense-position records: "
        f"{len(df):,}"
    )

    print(
        f"Latest NFL week: "
        f"{latest_week}"
    )

    if len(defenses) != 32:

        fail(
            "Expected 32 defenses; "
            f"found {len(defenses)}."
        )

    missing_positions = (
        ALLOWED_POSITIONS
        - positions
    )

    if missing_positions:

        fail(
            "Missing fantasy positions: "
            + ", ".join(
                sorted(
                    missing_positions
                )
            )
        )

    if len(df) != 128:

        fail(
            "Expected 128 "
            "defense-position records; "
            f"found {len(df)}."
        )

    if duplicates:

        fail(
            f"Found {duplicates} "
            "duplicate defense-position "
            "records."
        )

    if (
        df["games"] <= 0
    ).any():

        fail(
            "At least one defense "
            "has zero represented games."
        )

    if (
        df[
            "ppr_allowed_per_game"
        ]
        .isna()
        .any()
    ):

        fail(
            "Season PPR allowed/game "
            "contains missing values."
        )

    if (
        df[
            "last3_ppr_allowed_per_game"
        ]
        .isna()
        .any()
    ):

        fail(
            "Last-3 PPR allowed/game "
            "contains missing values."
        )

    print(
        "MATCHUP DATA VALIDATION PASSED"
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
        download_player_stats()
    )

    players = (
        normalize_player_data(
            raw
        )
    )

    print()

    print(
        f"Eligible player-week records: "
        f"{len(players):,}"
    )

    print(
        f"Teams represented: "
        f"{players['recent_team'].nunique()}/32"
    )

    print(
        f"Opponents represented: "
        f"{players['opponent'].nunique()}/32"
    )

    print(
        f"Latest week: "
        f"{int(players['week'].max())}"
    )

    matchups = (
        build_matchups(
            players
        )
    )

    records = (
        dataframe_records(
            matchups
        )
    )

    write_json(
        MATCHUP_OUTPUT,
        records,
    )

    print()
    print("=" * 72)
    print(
        "MATCHUP DATA UPDATE COMPLETE"
    )
    print("=" * 72)

    print(
        f"Records: "
        f"{len(records):,}"
    )

    print(
        f"Output: "
        f"{MATCHUP_OUTPUT}"
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

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

BASE_RELEASE_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download"
)

PLAYER_STATS_URL = (
    f"{BASE_RELEASE_URL}/stats_player/"
    f"stats_player_week_{SEASON}.csv"
)

SNAP_COUNTS_URL = (
    f"{BASE_RELEASE_URL}/snap_counts/"
    f"snap_counts_{SEASON}.csv"
)

ADV_PASS_URL = (
    f"{BASE_RELEASE_URL}/pfr_advstats/"
    f"advstats_week_pass_{SEASON}.csv"
)

ADV_RUSH_URL = (
    f"{BASE_RELEASE_URL}/pfr_advstats/"
    f"advstats_week_rush_{SEASON}.csv"
)

ADV_REC_URL = (
    f"{BASE_RELEASE_URL}/pfr_advstats/"
    f"advstats_week_rec_{SEASON}.csv"
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

MIN_SNAP_MATCH_RATE = 0.70


# ============================================================
# ERROR HANDLING
# ============================================================

def fail(message: str) -> None:

    print()
    print("=" * 72, file=sys.stderr)
    print(
        f"VALIDATION FAILED: {message}",
        file=sys.stderr,
    )
    print("=" * 72, file=sys.stderr)

    sys.exit(1)


# ============================================================
# DOWNLOAD
# ============================================================

def download_csv(
    url: str,
    label: str,
) -> pd.DataFrame:

    print()
    print("=" * 72)
    print(f"DOWNLOADING: {label}")
    print("=" * 72)
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
            f"Unable to download {label}: {exc}"
        )

    size = len(response.content)

    print(
        f"HTTP status: {response.status_code}"
    )

    print(
        f"Downloaded bytes: {size:,}"
    )

    print(
        "Content type:",
        response.headers.get(
            "content-type",
            "unknown",
        ),
    )

    beginning = (
        response.content[:500].lower()
    )

    if (
        b"<html" in beginning
        or b"<!doctype html" in beginning
    ):

        fail(
            f"{label} returned HTML "
            "instead of CSV."
        )

    if size < 500:

        fail(
            f"{label} download is "
            f"unexpectedly small "
            f"({size:,} bytes)."
        )

    try:

        df = pd.read_csv(
            io.BytesIO(
                response.content
            )
        )

    except Exception as exc:

        fail(
            f"Unable to parse "
            f"{label}: {exc}"
        )

    print(
        f"Parsed successfully: "
        f"{len(df):,} rows x "
        f"{len(df.columns):,} columns"
    )

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_name(value) -> str:

    if pd.isna(value):
        return ""

    value = str(value)

    value = unicodedata.normalize(
        "NFKD",
        value,
    )

    value = "".join(
        character
        for character in value
        if not unicodedata.combining(
            character
        )
    )

    value = (
        value
        .lower()
        .strip()
    )

    value = value.replace(
        "’",
        "'",
    )

    value = re.sub(
        r"[.'`-]",
        "",
        value,
    )

    value = re.sub(
        r"\b(jr|sr|ii|iii|iv|v)\b",
        "",
        value,
    )

    value = re.sub(
        r"[^a-z0-9]",
        "",
        value,
    )

    return value


def normalize_team(value) -> str:

    if pd.isna(value):
        return ""

    team = (
        str(value)
        .upper()
        .strip()
    )

    aliases = {
        "JAX": "JAC",
        "WSH": "WAS",
        "OAK": "LV",
        "SD": "LAC",
        "STL": "LA",
        "LAR": "LA",
    }

    return aliases.get(
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
# PLAYER DATA
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
            for column in team_candidates
            if column in df.columns
        ),
        None,
    )

    if team_column is None:

        fail(
            "No recognized team column "
            "in player stats."
        )

    if (
        team_column
        != "recent_team"
    ):

        df["recent_team"] = (
            df[team_column]
        )

    if (
        "player_display_name"
        not in df.columns
    ):

        candidates = [
            "player_name",
            "name",
        ]

        player_column = next(
            (
                column
                for column in candidates
                if column in df.columns
            ),
            None,
        )

        if player_column is None:

            fail(
                "No recognized player-name "
                "column."
            )

        df["player_display_name"] = (
            df[player_column]
        )

    required = {
        "player_id",
        "player_display_name",
        "position",
        "recent_team",
        "season",
        "week",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        fail(
            "Player stats missing: "
            + ", ".join(
                sorted(missing)
            )
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

    df["_join_name"] = (
        df["player_display_name"]
        .apply(normalize_name)
    )

    df["season"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    df["week"] = pd.to_numeric(
        df["week"],
        errors="coerce",
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
    ].copy()

    df["week"] = (
        df["week"]
        .astype(int)
    )

    return df


def validate_player_data(
    df: pd.DataFrame,
) -> None:

    if len(df) < 500:

        fail(
            "Player dataset contains "
            "too few records."
        )

    teams = set(
        df["recent_team"]
        .dropna()
        .unique()
    )

    if len(teams) != 32:

        fail(
            f"Expected 32 teams; "
            f"found {len(teams)}."
        )

    positions = set(
        df["position"]
        .dropna()
        .unique()
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

    latest_week = int(
        df["week"].max()
    )

    if not 1 <= latest_week <= 22:

        fail(
            f"Unexpected latest week: "
            f"{latest_week}"
        )

    print()
    print(
        "PLAYER DATA VALIDATION PASSED"
    )

    print(
        f"Player-week records: "
        f"{len(df):,}"
    )

    print(
        f"Teams: {len(teams)}/32"
    )

    print(
        f"Latest week: "
        f"{latest_week}"
    )


# ============================================================
# SNAP DATA
# ============================================================

def normalize_snap_data(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

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

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        fail(
            "Snap data missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    df["season"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    df["week"] = pd.to_numeric(
        df["week"],
        errors="coerce",
    )

    df = df[
        (df["season"] == SEASON)
        &
        (
            df["week"]
            .notna()
        )
    ].copy()

    df["week"] = (
        df["week"]
        .astype(int)
    )

    df["team"] = (
        df["team"]
        .apply(normalize_team)
    )

    df["opponent"] = (
        df["opponent"]
        .apply(normalize_team)
    )

    df["_join_name"] = (
        df["player"]
        .apply(normalize_name)
    )

    df["offense_snaps"] = (
        pd.to_numeric(
            df["offense_snaps"],
            errors="coerce",
        )
    )

    df["offense_pct"] = (
        pd.to_numeric(
            df["offense_pct"],
            errors="coerce",
        )
    )

    return df


def join_snap_data(
    players: pd.DataFrame,
    snaps: pd.DataFrame,
):

    print()
    print("=" * 72)
    print("JOINING SNAP COUNTS")
    print("=" * 72)

    snap_keep = [
        "_join_name",
        "team",
        "week",
        "opponent",
        "offense_snaps",
        "offense_pct",
    ]

    snap_join = (
        snaps[snap_keep]
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
        snap_join,
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

    matched = int(
        joined[
            "offense_snaps"
        ]
        .notna()
        .sum()
    )

    total = len(joined)

    match_rate = (
        matched / total
        if total
        else 0
    )

    print(
        f"Matched: "
        f"{matched:,}/{total:,}"
    )

    print(
        f"Snap match rate: "
        f"{match_rate:.1%}"
    )

    if (
        match_rate
        < MIN_SNAP_MATCH_RATE
    ):

        fail(
            "Snap match rate below "
            f"{MIN_SNAP_MATCH_RATE:.0%}."
        )

    return (
        joined,
        matched,
        match_rate,
    )


# ============================================================
# ADVANCED DATA
# ============================================================

def normalize_advanced_data(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:

    df = df.copy()

    required = {
        "season",
        "week",
        "team",
        "opponent",
        "pfr_player_name",
        "pfr_player_id",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        fail(
            f"{prefix} advanced data missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    df["season"] = pd.to_numeric(
        df["season"],
        errors="coerce",
    )

    df["week"] = pd.to_numeric(
        df["week"],
        errors="coerce",
    )

    df = df[
        (df["season"] == SEASON)
        &
        (
            df["week"]
            .notna()
        )
    ].copy()

    df["week"] = (
        df["week"]
        .astype(int)
    )

    df["team"] = (
        df["team"]
        .apply(normalize_team)
    )

    df["opponent"] = (
        df["opponent"]
        .apply(normalize_team)
    )

    df["_join_name"] = (
        df["pfr_player_name"]
        .apply(normalize_name)
    )

    protected = {
        "_join_name",
        "season",
        "week",
        "team",
        "opponent",
        "pfr_player_name",
        "pfr_player_id",
        "game_id",
        "pfr_game_id",
        "game_type",
    }

    rename_map = {}

    for column in df.columns:

        if column not in protected:

            rename_map[column] = (
                f"adv_{prefix}_{column}"
            )

    df = df.rename(
        columns=rename_map
    )

    df = df.rename(
        columns={
            "pfr_player_id":
                f"adv_{prefix}_pfr_player_id",
            "pfr_player_name":
                f"adv_{prefix}_pfr_player_name",
        }
    )

    return df


def join_advanced_data(
    players: pd.DataFrame,
    advanced: pd.DataFrame,
    prefix: str,
    eligibility_column: str | None,
):

    print()
    print("=" * 72)
    print(
        f"JOINING ADVANCED "
        f"{prefix.upper()}"
    )
    print("=" * 72)

    exclude = {
        "season",
        "opponent",
        "game_id",
        "pfr_game_id",
        "game_type",
    }

    keep_columns = [
        column
        for column
        in advanced.columns
        if column not in exclude
    ]

    advanced_join = (
        advanced[
            keep_columns
        ]
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
        advanced_join,
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
        suffixes=(
            "",
            f"_{prefix}",
        ),
    )

    joined.drop(
        columns=["team"],
        inplace=True,
        errors="ignore",
    )

    id_column = (
        f"adv_{prefix}_pfr_player_id"
    )

    matched_mask = (
        joined[id_column]
        .notna()
    )

    total_matches = int(
        matched_mask.sum()
    )

    if (
        eligibility_column
        and
        eligibility_column
        in joined.columns
    ):

        eligible_values = (
            pd.to_numeric(
                joined[
                    eligibility_column
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        eligible_mask = (
            eligible_values > 0
        )

    else:

        eligible_mask = (
            joined["position"]
            .notna()
        )

    eligible_count = int(
        eligible_mask.sum()
    )

    eligible_matches = int(
        (
            eligible_mask
            &
            matched_mask
        ).sum()
    )

    coverage = (
        eligible_matches
        / eligible_count
        if eligible_count
        else 0
    )

    print(
        f"Eligible player-weeks: "
        f"{eligible_count:,}"
    )

    print(
        f"Eligible matches: "
        f"{eligible_matches:,}"
    )

    print(
        f"Coverage: "
        f"{coverage:.1%}"
    )

    return (
        joined,
        {
            "eligible_records":
                eligible_count,
            "matched_records":
                eligible_matches,
            "coverage":
                coverage,
            "total_matches":
                total_matches,
        },
    )


# ============================================================
# DERIVED METRICS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):

    numerator = pd.to_numeric(
        numerator,
        errors="coerce",
    )

    denominator = pd.to_numeric(
        denominator,
        errors="coerce",
    )

    denominator = (
        denominator
        .replace(
            0,
            float("nan"),
        )
    )

    return (
        numerator
        / denominator
    )


def add_derived_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if (
        "carries" in df.columns
        and
        "receptions" in df.columns
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

    if (
        "carries" in df.columns
        and
        "targets" in df.columns
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

    if (
        "targets" in df.columns
        and
        "offense_snaps" in df.columns
    ):

        df["targets_per_snap"] = (
            safe_divide(
                df["targets"],
                df["offense_snaps"],
            )
        )

    if (
        "carries" in df.columns
        and
        "offense_snaps" in df.columns
    ):

        df["carries_per_snap"] = (
            safe_divide(
                df["carries"],
                df["offense_snaps"],
            )
        )

    if (
        "fantasy_points"
        in df.columns
    ):

        df[
            "fantasy_points_per_snap"
        ] = safe_divide(
            df["fantasy_points"],
            df["offense_snaps"],
        )

        df[
            "fantasy_points_per_100_snaps"
        ] = (
            df[
                "fantasy_points_per_snap"
            ]
            * 100
        )

    if (
        "fantasy_points_ppr"
        in df.columns
    ):

        df[
            "ppr_points_per_snap"
        ] = safe_divide(
            df["fantasy_points_ppr"],
            df["offense_snaps"],
        )

        df[
            "ppr_points_per_100_snaps"
        ] = (
            df[
                "ppr_points_per_snap"
            ]
            * 100
        )

    if (
        "receiving_yards"
        in df.columns
        and
        "targets"
        in df.columns
    ):

        df[
            "receiving_yards_per_target"
        ] = safe_divide(
            df["receiving_yards"],
            df["targets"],
        )

    if (
        "receptions" in df.columns
        and
        "targets" in df.columns
    ):

        df["catch_rate"] = (
            safe_divide(
                df["receptions"],
                df["targets"],
            )
        )

    if (
        "rushing_yards"
        in df.columns
        and
        "carries"
        in df.columns
    ):

        df[
            "rushing_yards_per_carry_calc"
        ] = safe_divide(
            df["rushing_yards"],
            df["carries"],
        )

    if (
        "adv_rush_rushing_yards_after_contact"
        in df.columns
        and
        "carries"
        in df.columns
    ):

        df[
            "yards_after_contact_per_carry"
        ] = safe_divide(
            df[
                "adv_rush_rushing_yards_after_contact"
            ],
            df["carries"],
        )

    if (
        "adv_rush_rushing_yards_before_contact"
        in df.columns
        and
        "carries"
        in df.columns
    ):

        df[
            "yards_before_contact_per_carry"
        ] = safe_divide(
            df[
                "adv_rush_rushing_yards_before_contact"
            ],
            df["carries"],
        )

    return df


# ============================================================
# OUTPUT
# ============================================================

def dataframe_records(
    df: pd.DataFrame,
):

    df = df.drop(
        columns=[
            "_join_name",
        ],
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

        temporary.replace(
            path
        )

    except Exception as exc:

        if temporary.exists():
            temporary.unlink()

        fail(
            f"Unable to write "
            f"{path}: {exc}"
        )


# ============================================================
# STATUS
# ============================================================

def build_status(
    df,
    snap_stats,
    pass_stats,
    rush_stats,
    rec_stats,
):

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

    teams = sorted(
        df["recent_team"]
        .dropna()
        .unique()
        .tolist()
    )

    position_counts = (
        players["position"]
        .value_counts()
        .to_dict()
    )

    return {

        "status":
            "validated",

        "season":
            SEASON,

        "latest_week":
            int(
                df["week"].max()
            ),

        "generated_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "player_week_records":
            int(len(df)),

        "unique_players":
            int(len(players)),

        "teams_found":
            int(len(teams)),

        "teams":
            teams,

        "position_counts": {
            str(key): int(value)
            for key, value
            in position_counts.items()
        },

        "sources": {

            "player_stats": {
                "status":
                    "loaded",
                "provider":
                    "nflverse",
            },

            "snap_counts": {
                "status":
                    "loaded",
                **snap_stats,
            },

            "advanced_passing": {
                "status":
                    "loaded",
                **pass_stats,
            },

            "advanced_rushing": {
                "status":
                    "loaded",
                **rush_stats,
            },

            "advanced_receiving": {
                "status":
                    "loaded",
                **rec_stats,
            },
        },

        "license":
            "CC BY 4.0",

        "attribution": (
            "NFL statistics provided "
            "through nflverse. "
            "See nflverse-data and "
            "applicable dataset "
            "documentation for source "
            "and attribution details."
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 72)

    print(
        f"NFL PLAYER DASHBOARD — "
        f"{SEASON} CONSOLIDATED UPDATE"
    )

    print("=" * 72)

    # --------------------------------------------------------
    # PLAYER STATS
    # --------------------------------------------------------

    player_raw = download_csv(
        PLAYER_STATS_URL,
        "Player Summary Stats",
    )

    players = (
        normalize_player_data(
            player_raw
        )
    )

    validate_player_data(
        players
    )

    # --------------------------------------------------------
    # SNAP COUNTS
    # --------------------------------------------------------

    snap_raw = download_csv(
        SNAP_COUNTS_URL,
        "Snap Counts",
    )

    snaps = normalize_snap_data(
        snap_raw
    )

    (
        joined,
        snap_matches,
        snap_rate,
    ) = join_snap_data(
        players,
        snaps,
    )

    snap_stats = {
        "matched_records":
            snap_matches,
        "coverage":
            round(
                snap_rate,
                4,
            ),
    }

    # --------------------------------------------------------
    # ADVANCED PASSING
    # --------------------------------------------------------

    pass_raw = download_csv(
        ADV_PASS_URL,
        "Advanced Passing",
    )

    pass_data = (
        normalize_advanced_data(
            pass_raw,
            "pass",
        )
    )

    (
        joined,
        pass_stats,
    ) = join_advanced_data(
        joined,
        pass_data,
        "pass",
        "attempts",
    )

    # --------------------------------------------------------
    # ADVANCED RUSHING
    # --------------------------------------------------------

    rush_raw = download_csv(
        ADV_RUSH_URL,
        "Advanced Rushing",
    )

    rush_data = (
        normalize_advanced_data(
            rush_raw,
            "rush",
        )
    )

    (
        joined,
        rush_stats,
    ) = join_advanced_data(
        joined,
        rush_data,
        "rush",
        "carries",
    )

    # --------------------------------------------------------
    # ADVANCED RECEIVING
    # --------------------------------------------------------

    rec_raw = download_csv(
        ADV_REC_URL,
        "Advanced Receiving",
    )

    rec_data = (
        normalize_advanced_data(
            rec_raw,
            "rec",
        )
    )

    (
        joined,
        rec_stats,
    ) = join_advanced_data(
        joined,
        rec_data,
        "rec",
        "targets",
    )

    # --------------------------------------------------------
    # DERIVED METRICS
    # --------------------------------------------------------

    joined = (
        add_derived_metrics(
            joined
        )
    )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if len(joined) != len(players):

        fail(
            "Advanced joins changed "
            "the number of player-week "
            "records."
        )

    teams = set(
        joined["recent_team"]
        .dropna()
        .unique()
    )

    if len(teams) != 32:

        fail(
            "Final dataset does not "
            "contain all 32 teams."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    records = (
        dataframe_records(
            joined
        )
    )

    status = build_status(
        joined,
        snap_stats,
        pass_stats,
        rush_stats,
        rec_stats,
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
    # REPORT
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("SUCCESS")
    print("=" * 72)

    print()
    print("PLAYER DATA")

    print(
        f"Player-week records: "
        f"{len(joined):,}"
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
    print("SNAP COUNTS")

    print(
        f"Matched: "
        f"{snap_matches:,}"
    )

    print(
        f"Coverage: "
        f"{snap_rate:.1%}"
    )

    print()
    print("ADVANCED PASSING")

    print(
        f"Eligible: "
        f"{pass_stats['eligible_records']:,}"
    )

    print(
        f"Matched: "
        f"{pass_stats['matched_records']:,}"
    )

    print(
        f"Coverage: "
        f"{pass_stats['coverage']:.1%}"
    )

    print()
    print("ADVANCED RUSHING")

    print(
        f"Eligible: "
        f"{rush_stats['eligible_records']:,}"
    )

    print(
        f"Matched: "
        f"{rush_stats['matched_records']:,}"
    )

    print(
        f"Coverage: "
        f"{rush_stats['coverage']:.1%}"
    )

    print()
    print("ADVANCED RECEIVING")

    print(
        f"Eligible: "
        f"{rec_stats['eligible_records']:,}"
    )

    print(
        f"Matched: "
        f"{rec_stats['matched_records']:,}"
    )

    print(
        f"Coverage: "
        f"{rec_stats['coverage']:.1%}"
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
    print(
        "FINAL VALIDATION: PASSED"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

SEASON = 2026

DATA_DIR = Path("data")

PLAYERS_FILE = DATA_DIR / "players.json"
MATCHUPS_FILE = DATA_DIR / "matchups.json"
SCHEDULE_FILE = DATA_DIR / "schedule.json"

OUTPUT_FILE = DATA_DIR / "player-matchups.json"

ALLOWED_POSITIONS = {
    "QB",
    "RB",
    "WR",
    "TE",
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

    if value is None:
        return ""

    team = str(value).upper().strip()

    return TEAM_ALIASES.get(
        team,
        team,
    )


def normalize_position(value) -> str:

    if value is None:
        return ""

    return (
        str(value)
        .upper()
        .strip()
    )


def safe_number(value):

    if value is None:
        return None

    if isinstance(
        value,
        (int, float),
    ):
        return value

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def load_json(path: Path):

    if not path.exists():

        fail(
            f"Required file does not exist: "
            f"{path}"
        )

    try:

        with path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(
                handle
            )

    except Exception as exc:

        fail(
            f"Unable to read {path}: "
            f"{exc}"
        )

    return data


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


def first_value(
    record: dict,
    candidates: list[str],
):

    for key in candidates:

        value = record.get(key)

        if (
            value is not None
            and
            value != ""
        ):
            return value

    return None


# ============================================================
# PLAYER FIELD HELPERS
# ============================================================

def get_player_name(
    record: dict,
):

    return first_value(
        record,
        [
            "player_name",
            "player_display_name",
            "display_name",
            "full_name",
            "name",
        ],
    )


def get_player_id(
    record: dict,
):

    return first_value(
        record,
        [
            "player_id",
            "gsis_id",
            "player_gsis_id",
            "id",
        ],
    )


def get_player_team(
    record: dict,
) -> str:

    value = first_value(
        record,
        [
            "recent_team",
            "team",
            "posteam",
        ],
    )

    return normalize_team(
        value
    )


def get_player_position(
    record: dict,
) -> str:

    return normalize_position(
        first_value(
            record,
            [
                "position",
                "position_group",
            ],
        )
    )


def get_player_week(
    record: dict,
):

    value = record.get(
        "week"
    )

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_inputs(
    players,
    matchups,
    schedule,
) -> None:

    print("=" * 72)
    print(
        "PLAYER MATCHUP DATA BUILD"
    )
    print("=" * 72)

    if not isinstance(
        players,
        list,
    ):

        fail(
            "players.json must contain "
            "a JSON array."
        )

    if not isinstance(
        matchups,
        list,
    ):

        fail(
            "matchups.json must contain "
            "a JSON array."
        )

    if not isinstance(
        schedule,
        list,
    ):

        fail(
            "schedule.json must contain "
            "a JSON array."
        )

    if not players:

        fail(
            "players.json is empty."
        )

    if not matchups:

        fail(
            "matchups.json is empty."
        )

    if not schedule:

        fail(
            "schedule.json is empty."
        )

    print(
        f"Player records loaded: "
        f"{len(players):,}"
    )

    print(
        f"Matchup records loaded: "
        f"{len(matchups):,}"
    )

    print(
        f"Schedule records loaded: "
        f"{len(schedule):,}"
    )


# ============================================================
# DETERMINE LATEST COMPLETED WEEK
# ============================================================

def determine_latest_week(
    players: list[dict],
) -> int:

    weeks = []

    for record in players:

        week = get_player_week(
            record
        )

        if week is not None:

            weeks.append(
                week
            )

    if not weeks:

        fail(
            "Unable to determine the "
            "latest completed week from "
            "players.json."
        )

    latest_week = max(
        weeks
    )

    if latest_week < 1:

        fail(
            "Latest completed week "
            "is invalid."
        )

    return latest_week


# ============================================================
# BUILD SCHEDULE LOOKUP
# ============================================================

def build_schedule_lookup(
    schedule: list[dict],
) -> dict:

    lookup = {}

    duplicate_count = 0

    for record in schedule:

        try:
            season = int(
                record.get(
                    "season"
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if season != SEASON:
            continue

        try:
            week = int(
                record.get(
                    "week"
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        team = normalize_team(
            record.get(
                "team"
            )
        )

        opponent = normalize_team(
            record.get(
                "opponent"
            )
        )

        if not team:
            continue

        key = (
            team,
            week,
        )

        if key in lookup:

            duplicate_count += 1

            continue

        lookup[key] = {

            "season":
                season,

            "week":
                week,

            "team":
                team,

            "opponent":
                opponent or None,

            "home_away":
                record.get(
                    "home_away"
                ),

            "game_date":
                record.get(
                    "game_date"
                ),

            "game_type":
                record.get(
                    "game_type"
                ),

        }

    if duplicate_count:

        fail(
            "Schedule lookup found "
            f"{duplicate_count} duplicate "
            "team/week records."
        )

    return lookup


# ============================================================
# BUILD MATCHUP LOOKUP
# ============================================================

def build_matchup_lookup(
    matchups: list[dict],
) -> dict:

    lookup = {}

    duplicate_count = 0

    for record in matchups:

        defense = normalize_team(
            record.get(
                "defense"
            )
        )

        position = (
            normalize_position(
                record.get(
                    "position"
                )
            )
        )

        if (
            not defense
            or
            position
            not in ALLOWED_POSITIONS
        ):
            continue

        key = (
            defense,
            position,
        )

        if key in lookup:

            duplicate_count += 1

            continue

        lookup[key] = record

    if duplicate_count:

        fail(
            "Matchup lookup found "
            f"{duplicate_count} duplicate "
            "defense/position records."
        )

    expected = (
        32
        *
        len(
            ALLOWED_POSITIONS
        )
    )

    if len(lookup) != expected:

        fail(
            "Expected "
            f"{expected} defense/position "
            "matchup records; found "
            f"{len(lookup)}."
        )

    return lookup


# ============================================================
# DETERMINE CURRENT PLAYER SNAPSHOT
# ============================================================

def latest_player_records(
    players: list[dict],
    latest_week: int,
) -> list[dict]:

    """
    players.json currently contains player-week
    records.

    For the upcoming matchup view we only need
    one current record per fantasy player.

    Prefer the player's latest available record
    at or before the latest completed NFL week.
    """

    latest_by_player = {}

    skipped_position = 0
    skipped_team = 0
    skipped_identity = 0

    for record in players:

        position = (
            get_player_position(
                record
            )
        )

        if (
            position
            not in ALLOWED_POSITIONS
        ):

            skipped_position += 1

            continue

        team = get_player_team(
            record
        )

        if not team:

            skipped_team += 1

            continue

        week = get_player_week(
            record
        )

        if (
            week is None
            or
            week > latest_week
        ):
            continue

        player_id = get_player_id(
            record
        )

        player_name = (
            get_player_name(
                record
            )
        )

        # Player ID is preferred because
        # names can occasionally collide.
        #
        # If an ID is unavailable, fall back
        # to name + position.

        if player_id:

            identity = (
                "ID",
                str(player_id),
            )

        elif player_name:

            identity = (
                "NAME",
                str(
                    player_name
                ).upper(),
                position,
            )

        else:

            skipped_identity += 1

            continue

        existing = (
            latest_by_player
            .get(
                identity
            )
        )

        if (
            existing is None
            or
            week >
            existing["_week"]
        ):

            latest_by_player[
                identity
            ] = {
                "_week":
                    week,

                "record":
                    record,
            }

    result = [
        value["record"]
        for value
        in latest_by_player.values()
    ]

    print()
    print(
        "CURRENT PLAYER SNAPSHOT"
    )

    print(
        f"Unique fantasy players: "
        f"{len(result):,}"
    )

    print(
        f"Skipped non-QB/RB/WR/TE "
        f"records: {skipped_position:,}"
    )

    print(
        f"Skipped records without "
        f"team: {skipped_team:,}"
    )

    print(
        f"Skipped records without "
        f"player identity: "
        f"{skipped_identity:,}"
    )

    if not result:

        fail(
            "No current fantasy player "
            "records were created."
        )

    return result


# ============================================================
# BUILD PLAYER MATCHUPS
# ============================================================

def build_player_matchups(
    players: list[dict],
    schedule_lookup: dict,
    matchup_lookup: dict,
    latest_week: int,
) -> list[dict]:

    upcoming_week = (
        latest_week + 1
    )

    print()
    print("=" * 72)
    print(
        "BUILDING UPCOMING PLAYER MATCHUPS"
    )
    print("=" * 72)

    print(
        f"Latest completed stats week: "
        f"{latest_week}"
    )

    print(
        f"Upcoming fantasy week: "
        f"{upcoming_week}"
    )

    current_players = (
        latest_player_records(
            players,
            latest_week,
        )
    )

    output = []

    bye_count = 0
    matched_count = 0
    missing_matchup_count = 0

    for player in current_players:

        player_name = (
            get_player_name(
                player
            )
        )

        player_id = (
            get_player_id(
                player
            )
        )

        team = (
            get_player_team(
                player
            )
        )

        position = (
            get_player_position(
                player
            )
        )

        schedule_record = (
            schedule_lookup.get(
                (
                    team,
                    upcoming_week,
                )
            )
        )

        # ----------------------------------------------------
        # BYE WEEK
        #
        # There is no team/week schedule record when a team
        # has a bye.
        # ----------------------------------------------------

        if schedule_record is None:

            bye_count += 1

            output.append(
                {
                    "season":
                        SEASON,

                    "through_week":
                        latest_week,

                    "upcoming_week":
                        upcoming_week,

                    "player_id":
                        player_id,

                    "player_name":
                        player_name,

                    "position":
                        position,

                    "team":
                        team,

                    "opponent":
                        None,

                    "home_away":
                        None,

                    "game_date":
                        None,

                    "bye":
                        True,

                    "matchup_available":
                        False,

                    "opponent_ppr_allowed_per_game":
                        None,

                    "opponent_matchup_rank":
                        None,

                    "opponent_last3_ppr_allowed_per_game":
                        None,

                    "opponent_last3_matchup_rank":
                        None,
                }
            )

            continue

        opponent = normalize_team(
            schedule_record.get(
                "opponent"
            )
        )

        matchup = (
            matchup_lookup.get(
                (
                    opponent,
                    position,
                )
            )
        )

        if matchup is None:

            missing_matchup_count += 1

            output.append(
                {
                    "season":
                        SEASON,

                    "through_week":
                        latest_week,

                    "upcoming_week":
                        upcoming_week,

                    "player_id":
                        player_id,

                    "player_name":
                        player_name,

                    "position":
                        position,

                    "team":
                        team,

                    "opponent":
                        opponent,

                    "home_away":
                        schedule_record.get(
                            "home_away"
                        ),

                    "game_date":
                        schedule_record.get(
                            "game_date"
                        ),

                    "bye":
                        False,

                    "matchup_available":
                        False,

                    "opponent_ppr_allowed_per_game":
                        None,

                    "opponent_matchup_rank":
                        None,

                    "opponent_last3_ppr_allowed_per_game":
                        None,

                    "opponent_last3_matchup_rank":
                        None,
                }
            )

            continue

        matched_count += 1

        output.append(
            {
                "season":
                    SEASON,

                "through_week":
                    latest_week,

                "upcoming_week":
                    upcoming_week,

                "player_id":
                    player_id,

                "player_name":
                    player_name,

                "position":
                    position,

                "team":
                    team,

                "opponent":
                    opponent,

                "home_away":
                    schedule_record.get(
                        "home_away"
                    ),

                "game_date":
                    schedule_record.get(
                        "game_date"
                    ),

                "bye":
                    False,

                "matchup_available":
                    True,

                "opponent_ppr_allowed_per_game":
                    safe_number(
                        matchup.get(
                            "ppr_allowed_per_game"
                        )
                    ),

                "opponent_matchup_rank":
                    safe_number(
                        matchup.get(
                            "season_rank"
                        )
                    ),

                "opponent_last3_ppr_allowed_per_game":
                    safe_number(
                        matchup.get(
                            "last3_ppr_allowed_per_game"
                        )
                    ),

                "opponent_last3_matchup_rank":
                    safe_number(
                        matchup.get(
                            "last3_rank"
                        )
                    ),
            }
        )

    print()
    print(
        f"Player matchup records: "
        f"{len(output):,}"
    )

    print(
        f"Matched upcoming games: "
        f"{matched_count:,}"
    )

    print(
        f"Players on bye: "
        f"{bye_count:,}"
    )

    print(
        f"Missing matchup records: "
        f"{missing_matchup_count:,}"
    )

    return output


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_output(
    records: list[dict],
    latest_week: int,
) -> None:

    print()
    print("=" * 72)
    print(
        "VALIDATING PLAYER MATCHUPS"
    )
    print("=" * 72)

    if not records:

        fail(
            "Player matchup output "
            "is empty."
        )

    upcoming_week = (
        latest_week + 1
    )

    teams = {
        record["team"]
        for record
        in records
        if record.get(
            "team"
        )
    }

    positions = {
        record["position"]
        for record
        in records
        if record.get(
            "position"
        )
    }

    bye_teams = {
        record["team"]
        for record
        in records
        if record.get(
            "bye"
        )
    }

    invalid_week = [
        record
        for record
        in records
        if record.get(
            "upcoming_week"
        )
        != upcoming_week
    ]

    invalid_positions = (
        positions
        -
        ALLOWED_POSITIONS
    )

    matched_non_bye = [
        record
        for record
        in records
        if (
            not record.get(
                "bye"
            )
            and
            record.get(
                "matchup_available"
            )
        )
    ]

    unmatched_non_bye = [
        record
        for record
        in records
        if (
            not record.get(
                "bye"
            )
            and
            not record.get(
                "matchup_available"
            )
        )
    ]

    print(
        f"Teams represented: "
        f"{len(teams)}/32"
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
        f"Upcoming week: "
        f"{upcoming_week}"
    )

    print(
        f"Bye teams: "
        f"{len(bye_teams)}"
    )

    if bye_teams:

        print(
            "Bye team list: "
            + ", ".join(
                sorted(
                    bye_teams
                )
            )
        )

    print(
        f"Matched non-bye players: "
        f"{len(matched_non_bye):,}"
    )

    print(
        f"Unmatched non-bye players: "
        f"{len(unmatched_non_bye):,}"
    )

    if invalid_week:

        fail(
            "At least one player matchup "
            "has the wrong upcoming week."
        )

    if invalid_positions:

        fail(
            "Output contains invalid "
            "positions: "
            + ", ".join(
                sorted(
                    invalid_positions
                )
            )
        )

    # We expect the current fantasy-player
    # snapshot to cover essentially all NFL
    # teams. Requiring at least 30 allows for
    # edge cases where a team has no qualifying
    # player in the source snapshot.

    if len(teams) < 30:

        fail(
            "Player matchup output only "
            f"represents {len(teams)} teams."
        )

    # A scheduled non-bye player should always
    # find the opponent/position matchup because
    # matchups.json contains all 32 defenses x
    # four fantasy positions.

    if unmatched_non_bye:

        examples = (
            unmatched_non_bye[:5]
        )

        example_text = "; ".join(
            (
                f"{item.get('player_name')} "
                f"{item.get('position')} "
                f"{item.get('team')} vs "
                f"{item.get('opponent')}"
            )
            for item
            in examples
        )

        fail(
            "Some scheduled players could "
            "not be matched to opponent "
            "defense data. Examples: "
            + example_text
        )

    print(
        "PLAYER MATCHUP VALIDATION PASSED"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    players = load_json(
        PLAYERS_FILE
    )

    matchups = load_json(
        MATCHUPS_FILE
    )

    schedule = load_json(
        SCHEDULE_FILE
    )

    validate_inputs(
        players,
        matchups,
        schedule,
    )

    latest_week = (
        determine_latest_week(
            players
        )
    )

    schedule_lookup = (
        build_schedule_lookup(
            schedule
        )
    )

    matchup_lookup = (
        build_matchup_lookup(
            matchups
        )
    )

    records = (
        build_player_matchups(
            players,
            schedule_lookup,
            matchup_lookup,
            latest_week,
        )
    )

    validate_output(
        records,
        latest_week,
    )

    write_json(
        OUTPUT_FILE,
        records,
    )

    print()
    print("=" * 72)
    print(
        "PLAYER MATCHUP BUILD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Through Week: "
        f"{latest_week}"
    )

    print(
        f"Upcoming Week: "
        f"{latest_week + 1}"
    )

    print(
        f"Records: "
        f"{len(records):,}"
    )

    print(
        f"Output: "
        f"{OUTPUT_FILE}"
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

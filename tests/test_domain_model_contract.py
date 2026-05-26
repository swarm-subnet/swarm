import json
from copy import deepcopy

from swarm.domain_model import (
    BENCHMARK_GROUP_ORDER,
    CHALLENGE_FAMILY_IDS,
    CHALLENGE_FAMILY_TO_INTERFACE_VERSIONS,
    CHALLENGE_INSTANCE_KEY_FIELDS,
    CHALLENGE_TYPE_TO_BENCHMARK_GROUP,
    CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE,
    EMISSIONS_STATES,
    ENVIRONMENT_TYPES,
    FAMILY_STATES,
    MARKET_VERTICAL_IDS,
    SKILL_IDS,
    UnknownChallengeFamilyError,
    domain_model_schema_path,
    filter_challenge_family_definitions,
    get_challenge_family_definition,
    get_policy_interface_contract,
    get_supported_interface_versions,
    load_challenge_family_registry,
)


def test_domain_model_schema_snapshot():
    expected = {
        "challenge_family_ids": [
            "cf_autopilot",
            "cf_search_and_rescue",
        ],
        "challenge_instance_key_fields": [
            "challenge_family",
            "environment_type",
            "benchmark_version",
            "seed",
        ],
        "skill_ids": [
            "sk_autonomous_navigation",
            "sk_search_and_rescue",
        ],
        "market_vertical_ids": [
            "mv_general_autonomy",
            "mv_public_safety",
        ],
        "family_states": [
            "incubating",
            "active",
            "archived",
        ],
        "emissions_states": [
            "incubating",
            "active",
            "saturated",
            "archived",
            "regression",
        ],
        "environment_types": [
            "city",
            "open",
            "mountain",
            "village",
            "warehouse",
            "forest",
        ],
        "challenge_type_to_environment_type": {
            1: "city",
            2: "open",
            3: "mountain",
            4: "village",
            5: "warehouse",
            6: "forest",
        },
        "challenge_type_to_benchmark_group": {
            1: "type1_city",
            2: "type2_open",
            3: "type3_mountain",
            4: "type4_village",
            5: "type5_warehouse",
            6: "type6_forest",
        },
        "benchmark_group_order": [
            "type1_city",
            "type2_open",
            "type3_mountain",
            "type4_village",
            "type5_warehouse",
            "type6_forest",
        ],
    }

    observed = {
        "challenge_family_ids": list(CHALLENGE_FAMILY_IDS),
        "challenge_instance_key_fields": list(CHALLENGE_INSTANCE_KEY_FIELDS),
        "skill_ids": list(SKILL_IDS),
        "market_vertical_ids": list(MARKET_VERTICAL_IDS),
        "family_states": list(FAMILY_STATES),
        "emissions_states": list(EMISSIONS_STATES),
        "environment_types": list(ENVIRONMENT_TYPES),
        "challenge_type_to_environment_type": dict(CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE),
        "challenge_type_to_benchmark_group": dict(CHALLENGE_TYPE_TO_BENCHMARK_GROUP),
        "benchmark_group_order": list(BENCHMARK_GROUP_ORDER),
    }

    assert observed == expected


def test_domain_model_schema_file_matches_runtime_exports():
    schema = json.loads(domain_model_schema_path().read_text(encoding="utf-8"))

    assert schema["entity_types"]["challenge_family"]["ids"] == list(CHALLENGE_FAMILY_IDS)
    assert schema["entity_types"]["challenge_instance"]["required_dimensions"] == list(
        CHALLENGE_INSTANCE_KEY_FIELDS
    )
    assert schema["entity_types"]["skill"]["ids"] == list(SKILL_IDS)
    assert schema["entity_types"]["market_vertical"]["ids"] == list(MARKET_VERTICAL_IDS)
    assert schema["enum_types"]["family_state"] == list(FAMILY_STATES)
    assert schema["enum_types"]["emissions_state"] == list(EMISSIONS_STATES)
    assert schema["enum_types"]["environment_type"] == list(ENVIRONMENT_TYPES)
    assert sorted(schema["challenge_families"]) == list(CHALLENGE_FAMILY_IDS)


def test_challenge_family_registry_contains_canonical_metadata():
    registry = load_challenge_family_registry()
    autopilot = get_challenge_family_definition("cf_autopilot", registry=registry)
    sar = get_challenge_family_definition("cf_search_and_rescue", registry=registry)

    assert autopilot["display"]["label"] == "Autopilot / Navigation"
    assert autopilot["family_state"] == "active"
    assert autopilot["emissions_state"] == "active"
    assert autopilot["score_schema_id"] == "ss_navigation_v1"
    assert list(get_supported_interface_versions("cf_autopilot", registry=registry)) == [
        "submission_zip.v1"
    ]

    assert sar["display"]["short_label"] == "SAR"
    assert sar["family_state"] == "incubating"
    assert sar["market_vertical_ids"] == ["mv_public_safety"]
    assert registry["score_schemas"]["ss_search_and_rescue_v1"]["primary_metric"] == (
        "mission_score"
    )

    assert CHALLENGE_FAMILY_TO_INTERFACE_VERSIONS == {
        "cf_autopilot": ("submission_zip.v1",),
        "cf_search_and_rescue": ("submission_zip.v1",),
    }


def test_get_challenge_family_definition_rejects_unknown_family():
    try:
        get_challenge_family_definition("cf_unknown")
    except UnknownChallengeFamilyError as exc:
        assert str(exc) == "'unknown_challenge_family:cf_unknown'"
    else:  # pragma: no cover
        raise AssertionError("Expected unknown family to be rejected")


def test_filter_challenge_family_definitions_handles_incubating_and_archived():
    registry = deepcopy(load_challenge_family_registry())
    registry["challenge_families"]["cf_archived_fixture"] = {
        **registry["challenge_families"]["cf_autopilot"],
        "family_id": "cf_archived_fixture",
        "display": {
            "label": "Archived Fixture",
            "short_label": "Archived",
            "slug": "archived-fixture",
        },
        "family_state": "archived",
        "emissions_state": "archived",
    }

    filtered_without_incubating = filter_challenge_family_definitions(
        registry["challenge_families"],
        include_incubating=False,
    )
    assert sorted(filtered_without_incubating) == [
        "cf_archived_fixture",
        "cf_autopilot",
    ]

    filtered_active_only = filter_challenge_family_definitions(
        registry["challenge_families"],
        include_incubating=False,
        include_archived=False,
    )
    assert sorted(filtered_active_only) == ["cf_autopilot"]


def test_policy_interface_contracts_are_registry_backed():
    contract = get_policy_interface_contract(
        "cf_search_and_rescue",
        "submission_zip.v1",
    )

    assert contract["contract_filename"] == "swarm_policy_contract.json"
    assert contract["entry_point"]["class_name"] == "DroneFlightController"
    assert contract["observation_space"]["fields"]["depth"]["shape"] == [128, 128, 1]
    assert contract["action_space"]["shape"] == [5]

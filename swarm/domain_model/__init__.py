from __future__ import annotations

import copy
import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping


_SCHEMA_RESOURCE = "benchmark_domain_model.schema.json"


class UnknownChallengeFamilyError(KeyError):
    """Raised when a challenge family id is not registered."""


class UnknownScoreSchemaError(KeyError):
    """Raised when a score schema id is not registered."""


class UnknownPolicyInterfaceError(KeyError):
    """Raised when a policy interface contract is not registered."""


def domain_model_schema_path() -> Path:
    return Path(str(files(__package__).joinpath(_SCHEMA_RESOURCE)))


def challenge_family_registry_path() -> Path:
    return domain_model_schema_path()


@lru_cache(maxsize=1)
def load_domain_model_schema() -> dict[str, Any]:
    schema_text = files(__package__).joinpath(_SCHEMA_RESOURCE).read_text(
        encoding="utf-8",
    )
    return json.loads(schema_text)


def load_challenge_family_registry() -> dict[str, Any]:
    return load_domain_model_schema()


def _deepcopy_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(mapping))


def filter_challenge_family_definitions(
    family_definitions: Mapping[str, Mapping[str, Any]],
    *,
    include_incubating: bool = True,
    include_archived: bool = True,
) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for family_id, family_definition in family_definitions.items():
        family_state = str(family_definition["family_state"])
        if family_state == "incubating" and not include_incubating:
            continue
        if family_state == "archived" and not include_archived:
            continue
        filtered[family_id] = _deepcopy_mapping(family_definition)
    return filtered


def list_challenge_family_definitions(
    *,
    include_incubating: bool = True,
    include_archived: bool = True,
    registry: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    active_registry = registry or load_challenge_family_registry()
    family_definitions = filter_challenge_family_definitions(
        active_registry["challenge_families"],
        include_incubating=include_incubating,
        include_archived=include_archived,
    )
    return tuple(family_definitions.values())


def get_challenge_family_definition(
    family_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_registry = registry or load_challenge_family_registry()
    family_definition = active_registry["challenge_families"].get(family_id)
    if family_definition is None:
        raise UnknownChallengeFamilyError(f"unknown_challenge_family:{family_id}")
    return _deepcopy_mapping(family_definition)


def get_score_schema(
    score_schema_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_registry = registry or load_challenge_family_registry()
    score_schema = active_registry["score_schemas"].get(score_schema_id)
    if score_schema is None:
        raise UnknownScoreSchemaError(f"unknown_score_schema:{score_schema_id}")
    return _deepcopy_mapping(score_schema)


def get_supported_interface_versions(
    family_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    family_definition = get_challenge_family_definition(family_id, registry=registry)
    return tuple(family_definition["supported_interface_versions"])


def get_family_screening_policy(
    family_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    family_definition = get_challenge_family_definition(family_id, registry=registry)
    return _deepcopy_mapping(family_definition.get("screening_policy", {}))


def get_family_benchmark_admission_policy(
    family_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    family_definition = get_challenge_family_definition(family_id, registry=registry)
    return _deepcopy_mapping(family_definition.get("benchmark_admission_policy", {}))


def get_family_visibility(
    family_id: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> str:
    family_definition = get_challenge_family_definition(family_id, registry=registry)
    return str(family_definition.get("visibility", "public"))


def policy_interface_key(family_id: str, interface_version: str) -> str:
    return f"{family_id}:{interface_version}"


def get_policy_interface_contract(
    family_id: str,
    interface_version: str,
    *,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active_registry = registry or load_challenge_family_registry()
    key = policy_interface_key(family_id, interface_version)
    contract = active_registry.get("policy_interfaces", {}).get(key)
    if contract is None:
        raise UnknownPolicyInterfaceError(
            f"unknown_policy_interface:{family_id}:{interface_version}"
        )
    return _deepcopy_mapping(contract)


_REGISTRY = load_challenge_family_registry()

SCHEMA_VERSION = str(_REGISTRY["schema_version"])
CHALLENGE_FAMILY_IDS = tuple(_REGISTRY["entity_types"]["challenge_family"]["ids"])
CHALLENGE_INSTANCE_KEY_FIELDS = tuple(
    _REGISTRY["entity_types"]["challenge_instance"]["required_dimensions"]
)
SKILL_IDS = tuple(_REGISTRY["entity_types"]["skill"]["ids"])
MARKET_VERTICAL_IDS = tuple(_REGISTRY["entity_types"]["market_vertical"]["ids"])
FAMILY_STATES = tuple(_REGISTRY["enum_types"]["family_state"])
EMISSIONS_STATES = tuple(_REGISTRY["enum_types"]["emissions_state"])
VISIBILITIES = tuple(_REGISTRY["enum_types"].get("visibility", ("public", "private")))
ENVIRONMENT_TYPES = tuple(_REGISTRY["enum_types"]["environment_type"])

CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE = {
    int(challenge_type): environment_type
    for challenge_type, environment_type in _REGISTRY["legacy_aliases"][
        "challenge_type_to_environment_type"
    ].items()
}
ENVIRONMENT_TYPE_TO_CHALLENGE_TYPE = {
    environment_type: challenge_type
    for challenge_type, environment_type in CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.items()
}
CHALLENGE_TYPE_IDS = tuple(CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE)
MIN_CHALLENGE_TYPE = min(CHALLENGE_TYPE_IDS)
MAX_CHALLENGE_TYPE = max(CHALLENGE_TYPE_IDS)

BENCHMARK_GROUP_TO_ENVIRONMENT_TYPE = dict(
    _REGISTRY["legacy_aliases"]["benchmark_group_to_environment_type"]
)
BENCHMARK_GROUP_ORDER = tuple(BENCHMARK_GROUP_TO_ENVIRONMENT_TYPE)
BENCHMARK_GROUP_TO_CHALLENGE_TYPE = {
    benchmark_group: ENVIRONMENT_TYPE_TO_CHALLENGE_TYPE[environment_type]
    for benchmark_group, environment_type in BENCHMARK_GROUP_TO_ENVIRONMENT_TYPE.items()
}
CHALLENGE_TYPE_TO_BENCHMARK_GROUP = {
    challenge_type: benchmark_group
    for benchmark_group, challenge_type in BENCHMARK_GROUP_TO_CHALLENGE_TYPE.items()
}

CHALLENGE_FAMILY_TO_ENVIRONMENT_TYPES = {
    family_id: tuple(family_definition["environment_types"])
    for family_id, family_definition in _REGISTRY["challenge_families"].items()
}
CHALLENGE_FAMILY_TO_SKILLS = {
    family_id: tuple(family_definition["skill_ids"])
    for family_id, family_definition in _REGISTRY["challenge_families"].items()
}
CHALLENGE_FAMILY_TO_MARKET_VERTICALS = {
    family_id: tuple(family_definition["market_vertical_ids"])
    for family_id, family_definition in _REGISTRY["challenge_families"].items()
}
CHALLENGE_FAMILY_TO_INTERFACE_VERSIONS = {
    family_id: tuple(family_definition["supported_interface_versions"])
    for family_id, family_definition in _REGISTRY["challenge_families"].items()
}
SCORE_SCHEMA_IDS = tuple(_REGISTRY["score_schemas"])
POLICY_INTERFACE_KEYS = tuple(_REGISTRY.get("policy_interfaces", {}))

__all__ = [
    "BENCHMARK_GROUP_ORDER",
    "BENCHMARK_GROUP_TO_CHALLENGE_TYPE",
    "BENCHMARK_GROUP_TO_ENVIRONMENT_TYPE",
    "CHALLENGE_FAMILY_IDS",
    "CHALLENGE_FAMILY_TO_ENVIRONMENT_TYPES",
    "CHALLENGE_FAMILY_TO_INTERFACE_VERSIONS",
    "CHALLENGE_FAMILY_TO_MARKET_VERTICALS",
    "CHALLENGE_FAMILY_TO_SKILLS",
    "CHALLENGE_INSTANCE_KEY_FIELDS",
    "CHALLENGE_TYPE_IDS",
    "CHALLENGE_TYPE_TO_BENCHMARK_GROUP",
    "CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE",
    "EMISSIONS_STATES",
    "ENVIRONMENT_TYPES",
    "ENVIRONMENT_TYPE_TO_CHALLENGE_TYPE",
    "FAMILY_STATES",
    "MARKET_VERTICAL_IDS",
    "MAX_CHALLENGE_TYPE",
    "MIN_CHALLENGE_TYPE",
    "SCHEMA_VERSION",
    "SCORE_SCHEMA_IDS",
    "SKILL_IDS",
    "VISIBILITIES",
    "UnknownChallengeFamilyError",
    "UnknownPolicyInterfaceError",
    "UnknownScoreSchemaError",
    "challenge_family_registry_path",
    "domain_model_schema_path",
    "filter_challenge_family_definitions",
    "get_family_benchmark_admission_policy",
    "get_challenge_family_definition",
    "get_family_screening_policy",
    "get_family_visibility",
    "get_policy_interface_contract",
    "get_score_schema",
    "get_supported_interface_versions",
    "list_challenge_family_definitions",
    "load_challenge_family_registry",
    "load_domain_model_schema",
    "policy_interface_key",
    "POLICY_INTERFACE_KEYS",
]

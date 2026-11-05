# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schema hints for decoding GLM's unquoted XML argument values.

This is not tool input validation. It preserves strings which look like JSON
literals, and uses argument constraints to disambiguate complete values. The
tool executor remains responsible for validating the complete argument object.
"""

from typing import Any
from urllib.parse import quote, urljoin

import regex as re
from jsonschema import Draft201909Validator, Draft202012Validator
from jsonschema.validators import validator_for
from referencing import Registry, Resource
from referencing.exceptions import Unresolvable
from referencing.jsonschema import DRAFT202012

_ALL_TYPES = frozenset(
    {"string", "null", "boolean", "integer", "number", "array", "object"}
)
_STRING_TYPE = frozenset({"string"})


class GlmToolArgumentSchema:
    def __init__(self, schema: Any, registry: Registry, resolver: Any, validator: Any):
        self._resolvable = True
        self.types = self._possible_types(schema, resolver, frozenset(), validator)
        self._validator = validator(schema, registry=registry)
        self._last_raw: str | None = None
        self._last_value: Any = None

    @property
    def is_string(self) -> bool:
        return self.types == _STRING_TYPE

    def select_value(self, raw: str, deserialized: Any) -> Any:
        # Cumulative streaming reparses completed arguments as later arguments
        # arrive. Validate an unchanged value only once.
        if raw != self._last_raw:
            self._last_value = self._select_value(raw, deserialized)
            self._last_raw = raw
        return self._last_value

    def _select_value(self, raw: str, deserialized: Any) -> Any:
        """Prefer the legacy JSON/Python interpretation if it satisfies schema.

        GLM emits strings without JSON quotes. Pure strings therefore retain
        their exact contents; nullable/mixed types wait for the complete value.
        For a mixed union, JSON null selects null, while JSON "null" selects a
        string. If neither candidate is valid, leave validation to the caller.
        """
        if self.is_string:
            return raw
        if isinstance(deserialized, str) and raw.strip() == deserialized:
            deserialized = raw
        if self._resolvable:
            try:
                if self._validator.is_valid(deserialized):
                    return deserialized
                if self._validator.is_valid(raw):
                    return raw
            except (Unresolvable, RecursionError):
                # An unavailable reference or non-productive reference cycle
                # must not discard the remaining arguments of the tool call.
                self._resolvable = False
        if "string" in self.types and self.types <= {"string", "null"}:
            return raw
        return deserialized

    def _possible_types(
        self,
        schema: Any,
        resolver: Any,
        seen: frozenset[int],
        validator: Any,
        enter: bool = True,
    ) -> frozenset[str]:
        """Conservative type upper bound, used only to commit string prefixes.

        Constraints such as pattern, not and conditionals are checked on the
        completed value by jsonschema. Ignoring them here can delay streaming,
        but cannot cause a non-string to be streamed as a quoted string.
        """
        if schema is False:
            return frozenset()
        if not isinstance(schema, dict):
            return _ALL_TYPES
        if id(schema) in seen:
            self._resolvable = False
            return _ALL_TYPES
        seen = seen | {id(schema)}
        validator = validator_for(schema, default=validator)
        if enter:
            resolver = resolver.in_subresource(
                Resource.from_contents(schema, default_specification=DRAFT202012)
            )
        types = _ALL_TYPES
        if "$ref" in schema:
            try:
                resolved = resolver.lookup(schema["$ref"])
                types &= self._possible_types(
                    resolved.contents, resolved.resolver, seen, validator, enter=False
                )
            except Unresolvable:
                self._resolvable = False
            # In older dialects $ref siblings are not applied.
            if validator not in (
                Draft202012Validator,
                Draft201909Validator,
            ):
                return types
        declared = schema.get("type")
        if isinstance(declared, str):
            declared = [declared]
        if isinstance(declared, list):
            allowed = set(declared)
            if "number" in allowed:
                allowed.add("integer")
            types &= allowed
        for keyword in ("enum", "const"):
            if keyword not in schema or keyword not in validator.VALIDATORS:
                continue
            values = schema[keyword] if keyword == "enum" else [schema[keyword]]
            allowed = {
                name
                for name in _ALL_TYPES
                if any(validator.TYPE_CHECKER.is_type(v, name) for v in values)
            }
            types &= allowed
        for keyword in ("allOf", "anyOf", "oneOf"):
            if keyword not in schema:
                continue
            branches = [
                self._possible_types(child, resolver, seen, validator)
                for child in schema[keyword]
            ]
            if keyword == "allOf":
                for branch in branches:
                    types &= branch
            else:
                types &= frozenset().union(*branches)
        return frozenset(types)


class GlmToolSchema:
    """Resolve argument schemas within one tool definition, without retrieval."""

    def __init__(self, schema: dict[str, Any]):
        self._schema = schema
        self._validator = validator_for(schema, default=Draft202012Validator)
        self._base_uri = urljoin(
            "https://vllm.invalid/tool-schema", schema.get("$id", "")
        )
        self._registry = Registry().with_resource(
            self._base_uri,
            Resource.from_contents(schema, default_specification=DRAFT202012),
        )
        self._resolver = self._registry.resolver(self._base_uri)
        self._locations: dict[int, str] = {}
        self._index_locations(schema, "")
        self._arguments: dict[str, GlmToolArgumentSchema] = {}

    def _index_locations(self, node: Any, pointer: str) -> None:
        if isinstance(node, dict):
            self._locations[id(node)] = pointer
            for key, value in node.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                self._index_locations(value, f"{pointer}/{escaped}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                self._index_locations(value, f"{pointer}/{index}")

    def argument(self, name: str) -> GlmToolArgumentSchema:
        if name not in self._arguments:
            projected = self._property_schema(
                self._schema,
                name,
                self._resolver,
                frozenset(),
                self._validator,
                enter=False,
            )
            self._arguments[name] = GlmToolArgumentSchema(
                projected, self._registry, self._resolver, self._validator
            )
        return self._arguments[name]

    def _reference(self, schema: Any) -> Any:
        if isinstance(schema, bool):
            return schema
        pointer = quote(self._locations[id(schema)], safe="/~")
        return {"$ref": f"{self._base_uri}#{pointer}"}

    def _property_schema(
        self,
        schema: Any,
        name: str,
        resolver: Any,
        seen: frozenset[int],
        validator: Any,
        enter: bool = True,
    ) -> Any:
        if isinstance(schema, bool):
            return schema
        if not isinstance(schema, dict) or id(schema) in seen:
            return True
        seen = seen | {id(schema)}
        validator = validator_for(schema, default=validator)
        if enter:
            resolver = resolver.in_subresource(
                Resource.from_contents(schema, default_specification=DRAFT202012)
            )
        constraints = []
        if "$ref" in schema:
            try:
                resolved = resolver.lookup(schema["$ref"])
                constraints.append(
                    self._property_schema(
                        resolved.contents,
                        name,
                        resolved.resolver,
                        seen,
                        validator,
                        enter=False,
                    )
                )
            except Unresolvable:
                pass
            if validator not in (Draft202012Validator, Draft201909Validator):
                return {"allOf": constraints} if constraints else True
        matched = False
        properties = schema.get("properties", {})
        if name in properties:
            matched = True
            constraints.append(self._reference(properties[name]))
        for pattern, child in schema.get("patternProperties", {}).items():
            if re.search(pattern, name):
                matched = True
                constraints.append(self._reference(child))
        if not matched and "additionalProperties" in schema:
            constraints.append(self._reference(schema["additionalProperties"]))
        for child in schema.get("allOf", []):
            constraints.append(
                self._property_schema(child, name, resolver, seen, validator)
            )
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                # Projecting one field loses the other fields which distinguish
                # object-level oneOf branches. Retain all possible alternatives;
                # field-level oneOf is preserved by the references above.
                constraints.append(
                    {
                        "anyOf": [
                            self._property_schema(
                                child, name, resolver, seen, validator
                            )
                            for child in schema[keyword]
                        ]
                    }
                )
        return {"allOf": constraints} if constraints else True

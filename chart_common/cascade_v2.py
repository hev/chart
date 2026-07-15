from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def encode_cascade_v2_map(value: Mapping[str, Any]) -> str:
    """Encode a cascade-v2 derived-label map for a gateway string column."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def decode_cascade_v2_map(value: str) -> dict[str, Any]:
    """Decode a cascade-v2 map read from its non-filterable string column."""
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("cascade-v2 map column must contain a JSON object")
    return decoded

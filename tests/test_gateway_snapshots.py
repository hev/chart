from chart_common.gateway import unnest_array_facets


def test_unnest_array_facets_explodes_and_merges():
    """hev/layer#151 mitigation: serialized-array buckets explode into
    per-element counts, empty arrays drop, scalars pass through and merge."""
    values = [
        {"value": "[]", "count": 87},
        {"value": '["medication_discontinued","medication_started"]', "count": 7},
        {"value": '["diagnosis_made","medication_discontinued"]', "count": 4},
        {"value": "medication_discontinued", "count": 2},  # already-unnested (post-fix)
        {"value": "not [json", "count": 1},
    ]
    out = {v["value"]: v["count"] for v in unnest_array_facets(values)}
    assert out["medication_discontinued"] == 13  # 7 + 4 + 2
    assert out["medication_started"] == 7
    assert out["diagnosis_made"] == 4
    assert out["not [json"] == 1
    assert "[]" not in out


def test_unnest_array_facets_passthrough_scalars():
    values = [{"value": "cardiology", "count": 5}, {"value": "other", "count": 2}]
    assert sorted(unnest_array_facets(values), key=lambda v: -v["count"]) == values

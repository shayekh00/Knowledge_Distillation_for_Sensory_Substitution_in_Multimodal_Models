import pandas as pd

from tools.audit_app.sampling import sample_audit_items


def _rows(question_type: str, sensor_counts: dict[str, int]) -> list[dict]:
    rows = []
    for sensor, count in sensor_counts.items():
        for i in range(count):
            rows.append({
                "question_id": f"{question_type}-{sensor}-{i}",
                "question_type": question_type,
                "sensor": sensor,
            })
    return rows


def test_caps_at_per_type_and_is_deterministic():
    items = pd.DataFrame(_rows("existence", {"kv1": 40, "kv2": 60}) + _rows("count", {"kv1": 10}))

    first = sample_audit_items(items, per_type=20, seed=42)
    second = sample_audit_items(items, per_type=20, seed=42)

    assert sorted(first["question_id"]) == sorted(second["question_id"])
    assert (first["question_type"] == "existence").sum() == 20
    assert (first["question_type"] == "count").sum() == 10  # pool smaller than per_type


def test_stratifies_by_sensor_proportionally():
    items = pd.DataFrame(_rows("existence", {"kv1": 25, "kv2": 75}))

    sampled = sample_audit_items(items, per_type=20, seed=42)

    counts = sampled["sensor"].value_counts()
    assert counts.get("kv1", 0) == 5   # 25% of 20
    assert counts.get("kv2", 0) == 15  # 75% of 20


def test_no_duplicate_rows_drawn():
    items = pd.DataFrame(_rows("left_right", {"kv1": 5, "kv2": 5}))

    sampled = sample_audit_items(items, per_type=150, seed=42)

    assert len(sampled) == len(items)  # capped at pool size
    assert sampled["question_id"].is_unique


def test_unstratified_draw_caps_per_type_and_is_deterministic():
    # M3FD's shape: no `sensor` column at all, so stratify_by is None.
    items = pd.DataFrame([
        {"question_id": f"existence-{i}", "question_type": "existence", "sequence_id": f"s{i % 4}"}
        for i in range(100)
    ] + [
        {"question_id": f"count-{i}", "question_type": "count", "sequence_id": "s0"}
        for i in range(7)
    ])

    first = sample_audit_items(items, per_type=20, seed=42, stratify_by=None)
    second = sample_audit_items(items, per_type=20, seed=42, stratify_by=None)

    assert list(first["question_id"]) == list(second["question_id"])
    assert (first["question_type"] == "existence").sum() == 20
    assert (first["question_type"] == "count").sum() == 7  # pool smaller than per_type
    assert first["question_id"].is_unique


def test_stratify_by_accepts_a_column_other_than_sensor():
    items = pd.DataFrame(
        [{"question_id": f"a{i}", "question_type": "existence", "split_group": "g1"} for i in range(25)]
        + [{"question_id": f"b{i}", "question_type": "existence", "split_group": "g2"} for i in range(75)]
    )

    sampled = sample_audit_items(items, per_type=20, seed=42, stratify_by="split_group")

    assert sampled["split_group"].value_counts().to_dict() == {"g2": 15, "g1": 5}

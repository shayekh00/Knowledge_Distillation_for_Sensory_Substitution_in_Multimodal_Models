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

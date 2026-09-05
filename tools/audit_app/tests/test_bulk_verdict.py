import pandas as pd

from tools.audit_app import bulk_verdict


def test_select_items_filters_superlative_variants(tmp_path, monkeypatch):
    """Prevent a depth-only bulk review from also selecting `largest` items."""
    audit_items = tmp_path / "audit_items.csv"
    pd.DataFrame([
        {"question_id": "closest", "question_type": "identify_superlative", "variant": "closest_camera"},
        {"question_id": "farthest", "question_type": "identify_superlative", "variant": "farthest_camera"},
        {"question_id": "largest", "question_type": "identify_superlative", "variant": "largest"},
        {"question_id": "depth", "question_type": "relative_depth", "variant": "closer"},
    ]).to_csv(audit_items, index=False)
    monkeypatch.setattr(bulk_verdict, "AUDIT_ITEMS_CSV", audit_items)

    selected = bulk_verdict.select_items(
        types=["identify_superlative"],
        variants=["closest_camera", "farthest_camera"],
        only_disagreements=False,
    )

    assert selected["question_id"].tolist() == ["closest", "farthest"]

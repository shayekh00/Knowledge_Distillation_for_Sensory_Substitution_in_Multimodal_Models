"""Manual test-set audit tool — FastAPI backend.

Implements the human side of DATASET_CREATION_PLAN.md §8.3: an annotator sees
the RGB image with the question's evidence objects outlined, the question,
and the gold answer, all at once, then marks it correct / incorrect /
ambiguous. Marking "incorrect" opens a text box for what the annotator
believes the right answer is; that text is spelling-corrected toward the
dataset's own controlled answer vocabulary (spelling.py) before being saved,
and the corrected value is returned to the UI so it can be reviewed and
re-edited.

Gold is shown immediately rather than after a blind guess: this was a
deliberate trade against the original §8.3 protocol, made explicitly by the
project owner, and it means `human_accuracy_vs_gold` in the stats/report no
longer measures independent agreement — see audit_store.compute_stats's
docstring for what it measures instead.

Usage::

    # 1. Once release/VQA-SUNRGBD-v2/rule_based/test.csv exists, draw the
    #    stratified audit sample (§8.3: 150 items/type):
    python -m tools.audit_app.sampling \
        --test-csv release/VQA-SUNRGBD-v2/rule_based/test.csv \
        --out audit/audit_items.csv

    # 2. Run the app (repeat per annotator, each picks their own annotator id
    #    in the UI — no separate server instance needed):
    python -m uvicorn tools.audit_app.main:app --port 8002 --reload

    # 3. Once both annotators are done, render the committed report:
    python -m tools.audit_app.report
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from tools.audit_app.audit_items import AuditItem, load_audit_items
from tools.audit_app.audit_store import (
    AuditResponse,
    append_response,
    compute_stats,
    list_annotator_ids,
    load_all_responses,
    load_responses,
    progress_for,
)
from tools.audit_app.scene_index import SceneIndex
from tools.audit_app.spelling import candidate_answers_for, correct_spelling

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
AUDIT_DIR = Path(os.environ.get("AUDIT_DIR", PROJECT_ROOT / "audit"))
AUDIT_ITEMS_CSV = Path(os.environ.get("AUDIT_ITEMS_CSV", AUDIT_DIR / "audit_items.csv"))
MODEL_ANSWERS_CSV = AUDIT_DIR / "model_answers.csv"
RESPONSES_DIR = AUDIT_DIR / "responses"
STATIC_DIR = Path(__file__).parent / "static"
CANONICAL_OBJECTS_CSV = DATA_DIR / "vocab" / "canonical_objects.csv"

SCENE_INDEX = SceneIndex(DATA_DIR / "index" / "scene_index.jsonl", DATASET_DIR)


def _load_canonical_display_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [row["display_name"].replace("_", " ") for row in csv.DictReader(csv_file)]


def _load_concept_display_names(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        return {
            row["canonical_concept"]: row["display_name"].replace("_", " ")
            for row in csv.DictReader(csv_file)
        }


CANONICAL_DISPLAY_NAMES = _load_canonical_display_names(CANONICAL_OBJECTS_CSV)
CONCEPT_DISPLAY_NAMES = _load_concept_display_names(CANONICAL_OBJECTS_CSV)

_ITEMS: list[AuditItem] = []
_ITEMS_BY_ID: dict[str, AuditItem] = {}
_LOAD_ERROR: str | None = None
_MODEL_HINTS: dict[str, dict] = {}


def _load_model_hints() -> None:
    """Optional triage layer from tools.audit_app.model_pass. A hint is a
    prioritisation signal only — never ground truth, never a second
    annotator (see model_pass.py's module docstring)."""
    global _MODEL_HINTS
    _MODEL_HINTS = {}
    if not MODEL_ANSWERS_CSV.is_file():
        return
    # Append-only log: a retried item has more than one row, last one wins.
    with MODEL_ANSWERS_CSV.open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            answer = (row.get("model_answer") or "").strip()
            error = (row.get("error") or "").strip()
            agrees = (row.get("agrees_with_gold") or "").strip().lower() == "true"
            _MODEL_HINTS[row["question_id"]] = {
                "model": (row.get("model") or "").strip(),
                "model_answer": answer,
                "model_reasoning": (row.get("model_reasoning") or "").strip(),
                "model_error": error,
                # "unavailable" keeps model failures out of the disagreement
                # queue: nothing for a human to adjudicate there.
                "model_status": "unavailable" if (not answer or error) else ("agrees" if agrees else "disagrees"),
            }


def _load_items() -> None:
    global _ITEMS, _ITEMS_BY_ID, _LOAD_ERROR
    _load_model_hints()
    if not AUDIT_ITEMS_CSV.is_file():
        _LOAD_ERROR = f"{AUDIT_ITEMS_CSV} does not exist yet — run tools.audit_app.sampling first."
        return
    try:
        _ITEMS = load_audit_items(AUDIT_ITEMS_CSV, SCENE_INDEX, CONCEPT_DISPLAY_NAMES)
        _ITEMS_BY_ID = {item.question_id: item for item in _ITEMS}
        _LOAD_ERROR = None
    except Exception as exc:  # surfaced to the UI, never crashes the app
        _LOAD_ERROR = str(exc)


_load_items()

app = FastAPI(title="VQA-SUNRGBD-v2 Audit Tool", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/status")
def get_status():
    return {
        "items_loaded": _LOAD_ERROR is None,
        "load_error": _LOAD_ERROR,
        "n_items": len(_ITEMS),
        "audit_items_csv": str(AUDIT_ITEMS_CSV),
        "scene_index_size": len(SCENE_INDEX),
    }


@app.post("/api/reload")
def reload_items():
    """Re-read audit_items.csv without restarting the server (e.g. after
    re-running the sampler)."""
    _load_items()
    return get_status()


@app.get("/api/items")
def get_items():
    if _LOAD_ERROR is not None:
        raise HTTPException(503, _LOAD_ERROR)
    items = []
    for item in _ITEMS:
        payload = item.to_public_dict()
        hint = _MODEL_HINTS.get(item.question_id)
        payload.update(hint or {"model_status": "unavailable", "model_answer": "",
                                 "model_reasoning": "", "model_error": "", "model": ""})
        items.append(payload)
    return {"items": items, "model_hints_loaded": bool(_MODEL_HINTS)}


@app.get("/api/model_summary")
def get_model_summary():
    """Per-type model/gold agreement over the sampled items.

    Read as a *triage* signal, not a label-quality score: for the
    depth-derived types the gold answer comes from measured depth while the
    model is guessing depth from a single RGB frame, so low agreement there
    mostly reflects the model's limitation rather than a label problem.
    """
    if not _MODEL_HINTS:
        return {"available": False, "by_type": {}}
    by_type: dict[str, dict] = {}
    for item in _ITEMS:
        hint = _MODEL_HINTS.get(item.question_id)
        if hint is None:
            continue
        bucket = by_type.setdefault(item.question_type, {"agrees": 0, "disagrees": 0, "unavailable": 0})
        bucket[hint["model_status"]] += 1
    for bucket in by_type.values():
        scored = bucket["agrees"] + bucket["disagrees"]
        bucket["agreement_rate"] = (bucket["agrees"] / scored) if scored else None
    return {"available": True, "by_type": by_type}


@app.get("/api/image/{image_id:path}")
def get_image(image_id: str):
    scene = SCENE_INDEX.get(image_id)
    if scene is None:
        raise HTTPException(404, f"Unknown image_id: {image_id!r}")
    if not scene.rgb_path.is_file():
        raise HTTPException(404, f"RGB file missing on disk: {scene.rgb_path}")
    return FileResponse(str(scene.rgb_path), media_type="image/jpeg")


@app.get("/api/polygons/{image_id:path}")
def get_polygons(image_id: str, objects: str = ""):
    scene = SCENE_INDEX.get(image_id)
    if scene is None:
        raise HTTPException(404, f"Unknown image_id: {image_id!r}")
    object_indices = {int(token) for token in objects.split(",") if token.strip() != ""}
    polygons = SCENE_INDEX.polygons_for(image_id, object_indices or None)
    return {
        "image_width": scene.image_width,
        "image_height": scene.image_height,
        "polygons": [
            {"object_index": polygon.object_index, "name": polygon.name, "x": polygon.x, "y": polygon.y}
            for polygon in polygons
        ],
    }


@app.get("/api/annotators")
def get_annotators():
    return {"annotator_ids": list_annotator_ids(RESPONSES_DIR)}


@app.get("/api/progress")
def get_progress(annotator: str):
    if _LOAD_ERROR is not None:
        raise HTTPException(503, _LOAD_ERROR)
    responses = load_responses(RESPONSES_DIR, annotator)
    return progress_for(_ITEMS, responses)


@app.get("/api/responses")
def get_annotator_responses(annotator: str):
    """question_id -> this annotator's saved response, so the UI can resume
    where it left off and let them revisit/edit an earlier item."""
    responses = load_responses(RESPONSES_DIR, annotator)
    return {qid: response.__dict__ for qid, response in responses.items()}


class SubmitResponseBody(BaseModel):
    question_id: str
    annotator_id: str
    own_answer: str
    verdict: str
    notes: str = ""


@app.post("/api/response")
def submit_response(body: SubmitResponseBody):
    item = _ITEMS_BY_ID.get(body.question_id)
    if item is None:
        raise HTTPException(404, f"Unknown question_id: {body.question_id!r}")
    if not body.annotator_id.strip():
        raise HTTPException(400, "annotator_id must not be blank")

    candidates = candidate_answers_for(item.question_type, CANONICAL_DISPLAY_NAMES)
    corrected_answer = correct_spelling(body.own_answer, candidates)

    try:
        response = AuditResponse.new(
            question_id=body.question_id,
            annotator_id=body.annotator_id.strip(),
            own_answer=corrected_answer,
            own_answer_raw=body.own_answer,
            verdict=body.verdict,
            notes=body.notes,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    append_response(RESPONSES_DIR, response)
    return {
        "progress": progress_for(_ITEMS, load_responses(RESPONSES_DIR, body.annotator_id.strip())),
        "saved": {
            "own_answer": response.own_answer,
            "own_answer_raw": response.own_answer_raw,
            "verdict": response.verdict,
            "notes": response.notes,
            "was_corrected": response.own_answer != response.own_answer_raw,
        },
    }


@app.get("/api/stats")
def get_stats():
    if _LOAD_ERROR is not None:
        raise HTTPException(503, _LOAD_ERROR)
    responses_by_annotator = load_all_responses(RESPONSES_DIR)
    stats = compute_stats(_ITEMS, responses_by_annotator)
    return {
        "annotator_ids": sorted(responses_by_annotator),
        "types": [stat.__dict__ for stat in stats],
    }

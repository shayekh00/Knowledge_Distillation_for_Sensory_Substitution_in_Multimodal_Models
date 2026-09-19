"""Manual test-set audit tool — FastAPI backend.

Implements the single-reviewer gold verification in DATASET_CREATION_PLAN.md §8.3: the reviewer sees
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

Which dataset is being audited is one named choice, `AUDIT_SOURCE` — see
sources.py for the table and for why M³FD could not just be three more path
overrides. It defaults to `sunrgbd`, and the individual path env vars still win
over whatever the source declares.

Usage::

    # 1. Once the release's test.csv exists, draw the stratified audit sample
    #    (§8.3: 150 items/type). --source fills in that release's paths:
    python -m tools.audit_app.sampling --source sunrgbd

    # 2. Run the app; the reviewer ID defaults to "solo":
    python -m uvicorn tools.audit_app.main:app --port 8002 --reload

    # 3. Once the reviewer is done, render the committed report:
    python -m tools.audit_app.report

Auditing M³FD instead — thermal shown first, `r` flips to the registered RGB
frame so the reviewer can check the overlay against both modalities as
M3FD_THERMAL_VQA_RUNBOOK.md requires::

    python -m tools.audit_app.sampling --source m3fd
    AUDIT_SOURCE=m3fd python -m uvicorn tools.audit_app.main:app --port 8002
"""
from __future__ import annotations

import csv
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
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
from tools.audit_app.arkit_rotation import (
    rotate_image, rotate_points, rotated_dimensions, rotation_index_for)
from tools.audit_app.scene_index import SceneIndex
from tools.audit_app.sources import active_source
from tools.audit_app.spelling import candidate_answers_for, correct_spelling

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset"
SOURCE = active_source()
AUDIT_DIR = Path(os.environ.get("AUDIT_DIR", SOURCE.audit_dir_path))
AUDIT_ITEMS_CSV = Path(os.environ.get("AUDIT_ITEMS_CSV", AUDIT_DIR / "audit_items.csv"))
MODEL_ANSWERS_CSV = AUDIT_DIR / "model_answers.csv"
RESPONSES_DIR = AUDIT_DIR / "responses"
STATIC_DIR = Path(__file__).parent / "static"
# The per-path env vars predate `AUDIT_SOURCE` and still win over it, so an
# existing invocation that pointed the app at a dataset by hand keeps working;
# new ones should just name the source.
CANONICAL_OBJECTS_CSV = Path(os.environ.get(
    "CANONICAL_OBJECTS_CSV", SOURCE.canonical_objects_path))
SCENE_INDEX_JSONL = Path(os.environ.get(
    "SCENE_INDEX_JSONL", SOURCE.scene_index_path))

SCENE_INDEX = SceneIndex(SCENE_INDEX_JSONL, DATASET_DIR)

# Browsers render these directly; anything else the source ships (M³FD's
# thermal frames are .bmp/.tif in some mirrors) is re-encoded to PNG below
# rather than sent as bytes the <img> tag would silently refuse to draw.
BROWSER_SAFE_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


def _load_canonical_display_names(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as csv_file:
        return [row["display_name"].replace("_", " ") for row in csv.DictReader(csv_file)]


def _load_concept_display_names(path: Path) -> dict[str, str]:
    """concept -> display name, tolerating either vocabulary column name.

    SUN-RGB-D's and ARKitScenes' vocabularies key on `canonical_concept`;
    `data/vocab_m3fd/canonical_objects.csv` keys on `concept`.
    """
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        fields = reader.fieldnames or []
        concept_column = "canonical_concept" if "canonical_concept" in fields else "concept"
        if concept_column not in fields:
            return {}
        return {
            row[concept_column]: row["display_name"].replace("_", " ")
            for row in reader
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
        _ITEMS = load_audit_items(AUDIT_ITEMS_CSV, SCENE_INDEX, CONCEPT_DISPLAY_NAMES,
                                  evidence_style=SOURCE.evidence_style)
        _ITEMS_BY_ID = {item.question_id: item for item in _ITEMS}
        _LOAD_ERROR = None
    except Exception as exc:  # surfaced to the UI, never crashes the app
        _LOAD_ERROR = str(exc)


_load_items()

app = FastAPI(title=f"{SOURCE.title} Audit Tool", version="1.0.0")
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
        "source": SOURCE.name,
        "source_title": SOURCE.title,
        # The UI shows a modality switch only when there is more than one.
        "modalities": list(SOURCE.modalities),
        "default_modality": SOURCE.default_modality,
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


def _arkit_rotation_for(scene) -> int:
    """0 for every source that is not ARKitScenes (SUN-RGB-D and M³FD are
    never rotated). `image_id` is `{video_id}/{frame_timestamp}` for
    ARKitScenes records."""
    if not SOURCE.display_rotation:
        return 0
    frame_timestamp = scene.image_id.rsplit("/", 1)[-1]
    return rotation_index_for(scene.rgb_path, frame_timestamp)


def _requested_modality(modality: str) -> str:
    """Validate a `?modality=` value against what this source actually has."""
    if not modality:
        return SOURCE.default_modality
    if not SOURCE.is_known_modality(modality):
        raise HTTPException(
            400, f"source {SOURCE.name!r} has no {modality!r} modality; "
                 f"available: {', '.join(SOURCE.modalities)}")
    return modality


def _geometry_scale(scene, modality: str) -> tuple[float, float]:
    """Factors taking indexed geometry into `modality`'s own frame.

    M³FD stores every box in the thermal frame while its RGB partner may be a
    different size, so overlaying evidence on the RGB image without this puts
    the boxes in the wrong place — exactly the registration error the reviewer
    is there to catch, faked by the viewer. Identity for single-modality
    sources, whose geometry and image are the same frame by construction.
    """
    indexed_width, indexed_height = scene.image_width, scene.image_height
    target_width, target_height = scene.frame_size_for(modality)
    if not indexed_width or not indexed_height:
        return 1.0, 1.0
    return target_width / indexed_width, target_height / indexed_height


@app.get("/api/image/{image_id:path}")
def get_image(image_id: str, modality: str = ""):
    scene = SCENE_INDEX.get(image_id)
    if scene is None:
        raise HTTPException(404, f"Unknown image_id: {image_id!r}")
    modality = _requested_modality(modality)
    image_path = scene.image_path(modality)
    if image_path is None:
        raise HTTPException(404, f"{image_id!r} has no {modality} image in the scene index")
    if not image_path.is_file():
        raise HTTPException(404, f"{modality.upper()} file missing on disk: {image_path}")
    # no-store: the URL for a given image_id never changes, so without this a
    # browser silently keeps serving whatever orientation it cached before the
    # rotation fix landed -- which looks exactly like the fix not working.
    no_cache = {"Cache-Control": "no-store, must-revalidate"}
    rotation = _arkit_rotation_for(scene)
    media_type = mimetypes.guess_type(image_path.name)[0] or ""
    if rotation == 0 and media_type in BROWSER_SAFE_IMAGE_TYPES:
        return FileResponse(str(image_path), media_type=media_type, headers=no_cache)
    # Display-only rectification (module docstring) -- rotate and/or re-encode
    # in memory, never touching the file on disk or anything the training/eval
    # pipeline reads.
    from io import BytesIO
    from PIL import Image
    buffer = BytesIO()
    rotate_image(Image.open(image_path), rotation).save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png",
                    headers=no_cache)


@app.get("/api/polygons/{image_id:path}")
def get_polygons(image_id: str, objects: str = "", modality: str = ""):
    scene = SCENE_INDEX.get(image_id)
    if scene is None:
        raise HTTPException(404, f"Unknown image_id: {image_id!r}")
    modality = _requested_modality(modality)
    object_indices = {int(token) for token in objects.split(",") if token.strip() != ""}
    polygons = SCENE_INDEX.polygons_for(image_id, object_indices or None)
    scale_x, scale_y = _geometry_scale(scene, modality)
    frame_width, frame_height = scene.frame_size_for(modality)
    rotation = _arkit_rotation_for(scene)
    width, height = rotated_dimensions(frame_width, frame_height, rotation)
    result = []
    for polygon in polygons:
        scaled_x = [value * scale_x for value in polygon.x]
        scaled_y = [value * scale_y for value in polygon.y]
        x, y = rotate_points(scaled_x, scaled_y, frame_width, frame_height, rotation)
        result.append({"object_index": polygon.object_index, "name": polygon.name, "x": x, "y": y})
    return {"image_width": width, "image_height": height, "modality": modality,
            "polygons": result}


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

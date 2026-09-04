"""
P1 of the VQA-SUNRGBD-v2 pipeline (see docs/DATASET_CREATION_PLAN.md, Rule V1-V4).

Builds the canonical object vocabulary from the raw annotation names seen
in data/index/scene_index.jsonl (written by build_index.py / P0):

  1. normalize every raw name (dataset_creation/v2/vocab.py: lowercase,
     strip trailing digits, singularize with an inflect-bug guard)
  2. merge known synonyms/spelling variants (data/vocab/synonyms.csv,
     hand-curated from the top-250 raw-name frequency table)
  3. canonical vocabulary = the 37 SUNRGBD segmentation classes (seg37,
     included regardless of frequency) UNION every merged concept with
     >=100 occurrences in the corpus (Rule V2)
  4. assign each canonical concept a category (for hard-negative sampling
     in later question generators) and an is_structural flag (Rule V3:
     wall/floor/ceiling/door_frame/window_frame/etc. are never an answer)
  5. record each concept's median visible size, from its non-border-touching
     instances only, to data/vocab/concept_typical_area.json — the frozen
     reference scene_objects.py compares a border-touching instance's area
     against, to tell a naturally-small-but-visible object apart from a
     large object that is mostly cropped out of frame (§13, crop gate)

Placeholder/failed-annotation labels ("unknown", "object") are excluded
outright, never merged into a real object class.

Anything not selected into the canonical set is not deleted: it still
exists as a raw/normalized name and still counts as "present" for
existence-question negatives (Rule V4). Borderline concepts (50-99
occurrences, not already pulled in by seg37) are written to
vocab_review_queue.csv for a human to promote in a later pass, per the
plan's P1 acceptance step ("hand review of top 300 normalised names").
"""
from __future__ import annotations

import csv
import json
import os
import statistics
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vocab import load_synonyms, normalize_raw_name  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
BUILD_LOG_DIR = os.path.join(REPO_ROOT, "build_log")

# An object concept needs at least this many clearly-visible (non-touching)
# instances corpus-wide before its median size is trusted as "typical" —
# see TYPICAL_AREA_PATH below.
MIN_SAMPLES_FOR_TYPICAL_AREA = 10
SCENE_INDEX_PATH = os.path.join(DATA_DIR, "index", "scene_index.jsonl")
SYNONYMS_PATH = os.path.join(DATA_DIR, "vocab", "synonyms.csv")
SEG37_MAT_PATH = os.path.join(REPO_ROOT, "dataset", "SUNRGBDtoolbox", "Metadata", "seg37list.mat")
CANONICAL_OBJECTS_PATH = os.path.join(DATA_DIR, "vocab", "canonical_objects.csv")
REVIEW_QUEUE_PATH = os.path.join(DATA_DIR, "vocab", "vocab_review_queue.csv")
TYPICAL_AREA_PATH = os.path.join(DATA_DIR, "vocab", "concept_typical_area.json")
CONFIG_PATH = os.path.join(DATA_DIR, "config.yaml")

FREQUENCY_THRESHOLD = 100
REVIEW_QUEUE_MIN_FREQUENCY = 50

# Placeholder / failed-annotation labels: never a real object, excluded
# outright regardless of frequency (both clear the frequency threshold).
EXCLUDED_CONCEPTS = {"unknown", "object"}

# concept (post-synonym-merge, still a single un-spaced token) -> display
# name using underscores for multi-word concepts (Rule Q3: underscores are
# replaced by spaces only at question/answer render time).
DISPLAY_NAME_OVERRIDES = {
    "filecabinet": "file_cabinet",
    "pictureframe": "picture_frame",
    "trashcan": "trash_can",
    "lightswitch": "light_switch",
    "remotecontrol": "remote_control",
    "floormat": "floor_mat",
    "nightstand": "night_stand",
    "showercurtain": "shower_curtain",
    "doorframe": "door_frame",
    "windowframe": "window_frame",
    "bedframe": "bed_frame",
    "cabinetdoor": "cabinet_door",
    "chairleg": "chair_leg",
    "tableleg": "table_leg",
    "doorhandle": "door_handle",
    "coffeetable": "coffee_table",
    "toiletpaper": "toilet_paper",
    "papertowel": "paper_towel",
    "firealarm": "fire_alarm",
    "fireextinguisher": "fire_extinguisher",
    "recyclebin": "recycle_bin",
    "switchboard": "switch_board",
    "pricetag": "price_tag",
    "tissuebox": "tissue_box",
    "stuffedanimal": "stuffed_animal",
    "bulletinboard": "bulletin_board",
    "dryeraseboard": "dry_erase_board",
    "waterbottle": "water_bottle",
    "flowerpot": "flower_pot",
    "windowsill": "window_sill",
    "mousepad": "mouse_pad",
    "cutting board": "cutting_board",
    "cuttingboard": "cutting_board",
    "endtable": "end_table",
    "tvstand": "tv_stand",
}

# category is used later for hard-negative sampling in the existence
# question generator (Rule 4.1: negatives should share the category of an
# object actually present). is_structural marks objects that Rule V3
# forbids as an answer to identification/counting/existence questions.
CATEGORY_AND_STRUCTURAL = {
    # -- structure (never an answer) --
    "wall": ("structure", True), "floor": ("structure", True), "ceiling": ("structure", True),
    "doorframe": ("structure", True), "windowframe": ("structure", True),
    "baseboard": ("structure", True), "pillar": ("structure", True), "column": ("structure", True),
    "roof": ("structure", True), "railing": ("structure", True), "partition": ("structure", True),
    "doorway": ("structure", True), "floormat": ("structure", False), "windowsill": ("structure", True),
    "tile": ("structure", True), "slab": ("structure", True), "step": ("structure", True),
    "stair": ("structure", True), "wallframe": ("structure", True),

    # -- furniture --
    "chair": ("furniture", False), "table": ("furniture", False), "sofa": ("furniture", False),
    "bed": ("furniture", False), "desk": ("furniture", False), "cabinet": ("furniture", False),
    "shelf": ("furniture", False), "bookshelf": ("furniture", False), "dresser": ("furniture", False),
    "stool": ("furniture", False), "bench": ("furniture", False), "ottoman": ("furniture", False),
    "armchair": ("furniture", False), "nightstand": ("furniture", False),
    "coffeetable": ("furniture", False), "drawer": ("furniture", False), "counter": ("furniture", False),
    "cupboard": ("furniture", False), "filecabinet": ("furniture", False), "bookcase": ("furniture", False),
    "closet": ("furniture", False), "bathtub": ("fixture", False), "toilet": ("fixture", False),
    "sink": ("fixture", False), "podium": ("furniture", False), "cart": ("furniture", False),
    "tvstand": ("furniture", False), "endtable": ("furniture", False),

    # -- fixture (fixed to the room, not furniture) --
    "door": ("fixture", False), "window": ("fixture", False), "faucet": ("fixture", False),
    "showercurtain": ("fixture", False), "outlet": ("fixture", False), "lightswitch": ("fixture", False),
    "switchboard": ("fixture", False), "doorhandle": ("fixture", False), "cabinetdoor": ("fixture", False),
    "vent": ("fixture", False), "thermostat": ("fixture", False), "firealarm": ("fixture", False),
    "fireextinguisher": ("fixture", False), "blind": ("fixture", False), "curtain": ("fixture", False),
    "doorknob": ("fixture", False), "fireplace": ("fixture", False), "drain": ("fixture", False),
    "knob": ("fixture", False), "handle": ("fixture", False), "tap": ("fixture", False),
    "socket": ("fixture", False), "heater": ("fixture", False), "airconditioner": ("fixture", False),

    # -- electronics / appliances --
    "monitor": ("electronics", False), "computer": ("electronics", False), "mouse": ("electronics", False),
    "keyboard": ("electronics", False), "laptop": ("electronics", False), "television": ("electronics", False),
    "cpu": ("electronics", False), "speaker": ("electronics", False), "printer": ("electronics", False),
    "phone": ("electronics", False), "telephone": ("electronics", False),
    "remotecontrol": ("electronics", False), "fan": ("electronics", False), "microwave": ("electronics", False),
    "oven": ("electronics", False), "refrigerator": ("electronics", False), "dishwasher": ("electronics", False),
    "stove": ("electronics", False), "projector": ("electronics", False), "screen": ("electronics", False),
    "desktop": ("electronics", False), "machine": ("electronics", False), "device": ("electronics", False),

    # -- container --
    "box": ("container", False), "bag": ("container", False), "basket": ("container", False),
    "trashcan": ("container", False), "recyclebin": ("container", False), "bin": ("container", False),
    "jar": ("container", False), "bottle": ("container", False), "container": ("container", False),
    "bucket": ("container", False), "backpack": ("container", False), "suitcase": ("container", False),
    "tissuebox": ("container", False), "purse": ("container", False),

    # -- tableware --
    "plate": ("tableware", False), "bowl": ("tableware", False), "cup": ("tableware", False),
    "mug": ("tableware", False), "glass": ("tableware", False), "tray": ("tableware", False),

    # -- decor --
    "picture": ("decor", False), "poster": ("decor", False), "painting": ("decor", False),
    "pictureframe": ("decor", False), "sign": ("decor", False), "plant": ("decor", False),
    "flower": ("decor", False), "sculpture": ("decor", False), "candlestick": ("decor", False),
    "photo": ("decor", False), "mirror": ("decor", False), "clock": ("decor", False),
    "flowerpot": ("decor", False), "vase": ("decor", False), "rack": ("furniture", False),
    "stand": ("furniture", False),

    # -- textile --
    "pillow": ("textile", False), "blanket": ("textile", False), "carpet": ("textile", False),
    "rug": ("textile", False), "clothes": ("textile", False), "cushion": ("textile", False),
    "towel": ("textile", False), "cloth": ("textile", False),

    # -- stationery --
    "paper": ("stationery", False), "book": ("stationery", False), "magazine": ("stationery", False),
    "binder": ("stationery", False), "pen": ("stationery", False), "marker": ("stationery", False),
    "tag": ("stationery", False), "pricetag": ("stationery", False), "label": ("stationery", False),
    "eraser": ("stationery", False), "stapler": ("stationery", False), "file": ("stationery", False),
    "napkin": ("stationery", False), "notice": ("stationery", False),

    # -- component (a part of a bigger object, not an object on its own) --
    "chairleg": ("component", False), "tableleg": ("component", False),

    # -- person --
    "person": ("person", False),

    # -- other --
    "toy": ("other", False), "shoe": ("other", False), "wire": ("other", False), "cord": ("other", False),
    "pipe": ("other", False), "toiletpaper": ("other", False), "papertowel": ("other", False),
    "wood": ("other", False), "stuffedanimal": ("other", False), "bulletinboard": ("decor", False),
    "chalkboard": ("decor", False), "blackboard": ("decor", False), "whiteboard": ("decor", False),
    "dryeraseboard": ("decor", False), "electricaloutlet": ("fixture", False),

    # -- additions found missing on the first pass over the real corpus --
    "lamp": ("fixture", False), "light": ("fixture", False), "board": ("decor", False),
    "frame": ("decor", False), "mat": ("textile", False), "mattress": ("furniture", False),
    "bedframe": ("furniture", False), "headboard": ("furniture", False), "pot": ("tableware", False),
    "tree": ("decor", False), "doll": ("other", False), "cable": ("other", False),
}


def load_seg37_concepts(synonym_map: dict) -> set:
    import scipy.io

    # seg37 class names use underscores as word separators ("night_stand",
    # "floor_mat"); our raw-name normalization joins words with no
    # separator ("nightstand", "floormat"), so strip underscores first or
    # these would end up as their own un-merged, zero-frequency concepts.
    seg37_raw = list(scipy.io.loadmat(SEG37_MAT_PATH, squeeze_me=True)["seg37list"])
    concepts = set()
    for raw_name in seg37_raw:
        normalized = normalize_raw_name(str(raw_name).replace("_", ""))
        concepts.add(synonym_map.get(normalized, normalized))
    return concepts


def display_name_for(concept: str) -> str:
    return DISPLAY_NAME_OVERRIDES.get(concept, concept)


def main() -> None:
    synonym_map = load_synonyms(SYNONYMS_PATH)
    seg37_concepts = load_seg37_concepts(synonym_map)

    records = [json.loads(line) for line in open(SCENE_INDEX_PATH)]

    concept_frequency = {}
    concept_normalized_variants = {}
    total_instances = 0
    for record in records:
        for obj in record["objects"]:
            normalized = normalize_raw_name(obj["raw_name"])
            if not normalized:
                continue
            total_instances += 1
            concept = synonym_map.get(normalized, normalized)
            if concept in EXCLUDED_CONCEPTS:
                continue
            concept_frequency[concept] = concept_frequency.get(concept, 0) + 1
            concept_normalized_variants.setdefault(concept, set()).add(normalized)

    frequent_concepts = {c for c, freq in concept_frequency.items() if freq >= FREQUENCY_THRESHOLD}
    final_concepts = sorted(seg37_concepts | frequent_concepts)

    with open(CONFIG_PATH) as config_file:
        min_area_frac = yaml.safe_load(config_file)["geometry"]["min_area_frac"]

    # Median visible size of each concept, measured only from instances that
    # are NOT touching the image border — i.e. presumed fully visible. This
    # is the frozen reference build_index.py's touches_border flag is
    # compared against (see scene_objects.py) to tell "naturally small
    # object, fully visible" apart from "large object, mostly cropped out of
    # frame": an absolute area threshold cannot make that distinction, since
    # both can land at the same area_frac. Frozen once here rather than
    # recomputed live by a generator, for the same reason
    # scene_type_cooccurrence.json is frozen (see its docstring in
    # existence.py and DATASET_CREATION_PLAN.md §13.15): a live recompute
    # would couple a test scene's crop verdict to which other scenes exist,
    # breaking the "test depends only on itself" invariance.
    visible_area_fracs_by_concept = defaultdict(list)
    for record in records:
        for obj in record["objects"]:
            if not obj.get("is_valid_polygon") or obj.get("touches_border"):
                continue
            area_frac = obj.get("area_frac")
            if area_frac is None or area_frac < min_area_frac:
                continue
            normalized = normalize_raw_name(obj["raw_name"])
            if not normalized:
                continue
            concept = synonym_map.get(normalized, normalized)
            if concept not in final_concepts:
                continue
            visible_area_fracs_by_concept[concept].append(area_frac)

    typical_area_by_concept = {
        concept: statistics.median(fracs)
        for concept, fracs in visible_area_fracs_by_concept.items()
        if len(fracs) >= MIN_SAMPLES_FOR_TYPICAL_AREA
    }
    os.makedirs(os.path.dirname(TYPICAL_AREA_PATH), exist_ok=True)
    with open(TYPICAL_AREA_PATH, "w") as typical_area_file:
        json.dump({
            "min_samples": MIN_SAMPLES_FOR_TYPICAL_AREA,
            "median_area_frac_by_concept": typical_area_by_concept,
        }, typical_area_file, indent=2, sort_keys=True)

    missing_category = [c for c in final_concepts if c not in CATEGORY_AND_STRUCTURAL]
    if missing_category:
        raise SystemExit(
            "Add a (category, is_structural) entry to CATEGORY_AND_STRUCTURAL for: "
            + ", ".join(sorted(missing_category))
        )

    os.makedirs(os.path.join(DATA_DIR, "vocab"), exist_ok=True)
    os.makedirs(BUILD_LOG_DIR, exist_ok=True)

    with open(CANONICAL_OBJECTS_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["canonical_concept", "display_name", "category", "is_structural", "source", "frequency"])
        for concept in final_concepts:
            category, is_structural = CATEGORY_AND_STRUCTURAL[concept]
            in_seg37 = concept in seg37_concepts
            in_frequent = concept in frequent_concepts
            source = "seg37+frequent" if (in_seg37 and in_frequent) else ("seg37" if in_seg37 else "frequent")
            writer.writerow([
                concept, display_name_for(concept), category, is_structural,
                source, concept_frequency.get(concept, 0),
            ])

    review_queue = sorted(
        (
            (freq, concept)
            for concept, freq in concept_frequency.items()
            if concept not in final_concepts and freq >= REVIEW_QUEUE_MIN_FREQUENCY
        ),
        reverse=True,
    )
    with open(REVIEW_QUEUE_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["concept", "frequency", "raw_normalized_variants_seen"])
        for freq, concept in review_queue:
            writer.writerow([concept, freq, "|".join(sorted(concept_normalized_variants.get(concept, [])))])

    covered_instances = sum(
        1
        for record in records
        for obj in record["objects"]
        if synonym_map.get(normalize_raw_name(obj["raw_name"]), normalize_raw_name(obj["raw_name"])) in final_concepts
    )
    coverage = covered_instances / total_instances if total_instances else 0.0

    report = {
        "total_object_instances": total_instances,
        "unique_normalized_names": len(concept_frequency) + len(EXCLUDED_CONCEPTS),
        "final_canonical_concepts": len(final_concepts),
        "from_seg37_only": len(seg37_concepts - frequent_concepts),
        "from_frequency_only": len(frequent_concepts - seg37_concepts),
        "from_both": len(seg37_concepts & frequent_concepts),
        "excluded_placeholder_concepts": sorted(EXCLUDED_CONCEPTS),
        "review_queue_size": len(review_queue),
        "instance_coverage_fraction": round(coverage, 4),
    }
    with open(os.path.join(BUILD_LOG_DIR, "p1_vocab_report.json"), "w") as report_file:
        json.dump(report, report_file, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nCanonical vocabulary written: {CANONICAL_OBJECTS_PATH} ({len(final_concepts)} concepts)")
    print(f"Review queue written: {REVIEW_QUEUE_PATH} ({len(review_queue)} concepts, freq 50-99)")
    print(f"Typical area table written: {TYPICAL_AREA_PATH} "
          f"({len(typical_area_by_concept)} / {len(final_concepts)} concepts with "
          f">={MIN_SAMPLES_FOR_TYPICAL_AREA} non-touching samples)")
    if coverage < 0.90:
        print(f"\nWARNING: instance coverage {coverage:.1%} is below the plan's 90% acceptance target.")


if __name__ == "__main__":
    main()

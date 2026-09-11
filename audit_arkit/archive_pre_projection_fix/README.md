# Archived 2026-09-11 — pre-projection-fix audit material

`audit_items.csv` here was sampled from VQA-ARKitScenes-v1.0, whose labels were
built with an inverted world-to-camera transform (`phase0_probe.world_to_camera`,
corrected 2026-09-11 — see its docstring). Every polygon, centroid, visibility
decision and therefore every generated question in that release is unreliable:
13 of these 38 items reference frames that contain no visible object at all
under the corrected projection.

`responses/solo.jsonl` holds 5 verdicts recorded against that material. They are
kept for the record but must not be counted toward any label-error rate: they
judged questions generated from bad geometry, and their `question_id`s refer to
different questions in the rebuilt release.

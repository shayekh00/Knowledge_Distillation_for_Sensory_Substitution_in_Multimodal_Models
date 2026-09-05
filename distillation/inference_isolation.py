"""Prove the deployed student is depth-only (plan §7.4).

The paper's central deployment claim is that the student needs no RGB at
inference. That has to be *demonstrated*, not asserted from reading the loader —
the legacy path's own augmentation flag looked correct and did nothing
(`implementation_audit.md` §A1), which is exactly how a confident reading goes
wrong.

Three independent checks, deliberately overlapping:

* :func:`assert_paths_denied` — make RGB and teacher-cache directories
  unreadable, then run a full evaluation pass. Anything that quietly reaches for
  them fails loudly instead of silently succeeding from a warm cache.
* :func:`assert_predictions_invariant_to_rgb` — mutate RGB paths while holding
  depth and questions fixed. Predictions must be **bitwise identical**.
* :func:`FileAccessTracer` — record what the loader actually opened, and compare
  it against the authorized set.

The second is the strongest single check: it does not depend on guessing which
directories matter.
"""
from __future__ import annotations

import contextlib
import os
import stat
from dataclasses import dataclass, field


@dataclass
class FileAccessTracer:
    """Record every path opened while active.

    Wraps `builtins.open`, which catches PIL, numpy, and plain file reads. It does
    not catch memory-mapped or C-level access, so a clean trace is supporting
    evidence rather than proof on its own — which is why §7.4 asks for three
    checks and not one.
    """
    opened: list = field(default_factory=list)
    _original: object = None

    def __enter__(self):
        import builtins
        self._original = builtins.open

        def traced_open(file, *args, **kwargs):
            self.opened.append(str(file))
            return self._original(file, *args, **kwargs)

        builtins.open = traced_open
        return self

    def __exit__(self, *exc_info):
        import builtins
        builtins.open = self._original
        return False

    def touched(self, fragment: str) -> list:
        return [path for path in self.opened if fragment in path]

    def assert_untouched(self, fragments) -> None:
        offenders = {fragment: self.touched(fragment) for fragment in fragments
                     if self.touched(fragment)}
        if offenders:
            raise AssertionError(
                "depth-only inference opened forbidden path(s): "
                + "; ".join(f"{fragment} -> {paths[:3]}" for fragment, paths in offenders.items()))


@contextlib.contextmanager
def paths_denied(directories):
    """Temporarily remove read permission from `directories`.

    Restores the original modes on exit even if the body raises. Running as root
    bypasses file permissions entirely, so callers should treat a pass under root
    as inconclusive — :func:`assert_paths_denied` says so explicitly.
    """
    original = {}
    try:
        for directory in directories:
            if os.path.exists(directory):
                original[directory] = stat.S_IMODE(os.stat(directory).st_mode)
                os.chmod(directory, 0o000)
        yield
    finally:
        for directory, mode in original.items():
            with contextlib.suppress(OSError):
                os.chmod(directory, mode)


def running_as_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def assert_paths_denied(run_inference, forbidden_directories) -> dict:
    """Run inference with the forbidden directories unreadable.

    Returns a report rather than a bare pass/fail, because a pass under root is
    not evidence: root ignores the permission bits, so the check has to be
    reported as inconclusive rather than quietly counted as a success.
    """
    if running_as_root():
        return {
            "status": "inconclusive",
            "reason": "running as root; permission bits do not restrict access. "
                      "Re-run as an unprivileged user, or rely on the path-mutation "
                      "and tracer checks instead.",
        }
    with paths_denied(forbidden_directories):
        run_inference()
    return {"status": "passed", "denied": list(forbidden_directories)}


def assert_predictions_invariant_to_rgb(run_inference, rows, rgb_field="image_path") -> dict:
    """Predictions must not change when RGB paths are mutated.

    Depth, questions, and every other input are held fixed; only the RGB path
    strings change, to values that do not exist. A model that truly ignores RGB
    produces bitwise-identical output. This check needs no guess about which
    directories matter, which is what makes it the strongest of the three.
    """
    baseline = list(run_inference(rows))
    mutated_rows = []
    for index, row in enumerate(rows):
        copy = dict(row)
        copy[rgb_field] = f"/nonexistent/rgb/{index}.jpg"
        mutated_rows.append(copy)
    mutated = list(run_inference(mutated_rows))

    if len(baseline) != len(mutated):
        raise AssertionError(
            f"prediction count changed under RGB mutation: {len(baseline)} vs {len(mutated)}")
    differing = [index for index, (before, after) in enumerate(zip(baseline, mutated))
                 if before != after]
    if differing:
        raise AssertionError(
            f"{len(differing)} prediction(s) changed when only the RGB path changed "
            f"(first at index {differing[0]}). The student is not depth-only.")
    return {"status": "passed", "n_items": len(baseline)}


@dataclass(frozen=True)
class IsolationReport:
    """Collected §7.4 evidence, for the run manifest."""
    path_denial: dict
    rgb_invariance: dict
    tracer_clean: bool
    depth_provenance: str

    def passed(self) -> bool:
        """RGB invariance is required. Path denial may be inconclusive under root,
        but must never be a failure."""
        return (self.rgb_invariance.get("status") == "passed"
                and self.path_denial.get("status") in ("passed", "inconclusive")
                and self.tracer_clean)

    def to_dict(self) -> dict:
        return {
            "path_denial": self.path_denial,
            "rgb_invariance": self.rgb_invariance,
            "tracer_clean": self.tracer_clean,
            "depth_provenance": self.depth_provenance,
            "passed": self.passed(),
        }

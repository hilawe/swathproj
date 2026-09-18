"""Mutation harness for verify_vgac_alongtrack.py: every assertion is watched failing.

Run from the repo root with the project venv:

    .venv/bin/python verification/test_alongtrack_mutations.py

WHY THIS EXISTS. The along-track findings were reported as "mutation-checked" while the mutations
themselves lived only in a shell session. That is the same preservation fault this project had just
removed from a joint-fit claim: a result cited as evidence whose procedure was not retained cannot
be re-run, audited, or trusted. A named assertion that nobody can watch fail is decoration.

HOW IT WORKS. Each case below names a defect, applies it as an exact string substitution to a COPY
of the script in a temporary directory, runs that copy, and requires it to exit nonzero with the
assertion message that is supposed to catch that defect. Requiring the SPECIFIC message matters:
the assertions overlap, and a mutation caught by the wrong one would otherwise report as healthy.

The copy runs with SWATHPROJ_ALONGTRACK_FAST=1, which skips the full-array scoring. The one case
that targets the full-array tail sets it back off.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "verification" / "verify_vgac_alongtrack.py"
# Honours the same override the script under test uses, so the missing-data path is testable
# rather than only reachable by deleting the granule.
DATA = Path(os.environ["SWATHPROJ_VGAC_DATA"]) if os.environ.get("SWATHPROJ_VGAC_DATA") \
    else REPO / "data" / "VGAC_sample_orbit.nc"

# (name, exact text to replace, replacement, fragment of the assertion that must fire, fast)
CASES = [
    ("time axis no longer linear",
     "    (a, b), *_ = np.linalg.lstsq(design, scan_time, rcond=None)",
     "    scan_time = scan_time + 1e-8*np.sin(idx/50.0)\n"
     "    (a, b), *_ = np.linalg.lstsq(design, scan_time, rcond=None)",
     "departs from a straight line", True),

    ("abstract forms no longer degenerate",
     "    step_b, rate_b = slope + NOMINAL_RATE * b, NOMINAL_RATE",
     "    step_b, rate_b = slope + NOMINAL_RATE * b * 1.000001, NOMINAL_RATE",
     "are NOT degenerate", True),

    ("latitude no longer pins the scan step",
     "    lo, hi = closure_step * (1 - 3.7e-5), closure_step * (1 + 2.9e-5)",
     "    lo, hi = closure_step * (1 + 1e-6), closure_step * (1 + 2.9e-5)",
     "not pinned by latitude at the precision claimed", True),

    ("latitude objective is flat",
     "    assert latitude_rms(closure_step * (1 + 1e-6)) > 10 * latitude_rms(closure_step), (",
     "    assert latitude_rms(closure_step * (1 + 1e-16)) > 10 * latitude_rms(closure_step), (",
     "objective is flat near the optimum", True),

    ("the 360-degree alias no longer holds",
     "        assert abs(latitude_rms(closure_step + 360.0 * k) - latitude_rms(closure_step)) < 1e-12, (",
     "        assert abs(latitude_rms(closure_step + 360.5 * k) - latitude_rms(closure_step)) < 1e-12, (",
     "alias no longer reproduces the objective", True),

    ("the compensating step stops being worse",
     "    m_substituted = median_residual(step_b, rate_b)",
     "    m_substituted = median_residual(step_a, rate_a)",
     "supposed to make things WORSE", True),

    ("held-out score scored with the nominal rate",
     "    held = residual(test_j, test_i, fitted)",
     "    held = residual(test_j, test_i, NOMINAL_RATE)",
     "the fit does not transfer even across halves", True),

    ("full-array tail regresses while the median holds",
     "    full = residual(jj, ii, fitted)",
     "    full = residual(jj, ii, fitted)\n    full[0] = 0.05",
     "full-array MAX", False),
]


def main():
    if not DATA.exists():
        # NOT a skip. This harness exists to prove the assertions fire, and a run that executed no
        # mutation has proved nothing. Returning zero here would let an automated evidence gate
        # report a successful mutation check that never ran, which is the failure mode this whole
        # project keeps rediscovering: unchecked must never read as passed.
        print(f"FAIL: {DATA} not present, so no mutation could run and nothing is verified.")
        print("      Set SWATHPROJ_VGAC_DATA to the granule, or run this only where it exists.")
        return 1

    source = TARGET.read_text()
    failures = []

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "verify_vgac_alongtrack.py"
        for name, old, new, fragment, fast in CASES:
            if old not in source:
                failures.append(f"{name}: the target text is no longer present, so this mutation "
                                f"tests nothing. Update it or remove the case.")
                print(f"  STALE   {name}")
                continue
            work.write_text(source.replace(old, new, 1))
            env = dict(os.environ)
            env["SWATHPROJ_ALONGTRACK_FAST"] = "1" if fast else "0"
            env["SWATHPROJ_VGAC_DATA"] = str(DATA)
            # the copy lives in a temp dir, so sys.path[0] is not the repo and swathproj would be
            # unimportable. An import failure would look like a caught mutation without this.
            env["PYTHONPATH"] = str(REPO)
            proc = subprocess.run([sys.executable, str(work)], capture_output=True, text=True,
                                  cwd=str(REPO), env=env)
            out = proc.stdout + proc.stderr
            if "SKIP" in out:
                # A run that did not execute must never be reported as a caught mutation, nor as a
                # survivor. This fired on the harness's first run, when the copy could not see the
                # data and exited zero through its skip path.
                failures.append(f"{name}: the mutated script SKIPPED rather than running. It "
                                f"proves nothing, and a skip must not be read as a result.")
                print(f"  SKIPPED  {name}")
            elif proc.returncode == 0:
                failures.append(f"{name}: the mutated script PASSED. The assertion meant to catch "
                                f"this defect does not.")
                print(f"  SURVIVED {name}")
            elif fragment not in out:
                failures.append(f"{name}: failed, but not with the assertion under test. Expected "
                                f"a message containing {fragment!r}. Got:\n"
                                + "\n".join("      " + q for q in out.strip().split("\n")[-4:]))
                print(f"  WRONG GUARD {name}")
            else:
                print(f"  caught   {name}")

    # the unmutated script must still pass, or the harness proves nothing
    proc = subprocess.run([sys.executable, str(TARGET)], capture_output=True, text=True,
                          cwd=str(REPO), env={**os.environ, "SWATHPROJ_ALONGTRACK_FAST": "1",
                                              "SWATHPROJ_VGAC_DATA": str(DATA)})
    if "SKIP" in (proc.stdout + proc.stderr):
        failures.append("the baseline run SKIPPED, so nothing here was verified")
        print("  BASELINE SKIPPED")
    elif proc.returncode != 0:
        failures.append("the UNMUTATED script fails, so every 'caught' above is meaningless:\n"
                        + proc.stdout + proc.stderr)
        print("  BASELINE FAILS")
    else:
        print("  baseline passes")

    if failures:
        print("\nMUTATION HARNESS FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print(f"\nall {len(CASES)} mutations caught by the specific assertion under test")
    return 0


if __name__ == "__main__":
    sys.exit(main())

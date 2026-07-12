# Cinematic Profile + cinematic_swell Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fifth loopback profile ("Cinematic") to the UI dropdown, built around a new `cinematic_swell` renderer: calm drifting floor + slow eased swells triggered only by strong kick hits, available in both loopback and AI-synced modes.

**Architecture:** New renderer in the shared `DmxEngineBase` (dmx_engine.py) following the golden_anthem accumulated-dt pattern; a new `profiles/cinematic.json` (auto-discovered by `app.py`, so the dropdown grows with zero backend/frontend code changes); registration in `music_light.py` ambient rotation and `llm_designer.py` prompt/repair gate. Spec: `docs/superpowers/specs/2026-07-12-cinematic-swell-profile-design.md`.

**Tech Stack:** Python 3.13 (plain procedural, no type hints), pytest, existing `dmx_punch.velocity_brightness`, `ema`, `lerp_color` helpers. Test with `.venv\Scripts\python.exe -m pytest -q` from repo root. Suite baseline before this plan: **37 passed**.

**Key existing anchors (verified 2026-07-12):**
- Renderer signature (all ambient renderers): `(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare, kick_color, accent_color, volume, cue, t)` — see `_render_golden_anthem` at `dmx_engine.py:894`.
- `self._beat_velocity` (0..1) is set in `process_audio` (`dmx_engine.py:1168`) before `_dispatch`.
- Constants block ends at `ANTHEM_DISCONTINUITY_THRESHOLD` (`dmx_engine.py:90`); state block ends at `self._ga_loud_ref` (`dmx_engine.py:252`); `VALID_BEHAVIORS` at `dmx_engine.py:104`; `_behavior_map` at `dmx_engine.py:264`.
- `ema(current, target, attack, decay)` and `lerp_color(c1, c2, t)` are module functions in dmx_engine.py.
- **User decision:** profile does NOT pin the renderer; auto-switching stays (declined `force_behavior`).

---

### Task 1: `cinematic_swell` renderer in dmx_engine.py (TDD)

**Files:**
- Modify: `dmx_engine.py` (constants after line 90, state after line 252, renderer after `_render_golden_anthem` ending line 941, `VALID_BEHAVIORS` line 104-111, `_behavior_map` around line 264)
- Test: `tests/test_cinematic_swell.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cinematic_swell.py`. Follow the construction pattern used in `tests/test_golden_anthem.py` (read it first; it builds a DRY_RUN engine — reuse its fixture/helper approach exactly, including setting `DMX_DRY_RUN=1` before import if that's how it does it).

```python
"""cinematic_swell: calm drifting floor + eased swells on strong kicks only.
Pins: trigger threshold, peak cap, smooth rise (no instant flash), decay back
to floor, velocity-scaled peaks, discontinuity guard, registration."""
import os
os.environ["DMX_DRY_RUN"] = "1"

import dmx_engine
from dmx_engine import (
    _cine_envelope, CINE_TRIGGER_VELOCITY, CINE_RISE_S, CINE_FALL_S,
    CINE_FLOOR_MIN, CINE_PEAK_MAX, VALID_BEHAVIORS,
)


def make_engine():
    # Mirror however test_golden_anthem.py constructs its engine (a minimal
    # DmxEngineBase subclass or direct instantiation under DRY_RUN).
    from test_golden_anthem import make_engine as make  # reuse if importable;
    return make()                                       # else copy its body.


def run_frames(eng, n, t0=0.0, velocity=0.0, kick_first=False, dt=0.012):
    """Drive the renderer directly for n frames; returns final t."""
    t = t0
    for i in range(n):
        eng._beat_velocity = velocity if (kick_first and i == 0) else 0.0
        eng._render_cinematic_swell(
            0.0, 0.0, 0.0, 0.1,                  # kick_i, snare_i, hihat_i, mid_i
            kick_first and i == 0, False,         # is_kick, is_snare
            (255, 140, 20), (0, 120, 140),        # kick_color, accent_color
            0.05, {"dimmer": 80}, t)              # volume, cue, t
        t += dt
    return t


def test_weak_hit_does_not_start_swell():
    eng = make_engine()
    run_frames(eng, 50)                                   # settle at floor
    floor_master = eng.out_master
    run_frames(eng, 50, t0=0.6, velocity=CINE_TRIGGER_VELOCITY - 0.1,
               kick_first=True)
    # A sub-threshold kick must not lift master meaningfully above the floor.
    assert eng.out_master < floor_master + 20.0


def test_strong_hit_swells_smoothly_and_caps():
    eng = make_engine()
    run_frames(eng, 50)
    prev = eng.out_master
    max_jump = 0.0
    peak = 0.0
    t = 0.6
    for i in range(200):                                  # ~2.4s of frames
        eng._beat_velocity = 0.9 if i == 0 else 0.0
        eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, i == 0, False,
                                    (255, 140, 20), (0, 120, 140),
                                    0.05, {"dimmer": 80}, t)
        max_jump = max(max_jump, abs(eng.out_master - prev))
        peak = max(peak, eng.out_master)
        prev = eng.out_master
        t += 0.012
    assert peak > 150.0                                   # swell clearly fired
    assert peak <= 255.0 * CINE_PEAK_MAX + 1.0            # hard cap holds
    assert max_jump < 30.0                                # eased, not a flash


def test_swell_decays_back_to_floor():
    eng = make_engine()
    run_frames(eng, 50)
    floor_master = eng.out_master
    # Trigger, then run well past rise+fall (+EMA slack).
    total_frames = int((CINE_RISE_S + CINE_FALL_S) / 0.012) + 300
    run_frames(eng, total_frames, t0=0.6, velocity=0.9, kick_first=True)
    assert abs(eng.out_master - floor_master) < 15.0


def test_stronger_hit_peaks_higher():
    def peak_for(v):
        eng = make_engine()
        run_frames(eng, 50)
        peak, t = 0.0, 0.6
        for i in range(200):
            eng._beat_velocity = v if i == 0 else 0.0
            eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, i == 0, False,
                                        (255, 140, 20), (0, 120, 140),
                                        0.05, {"dimmer": 80}, t)
            peak = max(peak, eng.out_master)
            t += 0.012
        return peak
    assert peak_for(1.0) > peak_for(CINE_TRIGGER_VELOCITY + 0.05) + 10.0


def test_discontinuity_guard_on_seek():
    eng = make_engine()
    run_frames(eng, 50)
    drift_before = eng._cs_drift_phase
    # One frame with t jumped 60s ahead: drift must advance <= one nominal frame.
    eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, False, False,
                                (255, 140, 20), (0, 120, 140),
                                0.05, {"dimmer": 80}, 60.6)
    assert abs(eng._cs_drift_phase - drift_before) < 0.01


def test_floor_never_dark_at_dimmer_zero():
    eng = make_engine()
    t = 0.0
    for _ in range(100):
        eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, False, False,
                                    (255, 140, 20), (0, 120, 140),
                                    0.05, {"dimmer": 0}, t)
        t += 0.012
    assert eng.out_master >= 255.0 * CINE_FLOOR_MIN - 5.0


def test_registered_in_engine():
    assert "cinematic_swell" in VALID_BEHAVIORS
    eng = make_engine()
    assert eng._behavior_map["cinematic_swell"] == eng._render_cinematic_swell
```

Adjust `make_engine`/import mechanics to match how `tests/test_golden_anthem.py` actually does it — same fixture style, same DRY_RUN setup. Keep every assertion above.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cinematic_swell.py -v`
Expected: FAIL at import — `ImportError: cannot import name '_cine_envelope'`.

- [ ] **Step 3: Add constants to dmx_engine.py**

Insert immediately after `ANTHEM_DISCONTINUITY_THRESHOLD` (line 90):

```python
# cinematic_swell renderer tuning. Film-score hits: a calm drifting floor,
# and a slow eased swell (rise + fall, ~2s total) fired only by STRONG kicks.
# Accumulated-dt like golden_anthem: swell/drift progress advances by clamped
# per-frame dt, so seeks and ambient-rotation re-entry cannot corrupt it.
CINE_TRIGGER_VELOCITY = 0.55  # min _beat_velocity to start a swell
CINE_RISE_S = 0.5             # eased rise duration (seconds)
CINE_FALL_S = 1.4             # eased fall duration (seconds)
CINE_FLOOR_MIN = 0.08         # never-dark floor (not dimmer-scaled; abyssal PWM lesson)
CINE_PEAK_MAX = 0.90          # hard cap on swell peak brightness fraction
CINE_DRIFT_PERIOD_S = 20.0    # idle floor drifts color_1 <-> color_2 this slowly
CINE_WHITE_PEAK = 100.0       # max white-channel lift at swell peak
CINE_DT_CLAMP = 0.1           # max seconds of progress per frame
CINE_DISCONTINUITY_THRESHOLD = 1.0  # bigger call-gap => one nominal frame
```

- [ ] **Step 4: Add the envelope helper (module function, after `_anthem_envelope`)**

```python
def _cine_envelope(age):
    """Eased swell envelope: smoothstep up over CINE_RISE_S, smoothstep down
    over CINE_FALL_S. `age` is seconds since trigger; returns 0..1.
    Returns 0.0 once the swell is finished (age >= rise+fall)."""
    if age < 0.0:
        return 0.0
    if age < CINE_RISE_S:
        x = age / CINE_RISE_S
        return x * x * (3.0 - 2.0 * x)
    fall_age = age - CINE_RISE_S
    if fall_age >= CINE_FALL_S:
        return 0.0
    x = 1.0 - fall_age / CINE_FALL_S
    return x * x * (3.0 - 2.0 * x)
```

- [ ] **Step 5: Add state in `DmxEngineBase.__init__`**

Insert immediately after `self._ga_loud_ref = 1e-6` (line 252):

```python
        # cinematic_swell renderer state
        self._cs_swell_age = None          # None = idle; else seconds since trigger
        self._cs_swell_peak = 0.0          # velocity-scaled peak fraction for the active swell
        self._cs_drift_phase = 0.0         # 0..1 idle floor drift position
        self._cs_last_render_t = None      # last t this renderer was called with
```

- [ ] **Step 6: Add the renderer (after `_render_golden_anthem`, i.e. after line 941)**

```python
    def _render_cinematic_swell(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                                kick_color, accent_color, volume, cue, t):
        """Film-score mode: a dim floor that drifts slowly between the palette
        colors, plus a wide eased swell toward the accent color fired only by
        STRONG kicks (velocity-gated). Rise ~0.5s / fall ~1.4s -- a slow-motion
        impact, never a flash. Accumulated dt (golden_anthem pattern): seeks
        and rotation re-entry advance progress by at most one nominal frame."""
        dimmer = (cue.get("dimmer", 60) if cue else 60) / 100.0

        if (self._cs_last_render_t is None or
                abs(t - self._cs_last_render_t) > CINE_DISCONTINUITY_THRESHOLD):
            dt = 0.012
        else:
            dt = min(max(t - self._cs_last_render_t, 0.0), CINE_DT_CLAMP)
        self._cs_last_render_t = t

        # --- Trigger: strong kicks only. Retrigger mid-swell only if the new
        # hit would out-peak what's left of the current swell (no stacking).
        if is_kick and self._beat_velocity >= CINE_TRIGGER_VELOCITY:
            new_peak = min(CINE_PEAK_MAX, velocity_brightness(self._beat_velocity) / 255.0)
            if self._cs_swell_age is None:
                self._cs_swell_age = 0.0
                self._cs_swell_peak = new_peak
            elif new_peak > self._cs_swell_peak * _cine_envelope(self._cs_swell_age):
                self._cs_swell_age = 0.0
                self._cs_swell_peak = new_peak

        # --- Advance swell + idle drift by clamped dt.
        swell = 0.0
        if self._cs_swell_age is not None:
            self._cs_swell_age += dt
            env = _cine_envelope(self._cs_swell_age)
            if self._cs_swell_age >= CINE_RISE_S + CINE_FALL_S:
                self._cs_swell_age = None
            swell = env * self._cs_swell_peak
        self._cs_drift_phase = (self._cs_drift_phase + dt / CINE_DRIFT_PERIOD_S) % 1.0

        # --- Floor: slow triangle-wave drift color_1 <-> color_2.
        tri = 1.0 - abs(2.0 * self._cs_drift_phase - 1.0)
        floor_color = lerp_color(kick_color, accent_color, tri * 0.6)
        # Floor is NOT dimmer-scaled (PWM-visibility lesson from abyssal_bloom).
        floor_b = CINE_FLOOR_MIN

        # --- Swell lifts brightness toward the accent color; never darkens floor.
        swell_color = lerp_color(floor_color, accent_color, min(1.0, swell * 1.5))
        brightness = max(floor_b, swell * dimmer)
        col = swell_color if swell > 0.0 else floor_color

        self.out_r = ema(self.out_r, col[0] * brightness, 0.08, 0.05)
        self.out_g = ema(self.out_g, col[1] * brightness, 0.08, 0.05)
        self.out_b = ema(self.out_b, col[2] * brightness, 0.08, 0.05)
        # White only near the swell peak (shimmer precedent from golden_anthem).
        white = CINE_WHITE_PEAK * max(0.0, swell - 0.7) / 0.3 * dimmer
        self.out_w = ema(self.out_w, white, 0.08, 0.06)
        self.out_master = ema(self.out_master, 255.0 * brightness, 0.08, 0.05)
        self.out_strobe = 0
```

- [ ] **Step 7: Register in `VALID_BEHAVIORS` and `_behavior_map`**

In `VALID_BEHAVIORS` (line 104-111) add `"cinematic_swell",` on the ambient line after `"golden_anthem",`. In `_behavior_map` (line 264+) add:

```python
            "cinematic_swell": self._render_cinematic_swell,
```

- [ ] **Step 8: Run the new tests**

Run: `.venv\Scripts\python.exe -m pytest tests/test_cinematic_swell.py -v`
Expected: 7 PASS. If `test_strong_hit_swells_smoothly_and_caps` fails on `max_jump`, loosen the EMA attack (0.08) is the knob — do NOT loosen the test bound above 40 without flagging it in your report.

- [ ] **Step 9: Run the full suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: **44 passed** (37 + 7), zero failures.

- [ ] **Step 10: Commit**

```bash
git add dmx_engine.py tests/test_cinematic_swell.py
git commit -m "feat: add cinematic_swell renderer (velocity-gated eased swells over drifting floor)"
```

Do NOT stage `playback_state.json`, `PROJECT_ARCHITECTURE.md`, or `PR_DESCRIPTION.md`.

---

### Task 2: Cinematic profile + loopback/LLM registration

**Files:**
- Create: `profiles/cinematic.json`
- Modify: `music_light.py:147` (ambient_pool), `music_light.py:303` (ambient_behaviors set)
- Modify: `llm_designer.py:~314` (prompt entry), `llm_designer.py:~457` (VALID_BEHAVIORS)

- [ ] **Step 1: Create `profiles/cinematic.json`**

```json
{
    "name": "Cinematic",
    "description": "Film-score mode. Slow immersive washes with wide dramatic palettes; big bass hits become slow-motion color swells, never flashes. Best for soundtracks, trailers, and emotional builds.",
    "gain_boost": 35.0,
    "volume_gate": 0.00005,
    "agc_thresh": 0.45,
    "kick_thresh": 0.10,
    "snare_thresh": 0.14,
    "onset_cooldown": 0.30,
    "palettes": [
        [ [255, 140, 20], [0, 120, 140] ],
        [ [180, 20, 30], [40, 90, 160] ],
        [ [255, 180, 40], [60, 30, 150] ],
        [ [200, 60, 0], [120, 180, 220] ]
    ],
    "color_cycle_mode": "time",
    "color_cycle_interval": 12.0,
    "rhythm_change_pct": 0.50,
    "deep_bass_enabled": true,
    "deep_bass_thresh": 0.80,
    "decay_speed": 0.92,
    "glow_thresh": 0.35,
    "beat_hold_frames": 8,
    "deep_bass_hold_frames": 8
}
```

(`color_cycle_*` / `rhythm_change_pct` are accepted-but-unused legacy keys — kept for schema consistency with the other four profiles.)

- [ ] **Step 2: Register in music_light.py**

At line 147, `ambient_pool` becomes:

```python
                ambient_pool = ["ocean_drift", "candlelight", "aurora_shimmer",
                               "sunset_fade", "abyssal_bloom", "golden_anthem",
                               "cinematic_swell"]
```

At line 303, add `"cinematic_swell",` to the `ambient_behaviors` set immediately after `"golden_anthem",`.

- [ ] **Step 3: Register in llm_designer.py**

Insert after the golden_anthem prompt block (ends line 314, before `=== NARRATIVE ARC TEMPLATE ===` at line 316), matching the neighbors' exact 3-line format:

```text
"cinematic_swell" — Calm dim floor drifting slowly between the palette colors; a STRONG bass hit triggers one slow, wide swell toward the accent color (rise ~0.5s, fall ~1.4s), like a film-score impact in slow motion. Weak beats do nothing. No strobe, no flashing.
  USE FOR: Cinematic tension-building sections, dramatic intros, half-time or stripped-back breakdowns, trailer-style moments.
  FEEL: Widescreen, slow-motion impact. Every hit feels earned.
```

In llm_designer's `VALID_BEHAVIORS` (~line 457) add `"cinematic_swell",` after `"golden_anthem",`.

- [ ] **Step 4: Repair-preservation check**

Run:

```bash
.venv/Scripts/python.exe -c "
from llm_designer import _validate_and_repair_plan
plan = {'show_name':'t','cues':[{'start_time':0,'end_time':10,'color_1':[255,140,20],'color_2':[0,120,140],'energy':5,'strobe':False,'behavior':'cinematic_swell','dimmer':80,'fade_in':1,'fade_out':1,'section_name':'intro','mood':'dark'}]}
print('behavior:', _validate_and_repair_plan(plan)['cues'][0]['behavior'])
"
```

Expected: `behavior: cinematic_swell` (NOT `beat_reactive` — if downgraded, the VALID_BEHAVIORS edit was missed).

- [ ] **Step 5: Profile discovery check**

Run: `.venv/Scripts/python.exe -c "import json,glob; [print(json.load(open(p))['name']) for p in sorted(glob.glob('profiles/*.json'))]"`
Expected output includes `Cinematic` plus the existing four names. (The dropdown is fed by `app.py list_profiles()` scanning this directory — no other change needed.)

- [ ] **Step 6: Dry-run smoke (synced path)**

Write a temp show JSON in the scratchpad (NOT the repo) with one full-track `cinematic_swell` cue over any WAV in `youtube_audio\` (same harness pattern as the golden_anthem Task 8 smoke: `DMX_DRY_RUN=1`, run `ai_show_player.py --show <temp.json>`, confirm `[DRY ...] cue=... ` logs and no traceback; sample `last_frame` mid-track and confirm non-zero floor frames).

- [ ] **Step 7: Full suite**

Run: `.venv\Scripts\python.exe -m pytest -q`
Expected: **44 passed**.

- [ ] **Step 8: Commit**

```bash
git add profiles/cinematic.json music_light.py llm_designer.py
git commit -m "feat: add Cinematic profile; register cinematic_swell in loopback rotation and LLM behavior set"
```

---

### Task 3: Final verification + handoff

**Files:** none modified (verification only).

- [ ] **Step 1: Full suite** — `.venv\Scripts\python.exe -m pytest -q` → **44 passed**.

- [ ] **Step 2: Pyflakes vs known baseline**

Run: `.venv\Scripts\python.exe -m pyflakes dmx_engine.py music_light.py ai_show_player.py llm_designer.py dmx_variety.py dmx_punch.py`
Expected: EXACTLY the 3 known baseline lines (unused `e` dmx_engine.py, unused `mids` music_light.py, lazy `boto3` llm_designer.py) — any NEW line is a defect to fix.

- [ ] **Step 3: Dry-run determinism** — run the Task 2 Step 6 smoke twice; `[DRY ...]` lines must be identical modulo timestamps.

- [ ] **Step 4: HARD CHECKPOINT — user hardware pass.** Hand to the user: pick **Cinematic** in the dropdown, play a soundtrack/mid-energy track. Confirm: calm drifting floor; strong bass hits become ~2s slow swells (not flashes); weak beats do nothing; loud sustained passages still route to punchy renderers (expected — auto-switching kept by user decision). Then finishing-a-development-branch (merge/PR decision). Do not proceed past this checkpoint without the user.

---

## Verification summary

| Check | Command | Expected |
|---|---|---|
| Unit tests | `pytest -q` | 44 passed |
| Lint | pyflakes 6 modules | 3 known baseline lines only |
| Repair gate | Task 2 Step 4 one-liner | `behavior: cinematic_swell` |
| Dropdown | Task 2 Step 5 | 5 profile names incl. Cinematic |
| Dry-run | Task 2 Step 6 ×2 | deterministic, no traceback |
| Feel | hardware pass (user) | slow swells, no flashes |

## Known risks (from spec)

1. `_beat_velocity` feel is hardware-adjudicated only — the DRY_RUN harness compresses time, so beat-driven triggering in dry-run is not representative (known caveat, documented at `ai_show_player.py:70-74`).
2. In loopback, ambient rotation only runs during very quiet passages, where strong kicks are rare — cinematic_swell may mostly show its floor there; the full effect shows best in synced shows. User explicitly chose to keep auto-switching (no profile pinning).

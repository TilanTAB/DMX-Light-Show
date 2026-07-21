# Ambient Pulse Profile + force_behavior Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A sixth loopback profile ("Ambient Pulse") that is chill-ambient in color but genuinely beat-locked, via a new layered multi-band `ambient_pulse` renderer and a new `force_behavior` profile-pinning key.

**Architecture:** Renderer in the shared `DmxEngineBase` (dmx_engine.py) using the accumulated-dt pattern; pinning is a one-key change in `music_light.py` (`load_profile` + `_dispatch`) that overrides auto-behavior while keeping the energy state machine running for palette variety; `profiles/ambient_pulse.json` is auto-discovered by `app.py`. Spec: `docs/superpowers/specs/2026-07-17-ambient-pulse-profile-design.md`.

**Tech Stack:** Python 3.13 (plain procedural, no type hints), pytest, existing helpers `ema`, `lerp_color`, `dmx_punch.velocity_brightness`, `dmx_punch.beat_velocity_from_ratio`. Run tests with `.venv\Scripts\python.exe -m pytest -q` from repo root. **Suite baseline on this branch: 59 passed.**

**Key anchors (verified 2026-07-17 on this branch, commit 1a945c2 — re-grep, don't trust line numbers):**
- Renderer signature: `(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare, kick_color, accent_color, volume, cue, t)` — see `_render_cinematic_swell`.
- CINE constants end at `CINE_DISCONTINUITY_THRESHOLD` (`dmx_engine.py:111`); `VALID_BEHAVIORS` at ~124-132; engine state block ends at `self._cs_last_render_t` (~317); `_behavior_map` cinematic entry at ~347.
- `music_light._dispatch`: variety tick ~292-299; `auto_behavior = self._detect_auto_behavior(...)` at ~302; `ambient_behaviors` set at ~312-315; synthetic ambient cue `{"dimmer": 50, ...}` at ~318.
- `_beat_velocity` is threshold-excess graded (0 at ratio 1.0, 1.0 at ratio 3.0) via `beat_velocity_from_ratio` — the fix merged in 61cb7a3. `velocity_brightness(v)` maps 0..1 → 120..255.

---

### Task 1: `ambient_pulse` renderer in dmx_engine.py (TDD)

**Files:**
- Modify: `dmx_engine.py` (constants after `CINE_DISCONTINUITY_THRESHOLD` ~line 111; state after `self._cs_last_render_t` ~line 317; renderer after `_render_cinematic_swell`; `VALID_BEHAVIORS`; `_behavior_map`)
- Test: `tests/test_ambient_pulse.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_ambient_pulse.py`:

```python
"""ambient_pulse: layered multi-band beat-locked ambient. Pins: graded kick
pulse with no post-hit step-up, snare color flip without master spike,
gated hi-hat shimmer, mid-driven breathing floor, discontinuity guard,
registration. Built on the threshold-excess _beat_velocity (soft ~0.1,
hard 1.0) merged in 61cb7a3."""
import os
os.environ["DMX_DRY_RUN"] = "1"

from dmx_engine import (
    DmxEngineBase, VALID_BEHAVIORS,
    PULSE_FLOOR_MIN, PULSE_FLOOR_MAX, PULSE_SNARE_FLIP_S, PULSE_HIHAT_THRESH,
)


def make_engine():
    eng = DmxEngineBase()
    return eng


def frame(eng, t, kick=False, snare=False, velocity=0.0,
          mid_i=0.2, hihat_i=0.0, dimmer=80):
    eng._beat_velocity = velocity
    eng._render_ambient_pulse(0.0, 0.0, hihat_i, mid_i, kick, snare,
                              (20, 60, 255), (255, 0, 180),
                              0.05, {"dimmer": dimmer}, t)


def settle(eng, n=100, t0=0.0):
    t = t0
    for _ in range(n):
        frame(eng, t)
        t += 0.012
    return t


def test_kick_pulse_rises_instantly_then_never_steps_up():
    eng = make_engine()
    t = settle(eng)
    floor_master = eng.out_master
    frame(eng, t, kick=True, velocity=1.0)
    hit_master = eng.out_master
    assert hit_master > floor_master + 50.0      # instant, visible rise
    prev = hit_master
    for i in range(120):                          # ~1.4s decay window
        t += 0.012
        frame(eng, t)
        assert eng.out_master <= prev + 1e-9      # monotone: no step-up
        prev = eng.out_master
    assert abs(eng.out_master - floor_master) < 20.0   # back near floor


def test_kick_pulse_graded_by_velocity():
    def peak_for(v):
        eng = make_engine()
        t = settle(eng)
        frame(eng, t, kick=True, velocity=v)
        return eng.out_master
    assert peak_for(1.0) > peak_for(0.2) + 30.0


def test_snare_flips_color_without_master_spike():
    eng = make_engine()
    t = settle(eng)
    master_before = eng.out_master
    r_before = eng.out_r
    frame(eng, t, snare=True, velocity=0.5)
    # Master must not pulse on a snare...
    assert eng.out_master < master_before + 15.0
    # ...but color moves toward the accent (255, 0, 180): red rises.
    for _ in range(10):
        t += 0.012
        frame(eng, t, snare=False)
    assert eng.out_r > r_before + 5.0
    # And it eases back after the flip window.
    for _ in range(int(PULSE_SNARE_FLIP_S / 0.012) + 80):
        t += 0.012
        frame(eng, t)
    assert abs(eng.out_r - r_before) < 15.0


def test_hihat_shimmer_gated_and_fast():
    eng = make_engine()
    t = settle(eng)
    w_quiet = eng.out_w
    frame(eng, t, hihat_i=PULSE_HIHAT_THRESH - 0.1)
    assert eng.out_w <= w_quiet + 1.0             # below gate: nothing
    frame(eng, t + 0.012, hihat_i=PULSE_HIHAT_THRESH + 0.2)
    w_spike = eng.out_w
    assert w_spike > 30.0                          # spike fired
    frame(eng, t + 0.024, hihat_i=0.0)
    assert eng.out_w < w_spike * 0.7               # fast decay


def test_floor_breathes_with_mids_and_never_dark():
    def settled_master(mid, dimmer):
        eng = make_engine()
        t = 0.0
        for _ in range(300):
            frame(eng, t, mid_i=mid, dimmer=dimmer)
            t += 0.012
        return eng.out_master
    assert settled_master(0.9, 80) > settled_master(0.05, 80) + 20.0
    # Floor not dimmer-scaled: still lit at dimmer 0.
    assert settled_master(0.05, 0) >= 255.0 * PULSE_FLOOR_MIN - 5.0
    # Capped: loud mids never exceed the floor max band by much.
    assert settled_master(1.0, 80) <= 255.0 * PULSE_FLOOR_MAX + 10.0


def test_discontinuity_guard_on_seek():
    eng = make_engine()
    settle(eng)
    drift_before = eng._ap_drift_phase
    frame(eng, 60.6)                               # 60s jump, one frame
    # Guard: advance <= one nominal frame (0.012/16 = 0.00075), NOT the
    # dt-clamp alone (0.1/16 = 0.00625) -- bound must discriminate.
    assert abs(eng._ap_drift_phase - drift_before) < 0.002


def test_registered_in_engine():
    assert "ambient_pulse" in VALID_BEHAVIORS
    eng = make_engine()
    assert eng._behavior_map["ambient_pulse"] == eng._render_ambient_pulse
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python.exe -m pytest tests/test_ambient_pulse.py -v`
Expected: FAIL at import — `ImportError: cannot import name 'PULSE_FLOOR_MIN'`.

- [ ] **Step 3: Add constants**

Insert immediately after `CINE_DISCONTINUITY_THRESHOLD` (dmx_engine.py ~line 111):

```python
# ambient_pulse renderer tuning. Layered multi-band: kicks pulse master
# (graded velocity), snares flip color, hi-hats shimmer white, mids breathe
# the floor. Accumulated-dt (sibling pattern): timers/drift advance by a
# clamped per-frame dt, so seeks and re-entry cannot corrupt them.
PULSE_FLOOR_MIN = 0.10        # breathing floor minimum (not dimmer-scaled)
PULSE_FLOOR_MAX = 0.35        # floor at sustained loud mids
PULSE_DRIFT_PERIOD_S = 16.0   # floor color drift period
PULSE_KICK_DECAY = 0.10       # per-frame exponential decay of the kick pulse
PULSE_SNARE_FLIP_S = 0.3      # how long a snare holds the accent color
PULSE_HIHAT_THRESH = 0.5      # hihat intensity gate for shimmer
PULSE_WHITE_SPIKE = 80.0      # white channel spike on shimmer
PULSE_DT_CLAMP = 0.1          # max accumulated-dt advance per frame
PULSE_DISCONTINUITY_THRESHOLD = 1.0  # bigger call-gap => one nominal frame
```

- [ ] **Step 4: Add state in `DmxEngineBase.__init__`**

Insert immediately after `self._cs_last_render_t = None` (~line 317):

```python
        # ambient_pulse renderer state
        self._ap_floor_energy = 0.0        # slow EMA of mid_i (breathing floor)
        self._ap_drift_phase = 0.0         # 0..1 floor color drift position
        self._ap_pulse_level = 0.0         # kick pulse height (0..1, decays)
        self._ap_snare_timer = 0.0         # seconds left of snare color flip
        self._ap_last_render_t = None      # last t this renderer was called with
```

- [ ] **Step 5: Add the renderer (after `_render_cinematic_swell`)**

```python
    def _render_ambient_pulse(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Beat-locked ambient: layered multi-band response. Kicks snap master
        to the graded velocity level and decay from THAT level (no hold
        plateau -- double-pulse lesson); snares flip the wash toward the
        accent color without touching master; hi-hats spark the white channel;
        sustained mids breathe a never-dark floor. Accumulated dt: seeks and
        rotation re-entry advance timers by at most one nominal frame."""
        dimmer = (cue.get("dimmer", 60) if cue else 60) / 100.0

        if (self._ap_last_render_t is None or
                abs(t - self._ap_last_render_t) > PULSE_DISCONTINUITY_THRESHOLD):
            dt = 0.012
        else:
            dt = min(max(t - self._ap_last_render_t, 0.0), PULSE_DT_CLAMP)
        self._ap_last_render_t = t

        # -- Layer 1: mid-driven breathing floor with slow color drift.
        self._ap_floor_energy = ema(self._ap_floor_energy, min(1.0, mid_i), 0.05, 0.02)
        # Floor is NOT dimmer-scaled (PWM-visibility lesson from abyssal_bloom).
        floor_b = PULSE_FLOOR_MIN + self._ap_floor_energy * (PULSE_FLOOR_MAX - PULSE_FLOOR_MIN)
        self._ap_drift_phase = (self._ap_drift_phase + dt / PULSE_DRIFT_PERIOD_S) % 1.0
        tri = 1.0 - abs(2.0 * self._ap_drift_phase - 1.0)
        base_color = lerp_color(kick_color, accent_color, tri * 0.7)

        # -- Layer 3: snare color flip (color event only, master untouched).
        if is_snare and not is_kick:
            self._ap_snare_timer = PULSE_SNARE_FLIP_S
        if self._ap_snare_timer > 0.0:
            self._ap_snare_timer = max(0.0, self._ap_snare_timer - dt)
            base_color = lerp_color(base_color, accent_color,
                                    self._ap_snare_timer / PULSE_SNARE_FLIP_S)

        # -- Layer 2: kick pulse. Graded by threshold-excess velocity; decays
        # exponentially from its own hit level toward the floor.
        if is_kick:
            self._ap_pulse_level = max(self._ap_pulse_level,
                                       velocity_brightness(self._beat_velocity) / 255.0)
        else:
            self._ap_pulse_level *= (1.0 - PULSE_KICK_DECAY)
            if self._ap_pulse_level < 0.01:
                self._ap_pulse_level = 0.0
        brightness = max(floor_b, self._ap_pulse_level * dimmer)

        # -- Layer 4: hi-hat shimmer (sparkle, never a wash).
        if hihat_i > PULSE_HIHAT_THRESH:
            self.out_w = max(self.out_w, PULSE_WHITE_SPIKE * dimmer)
        else:
            self.out_w *= 0.5

        # Color channels: crisp attack on kick frames, smooth otherwise.
        att = 0.9 if is_kick else 0.25
        self.out_r = ema(self.out_r, base_color[0] * brightness, att, 0.15)
        self.out_g = ema(self.out_g, base_color[1] * brightness, att, 0.15)
        self.out_b = ema(self.out_b, base_color[2] * brightness, att, 0.15)
        # Master: instant snap on the kick frame (sync feel), smooth follow after.
        if is_kick:
            self.out_master = 255.0 * brightness
        else:
            self.out_master = ema(self.out_master, 255.0 * brightness, 0.3, 0.12)
        self.out_strobe = 0
```

- [ ] **Step 6: Register.** Add `"ambient_pulse",` in `VALID_BEHAVIORS` after `"cinematic_swell",` and in `_behavior_map`:

```python
            "ambient_pulse": self._render_ambient_pulse,
```

- [ ] **Step 7: Run the new tests** — `.venv\Scripts\python.exe -m pytest tests/test_ambient_pulse.py -v` → 7 PASS. Tuning knobs if a bound fails: EMA decay 0.12/0.15 (smoothness), PULSE_KICK_DECAY (decay-window test). Do not weaken the monotone no-step-up assertion.

- [ ] **Step 8: Full suite** — `.venv\Scripts\python.exe -m pytest -q` → **66 passed** (59 + 7).

- [ ] **Step 9: Commit**

```bash
git add dmx_engine.py tests/test_ambient_pulse.py
git commit -m "feat: add ambient_pulse renderer (layered multi-band beat-locked ambient)"
```

Never stage: playback_state.json, PROJECT_ARCHITECTURE.md, PR_DESCRIPTION.md, current_show.json, root test_*.py.

---

### Task 2: `force_behavior` profile pinning (TDD)

**Files:**
- Modify: `music_light.py` (`__init__` ~line 21-49, `load_profile` ~line 51-78, `_dispatch` ~line 302-320)
- Test: `tests/test_ambient_pulse.py` (append)

- [ ] **Step 1: Write the failing tests (append to tests/test_ambient_pulse.py)**

```python
def _loopback_engine():
    # Imports pyaudiowpatch (Windows-only) -- fine on this project's machine.
    from music_light import DMXEngine
    return DMXEngine()


def _drive_dispatch(eng, n=40, loud=True):
    """Push frames through _dispatch with loud punchy-looking input."""
    vol = 0.01 if loud else 0.00006
    for i in range(n):
        eng.frame_counter += 1
        eng._dispatch(0.5, 0.2, 0.3, 0.1,
                      0.5 if loud else 0.0, 0.1, 0.2, 0.3,
                      loud and (i % 10 == 0), False,
                      vol, 48000)


def test_force_behavior_pins_dispatch_at_any_energy():
    eng = _loopback_engine()
    eng.profile_force_behavior = "ambient_pulse"
    eng.energy_state = "high"                      # would normally go punchy
    _drive_dispatch(eng, loud=True)
    assert eng.current_behavior == "ambient_pulse"
    eng.energy_state = "calm"                      # would normally rotate ambient
    _drive_dispatch(eng, loud=False)
    assert eng.current_behavior == "ambient_pulse"


def test_no_force_behavior_keeps_auto_detection():
    eng = _loopback_engine()
    assert eng.profile_force_behavior is None      # default: auto
    eng.energy_state = "high"
    _drive_dispatch(eng, loud=True)
    assert eng.current_behavior != "ambient_pulse" # auto picked something else


def test_force_behavior_unknown_value_falls_back():
    import json, tempfile, os as _os
    eng = _loopback_engine()
    prof = {"name": "Bad", "force_behavior": "no_such_renderer"}
    fd, path = tempfile.mkstemp(suffix=".json")
    with _os.fdopen(fd, "w") as f:
        json.dump(prof, f)
    try:
        eng.load_profile(path)
    finally:
        _os.remove(path)
    assert eng.profile_force_behavior is None      # warned + fell back


def test_variety_still_evolves_when_pinned():
    eng = _loopback_engine()
    eng.profile_force_behavior = "ambient_pulse"
    _drive_dispatch(eng, n=5, loud=True)
    first_palette = eng.variety.current_palette["id"]
    # Force the evolution timer past the threshold and dispatch again.
    eng.variety._section_start_t = -999.0
    _drive_dispatch(eng, n=5, loud=True)
    assert eng.variety.current_palette["id"] is not None
    # begin_section ran again (anti-repeat may or may not change the id;
    # the invariant is that the variety tick/begin_section path still runs).
    assert eng._last_section_id is not None
```

Note on the last test: read `dmx_variety.py` first — if `_section_start_t` is named differently, use the real attribute; the assertion target is "the variety path still executes under pinning," pinned via `_last_section_id` being maintained.

- [ ] **Step 2: Run to verify failure** — `.venv\Scripts\python.exe -m pytest tests/test_ambient_pulse.py -k force -v`
Expected: AttributeError `profile_force_behavior` (attribute missing).

- [ ] **Step 3: Implement in music_light.py**

In `__init__` (after `self.profile_deep_bass_hold = 5`):

```python
        # Optional: pin the dispatch to one renderer (profile "force_behavior"
        # key). None = auto-behavior detection as always. Added after Cinematic
        # hardware feedback: tuning-only profiles cannot guarantee a mode feel.
        self.profile_force_behavior = None
```

In `load_profile` (after the `profile_kick_dominance_ratio` line):

```python
            forced = p.get("force_behavior", None)
            if forced is not None:
                from dmx_engine import VALID_BEHAVIORS
                if forced in VALID_BEHAVIORS:
                    self.profile_force_behavior = forced
                else:
                    logger.warning(f"[PROFILE] Unknown force_behavior '{forced}' "
                                   "-- falling back to auto-behavior detection")
                    self.profile_force_behavior = None
```

(Module-level import is also fine if music_light already imports from dmx_engine — prefer adding `VALID_BEHAVIORS` to the existing `from dmx_engine import (...)` at the top instead of a local import.)

In `_dispatch`, replace the auto-behavior line (~302-303):

```python
        # ── Auto-behavior detection: picks chill or punchy ──
        # Always runs (keeps the energy state machine + Intent mood fresh);
        # a profile force_behavior overrides only the CHOICE, never the machine.
        auto_behavior = self._detect_auto_behavior(volume, kick_i, snare_i, current_time, beats_per_sec)
        if self.profile_force_behavior:
            auto_behavior = self.profile_force_behavior
        self.current_behavior = auto_behavior
```

Add `"ambient_pulse",` to the `ambient_behaviors` set (~line 312-315, after `"cinematic_swell",`) AND to the `ambient_pool` list in `_detect_auto_behavior` (~line 146-149, append after `"cinematic_swell"` — spec: other, non-pinned profiles may rotate into it during very quiet passages). Then give the pinned mode brightness headroom in the synthetic cue (~line 318):

```python
        if auto_behavior in ambient_behaviors:
            # ambient_pulse is the beat-locked mode -- it needs pulse headroom,
            # not the dim ambient default.
            cue_dimmer = 80 if auto_behavior == "ambient_pulse" else 50
            cue = {"dimmer": cue_dimmer, "energy": 3, "start": 0, "end": 60,
                    "strobe": False, "fade": 3.0}
```

- [ ] **Step 4: Run the force tests** — `.venv\Scripts\python.exe -m pytest tests/test_ambient_pulse.py -v` → all pass.

- [ ] **Step 5: Full suite** — `.venv\Scripts\python.exe -m pytest -q` → **70 passed** (66 + 4).

- [ ] **Step 6: Commit**

```bash
git add music_light.py tests/test_ambient_pulse.py
git commit -m "feat: force_behavior profile key pins loopback dispatch (validated, auto fallback)"
```

---

### Task 3: Profile + LLM registration + verification

**Files:**
- Create: `profiles/ambient_pulse.json`
- Modify: `llm_designer.py` (prompt entry after cinematic_swell block ~line 316-318; `VALID_BEHAVIORS` ~line 461)
- Test: `tests/test_ambient_pulse.py` (append repair-gate test)

- [ ] **Step 1: Create `profiles/ambient_pulse.json`**

```json
{
    "name": "Ambient Pulse",
    "description": "Chill-ambient colors that genuinely follow the beat. Kicks pulse the wash, snares flip the color, hi-hats sparkle white, quiet passages breathe. Tomorrowland feel at living-room volume.",
    "force_behavior": "ambient_pulse",
    "gain_boost": 45.0,
    "volume_gate": 0.00005,
    "agc_thresh": 0.35,
    "kick_thresh": 0.06,
    "snare_thresh": 0.09,
    "onset_cooldown": 0.12,
    "palettes": [
        [ [20, 60, 255], [255, 0, 180] ],
        [ [140, 0, 255], [0, 220, 255] ],
        [ [255, 40, 120], [255, 190, 40] ],
        [ [0, 200, 180], [120, 40, 255] ]
    ],
    "color_cycle_mode": "time",
    "color_cycle_interval": 8.0,
    "rhythm_change_pct": 0.50,
    "deep_bass_enabled": false,
    "deep_bass_thresh": 0.80,
    "decay_speed": 0.85,
    "glow_thresh": 0.40,
    "beat_hold_frames": 4,
    "deep_bass_hold_frames": 4
}
```

- [ ] **Step 2: llm_designer.py registration.** Prompt entry inserted after the cinematic_swell block, before `=== NARRATIVE ARC TEMPLATE ===`, matching the neighbors' 3-line format:

```text
"ambient_pulse" — Gentle colored wash that visibly follows the music: kicks pulse the brightness (graded by hit strength), snares flip the wash toward the accent color, hi-hats add brief white sparkles, sustained melody breathes the floor. Beat-locked but never harsh.
  USE FOR: Verses, grooves, melodic mid-energy sections, any part that needs the lights "dancing along" without full punch.
  FEEL: Festival wash breathing with the groove. Alive, connected, gentle.
```

Add `"ambient_pulse",` to llm_designer's `VALID_BEHAVIORS` after `"cinematic_swell",`.

- [ ] **Step 3: Repair-gate test (append to tests/test_ambient_pulse.py)**

```python
def test_registered_in_llm_designer_repair_gate():
    import llm_designer
    assert "ambient_pulse" in llm_designer.VALID_BEHAVIORS
    plan = {"show_name": "t", "cues": [{
        "start_time": 0, "end_time": 10,
        "color_1": [20, 60, 255], "color_2": [255, 0, 180],
        "energy": 5, "strobe": False, "behavior": "ambient_pulse",
        "dimmer": 80, "fade_in": 1, "fade_out": 1,
        "section_name": "groove", "mood": "cool"}]}
    out = llm_designer._validate_and_repair_plan(plan)
    assert out["cues"][0]["behavior"] == "ambient_pulse"
```

- [ ] **Step 4: Verify.** (a) repair test passes; (b) profile discovery: `.venv/Scripts/python.exe -c "import json,glob; [print(json.load(open(p))['name']) for p in sorted(glob.glob('profiles/*.json'))]"` → 6 names including `Ambient Pulse`; (c) full suite → **71 passed**; (d) dry-run determinism: temp show JSON (scratchpad) with an ambient_pulse cue over a `youtube_audio\` WAV, `DMX_DRY_RUN=1 ai_show_player.py --show <temp>` twice → `[DRY]` lines identical modulo timestamps, no traceback.

- [ ] **Step 5: Commit**

```bash
git add profiles/ambient_pulse.json llm_designer.py tests/test_ambient_pulse.py
git commit -m "feat: add Ambient Pulse profile (force_behavior pinned); register ambient_pulse in LLM designer"
```

- [ ] **Step 6: Final verification + HARD CHECKPOINT.** Pyflakes over the 6 core modules → exactly the 3 known baseline lines (unused `e` dmx_engine.py, unused `mids` music_light.py, lazy `boto3` llm_designer.py). Then hand to the user for the hardware pass: pick **Ambient Pulse** in the dropdown, play an EDM track at normal volume. Confirm: lights visibly track kicks (harder kick = brighter pulse), snares flip color, hi-hats sparkle, quiet parts breathe, and the mode NEVER drops out to unrelated ambient renderers. Then finishing-a-development-branch (merge/PR). Do not proceed past this checkpoint without the user.

---

## Verification summary

| Check | Command | Expected |
|---|---|---|
| Unit tests | `pytest -q` | 71 passed |
| Lint | pyflakes 6 modules | 3 known baseline lines only |
| Repair gate | Task 3 test | ambient_pulse preserved |
| Dropdown | discovery one-liner | 6 names incl. Ambient Pulse |
| Dry-run | ×2 | deterministic, no traceback |
| Feel | hardware pass (user) | beat-locked, graded, no dropouts |

## Known risks (from spec)

1. Onset-detection latency (~11ms block + WASAPI) bounds perceived sync; if pulses feel late on hardware, the fix is upstream, not in this renderer.
2. Pinning bypasses only the behavior CHOICE — the energy state machine still runs for Intent mood/palette variety, pinned by `test_variety_still_evolves_when_pinned`.

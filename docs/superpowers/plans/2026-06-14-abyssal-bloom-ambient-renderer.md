# `abyssal_bloom` Ambient Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `abyssal_bloom`, a "deep & sparse" ambient DMX renderer (near-black drifting floor + rare bass-reactive teal blooms + rarer white glints), wired into both the loopback auto-behavior rotation and the AI-synced show pipeline.

**Architecture:** One new renderer method (`_render_abyssal_bloom`) on the existing `DMXEngine` class in `music_light.py` (this branch is the pre-split monolith — one file, one class, one `_behavior_map` read by both the loopback and synced dispatch branches of `process_audio`). Three additive visual layers (floor / bloom / glint) driven by per-instance envelope state initialized in `__init__`. Eight total registration touch points across `music_light.py` (6) and `llm_designer.py` (2) — see Task 2.

**Tech Stack:** Python 3.13, no new dependencies (`math`, existing `ema`/`lerp_color` helpers already imported in `music_light.py`). No test framework exists in this repo at this commit; verification is CLI + hardware/eyeball, per the spec.

**Spec:** `docs/superpowers/specs/2026-06-14-abyssal-bloom-ambient-renderer-design.md`

---

## Reality check: how this gets verified

This repo has no `pytest`, no `unittest`, nothing. Two verification tiers:
- **Task 1** (pure math) is checked with a tiny **throwaway script** run via `python -c` — not a committed test file (nothing to import it; the codebase has no test convention to extend). It proves the envelope math in isolation before it's wired into the class.
- **Tasks 2–4** (wiring + prompt) are checked by **running the actual engine** (`python music_light.py --mode ...`) and reading logs / watching the fixture, per the spec's Testing section.

## File structure

| File | Change |
|---|---|
| `music_light.py` | Add `_render_abyssal_bloom` + its `__init__` state + all 4 registration points (`VALID_BEHAVIORS`, `_behavior_map`, `ambient_behaviors`, `ambient_pool`) |
| `llm_designer.py` | Add the `abyssal_bloom` prompt entry + add to its own separate `VALID_BEHAVIORS` |

No new files. `abyssal_bloom`'s parameters are module-level constants in `music_light.py` (per spec: not `profile_*`, to avoid new `load_profile` keys).

**Naming locked across tasks:**
- Renderer method: `_render_abyssal_bloom(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare, kick_color, accent_color, volume, cue, t)` — matches every other ambient renderer's signature exactly.
- Instance state: `self._ab_bloom_active`, `self._ab_bloom_t0`, `self._ab_last_bloom_t`, `self._ab_bloom_color`, `self._ab_glint_active`, `self._ab_glint_t0`, `self._ab_last_glint_t`, `self._ab_bass`, `self._ab_floor_phase_seed` (not needed — floor uses `t` directly, no extra phase state).
- Module constants (prefix `ABYSSAL_`): `ABYSSAL_FLOOR_MIN`, `ABYSSAL_FLOOR_MAX`, `ABYSSAL_BLOOM_BASE`, `ABYSSAL_BLOOM_MAX`, `ABYSSAL_BLOOM_BASS_GAIN`, `ABYSSAL_BLOOM_RISE`, `ABYSSAL_BLOOM_HOLD`, `ABYSSAL_BLOOM_FALL`, `ABYSSAL_BLOOM_GAP_MIN`, `ABYSSAL_BLOOM_INTERVAL`, `ABYSSAL_BLOOM_NUDGE_THRESH`, `ABYSSAL_GLINT_INTERVAL`, `ABYSSAL_GLINT_THRESH`.

---

## Task 1: Envelope math — standalone verification script

Prove the bloom/glint envelope and floor-drift math in isolation (no `DMXEngine` dependency) before it's embedded in the class. This is a throwaway script, deleted at the end of the task — the repo has no test convention to fold it into.

**Files:**
- Create (temporary): `scratch_abyssal_math.py` (repo root; deleted in Step 5)

- [ ] **Step 1: Write the throwaway verification script**

`scratch_abyssal_math.py`:
```python
"""Throwaway script to verify abyssal_bloom envelope math before embedding
it in DMXEngine. Deleted after verification — this repo has no test runner."""
import math

FLOOR_MIN, FLOOR_MAX = 0.03, 0.10
BLOOM_BASE, BLOOM_MAX, BLOOM_BASS_GAIN = 0.45, 0.70, 0.25
BLOOM_RISE, BLOOM_HOLD, BLOOM_FALL = 1.5, 0.4, 3.5
BLOOM_GAP_MIN, BLOOM_INTERVAL, BLOOM_NUDGE_THRESH = 3.0, 11.0, 0.25


def floor_brightness(t):
    wave1 = math.sin(t * 0.05) * 0.5 + 0.5
    wave2 = math.sin(t * 0.033 + 1.1) * 0.5 + 0.5
    blend = wave1 * 0.6 + wave2 * 0.4
    return FLOOR_MIN + (FLOOR_MAX - FLOOR_MIN) * blend


def bloom_envelope(age):
    """0..1 envelope value for a bloom of given age (seconds since trigger)."""
    if age < BLOOM_RISE:
        p = age / BLOOM_RISE
        return p * p * (3.0 - 2.0 * p)  # smoothstep rise
    if age < BLOOM_RISE + BLOOM_HOLD:
        return 1.0
    fall_age = age - BLOOM_RISE - BLOOM_HOLD
    if fall_age < BLOOM_FALL:
        p = 1.0 - (fall_age / BLOOM_FALL)
        return p * p * (3.0 - 2.0 * p)  # smoothstep fall
    return 0.0


def bloom_peak_brightness(bass_activity):
    return min(BLOOM_MAX, BLOOM_BASE + bass_activity * BLOOM_BASS_GAIN)


def next_bloom_gap(bass_activity):
    """Bass shortens the gap between blooms, floored at BLOOM_GAP_MIN."""
    return max(BLOOM_GAP_MIN, BLOOM_INTERVAL * (1.0 - bass_activity * 0.6))


# --- Assertions ---
assert 0.0 <= floor_brightness(0) <= FLOOR_MAX
assert FLOOR_MIN <= floor_brightness(12.3) <= FLOOR_MAX
assert abs(bloom_envelope(0.0)) < 1e-9, "envelope must start at 0"
assert bloom_envelope(BLOOM_RISE) == 1.0, "envelope must hit 1.0 at end of rise"
assert bloom_envelope(BLOOM_RISE + BLOOM_HOLD) == 1.0, "envelope holds at 1.0"
total_dur = BLOOM_RISE + BLOOM_HOLD + BLOOM_FALL
assert bloom_envelope(total_dur) == 0.0, "envelope must return to 0 after fall"
assert bloom_envelope(total_dur + 1.0) == 0.0, "envelope stays 0 past duration"
assert bloom_peak_brightness(0.0) == BLOOM_BASE
assert bloom_peak_brightness(1.0) == BLOOM_MAX, "bass=1.0 must cap at BLOOM_MAX exactly"
assert bloom_peak_brightness(5.0) == BLOOM_MAX, "bass_activity above 1.0 must still clamp to BLOOM_MAX"
assert next_bloom_gap(0.0) == BLOOM_INTERVAL
assert next_bloom_gap(1.0) == max(BLOOM_GAP_MIN, BLOOM_INTERVAL * 0.4)
assert next_bloom_gap(1.0) >= BLOOM_GAP_MIN, "gap must never go below the hard floor"

print("All abyssal_bloom envelope math assertions passed.")
```

- [ ] **Step 2: Run it**

Run: `.venv\Scripts\python.exe scratch_abyssal_math.py`
Expected: `All abyssal_bloom envelope math assertions passed.` with no `AssertionError`.

- [ ] **Step 3: If any assertion fails, fix the formula (not the assertion) and re-run**

The assertions encode the spec's actual requirements (envelope shape, bounded peak brightness, bounded gap) — if one fails, the math is wrong, not the check.

- [ ] **Step 4: Delete the throwaway script**

Run: `rm scratch_abyssal_math.py` (or `del scratch_abyssal_math.py` on cmd)

- [ ] **Step 5: No commit for this task** — it produced no permanent files. Proceed directly to Task 2.

---

## Task 2: `_render_abyssal_bloom` + `__init__` state + module constants

Add the renderer itself and its instance state to `music_light.py`. This task does **not** register the behavior anywhere yet (that's Task 3) — so after this task the method exists but is unreachable, which keeps the change reviewable in isolation.

**Files:**
- Modify: `music_light.py`

- [ ] **Step 1: Add the module-level constants**

Find the existing constants block that ends with the beat-detection constants (around the `ONSET_COOLDOWN = 0.12` / `AGC_SPEED = 0.015` / `LOOPBACK_*` lines, just before `DEFAULT_PALETTES`). Add immediately after `LOOPBACK_AGC_THRESH = 0.3   # Lower AGC threshold for loopback (vs 0.7 for synced)`:

```python

# abyssal_bloom renderer tuning (module constants, not profile_* — avoids
# adding new load_profile keys for a single renderer's parameters).
ABYSSAL_FLOOR_MIN = 0.03
ABYSSAL_FLOOR_MAX = 0.10
ABYSSAL_BLOOM_BASE = 0.45
ABYSSAL_BLOOM_MAX = 0.70
ABYSSAL_BLOOM_BASS_GAIN = 0.25
ABYSSAL_BLOOM_RISE = 1.5
ABYSSAL_BLOOM_HOLD = 0.4
ABYSSAL_BLOOM_FALL = 3.5
ABYSSAL_BLOOM_GAP_MIN = 3.0
ABYSSAL_BLOOM_INTERVAL = 11.0
ABYSSAL_BLOOM_NUDGE_THRESH = 0.25
ABYSSAL_GLINT_RISE = 0.2
ABYSSAL_GLINT_FALL = 0.6
ABYSSAL_GLINT_INTERVAL = 18.0
ABYSSAL_GLINT_THRESH = 0.5
```

- [ ] **Step 2: Add the instance state in `__init__`**

In `music_light.py`, find the `# Loopback direct-drive state` block (currently `self.peak_kick = 0.0` / `self.peak_snare = 0.0` / `self.peak_mid = 0.0`, lines 172-175). Add immediately after `self.peak_mid = 0.0`:

```python

        # abyssal_bloom renderer state
        self._ab_bass = 0.0                # smoothed bass activity (EMA of kick_i)
        self._ab_bloom_active = False
        self._ab_bloom_t0 = 0.0            # t when the current/last bloom started
        self._ab_last_bloom_t = -999.0      # t when the last bloom finished (for gap timing)
        self._ab_bloom_color = (0, 210, 210)
        self._ab_glint_active = False
        self._ab_glint_t0 = 0.0
        self._ab_last_glint_t = -999.0
```

(`-999.0` for the "last" timestamps means the very first frame is always eligible to bloom/glint — no artificial startup delay.)

- [ ] **Step 3: Add the renderer method**

Add this method directly after `_render_aurora_shimmer` (which ends around line 845, right before the `_detect_auto_behavior` docstring block). Match the existing indentation (methods are indented one level under `class DMXEngine:`):

```python
    def _render_abyssal_bloom(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Deep & sparse ambient: near-black drifting floor, rare bass-reactive
        teal blooms, rarer white glints. Restraint is the effect — only one
        bloom or glint active at a time, with a hard minimum gap between blooms
        regardless of bass energy."""
        dimmer = (cue.get("dimmer", 50) if cue else 50) / 100.0

        # --- Smoothed bass activity (not the raw per-frame kick_i) ---
        self._ab_bass = ema(self._ab_bass, min(1.0, kick_i), 0.2, 0.05)

        # --- Layer 1: floor (two slow non-harmonic sines, deep blue <-> violet) ---
        # Deliberately NOT scaled by `dimmer` here: FLOOR_MIN/MAX already ARE the
        # intended final 3-10% floor (the spec's "never fully black" guarantee).
        # Scaling it again by the cue's dimmer (~0.5 typical) would push it below
        # the visible floor of every sibling ambient renderer -- likely reading
        # as fully off on real LED hardware (PWM dead-zone).
        wave1 = math.sin(t * 0.05) * 0.5 + 0.5
        wave2 = math.sin(t * 0.033 + 1.1) * 0.5 + 0.5
        floor_blend = wave1 * 0.6 + wave2 * 0.4
        floor_brightness = (ABYSSAL_FLOOR_MIN +
                            (ABYSSAL_FLOOR_MAX - ABYSSAL_FLOOR_MIN) * floor_blend)
        deep_blue = (10, 20, 120)
        deep_violet = (60, 10, 130)
        floor_color = lerp_color(deep_blue, deep_violet, floor_blend)

        # --- Layer 2: bloom (rare teal swell, bass-reactive but bounded) ---
        bloom_gap = max(ABYSSAL_BLOOM_GAP_MIN,
                        ABYSSAL_BLOOM_INTERVAL * (1.0 - self._ab_bass * 0.6))
        if not self._ab_bloom_active and not self._ab_glint_active:
            time_since_last = t - self._ab_last_bloom_t
            timer_ready = time_since_last >= ABYSSAL_BLOOM_INTERVAL
            nudge_ready = (time_since_last >= bloom_gap and
                          kick_i > ABYSSAL_BLOOM_NUDGE_THRESH)
            if timer_ready or nudge_ready:
                self._ab_bloom_active = True
                self._ab_bloom_t0 = t
                self._ab_bloom_color = lerp_color(accent_color, (0, 210, 210), 0.6)

        bloom_brightness = 0.0
        if self._ab_bloom_active:
            age = t - self._ab_bloom_t0
            total = ABYSSAL_BLOOM_RISE + ABYSSAL_BLOOM_HOLD + ABYSSAL_BLOOM_FALL
            if age >= total:
                self._ab_bloom_active = False
                self._ab_last_bloom_t = t
            else:
                if age < ABYSSAL_BLOOM_RISE:
                    p = age / ABYSSAL_BLOOM_RISE
                    env = p * p * (3.0 - 2.0 * p)
                elif age < ABYSSAL_BLOOM_RISE + ABYSSAL_BLOOM_HOLD:
                    env = 1.0
                else:
                    fall_age = age - ABYSSAL_BLOOM_RISE - ABYSSAL_BLOOM_HOLD
                    p = 1.0 - (fall_age / ABYSSAL_BLOOM_FALL)
                    env = p * p * (3.0 - 2.0 * p)
                peak = min(ABYSSAL_BLOOM_MAX,
                          ABYSSAL_BLOOM_BASE + self._ab_bass * ABYSSAL_BLOOM_BASS_GAIN)
                bloom_brightness = env * peak * dimmer

        # --- Layer 3: glint (rarer white flare) ---
        # Mutually exclusive with bloom (spec: "only one bloom OR glint active
        # at a time") -- a glint can't arm while a bloom is mid-swell, and the
        # bloom-arming check above already excludes an active glint too.
        if not self._ab_glint_active and not self._ab_bloom_active:
            time_since_glint = t - self._ab_last_glint_t
            timer_ready = time_since_glint >= ABYSSAL_GLINT_INTERVAL
            hihat_ready = (time_since_glint >= ABYSSAL_BLOOM_GAP_MIN and
                          hihat_i > ABYSSAL_GLINT_THRESH)
            if timer_ready or hihat_ready:
                self._ab_glint_active = True
                self._ab_glint_t0 = t

        glint_brightness = 0.0
        if self._ab_glint_active:
            age = t - self._ab_glint_t0
            total = ABYSSAL_GLINT_RISE + ABYSSAL_GLINT_FALL
            if age >= total:
                self._ab_glint_active = False
                self._ab_last_glint_t = t
            else:
                if age < ABYSSAL_GLINT_RISE:
                    glint_brightness = (age / ABYSSAL_GLINT_RISE) * dimmer
                else:
                    fall_age = age - ABYSSAL_GLINT_RISE
                    glint_brightness = (1.0 - fall_age / ABYSSAL_GLINT_FALL) * dimmer

        # --- Compose: bloom/glint lift above the floor, never darken it ---
        r = max(floor_color[0] * floor_brightness, self._ab_bloom_color[0] * bloom_brightness)
        g = max(floor_color[1] * floor_brightness, self._ab_bloom_color[1] * bloom_brightness)
        b = max(floor_color[2] * floor_brightness, self._ab_bloom_color[2] * bloom_brightness)
        w = 200.0 * glint_brightness
        # 255.0 (not an arbitrary scalar) so floor_brightness's 0.03-0.10 maps
        # directly to "3-10% of full brightness" as the spec literally states.
        master = max(255.0 * floor_brightness, 200.0 * bloom_brightness, 220.0 * glint_brightness)

        self.out_r = ema(self.out_r, r, 0.05, 0.03)
        self.out_g = ema(self.out_g, g, 0.05, 0.03)
        self.out_b = ema(self.out_b, b, 0.05, 0.03)
        self.out_w = ema(self.out_w, w, 0.3, 0.15)
        self.out_master = ema(self.out_master, master, 0.05, 0.03)
        self.out_strobe = 0
```

- [ ] **Step 4: Verify the file still parses (method is unreachable but must not break the class)**

Run: `.venv\Scripts\python.exe -c "import ast; ast.parse(open('music_light.py', encoding='utf-8').read()); print('parses OK')"`
Expected: `parses OK`

- [ ] **Step 5: Verify instantiation still works and the new state exists**

Run:
```
.venv\Scripts\python.exe -c "import music_light; e = music_light.DMXEngine(); print('ab_bass=', e._ab_bass, '| bloom_active=', e._ab_bloom_active, '| has_renderer=', hasattr(e, '_render_abyssal_bloom'))"
```
Expected: `ab_bass= 0.0 | bloom_active= False | has_renderer= True`

- [ ] **Step 6: Commit**

```bash
git add music_light.py
git commit -m "feat: add abyssal_bloom renderer (not yet registered)

Deep & sparse ambient renderer: near-black drifting floor, rare bass-reactive
teal blooms (bounded by a hard min-gap + brightness cap), rarer white glints.
Method + __init__ state + module constants only -- not yet reachable via any
behavior map (see next commit)."
```

---

## Task 3: Register in `music_light.py` (all 4 remaining points)

Wires the renderer into `VALID_BEHAVIORS`, `_behavior_map`, `ambient_behaviors`, and `_detect_auto_behavior`'s `ambient_pool`. After this task, `abyssal_bloom` is playable in **both** loopback (auto-selected at very-low energy) and synced (if a cue names it) modes, because both branches of `process_audio` read the same `_behavior_map`.

**Files:**
- Modify: `music_light.py`

- [ ] **Step 1: Add to `VALID_BEHAVIORS`**

Find (near the top of the file, line 63):
```python
VALID_BEHAVIORS = {
    "blackout_punch", "slow_breathe", "bass_white_blast", "color_chase",
    "buildup_ramp", "static_wash", "strobe_blast", "fast_pulse",
    "beat_reactive", "rainbow_sweep", "instant_flash",
    # Ambient/chill behaviors
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer",
}
```
Replace with:
```python
VALID_BEHAVIORS = {
    "blackout_punch", "slow_breathe", "bass_white_blast", "color_chase",
    "buildup_ramp", "static_wash", "strobe_blast", "fast_pulse",
    "beat_reactive", "rainbow_sweep", "instant_flash",
    # Ambient/chill behaviors
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer", "abyssal_bloom",
}
```

- [ ] **Step 2: Add to `_behavior_map`**

Find (in `__init__`, line 224):
```python
        self._behavior_map = {
            "blackout_punch": self._render_blackout_punch,
            "slow_breathe": self._render_slow_breathe,
            "bass_white_blast": self._render_bass_white_blast,
            "color_chase": self._render_color_chase,
            "buildup_ramp": self._render_buildup_ramp,
            "static_wash": self._render_static_wash,
            "strobe_blast": self._render_strobe_blast,
            "fast_pulse": self._render_fast_pulse,
            "beat_reactive": self._render_beat_reactive,
            "rainbow_sweep": self._render_rainbow_sweep,
            "instant_flash": self._render_blackout_punch,
            # Ambient/chill behaviors
            "ocean_drift": self._render_ocean_drift,
            "candlelight": self._render_candlelight,
            "sunset_fade": self._render_sunset_fade,
            "aurora_shimmer": self._render_aurora_shimmer,
        }
```
Replace the last line before the closing brace with an added entry:
```python
            "ocean_drift": self._render_ocean_drift,
            "candlelight": self._render_candlelight,
            "sunset_fade": self._render_sunset_fade,
            "aurora_shimmer": self._render_aurora_shimmer,
            "abyssal_bloom": self._render_abyssal_bloom,
        }
```

- [ ] **Step 3: Add to the loopback `ambient_behaviors` set**

Find (inside `process_audio`, line 1245):
```python
            ambient_behaviors = {"ocean_drift", "candlelight", "sunset_fade",
                                 "aurora_shimmer", "slow_breathe", "static_wash",
                                 "buildup_ramp", "rainbow_sweep"}
```
Replace with:
```python
            ambient_behaviors = {"ocean_drift", "candlelight", "sunset_fade",
                                 "aurora_shimmer", "abyssal_bloom", "slow_breathe",
                                 "static_wash", "buildup_ramp", "rainbow_sweep"}
```

- [ ] **Step 4: Add to `_detect_auto_behavior`'s `ambient_pool`**

Find (inside `_detect_auto_behavior`, line 1021):
```python
                ambient_pool = ["ocean_drift", "candlelight", "aurora_shimmer", "sunset_fade"]
```
Replace with:
```python
                ambient_pool = ["ocean_drift", "candlelight", "aurora_shimmer",
                               "sunset_fade", "abyssal_bloom"]
```

- [ ] **Step 5: Verify with a live loopback run**

Run: `.venv\Scripts\python.exe music_light.py --mode loopback --profile profiles\chill_ambient.json`

Play quiet/ambient audio for at least 60 seconds (the `ambient_pool` rotates every 15s per `_detect_auto_behavior`, so `abyssal_bloom` should come up within one rotation cycle). Watch the log for:
```
[DIAG] ...
```
and confirm no `KeyError`/`AttributeError` is raised. Visually confirm the fixture goes to a dim violet/blue floor with occasional teal blooms during that behavior's 15s slot. Ctrl-C to stop.

Expected: no exceptions; the fixture visibly enters the deep/sparse look for roughly 1-in-5 of the ambient rotation slots.

- [ ] **Step 6: Commit**

```bash
git add music_light.py
git commit -m "feat: register abyssal_bloom in VALID_BEHAVIORS, behavior_map, and loopback ambient pool

Playable in both loopback (auto-selected via the ambient_pool rotation at
very-low energy) and synced mode (if a cue names it), since both dispatch
branches of process_audio share the same _behavior_map. Verified live on
loopback: no exceptions, fixture enters the deep/sparse look during its
15s rotation slot."
```

---

## Task 4: Wire into `llm_designer.py` (prompt + its own `VALID_BEHAVIORS`)

`llm_designer.py` has a **second, separate** `VALID_BEHAVIORS` set (line 351) that gates `_validate_and_repair_plan`. If a behavior isn't in it, the repair step **silently rewrites it to `beat_reactive`** (line 394) — no error, no log a user would notice. Missing this step is the single easiest way to ship a renderer the LLM can pick but that never actually plays.

**Files:**
- Modify: `llm_designer.py`

- [ ] **Step 1: Add the prompt description**

Find the `=== AMBIENT / CHILL BEHAVIORS ===` section (around line 178):
```python
"aurora_shimmer" — Three independent sine waves on R, G, B create a slowly evolving color field that never repeats.
  USE FOR: Ambient electronic, chillwave, post-rock, atmospheric sections.
  FEEL: Ethereal, mesmerizing. Like northern lights.

=== NARRATIVE ARC TEMPLATE ===
```
Insert a new entry directly after the `aurora_shimmer` block and before `=== NARRATIVE ARC TEMPLATE ===`:
```python
"aurora_shimmer" — Three independent sine waves on R, G, B create a slowly evolving color field that never repeats.
  USE FOR: Ambient electronic, chillwave, post-rock, atmospheric sections.
  FEEL: Ethereal, mesmerizing. Like northern lights.

"abyssal_bloom" — Near-black drifting floor with rare, slow teal blooms that swell and dissolve, and rarer white glints. Bass makes blooms more frequent and slightly brighter, but never lets it get busy.
  USE FOR: Deep breakdowns, the quietest intros/outros, meditative or minimal-techno sections.
  FEEL: Vast, restrained, oceanic-depths. Darkness as the canvas.

=== NARRATIVE ARC TEMPLATE ===
```

- [ ] **Step 2: Add to `llm_designer.py`'s own `VALID_BEHAVIORS`**

Find (line 351):
```python
VALID_BEHAVIORS = {
    "slow_breathe", "fast_pulse", "color_chase", "strobe_blast",
    "static_wash", "rainbow_sweep", "buildup_ramp", "instant_flash",
    "beat_reactive", "bass_white_blast", "blackout_punch",
    # Ambient/chill behaviors
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer",
}
```
Replace with:
```python
VALID_BEHAVIORS = {
    "slow_breathe", "fast_pulse", "color_chase", "strobe_blast",
    "static_wash", "rainbow_sweep", "buildup_ramp", "instant_flash",
    "beat_reactive", "bass_white_blast", "blackout_punch",
    # Ambient/chill behaviors
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer", "abyssal_bloom",
}
```

- [ ] **Step 3: Verify the repair function no longer downgrades it**

Run:
```
.venv\Scripts\python.exe -c "
import llm_designer
plan = {'show_name': 'x', 'cues': [{'section_name': 'Intro', 'start_time': 0, 'end_time': 10, 'behavior': 'abyssal_bloom', 'color_1': [0,0,0], 'color_2': [0,0,0]}]}
repaired = llm_designer._validate_and_repair_plan(plan)
print('behavior after repair:', repaired['cues'][0]['behavior'])
"
```
Expected: `behavior after repair: abyssal_bloom` (NOT `beat_reactive` — if you see `beat_reactive`, Step 2 didn't take effect).

- [ ] **Step 4: End-to-end verification (synced mode) — requires a show.json with an abyssal_bloom cue**

Hand-author a minimal show file for a WAV you already have (adjust the path/duration to a real file under `youtube_audio/`):
```
.venv\Scripts\python.exe -c "
import json, wave
wav_path = r'youtube_audio\Paul Kalkbrenner - Sky and Sand (Official Music Video).wav'
with wave.open(wav_path, 'rb') as wf:
    duration = wf.getnframes() / wf.getframerate()
show = {
    'audio_file': wav_path,
    'lighting_plan': {
        'show_name': 'abyssal_bloom smoke test',
        'cues': [{
            'section_name': 'Full', 'start_time': 0.0, 'end_time': duration,
            'color_1': [80, 0, 200], 'color_2': [0, 100, 180],
            'master_dimmer_percent': 50, 'fade_speed_seconds': 3.0,
            'strobe_allowed': False, 'energy_level': 2, 'behavior': 'abyssal_bloom'
        }],
        'phrases': []
    }
}
json.dump(show, open('current_show.json', 'w'), indent=2)
print('wrote current_show.json for', wav_path)
"
```
Then: `.venv\Scripts\python.exe music_light.py --mode synced --show current_show.json`

Expected: playback starts, the fixture shows the deep/sparse floor+bloom+glint look for the whole track, no exceptions in the log. Ctrl-C to stop (or let it finish).

- [ ] **Step 5: Commit**

```bash
git add llm_designer.py
git commit -m "feat: expose abyssal_bloom to the LLM (prompt + its own VALID_BEHAVIORS)

llm_designer.py has a separate VALID_BEHAVIORS from music_light.py's -- without
this entry, _validate_and_repair_plan would silently rewrite any LLM-chosen
abyssal_bloom cue to beat_reactive with no visible error. Verified the repair
function now preserves the behavior, and ran an end-to-end synced playback
smoke test."
```

---

## Task 5 (added post-implementation, from final holistic review): self-healing state reset

A final holistic review — run after Tasks 1-4 were each individually implemented and approved — caught an integration issue no task-scoped review could see: `_render_abyssal_bloom` is the **only stateful renderer** among otherwise-stateless ambient siblings, and nothing resets its `_ab_*` timers when the behavior is deselected and later reselected. Two concrete consequences:
1. **Loopback re-entry**: the `ambient_pool` rotates through 5 behaviors every 15s; while `abyssal_bloom` isn't selected, its `_ab_last_bloom_t`/`_ab_last_glint_t` freeze while the frame clock keeps advancing. On return (~every 75s in a sustained quiet passage), `time_since_last` is always far past `ABYSSAL_BLOOM_INTERVAL`, so it **always** fires an instant bloom — the opposite of "rare."
2. **Synced-mode seeks**: the existing `age = max(0.0, ...)` clamp (added earlier) prevents a brightness-spike on a small backward seek, but a large seek (forward or backward) still leaves `_ab_last_bloom_t`/`_ab_last_glint_t` desynced from the new `t` — causing either an instant forced bloom (forward seek) or up to tens of seconds of dead floor (backward seek).

**User decision: add a self-healing reset** (rather than accept-as-feature or defer). Mechanism: track the renderer's own last-called timestamp (`self._ab_last_render_t`); if the gap since the last call exceeds a threshold — covering re-entry after being deselected, and any seek in either direction — reset the bloom/glint state to "just had one" so the next natural bloom is still a full interval away, rather than firing immediately. This requires **no dispatch-layer changes** (matching the "no engine refactor" non-goal) — it's entirely self-contained within the renderer, detected purely from watching its own `t` sequence.

**Side effect (intentional, not a regression):** this also changes first-ever activation. Previously, `_ab_last_bloom_t = -999.0` in `__init__` made the very first call always immediately bloom-eligible ("no artificial startup delay"). Under the new mechanism, first activation is indistinguishable from "returning after a long gap" (since `_ab_last_render_t` starts unset), so it now goes through the same reset path and waits one full interval before its first bloom. This is arguably *more* consistent with "restraint is the effect" than the original instant-bloom-on-cold-start behavior — call this out in the implementation commit, not a silent behavior change.

### Task 5 implementation

**Files:**
- Modify: `music_light.py`

- [ ] **Step 1: Add the discontinuity threshold constant**

Add to the `ABYSSAL_*` constants block (after `ABYSSAL_GLINT_THRESH = 0.5`):
```python
ABYSSAL_DISCONTINUITY_THRESHOLD = 1.0  # seconds; a real audio-frame-to-frame
# gap while this behavior stays selected is ~0.01s. Anything bigger means
# this renderer was skipped (ambient_pool rotated away and back) or a seek
# happened -- either way, treat it as "just arrived" so blooms/glints reset
# to rare instead of firing instantly.
```

- [ ] **Step 2: Add the new instance state var in `__init__`**

In the `# abyssal_bloom renderer state` block, add one new field:
```python
        self._ab_last_render_t = None      # last t this renderer was actually called with (None = never)
```
(Alongside the existing `_ab_bass`, `_ab_bloom_active`, etc. — order within the block doesn't matter.)

- [ ] **Step 3: Add the discontinuity check as the first lines of `_render_abyssal_bloom`**

Immediately after the docstring, before the `dimmer = ...` line, insert:
```python
        # Self-healing reset: if this renderer wasn't called recently (skipped
        # while another ambient behavior was selected) or `t` jumped (a synced
        # seek in either direction), treat it as a fresh arrival rather than
        # letting stale timers fire an instant bloom/glint or desync for tens
        # of seconds. Covers both the loopback re-entry case and the seek case
        # with one mechanism -- no dispatch-layer changes needed.
        if (self._ab_last_render_t is None or
                abs(t - self._ab_last_render_t) > ABYSSAL_DISCONTINUITY_THRESHOLD):
            self._ab_bloom_active = False
            self._ab_glint_active = False
            self._ab_last_bloom_t = t
            self._ab_last_glint_t = t
        self._ab_last_render_t = t

```
This runs before the existing bloom-arming/glint-arming logic, so a detected discontinuity always takes effect before anything else in the frame.

- [ ] **Step 4: Verify no syntax break and instantiation still works**

Run: `.venv\Scripts\python.exe -c "import ast; ast.parse(open('music_light.py', encoding='utf-8').read()); print('parses OK')"` → expect `parses OK`.
Run: `.venv\Scripts\python.exe -c "import music_light; e = music_light.DMXEngine(); print('last_render_t=', e._ab_last_render_t)"` → expect `last_render_t= None`.

- [ ] **Step 5: Verify the re-entry fix directly**

```
.venv\Scripts\python.exe -c "
import music_light
e = music_light.DMXEngine()
cue = {'dimmer': 50}
# Simulate: renderer active for a bit, then a big gap (deselected for 60s), then called again.
e._render_abyssal_bloom(0.0,0,0,0,False,False,(80,0,200),(0,100,180),0.01,cue, 1.0)
e._ab_last_bloom_t = 1.0  # pretend a bloom just happened right before the gap
before = (e._ab_bloom_active, e.out_master)
e._render_abyssal_bloom(0.0,0,0,0,False,False,(80,0,200),(0,100,180),0.01,cue, 61.0)  # 60s later
print('bloom_active after 60s gap:', e._ab_bloom_active, '(should be False -- reset, not instantly re-armed)')
print('out_master after 60s gap:', round(e.out_master,2), '(should be small/floor-level, NOT a bloom spike)')
"
```
Expected: `bloom_active after 60s gap: False` and a small `out_master` (floor-level, roughly single digits to ~25) — NOT a value consistent with a bloom firing (which would be in the ~100+ range for `bloom_brightness` alone before EMA smoothing).

- [ ] **Step 6: Commit**

```bash
git add music_light.py
git commit -m "feat: self-heal abyssal_bloom timers across re-entry and seeks

Final holistic review found abyssal_bloom is the only stateful ambient
renderer, with no reset hook when deselected/reselected (loopback ambient_pool
rotation) or across a synced-mode seek -- causing a guaranteed instant bloom
on return, or a long dead patch / forced bloom after a seek. Fix: track the
renderer's own last-called t; a gap beyond ABYSSAL_DISCONTINUITY_THRESHOLD
(covers both re-entry and seeks, either direction) resets bloom/glint state
to 'just had one' instead of leaving stale timers. No dispatch-layer changes.

Side effect (intentional): first-ever activation now also waits one interval
before its first bloom, superseding the old -999.0 startup-eligibility trick
-- more consistent with 'restraint is the effect' than an instant cold-start
bloom."
```

---

## Self-review notes (coverage map)

| Spec requirement | Task |
|---|---|
| Floor layer (drift, faint presence) | Task 2 Step 3 |
| Bloom layer (envelope, bass-reactive, bounded) | Task 2 Step 3; math verified in Task 1 |
| Glint layer | Task 2 Step 3 |
| Restraint guard (single event, min gap, floor cap) | Task 1 assertions + Task 2 Step 3 (`bloom_gap`, `not self._ab_bloom_active` guard) |
| `bass_activity` smoothing | Task 2 Step 3 (`self._ab_bass = ema(...)`) |
| Module-level constants, not `profile_*` | Task 2 Step 1 |
| All 6 `music_light.py` registration points | Tasks 2 (state, method) + 3 (`VALID_BEHAVIORS`, `_behavior_map`, `ambient_behaviors`, `ambient_pool`) |
| Both `llm_designer.py` points (prompt + its own `VALID_BEHAVIORS`) | Task 4 |
| Loopback verification | Task 3 Step 5 |
| Synced verification | Task 4 Step 4 |
| Risk: per-instance state must exist before first frame | Task 2 Step 2 (added in `__init__`, before any registration makes the method reachable) |

**Out of scope (per spec Non-goals):** no engine refactor/dedup, no new fixture support, no strobe use in this renderer, no `profile_*` tunables for it.

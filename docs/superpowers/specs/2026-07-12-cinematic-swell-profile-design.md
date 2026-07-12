# Cinematic profile + cinematic_swell renderer — design spec

**Date:** 2026-07-12
**Branch:** `feature/immersive-ambient-profile`
**Status:** Approved by user (brainstorm 2026-07-12)

## Goal

A fifth loopback profile in the UI dropdown — **"Cinematic"** — slow and immersive, but *dynamic* like a film score rather than gentle background wash (that's Chill Ambient's job). Built around a new renderer, **`cinematic_swell`**: a calm ambient floor punctuated by slow, wide color swells triggered only by *strong* kick/bass hits. Available to both loopback (ambient rotation) and the AI-synced director (LLM-selectable behavior).

User decisions from brainstorm:
- Relation to Chill Ambient: **cinematic/dynamic slow mode** — wider color range, dramatic slow shifts, not darker/moodier.
- Beat response: **soft swells on strong hits only** — 1–2 s eased swell, never an instant flash.
- Needs a **new renderer** (no existing renderer matches hit-triggered slow swells).
- Scope: **both modes** (loopback + synced), following the golden_anthem registration pattern.

## Component 1 — `_render_cinematic_swell` (dmx_engine.py)

A shared-base renderer beside `_render_abyssal_bloom` / `_render_golden_anthem`.

### Behavior

- **Floor:** dim warm ambient wash of `color_1` at a never-dark floor (like ANTHEM_FLOOR_MIN precedent — floor NOT dimmer-scaled), with a very slow drift between `color_1` and `color_2` so idle passages still breathe.
- **Trigger:** a swell starts when a kick arrives with `self._beat_velocity >= CINE_TRIGGER_VELOCITY` (strong hits only — weak beats do nothing). While a swell is active, new triggers only *retrigger/extend* if the new hit is stronger than the current swell's remaining peak (no machine-gun stacking).
- **Envelope:** eased (smoothstep) rise over `CINE_RISE_S` (~0.5 s) to a velocity-scaled peak, then eased fall over `CINE_FALL_S` (~1.4 s) back to the floor. Total ~2 s — a film-score hit, never a strobe.
- **Color:** swell blends from the floor color toward the **accent** color from `current_colors()` (palette-aware, like golden_anthem); peak brightness scales with the triggering hit's velocity via `dmx_punch.velocity_brightness` semantics (reuse the module, don't inline 120+135·v — review debt from the last branch).
- **White channel:** small white lift near swell peak only (shimmer precedent from golden_anthem), scaled by dimmer.
- **Strobe:** always 0.
- **Time handling:** accumulated-phase / dt-clamped like golden_anthem (`CINE_DT_CLAMP`, discontinuity threshold ⇒ one nominal frame) so seeks and rotation re-entry are immune by construction. Swell progress advances by clamped per-frame dt, not absolute t.

### Constants (module-level `CINE_*`, NOT `profile_*` keys — per abyssal_bloom precedent)

| Constant | Value | Meaning |
|---|---|---|
| `CINE_TRIGGER_VELOCITY` | 0.55 | min `_beat_velocity` to start a swell |
| `CINE_RISE_S` | 0.5 | eased rise duration |
| `CINE_FALL_S` | 1.4 | eased fall duration |
| `CINE_FLOOR_MIN` | 0.08 | never-dark floor (not dimmer-scaled) |
| `CINE_PEAK_MAX` | 0.90 | hard cap on swell peak brightness fraction |
| `CINE_DRIFT_PERIOD_S` | 20.0 | idle floor color drift period |
| `CINE_WHITE_PEAK` | 100 | max white-channel lift at swell peak |
| `CINE_DT_CLAMP` | 0.1 | max phase advance per frame |
| `CINE_DISCONTINUITY_THRESHOLD` | 1.0 | call-gap ⇒ treat as one nominal frame |

State in `DmxEngineBase.__init__` (beside `_ga_*`): `_cs_swell_phase` (None = idle, else 0..1 through rise+fall), `_cs_swell_peak` (velocity-scaled target), `_cs_drift_phase`, `_cs_last_render_t`.

### Registration in dmx_engine.py

`"cinematic_swell"` in `VALID_BEHAVIORS` and `_behavior_map`.

## Component 2 — `profiles/cinematic.json`

New profile file (auto-discovered by `app.py` `list_profiles()` — dropdown gains a 5th entry with zero backend/frontend code changes).

- `name`: "Cinematic", description: film-score feel — slow immersive washes, wide dramatic palettes, soft swells on big hits.
- Tuning: `gain_boost` 35, `agc_thresh` 0.45, `kick_thresh` 0.10 / `snare_thresh` 0.14 (deliberately higher than Festival's 0.06 so only strong hits register), `onset_cooldown` 0.30 (swells shouldn't retrigger rapidly), `deep_bass_enabled` true / `deep_bass_thresh` 0.80, `decay_speed` 0.92, `glow_thresh` 0.35, `beat_hold_frames` 8, `deep_bass_hold_frames` 8, `color_cycle_mode` "time" / `color_cycle_interval` 12.0 (accepted-but-unused keys kept for schema compatibility), `volume_gate` 0.00005.
- Palettes (wide, cinematic contrast — warmer + wider than Chill Ambient's blue/purple):
  amber/teal `[255,140,20]/[0,120,140]`; deep red/steel blue `[180,20,30]/[40,90,160]`; gold/indigo `[255,180,40]/[60,30,150]`; ember/ice `[200,60,0]/[120,180,220]`.

## Component 3 — Registration (loopback + synced)

- `music_light.py`: append `"cinematic_swell"` to `ambient_pool` and to the `ambient_behaviors` set in `_dispatch` (after `"golden_anthem"`).
- `llm_designer.py`: prompt entry after the golden_anthem block, before `=== NARRATIVE ARC TEMPLATE ===`, same 3-line format (description / USE FOR: cinematic tension-building sections, dramatic intros, half-time breakdowns / FEEL: film-score, widescreen, slow-motion impact); plus `"cinematic_swell"` in llm_designer's own `VALID_BEHAVIORS` (the repair gate).

## Testing (TDD, tests/test_cinematic_swell.py)

1. Weak hit (`_beat_velocity < CINE_TRIGGER_VELOCITY`) does not start a swell — output stays at floor.
2. Strong hit starts a swell; master rises smoothly (no frame-to-frame jump > a bound) and peaks ≤ `CINE_PEAK_MAX`.
3. Swell decays back to floor within rise+fall duration (+EMA slack); floor holds at `CINE_FLOOR_MIN` when dimmer is 0.
4. Seek/re-entry: a 60 s jump in `t` advances swell/drift phase by ≤ one nominal frame (discontinuity guard).
5. Registration: `"cinematic_swell"` in `dmx_engine.VALID_BEHAVIORS`, `_behavior_map`, and llm_designer's `VALID_BEHAVIORS`; `_validate_and_repair_plan` preserves it.
6. Energy differentiation is N/A (hit-triggered, not envelope-followed) — instead: a stronger velocity yields a higher swell peak than a weaker one (both above trigger).

## Verification

- Full suite green (37 existing + new).
- `DMX_DRY_RUN=1` synced pass with a `cinematic_swell` cue: frames rise/fall around beats; deterministic across two runs.
- Hardware pass (user): pick "Cinematic" in the dropdown, play a film-score-like or mid-energy track; confirm calm floor + slow swells on big hits, nothing strobe-like.

## Risks

1. `_beat_velocity` is wall-clock/loopback-scale dependent (the DRY_RUN harness compresses time, so beat-driven velocity in dry-run is not representative — known caveat from the last branch). Trigger-threshold feel must be adjudicated on hardware.
2. In loopback, ambient behaviors are only selected during *very quiet* passages (`_detect_auto_behavior`), where strong kicks are rare by definition — cinematic_swell may mostly show its floor there, and loud passages under the Cinematic profile still route to punchy renderers. **User decision (2026-07-12): keep auto-switching; the profile only tunes.** A `force_behavior` profile-pinning mechanism was proposed and declined. The full swell effect shows best in synced shows; note for hardware pass.

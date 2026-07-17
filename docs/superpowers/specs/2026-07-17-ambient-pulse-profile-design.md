# Ambient Pulse profile + ambient_pulse renderer — design spec

**Date:** 2026-07-17
**Branch:** `feature/tomorrowland-pulse-profile` (based on main after the cinematic/velocity merge, commit 61cb7a3)
**Status:** Approved by user (brainstorm 2026-07-17)

## Goal

A sixth loopback profile — **"Ambient Pulse"** — with the chill-ambient color aesthetic but genuinely beat-locked: the user's hardware feedback on Cinematic was "not responsive, music not synced with the lights." Built around a new **`ambient_pulse`** renderer (layered multi-band response) plus a new **`force_behavior` profile-pinning mechanism** so the profile's feel is guaranteed at every loudness level.

User decisions from brainstorm:
- **Merged the cinematic/velocity branch to main first** — this design depends on the working threshold-excess `_beat_velocity` (`beat_velocity_from_ratio`, dmx_punch.py) and the hold/double-pulse fixes.
- **Pin the renderer** (`force_behavior` key) — reverses the earlier auto-switching-only decision, based on hardware evidence: profiles that only tune thresholds cannot deliver a mode identity.
- **Layered multi-band response** — kicks pulse brightness, snares flip color, hi-hats shimmer white, mids breathe the floor.

## Component 1 — `force_behavior` profile pinning

- `music_light.DMXEngine.__init__`: `self.profile_force_behavior = None`.
- `load_profile`: `self.profile_force_behavior = p.get("force_behavior", None)`. Validate against `dmx_engine.VALID_BEHAVIORS`; unknown value → log a warning and fall back to None (fail loud in the log, never crash the engine).
- `_dispatch`: when `profile_force_behavior` is set, use it as the behavior and skip `_detect_auto_behavior` entirely (energy states/ambient rotation bypassed). The Intent/variety tick still runs (palettes still evolve); mood/energy for the Intent come from the existing loopback mood map using the (still-updated) energy state machine — call `_detect_auto_behavior` is NOT needed for that; keep the existing energy-state update path if it is separate, else derive Intent mood from the pinned behavior's nature ("cool", energy 5). Implementation plan decides the minimal wiring, but the invariant is: **pinned behavior never changes, palette variety still evolves**.
- Existing profiles have no `force_behavior` key → behavior identical to today. No other profile gains the key in this project.

## Component 2 — `_render_ambient_pulse` (dmx_engine.py, shared base)

Layered multi-band renderer, signature identical to siblings
(`kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare, kick_color, accent_color, volume, cue, t`).

### Layers

1. **Mid-driven breathing floor.** A slow EMA of `mid_i` maps to floor brightness between `PULSE_FLOOR_MIN` (0.10, never dark, not dimmer-scaled — sibling precedent) and `PULSE_FLOOR_MAX` (0.35). Floor color drifts between `kick_color` and `accent_color` on an accumulated-dt phase (`PULSE_DRIFT_PERIOD_S` 16.0) with the standard discontinuity guard (gap > 1.0s ⇒ one nominal frame).
2. **Kick pulse.** On `is_kick`: master snaps instantly to `velocity_brightness(self._beat_velocity) * dimmer` (graded — soft kick ≈120, hard kick 255), then decays exponentially (`PULSE_KICK_DECAY` 0.10 per frame ≈ ~500ms perceptual) down to the floor level. Decay starts from the hit's own level — no hold plateau, no step-up (double-pulse lesson).
3. **Snare color flip.** On `is_snare` (and not is_kick): the wash color target snaps toward `accent_color` for `PULSE_SNARE_FLIP_S` (0.3s, accumulated-dt timer), then eases back to the drift color. Color event only — master unaffected.
4. **Hi-hat shimmer.** When `hihat_i > PULSE_HIHAT_THRESH` (0.5): `out_w` spikes to `PULSE_WHITE_SPIKE` (80) * dimmer and decays fast (0.5 decay factor per frame). Sparkle, never a wash.
5. **Strobe always 0.**

### Constants (module-level `PULSE_*`, not profile keys — sibling precedent)

| Constant | Value | Meaning |
|---|---|---|
| `PULSE_FLOOR_MIN` | 0.10 | breathing floor minimum (not dimmer-scaled) |
| `PULSE_FLOOR_MAX` | 0.35 | floor at sustained loud mids |
| `PULSE_DRIFT_PERIOD_S` | 16.0 | floor color drift period |
| `PULSE_KICK_DECAY` | 0.10 | per-frame exponential decay of the kick pulse |
| `PULSE_SNARE_FLIP_S` | 0.3 | how long a snare holds the accent color |
| `PULSE_HIHAT_THRESH` | 0.5 | hihat intensity gate for shimmer |
| `PULSE_WHITE_SPIKE` | 80.0 | white channel spike on shimmer |
| `PULSE_DT_CLAMP` | 0.1 | max accumulated-dt advance per frame |
| `PULSE_DISCONTINUITY_THRESHOLD` | 1.0 | call-gap ⇒ one nominal frame |

State in `DmxEngineBase.__init__` (beside `_cs_*`): `_ap_floor_energy` (mid EMA), `_ap_drift_phase`, `_ap_pulse_level` (current kick-pulse master above floor), `_ap_snare_timer`, `_ap_last_render_t`.

### Registration

`"ambient_pulse"` in dmx_engine `VALID_BEHAVIORS` + `_behavior_map`; music_light `ambient_pool` (other profiles may rotate into it) + `ambient_behaviors` set; llm_designer prompt entry (after cinematic_swell, before the narrative-arc marker; USE FOR: verses, grooves, melodic sections that need beat-locked but gentle response; FEEL: festival wash breathing with the groove) + llm_designer `VALID_BEHAVIORS`.

## Component 3 — `profiles/ambient_pulse.json`

- `name`: "Ambient Pulse"; description: chill-ambient colors that genuinely follow the beat — kicks pulse, snares flip color, hi-hats sparkle.
- **`"force_behavior": "ambient_pulse"`** — the new key.
- Sensitive detection (responsiveness is the point): `gain_boost` 45, `kick_thresh` 0.06, `snare_thresh` 0.09, `onset_cooldown` 0.12, `agc_thresh` 0.35, `volume_gate` 0.00005.
- `deep_bass_enabled` false (no white-blast events in this mode), `decay_speed` 0.85, `glow_thresh` 0.40, `beat_hold_frames` 4, `deep_bass_hold_frames` 4, legacy `color_cycle_*`/`rhythm_change_pct` kept for schema consistency.
- Palettes (Tomorrowland vivid): blue/magenta `[20,60,255]/[255,0,180]`; purple/cyan `[140,0,255]/[0,220,255]`; pink/gold `[255,40,120]/[255,190,40]`; teal/violet `[0,200,180]/[120,40,255]`.

## Testing (TDD, tests/test_ambient_pulse.py)

1. Kick pulse: instant rise on is_kick to the graded level; every following frame ≤ the hit frame (no step-up); decays toward floor within ~1s.
2. Graded: hard kick (ratio 3.0) peaks higher than soft kick (ratio 1.4).
3. Snare flip: is_snare shifts color channels toward accent without raising master; eases back after PULSE_SNARE_FLIP_S.
4. Hi-hat shimmer: out_w spikes only when hihat_i > threshold; decays fast; master unaffected.
5. Floor: breathes with mid_i (loud mids → brighter floor, capped at PULSE_FLOOR_MAX); never below PULSE_FLOOR_MIN at dimmer 0.
6. Discontinuity guard: 60s t-jump advances drift/timers ≤ one nominal frame.
7. `force_behavior`: profile dict with the key → `_dispatch` selects ambient_pulse at every energy state (drive the loopback engine with loud + quiet inputs); unknown value → warning + None fallback; profiles without the key → auto-detection unchanged (pin one existing test as regression).
8. Registration + repair gate: both VALID_BEHAVIORS sets, `_behavior_map`, `_validate_and_repair_plan` preserves the behavior.

## Verification

- Full suite green (59 existing + new).
- Dry-run determinism ×2 with an ambient_pulse cue.
- Hardware pass (user): pick "Ambient Pulse", play an EDM track at normal volume — lights must visibly track kicks (graded), snares flip color, hi-hats sparkle; no dropouts to unrelated ambient renderers at any volume.

## Risks

1. Onset-detection latency (~11ms audio block + WASAPI capture) bounds how "locked" pulses can feel; if hardware still feels late, the fix is upstream (block size/detection), not in this renderer.
2. `force_behavior` bypasses the energy state machine's dispatch, but the machine keeps running for Intent mood — if the wiring drifts, palettes could stop evolving under a pinned profile; the plan pins this with a test (variety tick still fires when pinned).

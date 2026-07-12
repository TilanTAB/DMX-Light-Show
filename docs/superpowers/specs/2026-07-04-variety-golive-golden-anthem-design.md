# Variety Go-Live + `golden_anthem` — Design

**Date:** 2026-07-04
**Status:** Approved (design walkthrough) — pending spec review
**Branch:** `feature/variety-wiring-golden-anthem` (off merged `main` @ `cba5e00`)

## Problem

Two things stand between the current code and the "Tomorrowland-like" goal the user has asked for twice:

1. **The variety machinery is built but dead.** `main` contains `dmx_variety.py` (VarietyEngine: curated palettes, anti-repeat, per-song seed, phrase-grid texture evolution — 11 passing unit tests) and `dmx_punch.py`, but the engine only *instantiates* VarietyEngine (`dmx_engine.py:195`) and never calls it. Colors still come from the old static plumbing (`music_light.py:310`, `ai_show_player.py:30/33`). The approved wiring plan (2026-06-13) went stale: written pre-merge, its later tasks were never executed.
2. **The user wants another Tomorrowland-flavored calm mode.** From four candidates, they chose **`golden_anthem`** — majestic gold swells cresting into white-gold shimmer ("hands in the air during the anthem"), with **ride-the-music** reactivity.

Decision (user): **wiring first, then the new mode** — new modes only feel "Tomorrowland" on top of the live variety layer, and `golden_anthem` then inherits palettes for free.

## Goals

- **Phase 1:** make VarietyEngine actually drive both modes (palette variety, per-song identity, loopback time-evolution) and unify punch (synced stops being mushy where it still is).
- **Hard checkpoint:** user verifies Phase 1 on hardware before Phase 2 begins.
- **Phase 2:** add `golden_anthem` on the live variety layer, with real pytest coverage (the suite exists on main now).

## Non-goals

- No new fixtures (single 8ch RGBW par), no IPC/app.py changes, no new LLM providers.
- No re-design of VarietyEngine/dmx_punch internals — they're built, tested, and reviewed; this wires them.
- No changes to the ambient/atmospheric renderers' *textures* (ocean_drift etc. keep their motion; only their color inputs change source).

## Verified current-state anchors (grepped on this branch, not assumed)

| Fact | Anchor |
|---|---|
| VarietyEngine instantiated, never called | `dmx_engine.py:195` (sole `self.variety` hit) |
| `mood` already carried into internal cues | `dmx_engine.py:868` |
| `self.show_bpm` declared but never set from a show | `dmx_engine.py:177`; no writes elsewhere |
| `set_song_seed` never called | no hits outside `dmx_variety.py` |
| `self.beats_per_sec` / `self._beat_velocity` set by base `process_audio` | `dmx_engine.py:170-171` + renderer uses at `:378,:422` |
| Old static color sourcing | `music_light.py:310`, `ai_show_player.py:30,33` |
| abyssal_bloom fully ported (incl. self-heal) | `dmx_engine.py:709+`, registration across all 3 files |

## Phase 1 — VarietyEngine go-live

Seven work items, all offline-verifiable via a dry-run harness before hardware:

1. **Dry-run harness.** `DMX_DRY_RUN=1` env var: `_init_hardware` skips USB (logs instead), `send_dmx` records `self.last_frame` instead of transferring; `run_synced_mode` gains a no-audio fast-iterate branch that logs cue transitions + frames. This is the offline proof channel for items 2–6.
2. **Per-song identity.** `load_ai_show` sets `self.show_bpm` from the plan's `song_metrics.bpm` (default 0.0 → VarietyEngine falls back to time-based phrasing) and calls `self.variety.set_song_seed(zlib.crc32((show_name or audio_file).encode("utf-8")))`. **Not Python's builtin `hash()`** — string hashing is salted per process (PYTHONHASHSEED), and every playback spawns a fresh worker process, so `hash()` would silently give a different seed each run and defeat replay determinism. `zlib.crc32` is stdlib, deterministic, and cheap. Same song → same palette sequence on every replay; different songs diverge.
3. **Synced director** (`ai_show_player._dispatch`): build `Intent(energy=cue["energy"], mood=cue.get("mood"), section_id=cue["name"], is_new_section=on cue change, bpm=self.show_bpm, strobe_allowed=cue["strobe"], seed_color=cue["color_1"])`; call `variety.tick(is_beat, bpm, t)` every frame and `begin_section(intent)` on cue change; source renderer colors from `variety.current_colors()` (cue still drives *behavior/energy/dimmer/strobe*; the LLM's `color_1` seeds the palette family instead of painting verbatim — decided in the original variety design).
4. **Loopback director** (`music_light._dispatch`): `Intent` from the energy state machine (mood map: calm→warm, building→cool, high→neon, dropping→euphoric; energy map: calm→2, building→5, high→8, dropping→6); `tick()` every frame; `begin_section` on state transition **or** when `tick()['seconds_in_section'] >= 16.0` (forced evolution at stable energy — fixes "loopback feels static"); colors from `current_colors()`; **retire `color_phase` cycling** (the R→B→G→W rotation and its profile plumbing become dead on this path — remove the dispatch usage, leave `load_profile` keys accepted-but-unused for profile-file compatibility).
5. **`_render_loopback_direct` rewired to palette colors.** Signature drops `color_idx`, takes `(color_1, color_2, accent)`: normal-beat snap uses `color_1` (kick) / `color_2` (snare) with white punch on kick; deep-bass combo blasts `accent`; between-beats glow uses `color_1`; breathing white floor unchanged. Kills the hardcoded R/B/G/W and combo tables ("colors repeat" — fixed at the punchy path).
6. **Punch unification + carried bug fix.** `_render_beat_reactive` and `_render_bass_white_blast` adopt `dmx_punch.velocity_brightness` + beat-hold + `afterglow` on non-beat frames (ambient renderers explicitly excluded — they are intentionally smooth). **Includes the deferred `bass_white_blast` fix:** its trailing unconditional `out_master` EMA currently overwrites `velocity_master` on the same frame — restructure so beat frames keep the velocity master and only non-beat frames EMA toward the wash level.
7. **abyssal_bloom: no changes.** Its bloom tint already blends from `accent_color`, so it inherits variety palettes automatically once the director passes palette colors. (Verify in dry-run logs, change nothing.)

### Checkpoint (hard gate — user on hardware)
- Loopback: palettes evolve (no fixed R/B/G/W), a stable-energy passage rotates look after ~16s, punch feel unchanged from today.
- Synced: show is palette-tinted per section, same song → same look across two replays, `beat_reactive`/`bass_white_blast` sections feel crisp.
- Regression knob if too busy/wrong: revert item 6's renderer adoption independently of items 3–5 (they're separate commits).

## Phase 2 — `golden_anthem`

A sixth calm-family renderer: majestic gold swells that crest into white-gold shimmer; brightest member of the calm family.

### Envelope — accumulated phase (seek-proof by construction)
State: `self._ga_phase` (0..1 cycle position), `self._ga_energy` (smoothed music energy), `self._ga_last_render_t`.
Per frame: `dt = t - _ga_last_render_t`; if `_ga_last_render_t is None` or `abs(dt) > ANTHEM_DISCONTINUITY_THRESHOLD (1.0s)` → treat `dt` as one nominal frame (self-heal, same lesson as abyssal_bloom, but here *by construction*: phase only ever advances by clamped `dt`, so no absolute-`t` subtraction can explode). Then `dt = min(max(dt, 0.0), 0.1)`; `_ga_phase = (_ga_phase + dt / period) % 1.0`.
Envelope value = smoothstep-shaped swell over the cycle (rise ~40% of period, crest hold ~10%, fall ~50%).

### Ride-the-music (bounded)
`self._ga_energy = ema(_ga_energy, min(1.0, volume * 2000 * 0.5 + mid_i * 0.5), 0.1, 0.03)` — slow, smoothed; no per-beat response.
- `period = ANTHEM_BASE_PERIOD (12.0s) - _ga_energy * (ANTHEM_BASE_PERIOD - ANTHEM_MIN_PERIOD (8.0s))` — louder passages swell somewhat faster, floor at 8s.
- `crest = min(ANTHEM_CREST_MAX (0.85), ANTHEM_CREST_BASE (0.55) + _ga_energy * ANTHEM_CREST_GAIN (0.30))` — louder passages crest brighter, hard cap.

### Color & channels
- `gold = lerp_color(current color_1 from variety, (255, 190, 80), 0.6)` — gold identity, variety-tinted (abyssal's teal-bias trick).
- Brightness = `max(ANTHEM_FLOOR_MIN (0.10), envelope * crest * dimmer)` — dimmer scales the swell, a fixed amber floor keeps it never-dark (PWM-visibility lesson: the floor term is *not* dimmer-scaled).
- White channel: shimmer only near the crest — `out_w` target proportional to `max(0, envelope - 0.8) / 0.2`.
- `out_strobe = 0` always. Slow EMAs (≈0.04/0.03) matching the calm family.

### Registration (9 points, same playbook as abyssal_bloom, current structure)
`dmx_engine.py`: ANTHEM_* constants · `_ga_*` state in `__init__` · the method · `_behavior_map["golden_anthem"]` · `VALID_BEHAVIORS`. `music_light.py`: `ambient_pool` · `ambient_behaviors`. `llm_designer.py`: prompt entry (USE FOR: finales, euphoric breakdowns, anthem/sing-along moments, sunset sets; FEEL: majestic, golden, hands-in-the-air) · its own `VALID_BEHAVIORS` (the silent-downgrade gate).

## Verification

- **Phase 1:** dry-run logs prove — palette changes on cue/section boundaries, anti-repeat across sections, identical palette sequence across two runs of the same show (seeding), loopback forced-evolution timing, `bass_white_blast` master values tracking velocity on beat frames. Existing 22 pytest tests stay green throughout. Then the hardware checkpoint above.
- **Phase 2:** new `tests/test_golden_anthem.py` — envelope bounded [0,1], crest cap respected at `_ga_energy=1`, period floor respected, discontinuity dt-clamp (60s gap advances phase by ≤ one nominal frame), floor minimum honored at `dimmer=0`. Plus dry-run frame logs and a final user hardware pass.
- Full suite green at every commit; conventional commits; each numbered item lands as its own commit for independent revert.

## Risks

1. **Punch regression in Phase 1 item 6** — the same tension flagged in the original variety design. Mitigation: separate commit, A/B by ear at the checkpoint, single-item revert path.
2. **Second stateful renderer.** `golden_anthem` carries swell phase across frames. Mitigated from day one by accumulated-phase design (no absolute-`t` math) + the discontinuity clamp — the exact class of bug that needed a post-hoc fix in abyssal_bloom is excluded structurally.
3. **Plan/tree drift (recurring theme).** Both prior projects hit stale-anchor surprises. Mitigation: every plan task re-verifies its anchors by grep before editing; the spec's anchor table above is the source of truth at design time only.

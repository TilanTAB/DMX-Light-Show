# `abyssal_bloom` — Deep & Sparse Ambient Renderer — Design

**Date:** 2026-06-14
**Status:** Draft for review
**Branch:** `feature/ambient-mode-research` (off `main`)

## Problem

The ambient/chill renderer family has four members — `ocean_drift` (cool watery), `candlelight` (warm flicker), `sunset_fade` (cinematic crossfade), `aurora_shimmer` (cosmic field). All four are **continuous washes**: the fixture is lit and gently moving most of the time. There is **no "deep & sparse" member** — one where darkness is the canvas and light is a rare event. That's the gap this fills.

## Goals

- Add one new atmospheric renderer, **`abyssal_bloom`**, in the calm family — *mostly darkness, punctuated by rare swells of light*.
- Make it feel **alive but restrained**: bass energy influences how often/bright the blooms are, without turning it into a beat-reactive effect.
- Usable in **both modes**: the loopback energy state machine can auto-select it at very-low energy, and the LLM can assign it to intros/breakdowns/outros in AI-designed shows.

## Non-goals

- No new fixture support — single 8-channel RGBW par (master/R/G/B/W/strobe), like every other renderer.
- No engine refactor. This branch is off `main` at the **pre-split monolith** — `music_light.py` alone contains both loopback and synced playback (dispatched via `is_loopback = elapsed_seconds is None` inside one shared `process_audio`, launched via a `--mode` flag). There is only one file, one `_behavior_map`, one `VALID_BEHAVIORS`. (A *later* branch/PR splits this into two files and then a shared base — out of scope here; corrected from an earlier draft of this spec that wrongly assumed the split already existed on `main`.)
- No strobe — this is a calm renderer; `out_strobe` stays 0.

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| What "mode" means | A **new renderer** (not a profile or a new family) |
| Vibe | **Deep & sparse** — darkness as canvas |
| Concept | A **mix** of three candidates: slow drifting floor + rare blooms + rarer glints |
| Integration | **Both modes** (loopback `ambient_pool` + LLM behavior set) |
| Audio reactivity | **More reactive** — bass scales bloom frequency *and* brightness (bounded; see §Tension) |
| Floor | **Faint presence** — never fully black (~3–10% violet/blue glow) |
| Name | `abyssal_bloom` (working name — open to change) |

## The renderer

A per-frame function matching the existing ambient signature:

```python
def _render_abyssal_bloom(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                          kick_color, accent_color, volume, cue, t):
```

It composes three additive layers, then writes `self.out_r/g/b/w/master` (strobe = 0).

### Layer 1 — Floor (always present)
Two slow non-harmonic sines drift the hue between a deep blue and a deep violet (reuse the `lerp_color` + dual-sine pattern from `ocean_drift`/`aurora_shimmer`). Brightness sits at `FLOOR_MIN..FLOOR_MAX` (≈ 3–10%) scaled by `cue.dimmer`. Slow EMA (attack/decay ≈ 0.02–0.03) so it never jerks. This is ~70%+ of the runtime.

### Layer 2 — Bloom (rare swell, the signature event)
An explicit envelope state machine on instance state (`self._ab_bloom_t0`, `self._ab_bloom_active`, `self._ab_last_bloom_t`, `self._ab_bloom_color`):
- **Arm condition:** idle, AND `(t - last_bloom_t) >= bloom_gap`, AND (the bloom timer elapsed **or** a bass nudge — `kick_i > BLOOM_NUDGE_THRESH`).
- **Envelope** from `age = t - bloom_t0`: smoothstep **rise** (~1.5 s) → brief **hold** → smoothstep **fade** (~3–4 s); when `age > rise+hold+fall`, go idle and stamp `last_bloom_t`.
- **Color:** teal/cyan biased — blend the cue's `accent_color` toward `(0, 210, 210)` so AI palettes still tint it.
- **Peak brightness:** `BLOOM_BASE + bass_activity * BLOOM_BASS_GAIN`, capped at `BLOOM_MAX` (≈ 70%).

The bloom value is `max()`-combined with the floor (a bloom lifts the fixture above the floor; it never darkens it).

### Layer 3 — Glint (rarest, punctuation)
A short white flare — fast rise (~0.2 s) and fade (~0.6 s) on `self.out_w` + a small master lift. Armed far less often (`glint_gap` ≈ 15–25 s) or by a hi-hat transient (`hihat_i > GLINT_THRESH`). Also `max()`-combined.

### Restraint guard
Only one bloom **or** glint active at a time; a hard `bloom_gap` floor (≈ 3 s minimum) regardless of bass; the floor never exceeds `FLOOR_MAX`. These three rules are what keep it "sparse" even under the reactive bass coupling.

### `bass_activity`
A smoothed bass measure, not a raw transient: `self._ab_bass = ema(self._ab_bass, min(1.0, kick_i), 0.2, 0.05)`. Smoothing means a single kick doesn't whipsaw the bloom rate; a sustained bass passage ramps it. (`kick_i` is still used raw for the instantaneous bloom *nudge*.)

### Parameters (module-level constants — not `profile_*`, to avoid adding new `load_profile` keys)
`FLOOR_MIN=0.03`, `FLOOR_MAX=0.10`, `BLOOM_BASE=0.45`, `BLOOM_MAX=0.70`, `BLOOM_BASS_GAIN=0.25`, `BLOOM_RISE=1.5`, `BLOOM_HOLD=0.4`, `BLOOM_FALL=3.5`, `BLOOM_GAP_MIN=3.0`, `BLOOM_INTERVAL=11.0`, `BLOOM_NUDGE_THRESH=0.25`, `GLINT_INTERVAL=18.0`, `GLINT_THRESH=0.5`. (Tunable during implementation against the hardware.)

## The "more reactive" vs "deep & sparse" tension

The user chose *more reactive* over the subtler default. Bass couples in two bounded ways: it **shortens the inter-bloom interval** (`interval * (1 - bass_activity*0.6)`, floored at `BLOOM_GAP_MIN`) and **raises peak bloom brightness** (capped at `BLOOM_MAX`). The bounds (min-gap + brightness cap + single-event guard) are deliberately what stop a loud passage from turning this into `beat_reactive`. If, on hardware, it still feels too busy, the single knob to turn is `BLOOM_GAP_MIN` up / `BLOOM_BASS_GAIN` down. This is the main thing to validate by eye.

## Integration (single file, single class — pre-split `music_light.py`)

Six registration points, **all in `music_light.py`** (confirmed: `VALID_BEHAVIORS` at line 63, `_behavior_map` in `__init__` at line 224, `ambient_behaviors` in the loopback branch of `process_audio` at line 1245, `ambient_pool` inside `_detect_auto_behavior` at line 1021):
1. `VALID_BEHAVIORS` — add `"abyssal_bloom"`.
2. New `self._ab_*` state vars in `__init__` (alongside the other loopback-direct state, e.g. near `self.peak_kick`).
3. The `_render_abyssal_bloom` method itself.
4. `_behavior_map["abyssal_bloom"] = self._render_abyssal_bloom` — this single map is read by **both** the loopback and synced branches of `process_audio` (confirmed at lines 1252 and 1279), so this one entry makes it playable in both modes without further branching.
5. The loopback `ambient_behaviors` set (line 1245) — so loopback routes it to the standard renderer dispatch, not `_render_loopback_direct`.
6. The loopback `_detect_auto_behavior` `ambient_pool` (line 1021) — so it's auto-selected at very-low energy.

Plus one change in `llm_designer.py`: add an `abyssal_bloom` entry to the `=== AMBIENT / CHILL BEHAVIORS ===` section of the prompt, so the LLM can assign it in synced shows.

## Data flow per frame
1. `process_audio` computes `kick_i…volume`, then branches on `is_loopback`: the loopback branch calls `_detect_auto_behavior` (which can return `"abyssal_bloom"` from the `ambient_pool` rotation) and, since it's in `ambient_behaviors`, dispatches via `_behavior_map` with a synthetic cue; the synced branch looks up the active cue and dispatches via the same `_behavior_map` if the cue's `behavior == "abyssal_bloom"`.
2. The renderer advances its floor sines, the bloom/glint state machines (reading `kick_i`/`hihat_i` for reactivity), composes the layers, and sets `self.out_*`.
3. `process_audio` sends the frame via `send_dmx` (one call, after the `if/else` branch).

## Error handling / edge cases
- Defensive cue access (`cue.get("dimmer", 50) if cue else 50`) — matches the other renderers; loopback passes a synthetic cue.
- `t` is monotonic per playback (loopback wall-ish, synced elapsed). On a synced **seek**, `t` jumps; the bloom state machine self-heals within one envelope (≤ ~5 s) — acceptable, no special handling.
- All math is bounded (`min`/`max` clamps on every channel) so no out-of-range DMX values.

## Testing / verification
`main` has **no test harness**, and the renderer needs the uDMX + audio to see. Verify by:
- **Loopback:** `python music_light.py --mode loopback --profile profiles/chill_ambient.json`, play quiet/ambient audio, confirm the energy machine drops to `calm` and rotates `abyssal_bloom` into the `ambient_pool`; eyeball the dark-floor-with-rare-blooms feel and that loud bass makes blooms more frequent **but still gapped**.
- **Synced:** hand-author or generate a `show.json` with an `abyssal_bloom` cue on an intro/breakdown, then `python music_light.py --mode synced --show shows/<id>/show.json`; confirm it renders.
- (Optional, low-cost) a tiny offline driver that calls `_render_abyssal_bloom` over synthetic `t`/`kick_i` and prints `out_*` to confirm the floor stays dim and blooms are gapped — no hardware needed.

## Risks
1. **Reactivity vs. calm.** The chosen "more reactive" coupling is the main way this could miss the brief — a busy room instead of a deep one. Mitigated by the bounds; validated by eye; one-knob fix.
2. **Six registration points, one file.** Easy to add the renderer but forget one of `VALID_BEHAVIORS` / `_behavior_map` / `ambient_behaviors` / `ambient_pool` / `__init__` state — each omission fails differently (KeyError, AttributeError, or silent non-selection). The implementation plan must check all six explicitly.
3. **Shared per-instance state.** The bloom/glint envelopes add `self._ab_*` fields the other ambient renderers don't have; must be initialised in `__init__` or the first frame throws `AttributeError`.

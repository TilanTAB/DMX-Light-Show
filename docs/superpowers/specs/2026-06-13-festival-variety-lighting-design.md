# Festival-Grade Variety & Evolution Lighting — Design

**Date:** 2026-06-13
**Status:** Draft for review
**Author:** brainstormed with Claude Code

## Problem

The current DMX show feels like a *reactive blinker*, not a *designed festival show*. Concretely, the lighting is monotonous along four axes (all four confirmed as in-scope by the user):

1. **Colors repeat.** Loopback hardcodes an R→B→G→W cycle (`color_phase` 0–3 in `music_light.py`); AI-synced shows keep landing on similar palettes because the LLM picks per-section colors with no memory.
2. **Texture is flat within a section.** A cue assigns one behavior to a 30 s chorus and the fixture does the identical move for 30 s. Nothing develops *inside* a section.
3. **Every song looks alike.** Different songs produce near-identical behavior sequences and color stories — there is no per-song identity.
4. **Loopback goes static.** The energy state machine settles into one behavior and holds it far too long when the music doesn't change much.

A secondary, already-documented defect compounds this: **synced mode lacks punch.** Loopback has a dedicated `_render_loopback_direct()` (velocity sensitivity, beat-hold, warm afterglow, breathing floor) that synced mode has no equivalent for, so synced shows feel smooth/mushy. Any variety work must *preserve and extend* punch, not dilute it.

## Goals

- Make both modes feel professionally *designed*: distinct, evolving looks instead of repetition — the "Variety & evolution" quality.
- Fix all four monotony axes with **one mechanism**, shared by both modes, so they cannot drift apart.
- Bring loopback's punch primitives to synced mode in the same change.
- Lay the foundation by **extracting the duplicated engine logic into a shared module** (user-approved).

## Non-goals

- **No multi-fixture support.** Target the single existing 8-channel RGBW par (master/R/G/B/W/strobe on uDMX). All gains come from timing, dynamics, contrast, and design language — not more hardware. The cue/render model deliberately emits *one* fixture's frame; no fixture-addressing abstraction (YAGNI).
- No change to the FastAPI/IPC/subprocess architecture. `app.py` still spawns the same two script names; the JSON-file IPC and the stderr→stdout watchdog are untouched.
- No new LLM provider work; Azure/Bedrock selection stays as-is.

## Key decisions

- **Hybrid (Approach 3):** two mode-specific *directors* normalize their input into a common `Intent`; one shared `VarietyEngine` owns all anti-monotony policy; renderers consume the engine's output. The LLM contributes design taste (mood/contrast) but never owns timing-critical variety.
- **Flat modules** at the repo root (`dmx_engine.py`, `dmx_variety.py`) to match the project's existing flat layout — not a package.
- **Directors stay in their existing files** (`music_light.py`, `ai_show_player.py`); everything else moves into the shared module.
- **Variety and punch are orthogonal:** variety decides *what color and when-in-the-phrase*; punch decides *how hard each beat lands*. This separation is the core invariant that prevents reintroducing mush.

---

## Architecture

```
INPUTS (mode-specific, stay in their files)
  ├─ Synced director   ai_show_player.py · _dispatch()   ── reads active LLM cue
  └─ Loopback director music_light.py    · _dispatch()   ── reads energy state machine
            │ both emit the SAME
            ▼
        Intent { energy, mood, section_id, is_new_section, bpm, strobe_allowed }
            │
            ▼
┌─ SHARED MODULE (extracted once — removes the 24-method duplication) ────────┐
│  VarietyEngine (dmx_variety.py)                                              │
│    • palette pick: library + anti-repeat memory + per-song seed             │
│    • texture phase: evolves on phrase boundaries (intra-section)            │
│    • time-evolution: keeps loopback moving at constant energy               │
│            │ colors + texture phase + punch params                          │
│            ▼                                                                 │
│  Renderers + shared punch (dmx_engine.py · DmxEngineBase)                    │
│  Audio + beat core: FFT/bands/AGC/flux/beat → BeatFrame (dmx_engine.py)      │
│            ▼                                                                 │
│  send_dmx() → USB worker thread (dmx_engine.py)                             │
└─────────────────────────────────────────────────────────────────────────────┘
            ▼
        Single RGBW par
```

### Module layout after extraction

- **`dmx_engine.py` — `DmxEngineBase`** (new). Holds everything currently duplicated: `_init_hardware`, `send_dmx`, `_dmx_worker`, `shutdown` (USB dispose stays in its `finally`), the IPC helpers (`_write_playback_state`, `_check_playback_command`, `_cleanup_ipc_files`), `load_ai_show`, all 14 `_render_*` methods, and `process_audio()`. `process_audio()` does the identical FFT/band/AGC/flux/beat-detection work, then calls an **abstract `_dispatch(beat_frame)`** that each subclass implements.
- **`dmx_variety.py` — `VarietyEngine` + `PALETTES`** (new). All anti-monotony policy and the curated palette library (data).
- **`music_light.py` — `class MusicLight(DmxEngineBase)`** (slimmed). Keeps `run_loopback_mode` (incl. WASAPI `p.terminate()` teardown), loopback `_dispatch()` (the energy state machine → `Intent`), profile loading, and color-cycling removal (superseded by VarietyEngine).
- **`ai_show_player.py` — `class AiShowPlayer(DmxEngineBase)`** (slimmed). Keeps `run_synced_mode`, synced `_dispatch()` (cue lookup → `Intent`), and transport-command handling.

---

## Component 1 — Shared-module extraction

A pure refactor with **no behavior change**, done first so later steps build on a single code path. The current divergence inside `process_audio()` (the `elapsed_seconds is None` branch) is replaced by the `_dispatch()` override seam. Subclasses provide the two things that genuinely differ: how a frame's lighting *intent* is derived (cue vs. state machine) and the run loop / teardown.

Load-bearing invariants preserved verbatim:
- `shutdown()` releases the libusb handle (`usb.util.dispose_resources`) in a `finally` on every path → lives in `dmx_engine.py`.
- `p.terminate()` on loopback teardown → stays in `music_light.py`'s run loop.
- `app.py` subprocess stdout merge + activity watchdog → untouched (script names unchanged).

---

## Component 2 — VarietyEngine (`dmx_variety.py`)

The single home for all four monotony fixes.

### Palette library
~12–16 curated *palette families*, each: `id`, `mood` (`warm`/`cool`/`neon`/`euphoric`/`dark`/…), `energy` suitability range, and `primary`/`secondary`/`accent` RGB triples. Example:

```python
{"id": "volcanic", "mood": "warm", "energy": [6, 10],
 "primary": [255, 60, 10], "secondary": [255, 150, 0], "accent": [255, 255, 255]}
```

These replace both the hardcoded R→B→G→W loopback cycle and serve as the curated set the synced director draws from.

### State
- `recent_palettes` — `deque(maxlen=4)` of recently used family ids → **anti-repeat** (fixes #1).
- `song_seed` + `rng` — a seeded RNG so a given song deterministically picks its look (fixes #3).
- `section_start_beat`, `phrase_index` — intra-section texture tracking (fixes #2).
- section clock (via `tick`'s `seconds_in_section`) — wall-clock drift driver the loopback director reads to force palette rotation (fixes #4).

### Methods
- `begin_section(intent)` — on a section/cue boundary: filter the library by `intent.energy` and exclude `recent_palettes`; then **if `intent.seed_color` is set, pick the palette whose primary is nearest that color** (LLM-color-seeds-palette), otherwise filter by `intent.mood` and pick via the seeded `rng`. Push to `recent_palettes`, reset `phrase_index`. Returns the chosen palette. Filters relax step-by-step so it never deadlocks.
- `on_phrase_boundary()` — advances `phrase_index`; the texture move is derived deterministically from `phrase_index` (swap accent in, flip color_1/color_2 roles, hue-shift the secondary) so a section *develops*. Beat-quantization is what makes evolution read as *intentional* rather than random.
- `current_colors()` — returns `(color_1, color_2, accent)` for this frame, modulated by `phrase_index`.
- `tick(is_beat, bpm, t)` — per frame: track beats, detect phrase boundaries (every 8 beats; time fallback when `bpm` is 0), and return `{phrase_boundary, seconds_in_section}`. The **loopback director** reads `seconds_in_section` and forces `begin_section` after ~16 s at constant energy (fixes #4); the engine reports, the director decides (explicit, not magic).

### Per-song identity (fix #3)
- **Synced:** `song_seed = hash(video_id or title + bpm_bucket)`. Same song → same look every time (also makes shows reproducible/debuggable); different songs diverge.
- **Loopback:** no song id exists, so identity is time-evolution (#4) rather than per-song. `song_seed` is per-session; the analog of "new song" is the evolution clock + energy transitions.

---

## Component 3 — The Intent contract

A small normalized object both directors produce and the engine consumes identically. This is the *only* coupling between modes and the variety layer; keeping it explicit is what prevents a second, divergent variety code path.

```python
Intent(
    energy,         # 1-10
    mood,           # "warm"|"cool"|"neon"|"euphoric"|"dark"|...  (optional; inferred if absent)
    section_id,     # changes on boundary
    is_new_section, # True triggers begin_section()
    bpm,            # for phrase detection
    strobe_allowed, # bool
    seed_color,     # optional [R,G,B] color hint (LLM color_1); None in loopback
)
```

- **Synced director** builds `Intent` from the active LLM cue: `energy_level`, optional `mood` (see Component 5), section boundary from cue change, `bpm` from show metadata, `strobe_allowed`, and `seed_color = cue["color_1"]` so the LLM's color choice seeds the palette family.
- **Loopback director** builds `Intent` from the energy state machine: `energy_state` → energy number, `mood` inferred deterministically from the energy band (e.g. low→`warm`, mid→`cool`, high→`neon`, peak→`euphoric`), "section" = energy-state transition, `bpm` from detected BPS.

---

## Component 4 — Renderer changes (parameter-driven + shared punch)

Renderers stop hardcoding colors and instead read `VarietyEngine.current_colors()`. The punch logic currently exclusive to `_render_loopback_direct()` — velocity → brightness (120–255), 4–5 frame beat-hold, warm afterglow decay (R 95% / G 88% / B 82%), breathing white floor — is lifted into shared pure functions (`velocity_brightness`, `afterglow` in `dmx_punch.py`) that **both** modes use.

Critically, `_render_loopback_direct()` itself currently ignores the colors passed to it and sets channels from a `color_idx` (the hardcoded R→B→G→W cycle). It is **rewired to consume the palette's `(color_1, color_2, accent)`** — without this, palette variety never reaches loopback's punchy path and pain point #1 survives. The standalone `color_phase` cycling is then dead and removed.

Result: synced mode finally gets punch, both modes get variety colors (including the loopback punchy path), and because variety touches only color/texture while punch governs beat response, the two never fight. The known "synced is mushy" complaint is fixed by the same change that adds variety.

---

## Component 5 — LLM schema / prompt enrichment (`llm_designer.py`)

Additive, backward-compatible. Add an optional **`mood`** field per cue to the `REQUIRED JSON FORMAT`, and a prompt instruction to assign a mood per section that *contrasts with neighbors*. The synced director maps cue `mood` → `Intent.mood`; if the LLM omits it, `_validate_and_repair_plan` defaults it from `energy_level` so nothing breaks.

**Color ownership (decided): the LLM's `color_1` *seeds* the palette rather than being used directly.** The synced director passes `seed_color = cue["color_1"]` and `begin_section` maps it to the nearest curated palette family, after which texture evolution + anti-repeat take over. This honors the LLM's design intent while still guaranteeing variety and anti-repetition — the original cue colors are no longer painted verbatim. The prompt's heavy "COLOR RULES" section can therefore be trimmed to a brief "pick an evocative `color_1` per section; it seeds the palette" hint. The LLM contributes taste (color intent + mood + contrast); the VarietyEngine owns final color, anti-repeat, and intra-section timing.

---

## Data flow (per audio frame)

1. `process_audio()` decodes audio, runs FFT/bands/AGC/flux, detects beats → `BeatFrame`.
2. Subclass `_dispatch(beat_frame)` builds `Intent`. If `is_new_section`, calls `VarietyEngine.begin_section(intent)`.
3. `VarietyEngine.tick(beat_frame)` advances phrase/evolution state; on a phrase boundary it applies a texture move.
4. The chosen renderer reads `current_colors()` and applies the shared punch functions (`velocity_brightness`, `afterglow`) to compute the 8-byte frame.
5. `send_dmx()` queues the frame for the USB worker thread.

---

## Error handling & failure modes

- **All palettes recently used / library too small** → relax anti-repeat (allow the oldest entry) rather than deadlock.
- **LLM omits `mood`** → director infers from energy (safe default); `_validate_and_repair_plan` fills it.
- **Wrong/zero BPM → bad phrase detection** → fall back to a time interval (~4 s) for texture moves instead of the beat grid.
- **Refactor regression** → each migration step is an independent git checkpoint; rollback = revert the step.
- **Hardware invariants** → USB dispose-in-`finally` and WASAPI `p.terminate()` retained exactly; verified by run after each step.

## Testing & verification

No automated tests or runner exist, and the renderers normally require uDMX hardware. To verify variety logic without lights, add a **`--dry-run`** mode to `dmx_engine.py` that runs `process_audio`/dispatch/VarietyEngine but routes `send_dmx` to stdout (palette id, behavior, colors per phrase) instead of USB. This makes palette selection, anti-repeat, and phrase evolution inspectable offline and gives the only objective check the project has. Hardware-in-the-loop verification remains "run the app and watch" per CLAUDE.md.

## Migration plan (5 safe checkpoints)

1. **Extract verbatim.** Move shared methods from `music_light.py` into `DmxEngineBase`; `MusicLight` subclasses it. Run loopback → confirm identical behavior.
2. **De-duplicate synced.** `AiShowPlayer` subclasses the same base; delete its copied methods. Run a synced show → confirm identical. *Duplication gone, behavior unchanged.*
3. **Wire VarietyEngine (inert).** Add `dmx_variety.py`; call `begin_section`/`tick` from both `_dispatch`es but renderers still ignore it. No visible change.
4. **Go live.** Renderers read VarietyEngine colors + shared punch helper. *This is where both modes should look dramatically better.* Verify via `--dry-run` then hardware.
5. **LLM enrichment.** Add the `mood` field + prompt/validation changes. Regenerate a show, confirm richer per-section mood.

## Risks & open questions

1. **Punch vs. variety regression.** The biggest risk is step 4 reintroducing mush. Mitigation: the orthogonality invariant + `--dry-run` + A/B by ear against current loopback.
2. **Refactor blast radius.** Extracting 24 methods across two 1000+ line files is the highest-churn part; mitigated by the verbatim-first, checkpoint-per-step sequence.
3. **Loopback "identity" is weaker than synced.** Without track boundaries, loopback variety is time/energy-driven, not per-song. Accept for v1; a future track-change detector could close the gap.

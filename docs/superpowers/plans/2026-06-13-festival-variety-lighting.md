# Festival Variety & Evolution Lighting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make both lighting modes feel like a designed festival show (variety + evolution + punch) by extracting the duplicated engine into a shared module and adding one `VarietyEngine` that both modes feed via a common `Intent`.

**Architecture:** Two mode-specific *directors* (`_dispatch` in each engine file) normalize their input into an `Intent`; a shared `VarietyEngine` owns palette selection (anti-repeat + per-song seed) and phrase-grid texture evolution; parameter-driven renderers consume its output plus a shared punch helper. The LLM contributes an optional per-cue `mood` but never owns timing-critical variety.

**Tech Stack:** Python 3.13 (`.venv`), numpy, pyusb, pyaudiowpatch; new dev dependency `pytest` for the pure-logic modules; hardware/integration verified via a `--dry-run` harness and live runs.

**Spec:** `docs/superpowers/specs/2026-06-13-festival-variety-lighting-design.md`

---

## Reality check: testing in this repo

This project has **no test runner** and the renderers need uDMX hardware + live audio. The plan splits verification accordingly:

- **Pure logic** (`Intent`, palette library, `VarietyEngine`, the punch math helper) → real `pytest` unit tests. No hardware. This is where TDD applies.
- **Refactor / integration / renderers** → characterization: run the app (or the `--dry-run` harness) and confirm behavior. The first two tasks are *verbatim moves* — "tests" are import smoke checks + a live run that must look identical.

Introducing `pytest` is a deliberate, additive choice (CLAUDE.md notes the repo has none today); it touches only the new `tests/` dir and dev tooling, never the runtime.

---

## File structure after this plan

| File | Responsibility | Status |
|------|----------------|--------|
| `dmx_engine.py` | `DmxEngineBase`: hardware, audio/beat core, IPC, 14 renderers, `process_audio()`, abstract `_dispatch()`, dry-run plumbing | **new** |
| `dmx_variety.py` | `Intent` + `PALETTES` library + `VarietyEngine` | **new** |
| `dmx_punch.py` | `velocity_brightness()` + `afterglow()` — shared punch math (velocity→brightness, warm-shifted decay) | **new** |
| `music_light.py` | `class DMXEngine(DmxEngineBase)`: loopback `_dispatch()` (energy state machine → Intent), `run_loopback_mode` (WASAPI teardown) | slimmed |
| `ai_show_player.py` | `class DMXEngine(DmxEngineBase)`: synced `_dispatch()` (cue lookup → Intent), `run_synced_mode` | slimmed |
| `llm_designer.py` | adds optional per-cue `mood` (prompt + schema + repair) | modified |
| `tests/test_intent.py` | unit tests for `Intent` | **new** |
| `tests/test_palettes.py` | unit tests for the palette library shape | **new** |
| `tests/test_variety_engine.py` | unit tests for `VarietyEngine` | **new** |
| `tests/test_punch.py` | unit tests for the punch helpers | **new** |
| `tests/test_llm_repair.py` | unit test for the mood-default repair | **new** |

**Naming locked across tasks** (use these exact names everywhere):
- `Intent` fields: `energy:int`, `mood:str`, `section_id`, `is_new_section:bool`, `bpm:float`, `strobe_allowed:bool`, `seed_color` (optional `[R,G,B]` color hint from the LLM; `None` in loopback).
- Palette family dict keys: `id`, `mood`, `energy` (`[lo, hi]`), `primary`, `secondary`, `accent` (each color `[R,G,B]`).
- `VarietyEngine` methods: `set_song_seed(seed)`, `begin_section(intent)`, `on_phrase_boundary()`, `current_colors()`, `tick(is_beat, bpm, t)`. Module helpers: `_color_distance(a, b)`, `_nearest_palette(candidates, color)`.
- `dmx_punch.py` functions: `velocity_brightness(beat_velocity)` → master 120..255; `afterglow(r, g, b, w)` → warm-shifted decayed tuple.

**Color ownership (decided):** in synced mode the LLM's `color_1` **seeds** the palette — `begin_section` maps it to the nearest curated family, then the engine evolves from there. The LLM no longer owns final colors; it expresses intent via `color_1` + `mood`. Loopback has no LLM color, so it selects purely by energy/mood/anti-repeat.

---

## Task 1: Test scaffolding (pytest)

**Files:**
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_smoke.py`
- Create: `pytest.ini`

- [ ] **Step 1: Install pytest into the existing venv**

Run: `.venv\Scripts\pip install pytest`
Expected: "Successfully installed pytest-..."

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 3: Un-ignore the `tests/` dir**

The repo's `.gitignore` has `test_*.py`, which matches `tests/test_intent.py` by basename — without this, every test file in this plan is silently skipped by `git add`. Add a negation **after** that rule (order matters in gitignore):

```gitignore
# Allow the real test suite (the test_*.py rule above targets throwaway probes)
!tests/
!tests/**
```

After this, plain `git add tests/...` works in every later task — no `-f` needed.

- [ ] **Step 4: Create `tests/__init__.py`** (empty file)

- [ ] **Step 5: Write a smoke test**

`tests/test_smoke.py`:
```python
def test_pytest_runs():
    assert 1 + 1 == 2
```

- [ ] **Step 6: Run it**

Run: `.venv\Scripts\python -m pytest -q`
Expected: `1 passed`

- [ ] **Step 7: Commit**

```bash
git add .gitignore pytest.ini tests/__init__.py tests/test_smoke.py
git status   # confirm tests/test_smoke.py is staged (not ignored)
git commit -m "test: add pytest scaffolding for new pure-logic modules"
```

---

## Task 2: Extract `DmxEngineBase` verbatim (Checkpoint 1)

Pure refactor, **zero behavior change**. Move the shared methods out of `music_light.py` into a new base class; make the loopback engine subclass it.

**Files:**
- Create: `dmx_engine.py`
- Modify: `music_light.py`

- [ ] **Step 1: Create `dmx_engine.py` with the base class and shared imports**

Move the module-level constants and shared imports (`os, sys, json, time, math, logging, threading`, `numpy`, `usb.core`, `usb.util`) and the `DMXEngine` class body from `music_light.py` into `dmx_engine.py`, renaming the class to `DmxEngineBase`. Move these methods **verbatim** (the 24 shared methods confirmed in the spec):
`__init__`, `_init_hardware`, `_dmx_worker`, `send_dmx`, `shutdown`, `_write_playback_state`, `_check_playback_command`, `_cleanup_ipc_files`, `load_ai_show`, `_get_active_cue`, `load_profile`, `process_audio`, all 14 `_render_*` methods, and `_render_loopback_direct`.

At the end of `process_audio`, replace the inline loopback dispatch (`music_light.py:1163-1199`, the auto-behavior + renderer/`_render_loopback_direct` block) with a single call to an abstract dispatch seam:

```python
        # Mode-specific color/behavior selection + renderer invocation.
        # Subclasses set self.out_* ; base then sends the frame.
        self._dispatch(kick_mag, snare_mag, mid_mag, hihat_mag,
                       kick_i, snare_i, hihat_i, mid_i,
                       is_kick, is_snare, volume, t,
                       elapsed_seconds=elapsed_seconds)

        self.send_dmx(self.out_master, self.out_r, self.out_g,
                      self.out_b, self.out_w, self.out_strobe)
```

Add the abstract method to `DmxEngineBase`:

```python
    def _dispatch(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                  kick_i, snare_i, hihat_i, mid_i,
                  is_kick, is_snare, volume, t, elapsed_seconds=None):
        raise NotImplementedError("Subclasses must implement _dispatch()")
```

> The exact slice of `process_audio` that moves to `_dispatch` is the per-mode tail. Keep everything above it (decode → FFT → bands → AGC → flux → indices → beat detection → BPS) in the base `process_audio`. Do not change any numbers.
>
> **One required addition (not a behavior change):** the BPS value `process_audio` computes is currently a local. Store it as `self.beats_per_sec` right where it's computed, so `_dispatch` can read it explicitly instead of guessing. (Loopback reads this; synced uses `self.show_bpm` instead — see Task 10.)

- [ ] **Step 2: Reduce `music_light.py` to a subclass**

Replace the old class with:

```python
import sys, time, logging
import pyaudio
from dmx_engine import DmxEngineBase, BLOCK_SIZE  # plus any loopback-only constants

logger = logging.getLogger(__name__)

class DMXEngine(DmxEngineBase):
    def _dispatch(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                  kick_i, snare_i, hihat_i, mid_i,
                  is_kick, is_snare, volume, t, elapsed_seconds=None):
        # --- VERBATIM: the loopback color-cycling + _detect_auto_behavior block
        #     (old music_light.py:1163-1199), unchanged. ---
        ...
```

Keep `_detect_auto_behavior`, the loopback color-cycling helpers, `run_loopback_mode` (with its WASAPI `p.terminate()` `finally`), and the `__main__` block in `music_light.py`. The `__main__` still says `engine = DMXEngine()` — unchanged.

> `_render_loopback_direct` now lives in the base, so the subclass calls `self._render_loopback_direct(...)` as before.

- [ ] **Step 3: Import smoke test**

Run: `.venv\Scripts\python -c "import dmx_engine, music_light; print('ok')"`
Expected: `ok` (no ImportError, no NameError)

- [ ] **Step 4: Live verification (loopback)**

Play any system audio, then run: `.venv\Scripts\python music_light.py --profile profiles/concert_punchy.json`
Expected: lights behave **identically** to before the refactor (same punch, same R→B→G→W cycling). Ctrl-C to stop; confirm clean shutdown log ("uDMX" released).

- [ ] **Step 5: Commit**

```bash
git add dmx_engine.py music_light.py
git commit -m "refactor: extract DmxEngineBase shared module from music_light"
```

---

## Task 3: De-duplicate `ai_show_player.py` (Checkpoint 2)

**Files:**
- Modify: `ai_show_player.py`

- [ ] **Step 1: Subclass the base, delete the duplicated copies**

Delete every method in `ai_show_player.py` that now lives in `DmxEngineBase` (the 24 shared ones, including its own copies of `process_audio`, `send_dmx`, `_init_hardware`, `shutdown`, the renderers, IPC helpers, `_get_active_cue`, `load_ai_show`). Replace the class header with:

```python
from dmx_engine import DmxEngineBase

class DMXEngine(DmxEngineBase):
    def _dispatch(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                  kick_i, snare_i, hihat_i, mid_i,
                  is_kick, is_snare, volume, t, elapsed_seconds=None):
        # --- VERBATIM: the synced cue-lookup + renderer block
        #     (old ai_show_player.py:901-918), unchanged. ---
        if self.synced_cues:
            cue = self._get_active_cue(elapsed_seconds)
            if cue:
                kick_color = cue["color_1"]; accent_color = cue["color_2"]
                behavior = cue.get("behavior", "beat_reactive")
            else:
                kick_color, accent_color = self.palettes[self.current_palette_idx]
                behavior = "beat_reactive"; cue = None
        else:
            kick_color, accent_color = self.palettes[self.current_palette_idx]
            behavior = "beat_reactive"; cue = None
        renderer = self._behavior_map.get(behavior, self._render_beat_reactive)
        renderer(kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                 kick_color, accent_color, volume, cue or {"energy": 7, "dimmer": 80}, t)
```

Keep `run_synced_mode` (cue logging, transport-command handling, WAV loop) and the `__main__` block.

- [ ] **Step 2: Import smoke test**

Run: `.venv\Scripts\python -c "import ai_show_player; print('ok')"`
Expected: `ok`

- [ ] **Step 3: Live verification (synced)**

With an existing show: `.venv\Scripts\python ai_show_player.py --show shows/<id>/show.json`
Expected: synced playback behaves **identically** to before. The duplication is now gone with no behavior change — safe checkpoint.

- [ ] **Step 4: Commit**

```bash
git add ai_show_player.py
git commit -m "refactor: ai_show_player subclasses DmxEngineBase, drop duplicated methods"
```

---

## Task 4: `--dry-run` harness (offline verification)

Lets later tasks be verified without uDMX hardware by feeding a WAV through the synced path and logging decisions instead of driving USB.

**Files:**
- Modify: `dmx_engine.py`
- Modify: `ai_show_player.py`

- [ ] **Step 1: Add dry-run plumbing to the base**

In `dmx_engine.py`, top of module:
```python
DRY_RUN = os.getenv("DMX_DRY_RUN") == "1"
```
In `_init_hardware`, before the USB lookup:
```python
        if DRY_RUN:
            logger.info("[DRY-RUN] Skipping uDMX init; frames will be logged.")
            self.dev = None
            return
```
In `send_dmx`, first lines:
```python
        if DRY_RUN:
            self.last_frame = (int(master), int(red), int(green), int(blue), int(white), int(strobe))
            return
```

- [ ] **Step 2: Add a dry-run branch to `run_synced_mode`**

In `ai_show_player.py`, inside `run_synced_mode`, when `DRY_RUN` is set, skip opening the PyAudio output stream and instead iterate WAV frames as fast as possible, still calling `process_audio(...)` and logging the active cue + `self.last_frame` on each cue change:

```python
        from dmx_engine import DRY_RUN
        if DRY_RUN:
            self._init_hardware()
            self.load_ai_show(show_file)
            wf = wave.open(self.audio_file, 'rb')
            sample_rate = wf.getframerate(); chunk = 1024
            frames_played = 0; last_cue = None
            data = wf.readframes(chunk)
            while data:
                elapsed = frames_played / sample_rate
                self.process_audio(data, elapsed_seconds=elapsed,
                                   input_format="int16", actual_sample_rate=sample_rate)
                cue = self._get_active_cue(elapsed)
                name = cue["name"] if cue else None
                if name != last_cue:
                    last_cue = name
                    logger.info(f"[DRY {elapsed:6.1f}s] cue={name} frame={self.last_frame}")
                frames_played += chunk
                data = wf.readframes(chunk)
            wf.close()
            return
```

- [ ] **Step 3: Verify the harness runs offline**

Run: `set DMX_DRY_RUN=1 && .venv\Scripts\python ai_show_player.py --show shows/<id>/show.json`
Expected: a stream of `[DRY ...s] cue=... frame=(m,r,g,b,w,s)` lines, no USB error, exits at end of WAV.

- [ ] **Step 4: Commit**

```bash
git add dmx_engine.py ai_show_player.py
git commit -m "feat: add DMX_DRY_RUN offline harness for synced path"
```

---

## Task 5: `Intent` contract

**Files:**
- Create: `dmx_variety.py`
- Test: `tests/test_intent.py`

- [ ] **Step 1: Write the failing test**

`tests/test_intent.py`:
```python
from dmx_variety import Intent

def test_intent_holds_fields():
    i = Intent(energy=8, mood="neon", section_id="chorus1",
               is_new_section=True, bpm=128.0, strobe_allowed=True)
    assert i.energy == 8
    assert i.mood == "neon"
    assert i.section_id == "chorus1"
    assert i.is_new_section is True
    assert i.bpm == 128.0
    assert i.strobe_allowed is True

def test_intent_defaults():
    i = Intent(energy=3, mood="warm", section_id="intro")
    assert i.is_new_section is False
    assert i.bpm == 0.0
    assert i.strobe_allowed is False
    assert i.seed_color is None

def test_intent_carries_seed_color():
    i = Intent(energy=8, mood="neon", section_id="drop", seed_color=[255, 0, 120])
    assert i.seed_color == [255, 0, 120]
```

- [ ] **Step 2: Run, verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_intent.py -q`
Expected: FAIL (`ModuleNotFoundError: dmx_variety` or `ImportError: Intent`).

- [ ] **Step 3: Implement `Intent`**

`dmx_variety.py`:
```python
"""Variety & evolution layer: Intent contract, palette library, VarietyEngine.

Shared by both engines. The two directors (loopback energy state machine and
synced LLM cue) each build an Intent; the VarietyEngine consumes it identically.
"""
import random
from collections import deque


class Intent:
    """Normalized lighting intent emitted by either director."""
    __slots__ = ("energy", "mood", "section_id", "is_new_section",
                 "bpm", "strobe_allowed", "seed_color")

    def __init__(self, energy, mood, section_id,
                 is_new_section=False, bpm=0.0, strobe_allowed=False,
                 seed_color=None):
        self.energy = energy
        self.mood = mood
        self.section_id = section_id
        self.is_new_section = is_new_section
        self.bpm = bpm
        self.strobe_allowed = strobe_allowed
        self.seed_color = seed_color  # optional [R,G,B] hint (LLM color_1); None in loopback
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_intent.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add -f dmx_variety.py tests/test_intent.py
git commit -m "feat: add Intent contract for variety layer"
```

---

## Task 6: Palette library

**Files:**
- Modify: `dmx_variety.py`
- Test: `tests/test_palettes.py`

- [ ] **Step 1: Write the failing test**

`tests/test_palettes.py`:
```python
from dmx_variety import PALETTES

def test_palettes_are_well_formed():
    assert len(PALETTES) >= 12
    ids = [p["id"] for p in PALETTES]
    assert len(ids) == len(set(ids)), "palette ids must be unique"
    for p in PALETTES:
        assert set(p) >= {"id", "mood", "energy", "primary", "secondary", "accent"}
        lo, hi = p["energy"]
        assert 1 <= lo <= hi <= 10
        for key in ("primary", "secondary", "accent"):
            r, g, b = p[key]
            assert all(0 <= c <= 255 for c in (r, g, b))

def test_palettes_cover_all_energy_bands():
    # every energy level 1..10 must be servable by at least one palette
    for e in range(1, 11):
        assert any(p["energy"][0] <= e <= p["energy"][1] for p in PALETTES)

def test_palettes_cover_core_moods():
    moods = {p["mood"] for p in PALETTES}
    assert {"warm", "cool", "neon", "euphoric", "dark"} <= moods
```

- [ ] **Step 2: Run, verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_palettes.py -q`
Expected: FAIL (`ImportError: PALETTES`).

- [ ] **Step 3: Implement the library**

Append to `dmx_variety.py`:
```python
# Curated palette families. Each: primary (kick color), secondary (snare/accent
# color), accent (combo/stab color). mood + energy range drive selection.
PALETTES = [
    {"id": "volcanic",   "mood": "warm",     "energy": [6, 10], "primary": [255, 60, 10],  "secondary": [255, 150, 0],  "accent": [255, 255, 255]},
    {"id": "ember",      "mood": "warm",     "energy": [2, 6],  "primary": [255, 90, 30],  "secondary": [200, 40, 60],  "accent": [255, 200, 120]},
    {"id": "candle",     "mood": "warm",     "energy": [1, 4],  "primary": [255, 140, 40], "secondary": [180, 70, 20],  "accent": [255, 220, 150]},
    {"id": "arctic",     "mood": "cool",     "energy": [4, 8],  "primary": [0, 180, 255],  "secondary": [10, 30, 180],  "accent": [255, 255, 255]},
    {"id": "deep_ocean", "mood": "cool",     "energy": [1, 5],  "primary": [10, 30, 180],  "secondary": [0, 120, 140],  "accent": [120, 200, 255]},
    {"id": "glacier",    "mood": "cool",     "energy": [3, 7],  "primary": [120, 200, 255],"secondary": [40, 90, 200],  "accent": [255, 255, 255]},
    {"id": "neon_pink",  "mood": "neon",     "energy": [6, 10], "primary": [255, 0, 120],  "secondary": [0, 220, 255],  "accent": [255, 255, 255]},
    {"id": "acid",       "mood": "neon",     "energy": [6, 10], "primary": [180, 255, 0],  "secondary": [255, 0, 200],  "accent": [255, 255, 255]},
    {"id": "violet_haze","mood": "neon",     "energy": [4, 9],  "primary": [130, 0, 255],  "secondary": [0, 220, 255],  "accent": [255, 120, 255]},
    {"id": "sunburst",   "mood": "euphoric", "energy": [5, 10], "primary": [255, 200, 0],  "secondary": [255, 0, 120],  "accent": [255, 255, 255]},
    {"id": "rave",       "mood": "euphoric", "energy": [7, 10], "primary": [0, 255, 100],  "secondary": [255, 0, 200],  "accent": [255, 255, 255]},
    {"id": "prism",      "mood": "euphoric", "energy": [4, 9],  "primary": [255, 0, 0],    "secondary": [0, 100, 255],  "accent": [0, 255, 100]},
    {"id": "midnight",   "mood": "dark",     "energy": [1, 5],  "primary": [40, 0, 80],    "secondary": [0, 40, 90],    "accent": [120, 80, 200]},
    {"id": "blood",      "mood": "dark",     "energy": [5, 10], "primary": [120, 0, 0],    "secondary": [200, 0, 40],   "accent": [255, 60, 60]},
    {"id": "forest",     "mood": "cool",     "energy": [1, 5],  "primary": [0, 120, 60],   "secondary": [40, 90, 30],   "accent": [150, 255, 180]},
    {"id": "aurora",     "mood": "euphoric", "energy": [2, 6],  "primary": [0, 255, 150],  "secondary": [80, 0, 255],   "accent": [0, 220, 255]},
]
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_palettes.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add dmx_variety.py
git add -f tests/test_palettes.py
git commit -m "feat: add curated palette library for variety engine"
```

---

## Task 7: `VarietyEngine` — selection, anti-repeat, seeding

**Files:**
- Modify: `dmx_variety.py`
- Test: `tests/test_variety_engine.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_variety_engine.py`:
```python
from dmx_variety import VarietyEngine, Intent, PALETTES

def _intent(energy=8, mood=None, sid="s", new=True, bpm=128.0):
    return Intent(energy=energy, mood=mood, section_id=sid,
                  is_new_section=new, bpm=bpm)

def test_begin_section_returns_palette_matching_energy():
    ve = VarietyEngine(seed=1)
    p = ve.begin_section(_intent(energy=2))
    lo, hi = p["energy"]
    assert lo <= 2 <= hi

def test_begin_section_respects_mood():
    ve = VarietyEngine(seed=1)
    p = ve.begin_section(_intent(energy=8, mood="neon"))
    assert p["mood"] == "neon"

def test_anti_repeat_avoids_recent_palettes():
    ve = VarietyEngine(seed=7)
    seen = [ve.begin_section(_intent()) ["id"] for _ in range(5)]
    # no palette repeats within a window of 4 selections
    for k in range(4, len(seen)):
        assert seen[k] not in seen[k-4:k]

def test_seeding_is_deterministic():
    a = VarietyEngine(seed=42)
    b = VarietyEngine(seed=42)
    seq_a = [a.begin_section(_intent())["id"] for _ in range(6)]
    seq_b = [b.begin_section(_intent())["id"] for _ in range(6)]
    assert seq_a == seq_b

def test_different_seeds_can_diverge():
    # Deterministic per seed; across a 6-section run two seeds should differ.
    # (Not a per-call guarantee, but with 16 palettes + anti-repeat this run
    #  is stable — asserted on the whole sequence, not a single pick.)
    ra = [VarietyEngine(seed=1).begin_section(_intent())["id"] for _ in range(1)]
    seq_a = VarietyEngine(seed=1)
    seq_b = VarietyEngine(seed=2)
    run_a = [seq_a.begin_section(_intent())["id"] for _ in range(6)]
    run_b = [seq_b.begin_section(_intent())["id"] for _ in range(6)]
    assert run_a != run_b

def test_seed_color_picks_nearest_family():
    # An intent carrying a magenta-ish seed_color should land on a neon/pink
    # family whose primary is close to it, deterministically (no rng).
    ve = VarietyEngine(seed=999)
    p = ve.begin_section(_intent(energy=8, mood=None))  # warm up recent buffer
    chosen = ve.begin_section(Intent(energy=8, mood=None, section_id="x",
                                     is_new_section=True, bpm=128.0,
                                     seed_color=[255, 0, 120]))
    # nearest primary to [255,0,120] among energy-8 candidates is neon_pink
    assert chosen["id"] == "neon_pink"

def test_relax_when_library_exhausted():
    tiny = PALETTES[:2]
    ve = VarietyEngine(palettes=tiny, seed=1)
    # more selections than the library size must not crash
    ids = [ve.begin_section(_intent(energy=tiny[0]["energy"][0]))["id"] for _ in range(5)]
    assert len(ids) == 5
```

> If `neon_pink` happens to be in the recent buffer when the seed-color test runs, anti-repeat would exclude it. The test warms up with a single prior selection at `seed=999`; if this proves flaky during implementation, assert instead that the chosen primary is within distance 60 of `[255,0,120]` rather than a fixed id.

- [ ] **Step 2: Run, verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_variety_engine.py -q`
Expected: FAIL (`ImportError: VarietyEngine`).

- [ ] **Step 3: Implement selection**

Append to `dmx_variety.py` (the two module helpers go near the top with `_shift_hue`; the class follows):
```python
def _color_distance(a, b):
    """Euclidean distance between two RGB triples."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _nearest_palette(candidates, color):
    """The candidate whose primary is closest to `color`. Deterministic."""
    return min(candidates, key=lambda p: _color_distance(p["primary"], color))


class VarietyEngine:
    """Owns all anti-monotony policy: palette selection (anti-repeat + per-song
    seed + optional LLM color seed) and phrase-grid texture evolution.
    Mode-agnostic — fed via Intent."""

    PHRASE_LEN_BEATS = 8
    TIME_PHRASE_FALLBACK_S = 4.0

    def __init__(self, palettes=PALETTES, seed=None):
        self._palettes = list(palettes)
        self._recent = deque(maxlen=4)          # recent palette ids (anti-repeat)
        self._rng = random.Random(seed)
        self.current_palette = self._palettes[0]
        self.phrase_index = 0
        self._beats_in_section = 0
        self._last_t = 0.0
        self._section_start_t = 0.0
        self._last_phrase_t = 0.0

    def set_song_seed(self, seed):
        self._rng = random.Random(seed)

    def begin_section(self, intent):
        """Pick a fresh palette for a new section and reset phrase state.

        Selection precedence:
          1. energy range + anti-repeat (always),
          2. mood match (only when there is no seed_color — color wins over mood),
          3. if seed_color given → nearest palette by primary color (deterministic);
             else → seeded random choice (per-song identity).
        Relaxes filters step-by-step if nothing matches, so it never deadlocks."""
        in_energy = [p for p in self._palettes
                     if p["energy"][0] <= intent.energy <= p["energy"][1]]
        fresh = [p for p in in_energy if p["id"] not in self._recent] or in_energy

        if intent.seed_color is None and intent.mood is not None:
            mood_match = [p for p in fresh if p["mood"] == intent.mood]
            candidates = mood_match or fresh
        else:
            candidates = fresh

        if not candidates:
            candidates = [p for p in self._palettes if p["id"] not in self._recent] \
                or list(self._palettes)

        if intent.seed_color is not None:
            chosen = _nearest_palette(candidates, intent.seed_color)
        else:
            chosen = self._rng.choice(candidates)

        self._recent.append(chosen["id"])
        self.current_palette = chosen
        self.phrase_index = 0
        self._beats_in_section = 0
        self._section_start_t = self._last_t
        self._last_phrase_t = self._last_t
        return chosen
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_variety_engine.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add dmx_variety.py
git add -f tests/test_variety_engine.py
git commit -m "feat: VarietyEngine palette selection with anti-repeat and seeding"
```

---

## Task 8: `VarietyEngine` — texture phase

**Files:**
- Modify: `dmx_variety.py`
- Modify: `tests/test_variety_engine.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_variety_engine.py`:
```python
def test_current_colors_returns_three_rgb():
    ve = VarietyEngine(seed=1)
    ve.begin_section(_intent(energy=8, mood="neon"))
    c1, c2, accent = ve.current_colors()
    for c in (c1, c2, accent):
        assert len(c) == 3 and all(0 <= v <= 255 for v in c)

def test_texture_evolves_across_phrases():
    ve = VarietyEngine(seed=1)
    ve.begin_section(_intent(energy=8, mood="neon"))
    look0 = ve.current_colors()
    ve.on_phrase_boundary()
    look1 = ve.current_colors()
    ve.on_phrase_boundary()
    look2 = ve.current_colors()
    # at least one of the first three phrases differs from phrase 0
    assert look1 != look0 or look2 != look0
    assert ve.phrase_index == 2
```

- [ ] **Step 2: Run, verify the new tests fail**

Run: `.venv\Scripts\python -m pytest tests/test_variety_engine.py -q`
Expected: FAIL (`AttributeError: on_phrase_boundary` / `current_colors`).

- [ ] **Step 3: Implement texture phase**

Add to `VarietyEngine`:
```python
    def on_phrase_boundary(self):
        """Advance one phrase; texture moves are derived from phrase_index so
        the section *develops* deterministically (beat-quantized = intentional)."""
        self.phrase_index += 1

    def current_colors(self):
        """(color_1, color_2, accent) for this frame, modulated by phrase_index.

        Phase pattern (cycles every 4 phrases):
          0: primary / secondary / accent
          1: primary / accent    / secondary   (swap accent in)
          2: secondary / primary / accent       (flip roles)
          3: primary / hue-shifted secondary / accent
        """
        p = self.current_palette
        prim, sec, acc = p["primary"], p["secondary"], p["accent"]
        phase = self.phrase_index % 4
        if phase == 0:
            return list(prim), list(sec), list(acc)
        if phase == 1:
            return list(prim), list(acc), list(sec)
        if phase == 2:
            return list(sec), list(prim), list(acc)
        return list(prim), _shift_hue(sec, 40), list(acc)
```

Add the module-level helper (near the top, after imports):
```python
def _shift_hue(rgb, degrees):
    """Rotate an RGB color's hue by `degrees`. Cheap, dependency-free."""
    import colorsys
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r * 255), int(g * 255), int(b * 255)]
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_variety_engine.py -q`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add dmx_variety.py tests/test_variety_engine.py
git commit -m "feat: VarietyEngine phrase-grid texture evolution"
```

---

## Task 9: `VarietyEngine` — `tick()` phrase detection + time fallback

**Files:**
- Modify: `dmx_variety.py`
- Modify: `tests/test_variety_engine.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_variety_engine.py`:
```python
def test_tick_fires_phrase_boundary_every_8_beats():
    ve = VarietyEngine(seed=1)
    ve.begin_section(_intent(bpm=120.0))
    boundaries = 0
    t = 0.0
    for beat in range(16):          # 16 beats -> 2 phrase boundaries
        t += 0.5
        ev = ve.tick(is_beat=True, bpm=120.0, t=t)
        if ev["phrase_boundary"]:
            boundaries += 1
    assert boundaries == 2
    assert ve.phrase_index == 2

def test_tick_time_fallback_when_no_bpm():
    ve = VarietyEngine(seed=1)
    ve.begin_section(_intent(bpm=0.0))
    fired = False
    t = 0.0
    while t < 4.5:                  # exceed TIME_PHRASE_FALLBACK_S with no beats
        t += 0.1
        ev = ve.tick(is_beat=False, bpm=0.0, t=t)
        fired = fired or ev["phrase_boundary"]
    assert fired
```

- [ ] **Step 2: Run, verify the new tests fail**

Run: `.venv\Scripts\python -m pytest tests/test_variety_engine.py -q`
Expected: FAIL (`AttributeError: tick`).

- [ ] **Step 3: Implement `tick`**

Add to `VarietyEngine`:
```python
    def tick(self, is_beat, bpm, t):
        """Per-frame update. Returns {'phrase_boundary': bool,
        'seconds_in_section': float}. Directors call this every frame and read
        current_colors(); they decide when to call begin_section()."""
        self._last_t = t
        boundary = False

        if bpm and bpm > 0:
            if is_beat:
                self._beats_in_section += 1
                if self._beats_in_section % self.PHRASE_LEN_BEATS == 0:
                    self.on_phrase_boundary()
                    boundary = True
                    self._last_phrase_t = t
        else:
            # No reliable beat grid → fall back to a wall-clock phrase interval.
            if t - self._last_phrase_t >= self.TIME_PHRASE_FALLBACK_S:
                self.on_phrase_boundary()
                boundary = True
                self._last_phrase_t = t

        return {"phrase_boundary": boundary,
                "seconds_in_section": t - self._section_start_t}
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_variety_engine.py -q`
Expected: `11 passed`

- [ ] **Step 5: Commit**

```bash
git add dmx_variety.py tests/test_variety_engine.py
git commit -m "feat: VarietyEngine tick() with phrase detection and time fallback"
```

---

## Task 10: Wire `VarietyEngine` into both directors (inert) (Checkpoint 3)

Build `Intent` in each `_dispatch`, drive the engine, log decisions via dry-run — but renderers still use the old colors, so no visible change yet.

**Files:**
- Modify: `dmx_engine.py` (construct engine in base `__init__`)
- Modify: `music_light.py` (loopback director builds Intent)
- Modify: `ai_show_player.py` (synced director builds Intent)

- [ ] **Step 1: Construct the engine and wire the per-song seed**

In `DmxEngineBase.__init__`, after existing state setup:
```python
        from dmx_variety import VarietyEngine
        self.variety = VarietyEngine()
        self._last_section_id = None
        self.show_bpm = 0.0          # set by load_ai_show (synced)
        self.beats_per_sec = 0.0     # set by process_audio (loopback); see Task 2
        self._evolution_secs = 16.0  # loopback: rotate palette after this long at one energy
```

In `DmxEngineBase.load_ai_show`, after the show dict is parsed, set the BPM and **seed the engine from the song's identity** so the same show always renders the same look (fixes "every song looks alike"):
```python
        metrics = show.get("lighting_plan", {}).get("song_metrics", {}) \
            or show.get("song_metrics", {})
        self.show_bpm = float(metrics.get("bpm", 0.0) or 0.0)
        seed_basis = show.get("show_name") or show.get("audio_file") or ""
        self.variety.set_song_seed(hash(seed_basis))
```
> Adjust the exact dict path to wherever `load_ai_show` already reads the plan — the point is a stable per-show seed (name or audio path) and the real BPM. If neither exists, `hash("")` is a stable fallback.

- [ ] **Step 2: Loopback director — Intent + time-evolution**

In `music_light.py` `_dispatch`, before the existing color/behavior block. Note: `tick` is called **first** so `seconds_in_section` is current; a new palette is forced either on an energy-state change **or** after `_evolution_secs` at constant energy (this is what fixes "loopback feels static"):
```python
        from dmx_variety import Intent
        bpm = self.beats_per_sec * 60.0   # explicit attr set in process_audio (Task 2)
        mood = {"calm": "warm", "building": "cool",
                "high": "neon", "dropping": "euphoric"}.get(self.energy_state, "cool")
        energy = {"calm": 2, "building": 5, "high": 8, "dropping": 6}.get(self.energy_state, 5)
        section_id = self.energy_state

        ev = self.variety.tick(is_beat=(is_kick or is_snare), bpm=bpm, t=t)
        force_evolve = ev["seconds_in_section"] >= self._evolution_secs
        is_new = (section_id != self._last_section_id) or force_evolve
        self._last_section_id = section_id
        if is_new:
            intent = Intent(energy=energy, mood=mood, section_id=section_id,
                            is_new_section=True, bpm=bpm, strobe_allowed=True)
            self.variety.begin_section(intent)   # resets section clock → evolution re-arms
```

- [ ] **Step 3: Synced director — Intent + LLM color seed**

In `ai_show_player.py` `_dispatch`, after resolving `cue`/`behavior`. The LLM's `color_1` becomes the palette **seed_color** (decided: LLM color seeds the palette):
```python
        from dmx_variety import Intent
        if cue:
            energy = int(cue.get("energy_level", cue.get("energy", 5)))
            mood = cue.get("mood")            # None until Task 13; fine
            section_id = cue.get("name", "")
            strobe = bool(cue.get("strobe_allowed", False))
            seed_color = cue.get("color_1")   # LLM intent → nearest palette family
        else:
            energy, mood, section_id, strobe, seed_color = 7, None, "_fallback", False, None
        bpm = self.show_bpm                    # explicit attr set in load_ai_show
        is_new = section_id != self._last_section_id
        self._last_section_id = section_id
        ev = self.variety.tick(is_beat=(is_kick or is_snare), bpm=bpm, t=t)
        if is_new:
            intent = Intent(energy=energy, mood=mood, section_id=section_id,
                            is_new_section=True, bpm=bpm, strobe_allowed=strobe,
                            seed_color=seed_color)
            self.variety.begin_section(intent)
```

- [ ] **Step 4: Verify via dry-run (decisions logged, no behavior change)**

Add a temporary log line at the end of each `_dispatch`:
```python
        if DRY_RUN and is_new:
            logger.info(f"[VARIETY] section={section_id} palette={self.variety.current_palette['id']} mood={mood} energy={energy}")
```
Run: `set DMX_DRY_RUN=1 && .venv\Scripts\python ai_show_player.py --show shows/<id>/show.json`
Expected: `[VARIETY] section=... palette=... ` lines, one per cue change, with **different** palettes across adjacent sections (anti-repeat working). Rendered frames unchanged from Task 4 output.

- [ ] **Step 5: Live verification — no visible change**

Run loopback and a synced show on hardware. Expected: identical to Task 3 (renderers still use old colors). This confirms the wiring is inert.

- [ ] **Step 6: Commit**

```bash
git add dmx_engine.py music_light.py ai_show_player.py
git commit -m "feat: wire VarietyEngine into both directors (inert, decisions logged)"
```

---

## Task 11: Shared punch helper

Extract the punch math (velocity → brightness, beat-hold, warm afterglow, breathing floor) from `_render_loopback_direct` into a reusable, unit-testable helper so synced renderers can use it too.

**Files:**
- Create: `dmx_punch.py`
- Test: `tests/test_punch.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_punch.py`:
```python
from dmx_punch import velocity_brightness, afterglow

def test_velocity_brightness_scales_between_floor_and_full():
    assert velocity_brightness(0.0) == 120.0          # soft beat -> dim floor
    assert velocity_brightness(1.0) == 255.0          # hard beat -> full
    mid = velocity_brightness(0.5)
    assert 120.0 < mid < 255.0

def test_velocity_brightness_clamps_above_one():
    assert velocity_brightness(5.0) == 255.0

def test_afterglow_shifts_warm():
    # warm shift: red decays slowest, blue fastest
    r, g, b, w = afterglow(200.0, 200.0, 200.0, 200.0)
    assert r > g > b
    assert w < 200.0
```

- [ ] **Step 2: Run, verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_punch.py -q`
Expected: FAIL (`ModuleNotFoundError: dmx_punch`).

- [ ] **Step 3: Implement the helper**

`dmx_punch.py`:
```python
"""Shared punch primitives: the velocity/beat-hold/afterglow math that made
loopback feel crisp, lifted out of _render_loopback_direct so synced renderers
can use it too. Pure functions — no engine state, easy to unit-test."""

VELOCITY_FLOOR = 120.0
VELOCITY_FULL = 255.0


def velocity_brightness(beat_velocity):
    """Map a 0..1 beat velocity to master brightness 120..255.
    Soft beats stay dim; hard beats blast. Values >1 clamp to full."""
    v = max(0.0, min(1.0, beat_velocity))
    return VELOCITY_FLOOR + (VELOCITY_FULL - VELOCITY_FLOOR) * v


def afterglow(r, g, b, w):
    """One frame of warm-shifted decay (R slowest, B fastest), matching the
    _render_loopback_direct tail (R 0.95 / G 0.88 / B 0.82 / W 0.80)."""
    return r * 0.95, g * 0.88, b * 0.82, w * 0.80
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_punch.py -q`
Expected: `3 passed`

- [ ] **Step 5: Refactor `_render_loopback_direct` to use the helper (no behavior change)**

In `dmx_engine.py`, replace the inline `velocity_brightness = 120.0 + (135.0 * beat_velocity)` with `from dmx_punch import velocity_brightness, afterglow` (top of module) and `vb = velocity_brightness(beat_velocity)`, and replace the afterglow block (`self.out_r *= 0.95 ; ... *= 0.88 ; ... *= 0.82 ; self.out_w *= 0.80`) with `self.out_r, self.out_g, self.out_b, self.out_w = afterglow(self.out_r, self.out_g, self.out_b, self.out_w)`. Run loopback on hardware → identical feel.

- [ ] **Step 6: Commit**

```bash
git add dmx_punch.py
git add -f tests/test_punch.py
git add dmx_engine.py
git commit -m "feat: extract shared punch helper; reuse in loopback renderer"
```

---

## Task 12: Renderers consume VarietyEngine colors + punch (Checkpoint 4)

The payoff. Renderers stop using the cue/cycle colors and read `self.variety.current_colors()`; synced gets punch via the shared helper.

**Files:**
- Modify: `dmx_engine.py` (renderers + both `_dispatch` color sourcing)

- [ ] **Step 1: Source colors from the variety engine in both directors**

In `music_light.py` `_dispatch`, replace the local color assignment with the palette's three colors, and pass them to the renderers:
```python
        kick_color, accent_color, combo_color = self.variety.current_colors()
```
- Ambient branch: call the standard renderer with `kick_color, accent_color` (as today).
- Punchy branch: call the rewired loopback renderer with the new signature (Step 2):
```python
            self._render_loopback_direct(
                kick_mag, snare_mag, mid_mag, hihat_mag,
                kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                kick_color, accent_color, combo_color, volume, t)
```

In `ai_show_player.py` `_dispatch`, replace the cue/`palettes[...]` color sourcing with `kick_color, accent_color, combo_color = self.variety.current_colors()`. The cue still drives `behavior`, `energy`, `strobe`, and `seed_color` (Task 10). Keep passing `kick_color, accent_color` into the renderer call.

- [ ] **Step 2: Rewire `_render_loopback_direct` to use palette colors (fixes the R→B→G→W repetition)**

This is the core loopback fix. The current renderer ignores the colors passed in and sets channels from `color_idx` (`music_light.py:858-861`), so palette variety never reaches the punchy path. Change its signature from `(..., color_1, color_2, volume, t, color_idx=0)` to `(..., color_1, color_2, accent, volume, t)` and replace the three `color_idx`-driven blocks in `dmx_engine.py`:

```python
    # ── DEEP BASS COMBO: accent blast ──
    if (is_kick or is_snare) and is_deep_bass:
        self.out_r, self.out_g, self.out_b = accent
        self.out_w = 255.0
        self.out_master = vb                 # vb = velocity_brightness(beat_velocity), Task 11
        self.out_strobe = 0
        self.beat_hold_frames = self.profile_deep_bass_hold
        return

    # ── NORMAL BEAT: palette color snap, white punch on kick ──
    if is_kick or is_snare:
        col = color_1 if is_kick else color_2
        self.out_r, self.out_g, self.out_b = col
        self.out_w = 255.0 if is_kick else 0.0
        self.out_master = vb
        self.out_strobe = 0
        self.beat_hold_frames = self.profile_beat_hold
        return
```
And the between-beats glow block (replacing the `color_idx` glow at `music_light.py:885-899`):
```python
        if bass_active:
            glow = (bass - self.profile_glow_thresh) * 0.4
            self.out_r, self.out_g, self.out_b = (c * glow for c in color_1)
            self.out_w = 0.0
            self.out_master = max(15, glow * 180)
        else:
            self.out_r, self.out_g, self.out_b, self.out_w = afterglow(
                self.out_r, self.out_g, self.out_b, self.out_w)
            self.out_master *= self.profile_decay_speed
```
> Keep the breathing-white-floor tail (`music_light.py:901-904`) unchanged. The loopback color-cycling helpers and `self.color_phase` are now dead — remove the `color_phase` increment from the loopback `_dispatch`; palette + phrase evolution replaces it.

- [ ] **Step 3: Add punch to the synced beat path**

In `dmx_engine.py`, in `_render_beat_reactive` and `_render_bass_white_blast` (the high-energy synced renderers), set master from velocity and apply afterglow on the non-beat frames, mirroring `_render_loopback_direct`:
```python
        from dmx_punch import velocity_brightness, afterglow
        beat_velocity = max(kick_i, snare_i)
        if is_kick or is_snare:
            self.out_master = velocity_brightness(beat_velocity)
            self.beat_hold_frames = getattr(self, "profile_beat_hold", 4)
        elif self.beat_hold_frames > 0:
            self.beat_hold_frames -= 1
            self.out_r, self.out_g, self.out_b, self.out_w = afterglow(
                self.out_r, self.out_g, self.out_b, self.out_w)
            self.out_master = max(self.out_master, 200.0)
```
> Apply only to the punchy renderers. Ambient renderers (`ocean_drift`, `candlelight`, `sunset_fade`, `aurora_shimmer`, `slow_breathe`, `static_wash`) must NOT get beat-hold — they are intentionally smooth.

- [ ] **Step 4: Verify variety + punch via dry-run**

Run: `set DMX_DRY_RUN=1 && .venv\Scripts\python ai_show_player.py --show shows/<id>/show.json`
Expected: within a single long cue, logged `frame=` colors **change across phrase boundaries** (texture evolution), and beat frames show master near 255 while off-beats decay (punch).

- [ ] **Step 5: Live verification — the real test**

Run a synced show and loopback on hardware.
Expected: loopback no longer cycles a fixed R→B→G→W — colors come from evolving palettes; synced now has punch (crisp beats, not mushy); both modes show evolving palettes within sections and fresh palettes across sections. A/B by ear against the pre-change loopback to confirm punch did not regress.

- [ ] **Step 6: Remove the temporary `[VARIETY]` debug log from Task 10 Step 4.**

- [ ] **Step 7: Commit**

```bash
git add dmx_engine.py music_light.py ai_show_player.py
git commit -m "feat: renderers consume VarietyEngine palette colors and shared punch"
```

---

## Task 13: LLM mood enrichment (Checkpoint 5)

Additive, backward-compatible: the LLM assigns a per-cue `mood`; missing values are repaired from energy.

**Files:**
- Modify: `llm_designer.py`
- Test: `tests/test_llm_repair.py`

- [ ] **Step 1: Write the failing test for the repair**

`tests/test_llm_repair.py`:
```python
from llm_designer import _default_mood_for_energy

def test_default_mood_for_energy_bands():
    assert _default_mood_for_energy(1) == "warm"
    assert _default_mood_for_energy(4) == "cool"
    assert _default_mood_for_energy(7) == "neon"
    assert _default_mood_for_energy(10) == "euphoric"
```

- [ ] **Step 2: Run, verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_llm_repair.py -q`
Expected: FAIL (`ImportError: _default_mood_for_energy`).

- [ ] **Step 3: Implement the helper + wire it into `_validate_and_repair_plan`**

In `llm_designer.py`:
```python
def _default_mood_for_energy(energy):
    """Fallback mood when the LLM omits one. Matches loopback's energy→mood map."""
    e = int(energy or 0)
    if e <= 3:
        return "warm"
    if e <= 6:
        return "cool"
    if e <= 8:
        return "neon"
    return "euphoric"
```
In `_validate_and_repair_plan`, for each cue:
```python
        if not cue.get("mood"):
            cue["mood"] = _default_mood_for_energy(cue.get("energy_level", 5))
```

- [ ] **Step 4: Run, verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_llm_repair.py -q`
Expected: `1 passed`

- [ ] **Step 5: Add `mood` to the prompt schema**

In the `=== REQUIRED JSON FORMAT ===` block of the prompt, add `"mood": "warm|cool|neon|euphoric|dark"` to the cue object, and add one line under `=== HARD CONSTRAINTS ===`:
```
10. Assign a "mood" to every cue. Adjacent cues SHOULD use contrasting moods.
```

- [ ] **Step 6: Run the full suite**

Run: `.venv\Scripts\python -m pytest -q`
Expected: all tests pass across the suite (`test_intent`, `test_palettes`, `test_variety_engine`, `test_punch`, `test_llm_repair`, `test_smoke`).

- [ ] **Step 7: End-to-end verification**

Generate a fresh show (`POST /api/shows/generate` or run `youtube_analyzer.py <url>`), confirm the resulting `show.json` cues carry a `mood`, then play it. Expected: per-section moods drive the palette families; richer, contrasting color story.

- [ ] **Step 8: Commit**

```bash
git add llm_designer.py
git add -f tests/test_llm_repair.py
git commit -m "feat: LLM assigns per-cue mood; repair defaults from energy"
```

---

## Self-review notes (coverage map)

| Goal / spec section | Where it's actually implemented |
|---|---|
| Shared-module extraction | 2, 3 |
| VarietyEngine: palettes | 6 |
| Colors repeat (#1) — loopback | **Task 12 Step 2** (rewire `_render_loopback_direct` to palette colors) + anti-repeat (7) |
| Colors repeat (#1) — synced | 7 (anti-repeat) + 12 Step 1 (renderer reads palette) |
| Same texture all section (#2) | 8 (phase), 9 (tick), 12 (renderer reads `current_colors`) |
| Every song looks alike (#3) | 7 (seed mechanism) + **Task 10 Step 1** (`set_song_seed` actually called from show identity) |
| Loopback feels static (#4) | **Task 10 Step 2** (`seconds_in_section` ≥ 16 s forces a new palette) |
| Intent contract (+ `seed_color`) | 5, 10 |
| LLM color seeds palette (decision) | 7 (`_nearest_palette`) + 10 Step 3 (`seed_color=cue["color_1"]`) |
| Renderer changes + shared punch | 11, 12 |
| LLM mood enrichment | 13 |
| Explicit BPM (no silent fallback) | 2 (`self.beats_per_sec`), 10 Step 1 (`self.show_bpm`) |
| Error handling (relax, mood/bpm fallback) | 7, 9, 13 |
| Testing / dry-run harness | 1, 4 |
| Invariants preserved (USB dispose, WASAPI terminate, watchdog) | 2, 3 (verbatim) |

**Open follow-ups (not in scope):** per-song seed for loopback (no track boundaries — loopback uses time-evolution instead); a loopback dry-run harness (needs live audio).

---

## Amendments applied (2026-06-13 self-review pass)

This plan was revised after a self-review caught gaps where the original would not have delivered the goals:
1. **`_render_loopback_direct` rewired to palette colors** (Task 12 Step 2) — without this, loopback's hardcoded R→B→G→W survived and pain point #1 was unfixed on the punchy path.
2. **`set_song_seed` is now actually called** (Task 10 Step 1) — per-song identity (#3) was previously implemented but never invoked.
3. **Loopback time-evolution wired** (Task 10 Step 2) — `seconds_in_section` now forces palette rotation at constant energy, fixing #4 (was reported by `tick` but unused).
4. **LLM color seeds the palette** (Tasks 5/7/10) — `Intent.seed_color` + `_nearest_palette` resolve the contradiction of discarding LLM colors while still asking for them.
5. **`.gitignore` `!tests/` negation** (Task 1) — test files would otherwise be silently skipped by `git add`.
6. **Explicit `self.beats_per_sec` / `self.show_bpm`** (Tasks 2/10) — replaced silent `getattr(..., 0.0)` fallbacks that would have disabled beat-grid phrasing without warning.
7. **Removed dead `recent_behaviors`; rewrote the flaky `test_different_seeds_diverge`** (Task 7).

# Variety Go-Live + `golden_anthem` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the inert VarietyEngine actually drive both playback modes (palette variety, per-song identity, loopback evolution, unified punch), then add the `golden_anthem` renderer on top of the live variety layer.

**Architecture:** Phase 1 (Tasks 1–6) wires `dmx_variety.VarietyEngine` into the two `_dispatch` directors via the `Intent` contract and unifies punch via `dmx_punch`, ending at a **hard hardware checkpoint the user must verify before Phase 2**. Phase 2 (Tasks 7–9) adds `golden_anthem` — an accumulated-phase gold-swell renderer that is seek-proof by construction — with pytest coverage.

**Tech Stack:** Python 3.13 (`.venv`), existing `dmx_variety`/`dmx_punch` modules (built + 14 tests passing), pytest (22 existing tests must stay green), `zlib.crc32` for process-stable seeding. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-04-variety-golive-golden-anthem-design.md`
**Branch:** `feature/variety-wiring-golden-anthem`

---

## Non-negotiable execution rules

1. **Re-verify every anchor before editing.** Line numbers below were verified on 2026-07-04 at commit `abf6af8`, but this codebase has burned us twice with stale anchors. Before each Edit: Grep for the quoted "Find" text and confirm it matches the live file. If it doesn't match exactly, STOP and report — do not improvise.
2. **Full suite green at every commit:** `.venv\Scripts\python.exe -m pytest -q` → currently `22 passed`; counts grow as tasks add tests. Never commit red.
3. Each task = one commit (Conventional Commits). End every commit message with:
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
4. Do not stage `playback_state.json`, `current_show.json`, `.superpowers/`, `PROJECT_ARCHITECTURE.md`, or `PR_DESCRIPTION.md`.
5. Windows: invoke Python as `.venv\Scripts\python.exe` from the repo root `D:\AIProjects\DMX`.

## Verified anchor map (2026-07-04, commit `abf6af8`)

| Anchor | Location |
|---|---|
| `ABYSSAL_*` constants block starts | `dmx_engine.py:51` |
| `VALID_BEHAVIORS` (engine) | `dmx_engine.py:88` |
| Base `__init__`: `beats_per_sec`/`_beat_velocity` | `dmx_engine.py:170-171` |
| Base `__init__`: `show_bpm = 0.0` | `dmx_engine.py:177` |
| Base `__init__`: variety block (`variety`, `_last_section_id`, `_evolution_secs`, `loopback_ambient`) | `dmx_engine.py:194-198` |
| `_behavior_map` | `dmx_engine.py:223` |
| `_render_bass_white_blast` (velocity-dilution bug: EMA at `:436` overwrites `velocity_master` from `:422/:426`) | `dmx_engine.py:415-437` |
| `_render_beat_reactive` (no velocity/hold yet) | `dmx_engine.py:536-567` |
| `load_ai_show` (internal cue dict incl. `"mood"` at `:868`; returns `audio_file` at `:874`) | `dmx_engine.py:830-877` |
| `process_audio` → `self._dispatch(...)` call | `dmx_engine.py:894` / `:1033` |
| Loopback subclass: `beat_hold_frames = 0` | `music_light.py:47` |
| `_render_loopback_direct` (**in the subclass**, `color_idx`-driven) | `music_light.py:175-273` |
| Loopback `_dispatch` (color-phase cycling `:283-307`, palette `:310`, ambient set `:324-326`, direct call passes `color_phase` `:336-341`) | `music_light.py:275-341` |
| Synced `_dispatch` (16-beat rotation `:21-22`, cue color sourcing `:27-34`) | `ai_show_player.py:16-38` |
| `run_synced_mode` (insert dry-run branch after the `[SYNCED] Playing:` logging, before `wave_mod.open`) | `ai_show_player.py:40-59` |
| `save_show` (does **not** persist `song_metrics` today) | `youtube_analyzer.py` (`def save_show`), call site near `:546-553` |
| LLM `VALID_BEHAVIORS` | `llm_designer.py:447` |
| VarietyEngine API: `set_song_seed(seed)`, `begin_section(intent)→palette`, `current_colors()→(c1,c2,accent)`, `tick(is_beat,bpm,t)→{'phrase_boundary','seconds_in_section'}` | `dmx_variety.py` |
| Punch API: `velocity_brightness(v)→120..255`, `afterglow(r,g,b,w)→tuple` | `dmx_punch.py` |

---

# PHASE 1 — VarietyEngine go-live

## Task 1: Dry-run harness (`DMX_DRY_RUN`)

Offline verification channel for everything that follows: skip USB, record frames, fast-iterate a synced show without audio output.

**Files:**
- Modify: `dmx_engine.py` (module flag, `_init_hardware`, `send_dmx`, `__init__`)
- Modify: `ai_show_player.py` (`run_synced_mode` dry-run branch)

- [ ] **Step 1: Add the module flag.** In `dmx_engine.py`, directly under the existing `logger = logging.getLogger(__name__)` line near the top:

```python
# Offline verification: DMX_DRY_RUN=1 skips USB init and records frames on
# self.last_frame instead of transferring. Used by the dry-run playback branch.
DRY_RUN = os.getenv("DMX_DRY_RUN") == "1"
```

- [ ] **Step 2: Record instead of send.** Grep `def _init_hardware` and `def send_dmx` in `dmx_engine.py`. Insert as the FIRST lines of each body:

In `_init_hardware`:
```python
        if DRY_RUN:
            logger.info("[DRY-RUN] Skipping uDMX init; frames recorded, not sent.")
            self.dev = None
            return
```
In `send_dmx`:
```python
        if DRY_RUN:
            self.last_frame = (int(master), int(red), int(green), int(blue), int(white), int(strobe))
            return
```

- [ ] **Step 3: Initialize the recording attr.** In the base `__init__` (near `dmx_engine.py:178`, after `self.audio_file = None`):
```python
        self.last_frame = None             # populated only in DRY_RUN mode
```

- [ ] **Step 4: Dry-run branch in `run_synced_mode`.** In `ai_show_player.py`, after the `[SYNCED] ... cues, behaviors:` logging block (anchor `:50-52`) and BEFORE `wf = wave_mod.open(audio_path, 'rb')`, insert:

```python
        from dmx_engine import DRY_RUN
        if DRY_RUN:
            wf = wave_mod.open(audio_path, 'rb')
            sample_rate = wf.getframerate()
            frames_played = 0
            last_cue = None
            data = wf.readframes(BLOCK_SIZE)
            while data:
                elapsed = frames_played / sample_rate
                self.process_audio(data, elapsed_seconds=elapsed,
                                   input_format="int16", actual_sample_rate=sample_rate)
                cue = self._get_active_cue(elapsed)
                name = cue["name"] if cue else None
                if name != last_cue:
                    last_cue = name
                    logger.info(f"[DRY {elapsed:6.1f}s] cue={name} "
                                f"palette={self.variety.current_palette['id']} "
                                f"frame={self.last_frame}")
                frames_played += BLOCK_SIZE
                data = wf.readframes(BLOCK_SIZE)
            wf.close()
            logger.info("[DRY-RUN] Completed synced pass.")
            return
```

- [ ] **Step 5: Verify offline.** Run:
`set DMX_DRY_RUN=1 && .venv\Scripts\python.exe ai_show_player.py --show "shows/Ava_Max_-_So_Am_I_Official_Music_Video/show.json"`
Expected: `[DRY-RUN] Skipping uDMX init`, then one `[DRY ...s] cue=... palette=... frame=(...)` line per cue, ending `[DRY-RUN] Completed synced pass.` — no USB errors, no audio device opened, no tracebacks. (Palette id will be static until Task 3 — that's expected.)

- [ ] **Step 6: Suite + commit.**
`.venv\Scripts\python.exe -m pytest -q` → `22 passed`.
```bash
git add dmx_engine.py ai_show_player.py
git commit -m "feat: add DMX_DRY_RUN offline harness (skip USB, record frames, fast synced pass)"
```

## Task 2: Per-song identity — process-stable seed + real BPM

**Files:**
- Modify: `youtube_analyzer.py` (persist `song_metrics` into show.json)
- Modify: `dmx_engine.py` (`load_ai_show` reads bpm + seeds; `import zlib`)
- Test: `tests/test_show_identity.py` (new)

- [ ] **Step 1: Write the failing test** — `tests/test_show_identity.py`:

```python
import json
import os
import tempfile

from dmx_engine import DmxEngineBase
from dmx_variety import Intent


def _write_show(dirpath, name="Test Song"):
    path = os.path.join(dirpath, "show.json")
    show = {
        "audio_file": "x.wav",
        "song_metrics": {"bpm": 128.0},
        "lighting_plan": {
            "show_name": name,
            "cues": [{
                "start_time": 0, "end_time": 10,
                "color_1": [255, 0, 120], "color_2": [0, 220, 255],
                "energy_level": 8, "strobe_allowed": False,
                "behavior": "beat_reactive", "master_dimmer_percent": 80,
                "fade_speed_seconds": 1.0, "section_name": "A", "mood": "neon",
            }],
            "phrases": [],
        },
    }
    with open(path, "w") as f:
        json.dump(show, f)
    return path


def _palette_run(show_path, n=4):
    engine = DmxEngineBase()
    engine.load_ai_show(show_path)
    assert engine.show_bpm == 128.0, "bpm must be read from the show file"
    return [engine.variety.begin_section(
                Intent(energy=8, mood=None, section_id=str(i), is_new_section=True)
            )["id"] for i in range(n)]


def test_same_show_same_palette_sequence_across_processes():
    # zlib.crc32 seeding: two fresh engines (simulating two worker processes)
    # must produce the identical palette sequence for the same show. Python's
    # builtin hash() would fail this across real processes (per-process salt).
    with tempfile.TemporaryDirectory() as d:
        p = _write_show(d)
        assert _palette_run(p) == _palette_run(p)


def test_different_shows_can_diverge():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = _palette_run(_write_show(d1, name="Song A"), n=6)
        b = _palette_run(_write_show(d2, name="Song B"), n=6)
        assert a != b
```

- [ ] **Step 2: Run to verify it fails.**
`.venv\Scripts\python.exe -m pytest tests/test_show_identity.py -q`
Expected: FAIL on `engine.show_bpm == 128.0` (stays `0.0` — nothing sets it yet).

- [ ] **Step 3: Persist `song_metrics` for future shows.** In `youtube_analyzer.py`, change `def save_show(ai_plan, audio_filepath):` to:

```python
def save_show(ai_plan, audio_filepath, song_metrics=None):
    """
    Saves the AI-generated lighting plan, audio file path, and song metrics
    (bpm etc.) to current_show.json so the player can load them in sync.
    """
    show_data = {
        "audio_file": os.path.abspath(audio_filepath),
        "song_metrics": song_metrics or {},
        "lighting_plan": ai_plan
    }
```
(keep the rest of the function body unchanged), and update its call site (grep `save_show(ai_plan`) to:
```python
        save_show(ai_plan, audio_file, telemetry.get("song_metrics"))
```
Old shows without the key fall back to bpm 0.0 → VarietyEngine's 4s time-phrasing. No migration needed.

- [ ] **Step 4: Read bpm + seed in `load_ai_show`.** In `dmx_engine.py`: add `import zlib` to the module imports. Then inside `load_ai_show`, directly after the line `plan = data.get("lighting_plan", {})` (anchor near `:840`):

```python
            # Per-song identity. NOT builtin hash(): Python salts string hashes
            # per process, and every playback is a fresh worker process, so
            # hash() would silently break same-song-same-look replay
            # determinism. zlib.crc32 is stdlib and process-stable.
            metrics = data.get("song_metrics", {}) or plan.get("song_metrics", {})
            self.show_bpm = float(metrics.get("bpm", 0.0) or 0.0)
            seed_basis = plan.get("show_name") or data.get("audio_file") or ""
            self.variety.set_song_seed(zlib.crc32(seed_basis.encode("utf-8")))
```

- [ ] **Step 5: Run to verify it passes.**
`.venv\Scripts\python.exe -m pytest tests/test_show_identity.py -q` → `2 passed`. Full suite → `24 passed`.

- [ ] **Step 6: Commit.**
```bash
git add youtube_analyzer.py dmx_engine.py tests/test_show_identity.py
git commit -m "feat: process-stable per-song seed (zlib.crc32) + persist/read show bpm"
```

## Task 3: Synced director — Intent wiring

**Files:**
- Modify: `ai_show_player.py` (`_dispatch` replaced; import `Intent`)

- [ ] **Step 1: Add the import.** In `ai_show_player.py`, after `from dmx_engine import DmxEngineBase, BLOCK_SIZE`:
```python
from dmx_variety import Intent
```

- [ ] **Step 2: Replace `_dispatch` wholesale.** Grep `def _dispatch` in `ai_show_player.py` (anchor `:16-38`) and replace the entire method with:

```python
    def _dispatch(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                  kick_i, snare_i, hihat_i, mid_i,
                  is_kick, is_snare, volume, sr, elapsed_seconds=None):
        t = elapsed_seconds

        cue = self._get_active_cue(elapsed_seconds) if self.synced_cues else None
        if cue:
            behavior = cue.get("behavior", "beat_reactive")
            section_id = cue.get("name", "")
            energy = int(cue.get("energy", 5))
            mood = cue.get("mood")
            strobe_allowed = bool(cue.get("strobe", False))
            seed_color = list(cue.get("color_1", (255, 255, 255)))
        else:
            behavior, section_id = "beat_reactive", "_fallback"
            energy, mood, strobe_allowed, seed_color = 7, None, False, None

        # Variety layer: LLM cue supplies intent (energy/mood/color seed); the
        # engine owns final color, anti-repeat, and phrase-grid texture.
        self.variety.tick(is_beat=(is_kick or is_snare), bpm=self.show_bpm, t=t)
        if section_id != self._last_section_id:
            self._last_section_id = section_id
            self.variety.begin_section(Intent(
                energy=energy, mood=mood, section_id=section_id,
                is_new_section=True, bpm=self.show_bpm,
                strobe_allowed=strobe_allowed, seed_color=seed_color))
        kick_color, accent_color, _accent = self.variety.current_colors()

        renderer = self._behavior_map.get(behavior, self._render_beat_reactive)
        renderer(kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                 kick_color, accent_color, volume, cue or {"energy": 7, "dimmer": 80}, t)
```
Note this deliberately deletes the old 16-beat `current_palette_idx` rotation and the `cue["color_1"]`-as-paint sourcing — LLM `color_1` now *seeds* the palette family instead (spec decision).

- [ ] **Step 3: Verify via dry-run.**
`set DMX_DRY_RUN=1 && .venv\Scripts\python.exe ai_show_player.py --show "shows/Ava_Max_-_So_Am_I_Official_Music_Video/show.json"`
Expected: each `[DRY ...]` cue line now shows a `palette=` id that **changes across cues** (anti-repeat) — no two adjacent cues with the same palette id. Run it TWICE: the palette sequence must be **identical across both runs** (seeding). No tracebacks.

- [ ] **Step 4: Suite + commit.** Full suite → `24 passed`.
```bash
git add ai_show_player.py
git commit -m "feat: synced director drives VarietyEngine via Intent (LLM color seeds palette)"
```

## Task 4: Loopback director + `_render_loopback_direct` palette rewire

One commit — the call site and the renderer signature must change together to stay compilable.

**Files:**
- Modify: `music_light.py` (`_dispatch`, `_render_loopback_direct`, `__init__` dead-state cleanup, import `Intent`)

- [ ] **Step 1: Import.** After `from dmx_engine import (...)` in `music_light.py`:
```python
from dmx_variety import Intent
```

- [ ] **Step 2: Replace `_dispatch`.** Grep `def _dispatch` in `music_light.py` (anchor `:275-341`) and replace the whole method with:

```python
    def _dispatch(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                  kick_i, snare_i, hihat_i, mid_i,
                  is_kick, is_snare, volume, sr, elapsed_seconds=None):
        current_time = time.time()
        beats_per_sec = self.beats_per_sec
        t = self.frame_counter * BLOCK_SIZE / sr

        # ── Variety layer: energy state acts as the "section"; a stable state
        # still evolves after _evolution_secs so loopback never goes static. ──
        bpm = beats_per_sec * 60.0
        mood = {"calm": "warm", "building": "cool",
                "high": "neon", "dropping": "euphoric"}.get(self.energy_state, "cool")
        energy = {"calm": 2, "building": 5, "high": 8, "dropping": 6}.get(self.energy_state, 5)

        ev = self.variety.tick(is_beat=(is_kick or is_snare), bpm=bpm, t=t)
        force_evolve = ev["seconds_in_section"] >= self._evolution_secs
        if self.energy_state != self._last_section_id or force_evolve:
            self._last_section_id = self.energy_state
            self.variety.begin_section(Intent(
                energy=energy, mood=mood, section_id=self.energy_state,
                is_new_section=True, bpm=bpm, strobe_allowed=True))
        kick_color, accent_color, combo_color = self.variety.current_colors()

        # ── Auto-behavior detection: picks chill or punchy ──
        auto_behavior = self._detect_auto_behavior(volume, kick_i, snare_i, current_time, beats_per_sec)
        self.current_behavior = auto_behavior

        # Write IPC state every ~50 frames (~1s) so the UI shows the sub-mode
        if self.frame_counter % 50 == 0:
            self.playback_state = "loopback"
            self.current_cue_name = f"{self.energy_state} | {auto_behavior}"
            self._write_playback_state()

        # Ambient/chill behaviors → use the standard renderer dispatch
        ambient_behaviors = {"ocean_drift", "candlelight", "sunset_fade",
                             "aurora_shimmer", "abyssal_bloom", "slow_breathe",
                             "static_wash", "buildup_ramp", "rainbow_sweep"}

        if auto_behavior in ambient_behaviors:
            cue = {"dimmer": 50, "energy": 3, "start": 0, "end": 60,
                    "strobe": False, "fade": 3.0}
            renderer = self._behavior_map.get(auto_behavior, self._render_beat_reactive)
            renderer(kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                     kick_color, accent_color, volume, cue, t)
        else:
            # Punchy behaviors → loopback direct renderer (palette-driven)
            self._render_loopback_direct(
                kick_mag, snare_mag, mid_mag, hihat_mag,
                kick_i, snare_i, hihat_i, mid_i,
                is_kick, is_snare,
                kick_color, accent_color, combo_color, volume, t)
```
This removes the whole R→B→G→W `color_phase` cycling block and the static palette lookup.

- [ ] **Step 3: Rewire `_render_loopback_direct`.** At `music_light.py:175`, change the signature from `(..., color_1, color_2, volume, t, color_idx=0)` to:
```python
    def _render_loopback_direct(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                                kick_i, snare_i, hihat_i, mid_i,
                                is_kick, is_snare,
                                color_1, color_2, accent, volume, t):
```
Then replace the three `color_idx`-driven blocks inside it (grep each "Find" text to anchor):

**Deep-bass combo** (Find: `combo = color_idx % 4` — replace that whole if/elif chain and the channel writes with):
```python
            self.out_r, self.out_g, self.out_b = accent
            self.out_w = 255.0
```
(keep the following `self.out_master = velocity_brightness` / `self.out_strobe = 0` / `self.beat_hold_frames = self.profile_deep_bass_hold` / `return` lines unchanged).

**Normal beat** (Find: `self.out_r = 255.0 if color_idx == 0 else 0` — replace the four `color_idx` channel lines with):
```python
            col = color_1 if is_kick else color_2
            self.out_r, self.out_g, self.out_b = col
            self.out_w = 255.0 if is_kick else 0.0
```

**Between-beats glow** (Find: `self.out_r = 255.0 * glow if color_idx == 0 else 0` — replace the four `color_idx` lines with):
```python
            self.out_r = color_1[0] * glow
            self.out_g = color_1[1] * glow
            self.out_b = color_1[2] * glow
            self.out_w = 0.0
```
Keep the hold/afterglow tail and the breathing white floor untouched.

- [ ] **Step 4: Delete dead color-cycling state.** In `music_light.__init__`, grep and delete these now-unused lines (verify no remaining references first with grep): `self.prev_bps = 0.0`, `self.bps_check_time = 0.0`, `self.color_phase = 0`, `self.last_color_change = 0.0`. Leave the `profile_color_cycle_*` / `profile_rhythm_change_pct` attributes and their `load_profile` keys in place (profile-file compatibility — accepted but unused).

- [ ] **Step 5: Verify.** Parse + import + instantiation:
`.venv\Scripts\python.exe -c "import music_light; e = music_light.DMXEngine(); print('ok', e._evolution_secs)"` → `ok 16.0`.
Grep `color_idx|color_phase` in `music_light.py` → **zero hits** (all retired).
Full suite → `24 passed`.

- [ ] **Step 6: Commit.**
```bash
git add music_light.py
git commit -m "feat: loopback director drives VarietyEngine; palette-colored direct renderer (retire R->B->G->W)"
```

## Task 5: Punch unification + `bass_white_blast` velocity-dilution fix

**Files:**
- Modify: `dmx_engine.py` (`__init__` hold-state defaults, both punchy renderers, punch imports)
- Modify: `music_light.py` (remove duplicate `beat_hold_frames` init)
- Test: `tests/test_punch_wiring.py` (new)

- [ ] **Step 1: Write the failing test** — `tests/test_punch_wiring.py`:

```python
from dmx_engine import DmxEngineBase


def _engine():
    e = DmxEngineBase()
    e._beat_velocity = 1.0
    return e


def test_bass_white_blast_kick_master_owned_by_velocity():
    # The dilution bug: a trailing unconditional EMA used to overwrite the
    # velocity master on the SAME frame. On a kick with velocity 1.0 and
    # dimmer 0.8, master must be exactly 255 * 0.8.
    e = _engine()
    cue = {"energy": 7, "dimmer": 80}
    e._render_bass_white_blast(1.0, 0, 0, 0, True, False,
                               (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    assert e.out_master == 255.0 * 0.8

def test_bass_white_blast_soft_kick_dimmer_floor():
    e = _engine()
    e._beat_velocity = 0.0
    cue = {"energy": 7, "dimmer": 80}
    e._render_bass_white_blast(0.2, 0, 0, 0, True, False,
                               (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    assert e.out_master == 120.0 * 0.8

def test_beat_reactive_beat_master_owned_by_velocity():
    e = _engine()
    cue = {"energy": 7, "dimmer": 80}
    e._render_beat_reactive(1.0, 0, 0, 0, True, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    assert e.out_master == 255.0 * 0.8

def test_beat_hold_keeps_master_bright():
    e = _engine()
    cue = {"energy": 7, "dimmer": 80}
    e._render_beat_reactive(1.0, 0, 0, 0, True, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    e._render_beat_reactive(0.0, 0, 0, 0, False, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.02)
    assert e.out_master >= 200.0 * 0.8   # hold frame stays bright
    assert e.beat_hold_frames == e.profile_beat_hold - 1
```

- [ ] **Step 2: Run to verify it fails.**
`.venv\Scripts\python.exe -m pytest tests/test_punch_wiring.py -q`
Expected: FAIL — first test gets an EMA-diluted master (≈171 after one frame, not 204), and `DmxEngineBase` has no `profile_beat_hold`.

- [ ] **Step 3: Base hold-state defaults.** In `dmx_engine.py` `__init__`, after `self.profile_kick_dominance_ratio = 1.5` (anchor `:193`):
```python
        # Beat-hold shared by the punchy renderers (loopback overrides via profile)
        self.profile_beat_hold = 4
        self.beat_hold_frames = 0
```
In `music_light.py` `__init__`, delete the now-duplicate `self.beat_hold_frames = 0` (anchor `:47`) — the subclass's `profile_beat_hold` override stays.

- [ ] **Step 4: Punch import.** Add to `dmx_engine.py` module imports:
```python
from dmx_punch import velocity_brightness, afterglow
```

- [ ] **Step 5: Fix `_render_bass_white_blast`.** Replace the whole method (anchor `:415-437`) with:

```python
    def _render_bass_white_blast(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                                 kick_color, accent_color, volume, cue, t):
        """WHITE LED blasts on every kick. Colored wash underneath from mids."""
        energy = cue.get("energy", 7) if cue else 7
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0
        energy_scale = 0.5 + (energy / 10.0)  # 0.6 to 1.5

        wash_brightness = max(mid_i * 0.4 * energy_scale, 0.1)
        snare_boost = 0.6 * energy_scale if is_snare else 0.0
        self.out_r = ema(self.out_r, kick_color[0] * wash_brightness + accent_color[0] * snare_boost, 0.3, 0.08)
        self.out_g = ema(self.out_g, kick_color[1] * wash_brightness + accent_color[1] * snare_boost, 0.3, 0.08)
        self.out_b = ema(self.out_b, kick_color[2] * wash_brightness + accent_color[2] * snare_boost, 0.3, 0.08)

        if is_kick:
            # Beat frame: velocity OWNS master. (Previously a trailing
            # unconditional EMA overwrote this on the same frame -- the
            # velocity-dilution bug found in review.)
            self.out_w = 255.0 * dimmer
            self.out_master = velocity_brightness(self._beat_velocity) * dimmer
            self.beat_hold_frames = self.profile_beat_hold
        elif self.beat_hold_frames > 0:
            self.beat_hold_frames -= 1
            self.out_w *= 0.80
            self.out_master = max(self.out_master, 200.0 * dimmer)
        else:
            self.out_w = ema(self.out_w, 0, 0, 0.35)
            self.out_master = ema(self.out_master, max(120.0 * dimmer, volume * 4000), 0.5, 0.15)
        self.out_strobe = 0
```

- [ ] **Step 6: Add punch to `_render_beat_reactive`.** Replace its EMA/output tail — from `is_beat = is_kick or is_snare` through the `self.out_master = ema(...)` line (anchor `:553-561`) — with:

```python
        is_beat = is_kick or is_snare
        att = 0.95 if is_beat else 0.15
        dec = 0.25 if (kick_i > 0.1 or snare_i > 0.1) else 0.06

        self.out_r = ema(self.out_r, tr, att, dec)
        self.out_g = ema(self.out_g, tg, att, dec)
        self.out_b = ema(self.out_b, tb, att, dec)
        self.out_w = ema(self.out_w, tw, 0.9 if is_kick else 0.3, 0.25)

        if is_beat:
            self.out_master = velocity_brightness(self._beat_velocity) * dimmer
            self.beat_hold_frames = self.profile_beat_hold
        elif self.beat_hold_frames > 0:
            self.beat_hold_frames -= 1
            self.out_r, self.out_g, self.out_b, self.out_w = afterglow(
                self.out_r, self.out_g, self.out_b, self.out_w)
            self.out_master = max(self.out_master, 200.0 * dimmer)
        else:
            self.out_master = ema(self.out_master, tm, 0.4, 0.12)
```
Keep the trailing strobe block unchanged. Ambient renderers get NO changes (intentionally smooth).

- [ ] **Step 7: Run to verify pass + no regression.**
`.venv\Scripts\python.exe -m pytest tests/test_punch_wiring.py -q` → `4 passed`. Full suite → `28 passed`.
Dry-run once more (Task 3 command): still clean, frames on beat-heavy cues show master tracking velocity.

- [ ] **Step 8: Commit.**
```bash
git add dmx_engine.py music_light.py tests/test_punch_wiring.py
git commit -m "feat: unify punch via dmx_punch; fix bass_white_blast velocity dilution"
```

## Task 6: Phase-1 wrap — abyssal_bloom inheritance check + HARD CHECKPOINT

**Files:** none (verification only)

- [ ] **Step 1: abyssal_bloom inherits variety (no code change expected).** Dry-run a show containing an `abyssal_bloom` cue (hand-write one with the Task 2 test's JSON shape if no library show has one, pointing at a real WAV under `youtube_audio/`). Expected: its bloom tint follows the section palette's accent (visible in `frame=` colors), because `_render_abyssal_bloom` already blends from `accent_color`. If a code change seems needed — STOP and report; the spec says none is.

- [ ] **Step 2: Full suite** → `28 passed`. `git status --short` shows no unstaged code changes.

- [ ] **Step 3: STOP — hand back for the hardware checkpoint.** Report to the user; do NOT begin Task 7. The user verifies on the fixture:
  - Loopback: palettes evolve (no fixed R/B/G/W); a stable-energy passage rotates look after ~16s; punch feel unchanged.
  - Synced: sections palette-tinted; same song → identical look across two replays; `beat_reactive`/`bass_white_blast` crisp.
  - If punch feels wrong: `git revert` the Task 5 commit only (independent of Tasks 3–4).

---

# PHASE 2 — `golden_anthem` (only after the user passes the checkpoint)

## Task 7: `golden_anthem` renderer (TDD) + engine registration

**Files:**
- Modify: `dmx_engine.py` (ANTHEM constants, `_anthem_envelope`, `_ga_*` state, method, `_behavior_map`, `VALID_BEHAVIORS`)
- Test: `tests/test_golden_anthem.py` (new)

- [ ] **Step 1: Write the failing tests** — `tests/test_golden_anthem.py`:

```python
from dmx_engine import (DmxEngineBase, _anthem_envelope,
                        ANTHEM_CREST_BASE, ANTHEM_CREST_GAIN, ANTHEM_CREST_MAX)


def test_envelope_bounded_and_shaped():
    vals = [_anthem_envelope(i / 200.0) for i in range(200)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert abs(_anthem_envelope(0.0)) < 1e-9          # starts dark
    assert _anthem_envelope(0.45) == 1.0              # crest hold (0.4..0.5)
    assert _anthem_envelope(0.999) < 0.01             # returns to ~0


def test_crest_lift_is_hard_capped():
    assert min(ANTHEM_CREST_MAX, ANTHEM_CREST_BASE + 1.0 * ANTHEM_CREST_GAIN) == ANTHEM_CREST_MAX
    assert ANTHEM_CREST_BASE + 0.0 * ANTHEM_CREST_GAIN == ANTHEM_CREST_BASE


def test_discontinuity_advances_one_nominal_frame():
    # Accumulated-phase design: a 60s time jump (seek / deselected-and-back)
    # must advance the swell by ~one frame, never fast-forward whole cycles.
    e = DmxEngineBase()
    cue = {"dimmer": 60}
    e._render_golden_anthem(0, 0, 0, 0, False, False,
                            (255, 0, 120), (0, 220, 255), 0.0, cue, 10.0)
    before = e._ga_phase
    e._render_golden_anthem(0, 0, 0, 0, False, False,
                            (255, 0, 120), (0, 220, 255), 0.0, cue, 70.0)
    assert e._ga_phase - before < 0.01


def test_floor_keeps_light_alive_at_zero_dimmer():
    e = DmxEngineBase()
    cue = {"dimmer": 0}
    for i in range(60):
        e._render_golden_anthem(0, 0, 0, 0, False, False,
                                (255, 0, 120), (0, 220, 255), 0.0, cue, i * 0.012)
    assert e.out_master > 0.0                          # never fully dark
    assert e.out_strobe == 0


def test_registered_in_engine():
    e = DmxEngineBase()
    import dmx_engine
    assert "golden_anthem" in dmx_engine.VALID_BEHAVIORS
    assert e._behavior_map["golden_anthem"] == e._render_golden_anthem
```

- [ ] **Step 2: Run to verify failure.**
`.venv\Scripts\python.exe -m pytest tests/test_golden_anthem.py -q` → FAIL (`ImportError: _anthem_envelope`).

- [ ] **Step 3: Constants.** In `dmx_engine.py`, directly after the ABYSSAL block (grep its last line — the multi-line `ABYSSAL_DISCONTINUITY_THRESHOLD` comment):

```python
# golden_anthem renderer tuning. Accumulated-phase swell: phase only advances
# by a clamped per-frame dt, so seeks/re-entry cannot corrupt it by design.
ANTHEM_BASE_PERIOD = 12.0     # seconds per swell at zero music energy
ANTHEM_MIN_PERIOD = 8.0       # loud passages swell faster, never below this
ANTHEM_CREST_BASE = 0.55      # crest brightness at zero music energy
ANTHEM_CREST_GAIN = 0.30      # how much sustained energy lifts the crest
ANTHEM_CREST_MAX = 0.85       # hard cap
ANTHEM_FLOOR_MIN = 0.10       # never-dark amber floor (not dimmer-scaled)
ANTHEM_GOLD = (255, 190, 80)  # identity color; palette color_1 blends toward it
ANTHEM_DT_CLAMP = 0.1         # max seconds of phase advance per frame
ANTHEM_DISCONTINUITY_THRESHOLD = 1.0  # bigger call-gap => treat as one nominal frame
```

And the module-level envelope function (place near the other pure helpers, e.g. after `gamma_correct` — grep `def gamma_correct`):

```python
def _anthem_envelope(phase):
    """golden_anthem swell shape over one 0..1 cycle: smoothstep rise (40%),
    crest hold (10%), smoothstep fall (50%)."""
    if phase < 0.4:
        p = phase / 0.4
        return p * p * (3.0 - 2.0 * p)
    if phase < 0.5:
        return 1.0
    p = 1.0 - (phase - 0.5) / 0.5
    return p * p * (3.0 - 2.0 * p)
```

- [ ] **Step 4: State.** In `__init__`, after the `_ab_last_render_t` line (anchor `:211`):
```python
        # golden_anthem renderer state
        self._ga_phase = 0.0               # 0..1 position in the swell cycle
        self._ga_energy = 0.0              # smoothed music energy (volume+mids EMA)
        self._ga_last_render_t = None      # last t this renderer was called with
```

- [ ] **Step 5: The renderer.** Add after `_render_abyssal_bloom` (grep its final `self.out_strobe = 0`):

```python
    def _render_golden_anthem(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Majestic gold swells cresting into white-gold shimmer -- the
        "hands in the air during the anthem" moment. Rides the music: a slow
        volume/mids EMA lifts crest brightness (hard-capped) and quickens the
        swell (period floor). No per-beat response, no strobe. Accumulated
        phase: position advances only by a clamped per-frame dt, so seeks and
        re-entry gaps cannot corrupt it (unlike absolute-t envelope math)."""
        dimmer = (cue.get("dimmer", 60) if cue else 60) / 100.0

        # dt from our own call cadence; a discontinuity (deselected, or a
        # seek in either direction) counts as one nominal frame.
        if (self._ga_last_render_t is None or
                abs(t - self._ga_last_render_t) > ANTHEM_DISCONTINUITY_THRESHOLD):
            dt = 0.012
        else:
            dt = min(max(t - self._ga_last_render_t, 0.0), ANTHEM_DT_CLAMP)
        self._ga_last_render_t = t

        # Slow smoothed music energy -- rides passages, ignores single hits.
        self._ga_energy = ema(self._ga_energy,
                              min(1.0, volume * 1000.0 + mid_i * 0.5), 0.1, 0.03)

        period = ANTHEM_BASE_PERIOD - self._ga_energy * (ANTHEM_BASE_PERIOD - ANTHEM_MIN_PERIOD)
        self._ga_phase = (self._ga_phase + dt / period) % 1.0

        env = _anthem_envelope(self._ga_phase)
        crest = min(ANTHEM_CREST_MAX, ANTHEM_CREST_BASE + self._ga_energy * ANTHEM_CREST_GAIN)
        # Floor is NOT dimmer-scaled (PWM-visibility lesson from abyssal_bloom).
        brightness = max(ANTHEM_FLOOR_MIN, env * crest * dimmer)

        gold = lerp_color(kick_color, ANTHEM_GOLD, 0.6)  # variety-tinted gold
        shimmer = max(0.0, env - 0.8) / 0.2              # white only near the crest

        self.out_r = ema(self.out_r, gold[0] * brightness, 0.04, 0.03)
        self.out_g = ema(self.out_g, gold[1] * brightness, 0.04, 0.03)
        self.out_b = ema(self.out_b, gold[2] * brightness, 0.04, 0.03)
        self.out_w = ema(self.out_w, 120.0 * shimmer * dimmer, 0.06, 0.05)
        self.out_master = ema(self.out_master, 255.0 * brightness, 0.04, 0.03)
        self.out_strobe = 0
```

- [ ] **Step 6: Engine registration.** Add `"golden_anthem",` to `VALID_BEHAVIORS` (anchor `:88`, end of the ambient group after `"abyssal_bloom"`), and to `_behavior_map` (anchor `:223`, after the `"abyssal_bloom"` entry):
```python
            "golden_anthem": self._render_golden_anthem,
```

- [ ] **Step 7: Run to verify pass.**
`.venv\Scripts\python.exe -m pytest tests/test_golden_anthem.py -q` → `5 passed`. Full suite → `33 passed`.

- [ ] **Step 8: Commit.**
```bash
git add dmx_engine.py tests/test_golden_anthem.py
git commit -m "feat: add golden_anthem renderer (accumulated-phase gold swells, ride-the-music)"
```

## Task 8: `golden_anthem` — loopback + LLM registration

**Files:**
- Modify: `music_light.py` (`ambient_pool`, `ambient_behaviors`)
- Modify: `llm_designer.py` (prompt entry + its own `VALID_BEHAVIORS`)

- [ ] **Step 1: Loopback pool.** In `music_light.py` `_detect_auto_behavior`, grep `ambient_pool = ` and extend:
```python
                ambient_pool = ["ocean_drift", "candlelight", "aurora_shimmer",
                               "sunset_fade", "abyssal_bloom", "golden_anthem"]
```
And in `_dispatch`'s `ambient_behaviors` set (from Task 4), add `"golden_anthem"` after `"abyssal_bloom"`.

- [ ] **Step 2: LLM prompt entry.** In `llm_designer.py`, grep the `"abyssal_bloom" —` prompt block and insert after it (before `=== NARRATIVE ARC TEMPLATE ===`):
```
"golden_anthem" — Majestic slow gold swells that crest into white-gold shimmer, riding the music's sustained energy (louder passages crest brighter and swell faster, hard-capped). No beat flashing.
  USE FOR: Finales, euphoric sing-along/anthem moments, emotional breakdowns, sunset sets.
  FEEL: Majestic, golden, hands-in-the-air. The whole crowd swaying as one.
```

- [ ] **Step 3: LLM `VALID_BEHAVIORS`** (anchor `llm_designer.py:447` — the *silent-downgrade gate*: a behavior missing here gets rewritten to `beat_reactive` with no error). Add `"golden_anthem",` after `"abyssal_bloom",` in the ambient group.

- [ ] **Step 4: Verify the repair preserves it.**
```
.venv\Scripts\python.exe -c "import llm_designer; p = llm_designer._validate_and_repair_plan({'show_name':'x','cues':[{'section_name':'F','start_time':0,'end_time':10,'behavior':'golden_anthem','color_1':[255,190,80],'color_2':[80,0,200]}],'phrases':[]}); print('behavior:', p['cues'][0]['behavior'])"
```
Expected: `behavior: golden_anthem` (NOT `beat_reactive`).

- [ ] **Step 5: Dry-run smoke.** Hand-write a `current_show.json` (Task 2 test JSON shape, real WAV path from `youtube_audio/`, `behavior: "golden_anthem"`, full-track cue), then:
`set DMX_DRY_RUN=1 && .venv\Scripts\python.exe ai_show_player.py --show current_show.json`
Expected: `[DRY 0.0s] cue=... palette=...` then clean completion; sampled `frame=` values show warm gold-dominant R/G channels rising and falling. No tracebacks.

- [ ] **Step 6: Suite + commit.** Full suite → `33 passed`.
```bash
git add music_light.py llm_designer.py
git commit -m "feat: register golden_anthem in loopback rotation and LLM behavior set"
```

## Task 9: Final verification + handoff

- [ ] **Step 1:** Full suite → `33 passed`. `.venv\Scripts\python.exe -m pyflakes dmx_engine.py music_light.py ai_show_player.py llm_designer.py` → no NEW findings vs. the known baseline (an intentional `boto3` noqa import, pre-existing unused locals).
- [ ] **Step 2:** Dry-run both a normal library show and the golden_anthem show once more; confirm palette determinism across two runs of the same show.
- [ ] **Step 3: STOP — hand back to the user** for the Phase-2 hardware pass (watch a full golden_anthem swell cycle ride a loud vs. quiet passage) and the merge/PR decision (finishing-a-development-branch).

---

## Self-review notes (coverage map)

| Spec requirement | Task |
|---|---|
| Dry-run harness | 1 |
| `zlib.crc32` seed (process-stable) + bpm persist/read | 2 (incl. `save_show` gap found during anchor verification: show.json never contained bpm) |
| Synced director Intent wiring (LLM color seeds palette) | 3 |
| Loopback director + 16s forced evolution + retire color cycling | 4 |
| `_render_loopback_direct` palette rewire (kills R→B→G→W) | 4 (merged with director — signature and call site must change together; deviation from spec's "each item its own commit" noted and justified) |
| Punch unification + `bass_white_blast` dilution fix | 5 |
| abyssal_bloom inherits variety, no code change | 6 |
| Hard hardware checkpoint before Phase 2 | 6 Step 3 |
| golden_anthem: accumulated phase, bounded lift, floor, gold tint, no strobe | 7 |
| golden_anthem: 9-point registration | 7 (engine: 5 points) + 8 (loopback: 2, LLM: 2) |
| pytest coverage (identity, punch, anthem) | 2, 5, 7 |
| Suite green every commit; conventional commits; revert knob | execution rules + Task 6 |

**Known deviations from spec:** spec items 4 and 5 (loopback director / direct-renderer rewire) land as ONE commit (Task 4) because the renderer's signature change and its only call site cannot compile independently.

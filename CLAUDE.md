# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Windows-only, AI-powered DMX stage-lighting controller. A Python (FastAPI) backend plus a React 19 / Vite 7 PWA frontend drive a uDMX USB adapter in sync with audio, in two modes:

- **Live loopback** — captures system audio via WASAPI and reacts in real time.
- **AI-synced** — YouTube URL → yt-dlp/ffmpeg → FFT analysis → LLM-designed cue plan → WAV playback driving the lights.

## Process architecture

`app.py` (FastAPI on `:8000`) is the only long-running process. Engines run as **subprocesses** and talk to it via **JSON files, not HTTP**. Mode is determined by which script is spawned — there is no `--mode` flag:

- `music_light.py` — live loopback engine. Spawned by `POST /api/loopback`.
- `ai_show_player.py` — synced AI-show playback. Spawned by `POST /api/play`.
- `youtube_analyzer.py` — download + spectral analysis. Spawned by `POST /api/shows/generate`.
- `llm_designer.py` — LLM cue-plan layer. Imported by `youtube_analyzer.py`, not a worker.

IPC files are written atomically (`.tmp` write + `os.replace`) — preserve that pattern:

- `playback_state.json` — engine → API (position, state, cue).
- `playback_command.json` — API → engine (seek/pause/resume/stop). Only `ai_show_player.py` acts on commands; transport controls are no-ops in loopback.

## The big trap: duplicated engines (being removed on `feature/pyinstaller-packaging`)

`music_light.py` and `ai_show_player.py` were split from one file and share many identically-**named** methods with **no shared module** — but they have already **silently diverged**. Verified by AST diff of the two `DMXEngine` classes (2026-06-13):

- **17 methods are byte-identical** (`send_dmx`, `_init_hardware`, `_dmx_worker`, `shutdown`, the three IPC helpers, `_write_playback_state`, and 9 of the 14 renderers — all the ambient/atmospheric ones + `strobe_blast`).
- **7 methods have drifted**: `__init__`, `load_ai_show`, `process_audio`, and the four *punchy* renderers `_render_beat_reactive`, `_render_bass_white_blast`, `_render_blackout_punch`, `_render_fast_pulse`.

The drift is **bidirectional** — neither file is the superset:
- `process_audio`: loopback uses `profile_gain_boost` (≈50×) + `profile_agc_thresh`; synced uses gain ×1 + hardcoded `agc_thresh = 0.7`. Synced also computes `self._beat_velocity` and rotates palettes every 16 beats; loopback does color-phase cycling instead.
- `_render_bass_white_blast` / `_render_blackout_punch` / `_render_fast_pulse`: the **synced** copies already scale master by `self._beat_velocity`; the loopback copies use flat `255`. So **synced is *not* the mushy one for these** — the old claim that "synced lacks velocity sensitivity" is wrong.
- `_render_beat_reactive`: the **loopback** copy is richer (ambient floor + `energy_state`-driven warm/cool color-temperature shift); synced stripped both.
- Loopback alone has `_render_loopback_direct`, `_detect_auto_behavior`, `load_profile`, `run_loopback_mode`; synced alone has `_get_active_cue`, `run_synced_mode`.

Until the shared-module refactor (`dmx_engine.py` / `dmx_variety.py` / `dmx_punch.py`, in progress on this branch) lands, any change to a *shared* method must be applied to **both files** — and check the diff first, because "identical by name" is not "identical."

## Commands

Dev (two terminals):

```
.venv\Scripts\python.exe app.py     # backend on :8000
cd frontend && npm run dev          # Vite on :5173, proxies /api -> :8000
```

`start.bat` launches both and opens the browser; `stop.bat` kills both.

Build the distributable: `build.bat` from the repo root. Order matters: it builds the frontend first (`npm run build` → `frontend/dist/`) because `dmx_app.spec` bundles `frontend/dist`, then runs PyInstaller, then copies the `ffmpeg.exe`/`yt-dlp.exe` sidecars into `dist\dmx_app\`. Output is a one-folder bundle with 4 exes (`dmx_app.exe` + three `*_worker.exe`).

Lint: `cd frontend && npm run lint` (ESLint). There is no Python linter or formatter configured.

## Dependencies

There is **no `requirements.txt` or `pyproject.toml`** — the `pip install -r requirements.txt` hint printed by `build.bat` points at a file that doesn't exist. The Python 3.13 venv must live at `.venv\` because `app.py` hardcodes `.venv\Scripts\python.exe` when spawning workers in dev. Install:

```
.venv\Scripts\pip install fastapi uvicorn pydantic python-dotenv numpy pyaudiowpatch pyusb requests
```

`boto3` is needed only when `LLM_PROVIDER=bedrock` (it is imported lazily).

## Testing

A tracked pytest suite lives in `tests/` (run `.venv\Scripts\python.exe -m pytest -q` from the repo root; `pytest.ini` configures discovery, `tests/conftest.py` sets `DMX_DRY_RUN=1` before any engine import). It covers the variety/punch modules, renderer behavior (envelope math, discontinuity guards, velocity grading), profile pinning, and the LLM repair gate — extend it TDD-style when touching those areas. The engines also support an offline harness: `DMX_DRY_RUN=1` skips USB init and records frames instead of sending them (see `ai_show_player.py`'s dry-run branch for deterministic no-hardware playback verification).

The gitignored `test_*.py` files at the **repo root** are a different thing: throwaway manual hardware/API probes (some contain hardcoded credentials — never commit or extend them). Hardware-feel changes still need verification by running the app on the uDMX.

## Hardware & sidecar binaries

- **Windows-only**: WASAPI loopback via `pyaudiowpatch`.
- The uDMX adapter (VID `0x16C0`, PID `0x05DC`) is hardcoded; engines raise `RuntimeError` without it — there is no simulation mode. `app.py` itself imports no USB/audio libraries, so the server, UI, YouTube download, and AI generation all work without hardware — only the playback/loopback subprocesses need it.
- `ffmpeg.exe` and `yt-dlp.exe` are gitignored sidecars downloaded manually; they live at the repo root (dev) or beside the exe (packaged). `libusb-1.0.dll` is required at runtime for USB.

## Load-bearing invariants — do not reintroduce these bugs

- `shutdown()` must release the libusb handle (`usb.util.dispose_resources`) on **every** exit path — it lives in a `finally`; keep it there. Leaking it makes the next run report a ghost "uDMX not found".
- `p.terminate()` must run on loopback teardown — it releases the WASAPI COM port binding; leaked bindings eventually block new WASAPI connections.
- `app.py` merges worker stderr into stdout and streams it with an activity watchdog (see the "M4 FIX" comments) to avoid a >64 KB pipe-buffer deadlock. Preserve this when touching subprocess code.

## Config & conventions

- Secrets go in `.env` (gitignored); the template is `.env.example`. `LLM_PROVIDER` selects `azure` (default — called via raw `requests.post`, **not** the `openai` SDK) or `bedrock` (boto3 Converse API, lazily imported with a module-level client cache).
- Commits follow Conventional Commits (`feat:`, `fix:`, `refactor:`); feature branches are `feature/<kebab-case>`.
- `playback_state.json` is a runtime artifact tracked by mistake — it shows up dirty after every run; don't commit its churn.
- Python style: plain procedural/class code using `threading` (no asyncio), no type hints — match the existing style.

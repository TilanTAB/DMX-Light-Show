# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Windows-only, AI-powered DMX stage-lighting controller: a Python (FastAPI) backend + React 19 / Vite 7 frontend that drives a physical uDMX USB fixture in sync with audio. Two modes:

- **Loopback (live)**: captures system audio via WASAPI and reacts in real time.
- **AI-synced**: YouTube URL → spectral analysis → LLM-designed cues → WAV playback driving the lights.

## Architecture you must know before editing

`app.py` is the only long-running process (FastAPI on `:8000`). It spawns the engines as **subprocesses** and communicates with them through **JSON files, not HTTP**.

- `music_light.py` — **loopback engine only**. Spawned by `POST /api/loopback`.
- `ai_show_player.py` — **AI-synced playback engine only**. Spawned by `POST /api/play`.
- `youtube_analyzer.py` — download + spectral analysis. Spawned by `POST /api/shows/generate`.
- `llm_designer.py` — turns analysis telemetry into a cue plan.

There is no `--mode` flag anymore — the mode is determined by *which script* runs.

⚠️ **`music_light.py` and `ai_show_player.py` were split from one engine and still share ~80% duplicated code with no common module.** Any change to shared logic — the renderer methods, `send_dmx`, the gamma LUT, hardware init, or the IPC helpers — **must be applied to BOTH files** or they silently diverge. (Extracting a shared module is desirable but hasn't been done.)

**IPC files** (written atomically via a `.tmp` write + `os.replace`):
- `playback_state.json` — engine → API (position, state, cue, behavior).
- `playback_command.json` — API → engine (seek/pause/resume/stop). Only `ai_show_player.py` acts on these; `music_light.py` defines the handler but never calls it, so transport controls are no-ops in loopback.

## Commands

Run in dev (two processes):
```
.venv\Scripts\python.exe app.py     # backend on :8000
cd frontend && npm run dev          # Vite dev server on :5173, proxies /api -> :8000
```
`start.bat` launches both and opens the browser; `stop.bat` kills both by port.

Build the distributable (from repo root):
```
build.bat
```
It builds the frontend **first** (`npm run build` → `frontend/dist/`), then runs PyInstaller (`dmx_app.spec`), then copies the `ffmpeg.exe` / `yt-dlp.exe` sidecars into `dist\dmx_app\`. The frontend must be built before PyInstaller because the spec bundles `frontend/dist`. Output is a one-folder, 4-exe bundle (`dmx_app.exe` + three `*_worker.exe`). The committed `dist/` is stale — rebuild instead of trusting it.

Lint (frontend only): `cd frontend && npm run lint`. There is **no** Python linter or formatter configured.

Worker CLIs (rarely invoked by hand):
```
python music_light.py [--profile profiles/concert_punchy.json] [--show shows/<id>/show.json]
python ai_show_player.py --show shows/<id>/show.json
python youtube_analyzer.py "<youtube-url>"
```

## Dependencies

**No `requirements.txt` or `pyproject.toml` exists** (Python 3.13 venv at `.venv\`). The `pip install -r requirements.txt` hint inside `build.bat` points at a file that isn't there. Install deps explicitly (list taken from `start.bat`):
```
.venv\Scripts\pip install fastapi uvicorn pydantic python-dotenv numpy pyaudiowpatch pyusb requests
```
Add `boto3` only when using `LLM_PROVIDER=bedrock`. The dev worker spawn hardcodes `.venv\Scripts\python.exe`, so the venv must live at `.venv\`.

## Testing

There are **no automated tests** and no test runner. `npm test` does not exist and `pytest` has nothing to collect. The `test_*.py` files are gitignored, throwaway manual hardware/Azure probes (some with hardcoded credentials) — not a suite. Verify changes by running the app.

## Hardware & external binaries

- **Windows-only**: WASAPI loopback (`pyaudiowpatch`), plus `yt-dlp.exe` + `curl.exe` + `ffmpeg.exe`. Any Linux/Mac notes in the README are aspirational.
- The **uDMX adapter** (VID `0x16C0`, PID `0x05DC`) is hardcoded. The engines **raise `RuntimeError` if it's missing** — there is no simulation mode. However, `app.py` imports no USB/audio libraries, so the **server, UI, YouTube download, and AI generation all work without hardware connected** — only the playback/loopback subprocesses require it.
- `ffmpeg.exe` and `yt-dlp.exe` are gitignored sidecars you download manually; they live at the repo root (dev) or beside the exe (packaged). `libusb-1.0.dll` is required at runtime for the USB adapter.

## Load-bearing invariants — do not reintroduce these bugs

- **USB handle cleanup ("C3 fix")**: `shutdown()` must release the libusb handle on every exit path (it lives in a `finally`). Skip it and the *next* run reports a ghost "uDMX not found".
- **WASAPI cleanup ("C1 fix")**: `p.terminate()` must run on loopback teardown, or Windows refuses new WASAPI connections after ~15 leaks.
- **Subprocess stdout**: `app.py` merges worker stderr→stdout and streams it with an activity watchdog to avoid a >64 KB pipe-buffer deadlock. Preserve this when touching subprocess code.

## Config & conventions

- Secrets go in `.env` (gitignored); the template is `.env.example`. `LLM_PROVIDER` selects `azure` (default — called via raw `requests`, **not** the `openai` SDK) or `bedrock` (boto3 Converse API, imported lazily).
- Commits follow **Conventional Commits** (`feat:`, `fix:`, `refactor:`, with optional scopes). Feature branches are `feature/<kebab-case>`.
- `playback_state.json` is a runtime artifact tracked by mistake — it shows up dirty after every run; don't commit its churn.

"""
DMX Show Manager — FastAPI REST API
Manages a library of pre-compiled AI-generated lighting shows,
controls playback (synced WAV or live loopback), and triggers new show generation.

Run with: python app.py
Auto-docs: http://localhost:8000/docs
"""
import os
import re
import json
import time
import shutil
import subprocess
import threading
import logging
import wave
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ==========================================
# LOGGING SETUP
# ==========================================
os.makedirs("logs", exist_ok=True)
logger = logging.getLogger("dmx")
logger.setLevel(logging.DEBUG)

# File handler — detailed log with timestamps
fh = logging.FileHandler("logs/generation.log", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
logger.addHandler(fh)

# Console handler — shorter format  
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)

# ==========================================
# APP SETUP
# ==========================================
app = FastAPI(
    title="DMX Show Manager",
    description="REST API for managing AI-generated DMX lighting shows",
    version="1.0.0"
)

# Allow ALL origins — safe for a local-only tool.
# CORS is a browser policy, not a server security mechanism.
# Your API is equally accessible via curl regardless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Any origin (React dev server, browser, etc.)
    allow_credentials=True,
    allow_methods=["*"],       # GET, POST, DELETE, etc.
    allow_headers=["*"],
)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOWS_DIR = os.path.join(BASE_DIR, "shows")
AUDIO_DIR = os.path.join(BASE_DIR, "youtube_audio")
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")

os.makedirs(SHOWS_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

# ==========================================
# GLOBAL STATE (process tracking)
# ==========================================
_active_process = None
_active_lock = threading.Lock()
_generation_lock = threading.Lock()  # C2 FIX: separate lock for generation status
_generation_status = {"active": False, "message": "", "progress": 0}

# I6 FIX: In-memory index of video_id → show_id (built on startup, updated on add/delete)
_video_index: dict[str, str] = {}


# ==========================================
# PYDANTIC REQUEST MODELS
# FastAPI uses these to auto-validate incoming JSON.
# If a field is missing or the wrong type, the client
# gets a clear 422 error automatically — no manual checks.
# ==========================================
class GenerateRequest(BaseModel):
    url: str
    regenerate: bool = False
    start_time: float | None = None  # Optional trim start (seconds)
    end_time: float | None = None    # Optional trim end (seconds)

class PlayRequest(BaseModel):
    show_id: str

class LoopbackRequest(BaseModel):
    show_id: str | None = None
    profile_id: str | None = None

class SeekRequest(BaseModel):
    position: float  # Seconds to seek to

# IPC file paths (shared with music_light.py)
_PLAYBACK_STATE_FILE = os.path.join(BASE_DIR, "playback_state.json")
_PLAYBACK_COMMAND_FILE = os.path.join(BASE_DIR, "playback_command.json")


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def _extract_video_id(url: str) -> str | None:
    """
    Extract the YouTube video ID from various URL formats:
      - https://www.youtube.com/watch?v=dQw4w9WgXcQ
      - https://youtu.be/dQw4w9WgXcQ
      - https://youtube.com/shorts/dQw4w9WgXcQ
      - https://www.youtube.com/embed/dQw4w9WgXcQ
    Returns the 11-character video ID or None.
    """
    patterns = [
        r'(?:youtube\.com/watch\?.*v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_audio_duration(filepath: str) -> float:
    """Get duration of a WAV file in seconds."""
    try:
        with wave.open(filepath, 'rb') as wf:
            return round(wf.getnframes() / wf.getframerate(), 1)
    except Exception:
        return 0.0


def _save_show_to_library(show_json_path: str, name: str | None = None) -> str:
    """
    Copy a show.json and its audio file into the shows/ library.
    Returns the show_id (folder name).
    Raises ValueError with context on failure (I5 FIX).
    """
    try:
        with open(show_json_path, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise ValueError(f"Cannot read show file '{show_json_path}': {e}")

    audio_src = data.get("audio_file", "")
    if not audio_src:
        raise ValueError("Show JSON has no 'audio_file' field")
    if not os.path.exists(audio_src):
        raise ValueError(f"Audio file not found: {audio_src}")

    # Generate a clean folder name from the audio filename
    audio_basename = os.path.splitext(os.path.basename(audio_src))[0]
    raw_name = name if name else audio_basename
    show_id = "".join(c for c in raw_name if c.isalnum() or c in " -_").strip().replace(" ", "_")

    show_dir = os.path.join(SHOWS_DIR, show_id)
    os.makedirs(show_dir, exist_ok=True)

    # Copy audio file into show directory
    audio_dest = os.path.join(show_dir, os.path.basename(audio_src))
    if not os.path.exists(audio_dest):
        shutil.copy2(audio_src, audio_dest)

    data["audio_file"] = os.path.abspath(audio_dest)

    # Build metadata
    duration = _get_audio_duration(audio_dest)
    plan = data.get("lighting_plan", {})
    phrases = plan.get("phrases", [])
    cues = plan.get("cues", [])

    existing_video_id = data.get("metadata", {}).get("video_id")
    source_url = data.get("source_url", "")
    video_id = existing_video_id or _extract_video_id(source_url) or None

    metadata = {
        "title": audio_basename,
        "show_id": show_id,
        "video_id": video_id,
        "duration_seconds": duration,
        "num_phrases": len(phrases),
        "num_cues": len(cues),
        "phrases": [p.get("name", "Unknown") for p in phrases],
        "show_name": plan.get("show_name", audio_basename),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    data["metadata"] = metadata

    show_file = os.path.join(show_dir, "show.json")
    with open(show_file, 'w') as f:
        json.dump(data, f, indent=2)

    # I6 FIX: Update in-memory index
    if video_id:
        _video_index[video_id] = show_id

    return show_id


def _kill_active_process():
    """Terminate any currently playing music_light.py process."""
    global _active_process
    if _active_process and _active_process.poll() is None:
        _active_process.terminate()
        try:
            _active_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _active_process.kill()
        _active_process = None
    # FIX #5: Purge stale IPC files to prevent ghost state
    for ipc_file in (_PLAYBACK_STATE_FILE, _PLAYBACK_COMMAND_FILE,
                     _PLAYBACK_STATE_FILE + ".tmp", _PLAYBACK_COMMAND_FILE + ".tmp"):
        try:
            if os.path.exists(ipc_file):
                os.remove(ipc_file)
        except OSError:
            pass


def _resolve_show_path(show_id: str) -> str:
    """Resolve show directory path with path-traversal protection.
    Returns the validated absolute show dir path.
    Raises HTTPException(400) if the path escapes SHOWS_DIR."""
    show_dir = os.path.realpath(os.path.join(SHOWS_DIR, show_id))
    if not show_dir.startswith(os.path.realpath(SHOWS_DIR)):
        raise HTTPException(400, "Invalid show ID")
    return show_dir


def _build_video_index():
    """I6 FIX: Build in-memory video_id → show_id index on startup."""
    _video_index.clear()
    if not os.path.exists(SHOWS_DIR):
        return
    for folder in os.listdir(SHOWS_DIR):
        show_file = os.path.join(SHOWS_DIR, folder, "show.json")
        if not os.path.isfile(show_file):
            continue
        try:
            with open(show_file, 'r') as fh:
                data = json.load(fh)
            vid = data.get("metadata", {}).get("video_id")
            if vid:
                _video_index[vid] = folder
        except Exception:
            pass
    logger.info(f"Video index built: {len(_video_index)} entries")


def _find_existing_show_by_audio(url: str) -> dict | None:
    """
    Check if a show already exists for the SAME YouTube video.
    Uses in-memory index for O(1) lookup (I6 FIX).
    """
    video_id = _extract_video_id(url)
    if not video_id:
        return None
    
    # O(1) lookup from in-memory index
    if video_id in _video_index:
        folder = _video_index[video_id]
        show_file = os.path.join(SHOWS_DIR, folder, "show.json")
        if os.path.isfile(show_file):
            try:
                with open(show_file, 'r') as fh:
                    data = json.load(fh)
                audio_path = data.get("audio_file", "")
                return {
                    "show_id": folder,
                    "title": data.get("metadata", {}).get("title", folder),
                    "audio_file": audio_path if os.path.exists(audio_path) else None,
                }
            except Exception:
                pass
    
    # Fallback: Check if raw audio exists in youtube_audio/
    if os.path.exists(AUDIO_DIR):
        for f in os.listdir(AUDIO_DIR):
            if f.endswith('.wav') and video_id in f:
                return {
                    "show_id": None,
                    "audio_file": os.path.join(AUDIO_DIR, f),
                    "title": None,
                }
    
    return None


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/api/shows")
def list_shows():
    """Return all saved shows in the library."""
    shows = []
    if not os.path.exists(SHOWS_DIR):
        return shows

    for folder in sorted(os.listdir(SHOWS_DIR)):
        show_file = os.path.join(SHOWS_DIR, folder, "show.json")
        if not os.path.isfile(show_file):
            continue
        try:
            with open(show_file, 'r') as f:
                data = json.load(f)

            meta = data.get("metadata", {})
            plan = data.get("lighting_plan", {})
            phrases = plan.get("phrases", [])
            cues = plan.get("cues", [])

            # Color palette preview from phrases or cues
            palette_preview = []
            source = cues[:4] if cues else phrases[:4]
            for p in source:
                palette_preview.append({
                    "name": p.get("section_name", p.get("name", "?")),
                    "color_1": p.get("color_1", [255, 255, 255]),
                    "color_2": p.get("color_2", [255, 255, 255]),
                    "energy": p.get("energy_level", p.get("energy", 3)),
                    "behavior": p.get("behavior", "static_wash"),
                })

            shows.append({
                "show_id": folder,
                "title": meta.get("title", folder),
                "show_name": meta.get("show_name", ""),
                "duration": meta.get("duration_seconds", 0),
                "num_phrases": meta.get("num_phrases", len(phrases)),
                "num_cues": meta.get("num_cues", len(cues)),
                "phrases": meta.get("phrases", []),
                "palette_preview": palette_preview,
                "created_at": meta.get("created_at", "Unknown"),
                "has_audio": os.path.exists(data.get("audio_file", "")),
            })
        except Exception as e:
            print(f"[WARN] Failed to read show {folder}: {e}")

    return shows


@app.post("/api/shows/import-current")
def import_current_show():
    """Import the current_show.json into the library."""
    current = os.path.join(BASE_DIR, "current_show.json")
    if not os.path.exists(current):
        raise HTTPException(404, "No current_show.json found. Generate a show first.")

    try:
        show_id = _save_show_to_library(current)
    except ValueError as e:
        raise HTTPException(400, str(e))

    return {"message": f"Imported as '{show_id}'", "show_id": show_id}


@app.post("/api/shows/generate")
def generate_show(req: GenerateRequest):
    """
    Generate a new show from a YouTube URL.
    
    If the audio already exists and regenerate=False, returns the existing show.
    If regenerate=True, skips the download but re-runs AI analysis + generation.
    """
    global _generation_status

    if not req.url.strip():
        raise HTTPException(400, "No YouTube URL provided.")

    # C2 FIX: Atomic check-and-set with lock
    with _generation_lock:
        if _generation_status["active"]:
            raise HTTPException(409, "A show is already being generated. Please wait.")
        _generation_status = {"active": True, "message": "Starting...", "progress": 0}

    # Check if we already have this song
    # FIX #1: Skip dedup when trim range is specified (different segment = different show)
    has_trim = req.start_time is not None or req.end_time is not None
    existing = None if has_trim else _find_existing_show_by_audio(req.url.strip())

    if existing and existing.get("show_id") and not req.regenerate:
        with _generation_lock:
            _generation_status = {"active": False, "message": "", "progress": 0}
        return {
            "message": f"Show already exists: '{existing['title']}'",
            "status": "exists",
            "show_id": existing["show_id"],
            "title": existing["title"],
        }

    def _run_generation():
        global _generation_status
        proc = None
        
        current = os.path.join(BASE_DIR, "current_show.json")
        
        if os.path.exists(current):
            os.remove(current)
            logger.info("Removed stale current_show.json")
        
        cmd = [PYTHON_EXE, "-u", "youtube_analyzer.py"]
        
        if existing and existing.get("audio_file") and os.path.exists(existing["audio_file"]):
            with _generation_lock:
                _generation_status = {"active": True, "message": "Regenerating AI plan (audio cached)...", "progress": 40}
            cmd.extend(["--skip-download", "--audio", existing["audio_file"]])
            logger.info(f"Regenerating with cached audio: {existing['audio_file']}")
        else:
            with _generation_lock:
                _generation_status = {"active": True, "message": "Starting download pipeline...", "progress": 5}
            cmd.append(req.url.strip())
            logger.info(f"Full generation for URL: {req.url.strip()}")
        
        # Pass time range if specified
        if req.start_time is not None:
            cmd.extend(["--start", str(req.start_time)])
        if req.end_time is not None:
            cmd.extend(["--end", str(req.end_time)])
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        # I3 FIX: Watchdog timer kills hung subprocess after 300s
        def _watchdog():
            nonlocal proc
            import time as _t
            _t.sleep(300)
            if proc and proc.poll() is None:
                logger.error("Watchdog: killing hung subprocess after 300s")
                proc.kill()
        
        try:
            proc = subprocess.Popen(
                cmd, cwd=BASE_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            
            watchdog = threading.Thread(target=_watchdog, daemon=True)
            watchdog.start()
            
            all_output = []
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                all_output.append(line)
                logger.debug(f"[GEN] {line}")
                
                with _generation_lock:
                    if "[DOWNLOAD]" in line or "[YT-DLP]" in line:
                        try:
                            parts = line.split("]", 1)[-1].strip().split()
                            if parts:
                                first_num = parts[0].replace("%", "")
                                dl_pct = float(first_num)
                                if 0 <= dl_pct <= 100:
                                    mapped = 5 + (dl_pct / 100) * 25
                                    _generation_status["message"] = f"Downloading audio... {dl_pct:.0f}%"
                                    _generation_status["progress"] = int(mapped)
                        except (ValueError, IndexError):
                            pass
                    elif "[CONVERT]" in line:
                        _generation_status["message"] = "Converting to WAV format..."
                        _generation_status["progress"] = 32
                    elif "Download phase complete" in line:
                        _generation_status["message"] = "Download complete! Starting analysis..."
                        _generation_status["progress"] = 35
                    elif "Downloading" in line:
                        _generation_status["message"] = "Downloading audio from YouTube..."
                        _generation_status["progress"] = 8
                    elif "Using existing" in line or "Using provided" in line or "Skipping download" in line:
                        _generation_status["message"] = "Using cached audio file"
                        _generation_status["progress"] = 35
                    elif "deep audio analysis" in line.lower() or "Analyzing" in line:
                        _generation_status["message"] = "Analyzing audio structure (FFT spectral analysis)..."
                        _generation_status["progress"] = 45
                    elif "Deep analysis complete" in line or "Detected" in line:
                        _generation_status["message"] = f"Audio analysis done. {line.split('Detected')[-1].strip() if 'Detected' in line else ''}"
                        _generation_status["progress"] = 55
                    elif "TELEMETRY" in line:
                        _generation_status["message"] = "Audio telemetry extracted!"
                        _generation_status["progress"] = 60
                    elif "Sending" in line and ("GPT" in line or "Azure" in line or "OpenAI" in line):
                        _generation_status["message"] = "Sending to Azure GPT for lighting design..."
                        _generation_status["progress"] = 65
                    elif "AI DMX LIGHTING" in line or "LIGHTING SCRIPT" in line:
                        _generation_status["message"] = "AI lighting plan received!"
                        _generation_status["progress"] = 80
                    elif "Show saved" in line:
                        _generation_status["message"] = "Show file saved!"
                        _generation_status["progress"] = 85
                    elif "Error" in line or "Failed" in line or "[-]" in line:
                        _generation_status["message"] = line
                        logger.warning(f"Potential error: {line}")
                    
            proc.wait(timeout=30)  # stdout already closed, so this is just cleanup
            
            stderr = proc.stderr.read() if proc.stderr else ""
            if stderr:
                logger.warning(f"stderr output: {stderr[-500:]}")
            
            if proc.returncode != 0:
                error_msg = stderr[-300:] if stderr else (all_output[-1] if all_output else "Unknown error")
                logger.error(f"youtube_analyzer.py exited with code {proc.returncode}: {error_msg}")
                with _generation_lock:
                    _generation_status = {"active": False, "message": f"Error: {error_msg}", "progress": 0}
                return

            if not os.path.exists(current):
                logger.error("youtube_analyzer.py returned 0 but current_show.json was NOT created")
                with _generation_lock:
                    _generation_status = {"active": False, "message": "Generation failed — no show file created.", "progress": 0}
                return

            with _generation_lock:
                _generation_status["message"] = "Saving to library..."
                _generation_status["progress"] = 90

            with open(current, 'r') as f:
                show_data = json.load(f)
            
            logger.info(f"current_show.json audio_file: {show_data.get('audio_file', 'MISSING')}")
            
            show_data["source_url"] = req.url.strip()
            video_id = _extract_video_id(req.url.strip())
            if video_id:
                show_data.setdefault("metadata", {})["video_id"] = video_id
            with open(current, 'w') as f:
                json.dump(show_data, f, indent=2)
            
            try:
                show_id = _save_show_to_library(current)
            except ValueError as e:
                logger.error(f"Failed to save to library: {e}")
                with _generation_lock:
                    _generation_status = {"active": False, "message": f"Error: {e}", "progress": 0}
                return

            logger.info(f"Show saved to library: {show_id}")
            with _generation_lock:
                _generation_status = {
                    "active": False,
                    "message": f"Done! Show '{show_id}' saved.",
                    "progress": 100,
                    "show_id": show_id,
                }
        except Exception as e:
            logger.exception(f"Fatal generation error: {e}")
            with _generation_lock:
                _generation_status = {"active": False, "message": f"Fatal: {e}", "progress": 0}

    thread = threading.Thread(target=_run_generation, daemon=True)
    thread.start()
    
    skip_msg = "Regenerating AI plan (reusing cached audio)..." if existing else "Full generation started..."
    return {"message": skip_msg, "status": "processing"}


@app.get("/api/shows/generate/status")
def generation_status():
    """Poll the current show generation status."""
    with _generation_lock:
        return dict(_generation_status)  # Return a copy


@app.post("/api/play")
def play_show(req: PlayRequest):
    """Start synced WAV playback of a saved show."""
    # FIX #4: Path traversal protection
    show_dir = _resolve_show_path(req.show_id)
    show_file = os.path.join(show_dir, "show.json")
    if not os.path.isfile(show_file):
        raise HTTPException(404, f"Show '{req.show_id}' not found.")

    with _active_lock:
        _kill_active_process()
        global _active_process
        _active_process = subprocess.Popen(
            [PYTHON_EXE, "music_light.py", "--mode", "synced", "--show", show_file],
            cwd=BASE_DIR
        )
    return {"message": f"Playing '{req.show_id}' (synced mode)", "pid": _active_process.pid}


@app.post("/api/loopback")
def start_loopback(req: LoopbackRequest):
    """Start live WASAPI loopback capture mode."""
    cmd = [PYTHON_EXE, "music_light.py", "--mode", "loopback"]

    if req.show_id:
        show_dir = _resolve_show_path(req.show_id)
        show_file = os.path.join(show_dir, "show.json")
        if os.path.isfile(show_file):
            cmd.extend(["--show", show_file])

    # Add profile if specified
    if req.profile_id:
        profile_path = os.path.join(PROFILES_DIR, req.profile_id + ".json")
        if os.path.isfile(profile_path):
            cmd.extend(["--profile", profile_path])

    with _active_lock:
        _kill_active_process()
        global _active_process
        _active_process = subprocess.Popen(cmd, cwd=BASE_DIR)

    return {"message": "Loopback mode started.", "pid": _active_process.pid}


@app.post("/api/stop")
def stop_playback():
    """Stop any currently running playback."""
    with _active_lock:
        was_playing = _active_process is not None and _active_process.poll() is None
        _kill_active_process()

    return {"message": "Playback stopped." if was_playing else "Nothing was playing."}


@app.get("/api/playback/position")
def get_playback_position():
    """Read current playback position from the IPC state file."""
    if not os.path.exists(_PLAYBACK_STATE_FILE):
        return {"position": 0, "duration": 0, "state": "stopped", "cue_name": "", "behavior": ""}
    try:
        with open(_PLAYBACK_STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {"position": 0, "duration": 0, "state": "stopped", "cue_name": "", "behavior": ""}


def _write_playback_command(command: dict):
    """Write a command file for music_light.py to pick up."""
    cmd_path = _PLAYBACK_COMMAND_FILE
    tmp = cmd_path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(command, f)
    os.replace(tmp, cmd_path)


@app.post("/api/playback/seek")
def seek_playback(req: SeekRequest):
    """Seek to a position in the current synced playback."""
    with _active_lock:
        if _active_process is None or _active_process.poll() is not None:
            raise HTTPException(400, "No playback active")
    _write_playback_command({"command": "seek", "position": req.position})
    return {"message": f"Seeking to {req.position:.1f}s"}


@app.post("/api/playback/pause")
def pause_playback():
    """Pause the current synced playback."""
    with _active_lock:
        if _active_process is None or _active_process.poll() is not None:
            raise HTTPException(400, "No playback active")
    _write_playback_command({"command": "pause"})
    return {"message": "Paused"}


@app.post("/api/playback/resume")
def resume_playback():
    """Resume paused synced playback."""
    with _active_lock:
        if _active_process is None or _active_process.poll() is not None:
            raise HTTPException(400, "No playback active")
    _write_playback_command({"command": "resume"})
    return {"message": "Resumed"}


@app.get("/api/status")
def playback_status():
    """Get current playback and generation status."""
    with _active_lock:
        is_playing = _active_process is not None and _active_process.poll() is None

    # Include position data if available
    position_data = {}
    if is_playing and os.path.exists(_PLAYBACK_STATE_FILE):
        try:
            with open(_PLAYBACK_STATE_FILE, 'r') as f:
                position_data = json.load(f)
        except Exception:
            pass

    return {
        "playing": is_playing,
        "generation": _generation_status,
        "playback": position_data,
    }


@app.delete("/api/shows/{show_id}")
def delete_show(show_id: str):
    """Delete a show from the library."""
    show_dir = _resolve_show_path(show_id)
    if not os.path.isdir(show_dir):
        raise HTTPException(404, "Show not found.")

    # I6 FIX: Remove from index
    for vid, sid in list(_video_index.items()):
        if sid == show_id:
            del _video_index[vid]
            break

    shutil.rmtree(show_dir)
    return {"message": f"Show '{show_id}' deleted."}


# ==========================================
# PROFILE ENDPOINTS
# ==========================================

@app.get("/api/profiles")
def list_profiles():
    """List all available lighting profiles."""
    profiles = []
    if os.path.isdir(PROFILES_DIR):
        for fname in sorted(os.listdir(PROFILES_DIR)):
            if fname.endswith(".json"):
                fpath = os.path.join(PROFILES_DIR, fname)
                try:
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    profiles.append({
                        "id": fname.replace(".json", ""),
                        "name": data.get("name", fname),
                        "description": data.get("description", ""),
                    })
                except Exception:
                    profiles.append({"id": fname.replace(".json", ""), "name": fname, "description": "Error reading profile"})
    return {"profiles": profiles}


@app.get("/api/profiles/{profile_id}")
def get_profile(profile_id: str):
    """Get full profile details."""
    # Prevent path traversal
    if "/" in profile_id or "\\" in profile_id or ".." in profile_id:
        raise HTTPException(400, "Invalid profile ID")
    fpath = os.path.join(PROFILES_DIR, profile_id + ".json")
    if not os.path.isfile(fpath):
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    with open(fpath, 'r') as f:
        return json.load(f)


@app.post("/api/profiles")
def save_profile(profile: dict):
    """Save or update a profile. Body must include 'name'."""
    name = profile.get("name")
    if not name:
        raise HTTPException(400, "Profile must have a 'name' field")
    # Create a safe filename from the name
    safe_id = re.sub(r'[^a-z0-9_]', '_', name.lower().strip()).strip('_')
    if not safe_id:
        raise HTTPException(400, "Profile name produces invalid ID")
    fpath = os.path.join(PROFILES_DIR, safe_id + ".json")
    with open(fpath, 'w') as f:
        json.dump(profile, f, indent=2)
    return {"message": f"Profile '{name}' saved.", "id": safe_id}


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    """Delete a profile."""
    if "/" in profile_id or "\\" in profile_id or ".." in profile_id:
        raise HTTPException(400, "Invalid profile ID")
    fpath = os.path.join(PROFILES_DIR, profile_id + ".json")
    if not os.path.isfile(fpath):
        raise HTTPException(404, f"Profile '{profile_id}' not found")
    os.remove(fpath)
    return {"message": f"Profile '{profile_id}' deleted."}


@app.post("/api/profiles/{profile_id}/activate")
def activate_profile(profile_id: str):
    """Set active profile and restart loopback with it."""
    if "/" in profile_id or "\\" in profile_id or ".." in profile_id:
        raise HTTPException(400, "Invalid profile ID")
    profile_path = os.path.join(PROFILES_DIR, profile_id + ".json")
    if not os.path.isfile(profile_path):
        raise HTTPException(404, f"Profile '{profile_id}' not found")

    # Restart loopback with this profile
    cmd = [PYTHON_EXE, "music_light.py", "--mode", "loopback", "--profile", profile_path]

    with _active_lock:
        _kill_active_process()
        global _active_process
        _active_process = subprocess.Popen(cmd, cwd=BASE_DIR)

    with open(profile_path, 'r') as f:
        profile_name = json.load(f).get("name", profile_id)

    return {"message": f"Profile '{profile_name}' activated.", "pid": _active_process.pid}

# ==========================================
# STARTUP
# ==========================================
if __name__ == "__main__":
    # I6 FIX: Build video index on startup
    _build_video_index()

    # Auto-import current_show.json on first run
    current = os.path.join(BASE_DIR, "current_show.json")
    if os.path.exists(current) and not os.listdir(SHOWS_DIR):
        try:
            sid = _save_show_to_library(current)
            print(f"[+] Auto-imported current_show.json as '{sid}'")
        except ValueError as e:
            print(f"[WARN] Auto-import failed: {e}")

    print("=" * 50)
    print("  DMX Show Manager API (FastAPI)")
    print("  API:  http://localhost:8000/api/shows")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)

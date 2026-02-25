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

os.makedirs(SHOWS_DIR, exist_ok=True)

# ==========================================
# GLOBAL STATE (process tracking)
# ==========================================
_active_process = None
_active_lock = threading.Lock()
_generation_status = {"active": False, "message": "", "progress": 0}


# ==========================================
# PYDANTIC REQUEST MODELS
# FastAPI uses these to auto-validate incoming JSON.
# If a field is missing or the wrong type, the client
# gets a clear 422 error automatically — no manual checks.
# ==========================================
class GenerateRequest(BaseModel):
    url: str
    regenerate: bool = False  # If True, skip download but re-run AI generation

class PlayRequest(BaseModel):
    show_id: str

class LoopbackRequest(BaseModel):
    show_id: str | None = None


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


def _save_show_to_library(show_json_path: str, name: str | None = None) -> str | None:
    """
    Copy a show.json and its audio file into the shows/ library.
    Returns the show_id (folder name) or None on failure.
    """
    with open(show_json_path, 'r') as f:
        data = json.load(f)

    audio_src = data.get("audio_file", "")
    if not os.path.exists(audio_src):
        return None

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

    # Update the show JSON to point to the local copy
    data["audio_file"] = os.path.abspath(audio_dest)

    # Build metadata
    duration = _get_audio_duration(audio_dest)
    plan = data.get("lighting_plan", {})
    phrases = plan.get("phrases", [])
    cues = plan.get("cues", [])

    # Preserve video_id if it was set by a previous generation
    existing_video_id = data.get("metadata", {}).get("video_id")
    # Also try to extract from source_url if present
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


def _find_existing_show_by_audio(url: str) -> dict | None:
    """
    Check if a show already exists for the SAME YouTube video.
    Matches by video ID extracted from the URL — not by filename.
    Returns show info dict or None.
    """
    video_id = _extract_video_id(url)
    
    if not video_id:
        # Can't extract ID (not a valid YouTube URL) — skip matching
        return None
    
    # Pass 1: Check shows in library by stored video_id
    for folder in os.listdir(SHOWS_DIR):
        show_file = os.path.join(SHOWS_DIR, folder, "show.json")
        if not os.path.isfile(show_file):
            continue
        try:
            with open(show_file, 'r') as fh:
                data = json.load(fh)
            stored_id = data.get("metadata", {}).get("video_id")
            if stored_id == video_id:
                audio_path = data.get("audio_file", "")
                return {
                    "show_id": folder,
                    "title": data.get("metadata", {}).get("title", folder),
                    "audio_file": audio_path if os.path.exists(audio_path) else None,
                }
        except Exception:
            pass
    
    # Pass 2: Check if raw audio exists in youtube_audio/ with the video ID in filename
    # (yt-dlp appends [VIDEO_ID] to filenames by default)
    if os.path.exists(AUDIO_DIR):
        for f in os.listdir(AUDIO_DIR):
            if f.endswith('.wav') and video_id in f:
                return {
                    "show_id": None,  # Audio exists but no show yet
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

    show_id = _save_show_to_library(current)
    if not show_id:
        raise HTTPException(400, "Failed to import — audio file missing.")

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

    if _generation_status["active"]:
        raise HTTPException(409, "A show is already being generated. Please wait.")

    # Check if we already have this song
    existing = _find_existing_show_by_audio(req.url.strip())

    if existing and existing.get("show_id") and not req.regenerate:
        # Show already exists and user didn't ask to regenerate
        return {
            "message": f"Show already exists: '{existing['title']}'",
            "status": "exists",
            "show_id": existing["show_id"],
            "title": existing["title"],
        }

    def _run_generation():
        global _generation_status
        
        current = os.path.join(BASE_DIR, "current_show.json")
        
        # DELETE stale current_show.json BEFORE starting.
        # If the subprocess fails, we must NOT reuse old data.
        if os.path.exists(current):
            os.remove(current)
            logger.info("Removed stale current_show.json")
        
        # Build the command for youtube_analyzer.py
        cmd = [PYTHON_EXE, "-u", "youtube_analyzer.py"]  # -u = unbuffered output
        
        if existing and existing.get("audio_file") and os.path.exists(existing["audio_file"]):
            _generation_status = {"active": True, "message": "Regenerating AI plan (audio cached)...", "progress": 40}
            cmd.extend(["--skip-download", "--audio", existing["audio_file"]])
            logger.info(f"Regenerating with cached audio: {existing['audio_file']}")
        else:
            _generation_status = {"active": True, "message": "Starting download pipeline...", "progress": 5}
            cmd.append(req.url.strip())
            logger.info(f"Full generation for URL: {req.url.strip()}")
        
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            proc = subprocess.Popen(
                cmd, cwd=BASE_DIR,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1
            )
            
            all_output = []
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                all_output.append(line)
                logger.debug(f"[GEN] {line}")
                
                # Parse progress from youtube_analyzer.py output
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
                    
            proc.wait(timeout=300)
            
            stderr = proc.stderr.read() if proc.stderr else ""
            if stderr:
                logger.warning(f"stderr output: {stderr[-500:]}")
            
            if proc.returncode != 0:
                error_msg = stderr[-300:] if stderr else (all_output[-1] if all_output else "Unknown error")
                logger.error(f"youtube_analyzer.py exited with code {proc.returncode}: {error_msg}")
                _generation_status = {"active": False, "message": f"Error: {error_msg}", "progress": 0}
                return

            # Validate that current_show.json was actually written by this run
            if not os.path.exists(current):
                logger.error("youtube_analyzer.py returned 0 but current_show.json was NOT created")
                _generation_status = {"active": False, "message": "Generation failed — no show file created.", "progress": 0}
                return

            _generation_status["message"] = "Saving to library..."
            _generation_status["progress"] = 90

            # Inject source URL and video ID
            with open(current, 'r') as f:
                show_data = json.load(f)
            
            logger.info(f"current_show.json audio_file: {show_data.get('audio_file', 'MISSING')}")
            
            show_data["source_url"] = req.url.strip()
            video_id = _extract_video_id(req.url.strip())
            if video_id:
                show_data.setdefault("metadata", {})["video_id"] = video_id
            with open(current, 'w') as f:
                json.dump(show_data, f, indent=2)
            
            show_id = _save_show_to_library(current)
            logger.info(f"Show saved to library: {show_id}")
            _generation_status = {
                "active": False,
                "message": f"Done! Show '{show_id}' saved.",
                "progress": 100,
                "show_id": show_id,
            }
        except subprocess.TimeoutExpired:
            logger.error("Generation timed out after 5 minutes")
            _generation_status = {"active": False, "message": "Timed out after 5 minutes.", "progress": 0}
        except Exception as e:
            logger.exception(f"Fatal generation error: {e}")
            _generation_status = {"active": False, "message": f"Fatal: {e}", "progress": 0}

    thread = threading.Thread(target=_run_generation, daemon=True)
    thread.start()
    
    skip_msg = "Regenerating AI plan (reusing cached audio)..." if existing else "Full generation started..."
    return {"message": skip_msg, "status": "processing"}


@app.get("/api/shows/generate/status")
def generation_status():
    """Poll the current show generation status."""
    return _generation_status


@app.post("/api/play")
def play_show(req: PlayRequest):
    """Start synced WAV playback of a saved show."""
    show_file = os.path.join(SHOWS_DIR, req.show_id, "show.json")
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
        show_file = os.path.join(SHOWS_DIR, req.show_id, "show.json")
        if os.path.isfile(show_file):
            cmd.extend(["--show", show_file])

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


@app.get("/api/status")
def playback_status():
    """Get current playback and generation status."""
    with _active_lock:
        is_playing = _active_process is not None and _active_process.poll() is None

    return {
        "playing": is_playing,
        "generation": _generation_status,
    }


@app.delete("/api/shows/{show_id}")
def delete_show(show_id: str):
    """Delete a show from the library."""
    show_dir = os.path.join(SHOWS_DIR, show_id)
    if not os.path.isdir(show_dir):
        raise HTTPException(404, "Show not found.")

    shutil.rmtree(show_dir)
    return {"message": f"Show '{show_id}' deleted."}


# ==========================================
# STARTUP
# ==========================================
if __name__ == "__main__":
    # Auto-import current_show.json on first run
    current = os.path.join(BASE_DIR, "current_show.json")
    if os.path.exists(current) and not os.listdir(SHOWS_DIR):
        sid = _save_show_to_library(current)
        print(f"[+] Auto-imported current_show.json as '{sid}'")

    print("=" * 50)
    print("  DMX Show Manager API (FastAPI)")
    print("  API:  http://localhost:8000/api/shows")
    print("  Docs: http://localhost:8000/docs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)

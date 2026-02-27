import os
import re
import json
import time
import wave
import numpy as np
import llm_designer

YOUTUBE_AUDIO_DIR = "youtube_audio"

# Whitelist of valid YouTube URL patterns
_YOUTUBE_URL_PATTERN = re.compile(
    r'^https?://(www\.)?(youtube\.com/(watch|shorts|embed)|youtu\.be/|music\.youtube\.com/)'
)
SHOW_FILE = "current_show.json"


def trim_audio_range(audio_path, start_sec=None, end_sec=None):
    """
    Trim a WAV file to [start_sec, end_sec] using ffmpeg.
    Returns path to trimmed file, or original path if no trimming needed.
    """
    if start_sec is None and end_sec is None:
        return audio_path

    import subprocess

    base, ext = os.path.splitext(audio_path)
    trimmed_path = f"{base}_trimmed{ext}"

    cmd = ["ffmpeg", "-y", "-i", audio_path]
    if start_sec is not None:
        cmd.extend(["-ss", str(start_sec)])
    if end_sec is not None:
        cmd.extend(["-to", str(end_sec)])
    # Copy codec for instant trim (no re-encoding)
    cmd.extend(["-c", "copy", trimmed_path])

    print(f"[+] Trimming audio: {_fmt_time(start_sec or 0)} → {_fmt_time(end_sec) if end_sec else 'end'}", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            print(f"[-] ffmpeg trim failed: {result.stderr[-300:]}", flush=True)
            return audio_path  # Fall back to untrimmed
        print(f"[+] Trimmed audio saved: {os.path.basename(trimmed_path)}", flush=True)
        return trimmed_path
    except Exception as e:
        print(f"[-] Trim error: {e}", flush=True)
        return audio_path


def _fmt_time(seconds):
    """Format seconds as M:SS for display."""
    if seconds is None:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def download_youtube_audio(url):
    """
    Downloads the HIGHEST QUALITY audio from a YouTube video via subprocess.
    Uses native curl.exe and yt-dlp.exe to bypass the completely broken 
    Python IPv6/SSL networking stack on this Windows machine.
    Outputs a high-fidelity WAV file (44.1kHz, 16-bit stereo) via ffmpeg.
    """
    import subprocess
    os.makedirs(YOUTUBE_AUDIO_DIR, exist_ok=True)

    # C4 FIX: Validate URL before passing to subprocess
    if not _YOUTUBE_URL_PATTERN.match(url):
        print(f"[-] Invalid YouTube URL: {url}", flush=True)
        return None

    print(f"\n[+] Downloading BEST quality audio from YouTube...", flush=True)
    print(f"    URL: {url}", flush=True)

    # Use standalone yt-dlp.exe but force curl to do the actual downloading
    cmd = [
        ".\\yt-dlp.exe",
        "--newline",             # CRITICAL: output progress on new lines, not \r overwrites
        "--progress",            # Always show progress even when piped
        "--downloader", "curl",
        "--downloader-args", "-k",  # Skip SSL check where urllib hangs
        "--format", "bestaudio/best",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--postprocessor-args", "ffmpeg:-ar 44100", 
        "--no-playlist",
        "--force-overwrites",
        "-o", os.path.join(YOUTUBE_AUDIO_DIR, "%(title)s.%(ext)s"),
        url
    ]

    start = time.time()
    try:
        # Use subprocess.Popen to capture yt-dlp progress in real-time.
        # yt-dlp writes progress to stderr — we merge it into stdout so
        # the FastAPI backend can parse and forward it to the React UI.
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # yt-dlp progress: [download]  45.2% of 5.23MiB at 1.2MiB/s ETA 00:03
            if "[download]" in line and "%" in line:
                print(f"[DOWNLOAD] {line}", flush=True)
            elif "[ExtractAudio]" in line or "ffmpeg" in line.lower():
                print(f"[CONVERT] Converting to WAV...", flush=True)
            elif "Deleting original" in line or "has already been" in line:
                pass  # Skip noise
            else:
                print(f"[YT-DLP] {line}", flush=True)
        
        proc.wait()
        
        if proc.returncode != 0:
            print(f"\n[-] Download failed with exit code {proc.returncode}", flush=True)
            return None
            
    except Exception as e:
        print(f"\n[-] Critical download error: {e}", flush=True)
        return None

    elapsed = time.time() - start
    print(f"[+] Download phase complete! ({elapsed:.1f}s)", flush=True)

    # Find the newest WAV file in the output directory
    files = [os.path.join(YOUTUBE_AUDIO_DIR, f) for f in os.listdir(YOUTUBE_AUDIO_DIR) if f.endswith('.wav')]
    if not files:
        print("[-] No WAV file found after download! Did ffmpeg fail?", flush=True)
        return None

    newest = max(files, key=os.path.getctime)
    
    # Print file quality info
    try:
        with wave.open(newest, 'rb') as wf:
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            dur = wf.getnframes() / sr
            print(f"    Validated Quality: {sr}Hz, {sw*8}-bit, {'Stereo' if ch == 2 else 'Mono'}, {dur:.1f}s", flush=True)
    except Exception as e:
        print(f"    Warning: WAV header check failed ({e})", flush=True)
        
    return newest


def analyze_audio_structure(filepath):
    """
    Professional-grade audio structure analysis using pure Python + Numpy.
    Extracts a detailed spectral timeline with per-segment frequency band energy,
    structural section detection, vocal presence, and energy arcs.
    """
    start_time = time.time()
    print(f"[+] Running deep audio analysis...", flush=True)
    
    # Read the WAV file
    with wave.open(filepath, 'rb') as wf:
        n_frames = wf.getnframes()
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        audio_data = wf.readframes(n_frames)
        
    y = np.frombuffer(audio_data, dtype=np.int16)
    if n_channels == 2:
        y = y.reshape(-1, 2).mean(axis=1)
    y = y.astype(np.float32) / 32768.0
    
    total_duration = n_frames / sr
    
    # ================================================================
    # STEP 1: Spectral Timeline (2-second segments)
    # For each segment, compute energy in 4 frequency bands:
    #   Sub-Bass (20-60Hz)   → Kick drums, 808s
    #   Bass (60-250Hz)      → Bass guitar, bass synths
    #   Mids (250-4000Hz)    → Vocals, guitars, keys, snares
    #   Highs (4000-16000Hz) → Hi-hats, cymbals, air, brightness
    # ================================================================
    segment_duration = 2.0  # seconds per analysis window
    segment_size = int(sr * segment_duration)
    num_segments = len(y) // segment_size
    
    timeline = []
    rms_values = []
    bass_energies = []
    
    for seg_idx in range(num_segments):
        chunk = y[seg_idx * segment_size : (seg_idx + 1) * segment_size]
        seg_time = round(seg_idx * segment_duration, 1)
        
        # RMS volume
        rms = float(np.sqrt(np.mean(chunk**2)))
        rms_values.append(rms)
        
        # FFT for frequency analysis
        N = len(chunk)
        yf = np.abs(np.fft.fft(chunk)[:N // 2])
        freqs = np.fft.fftfreq(N, 1 / sr)[:N // 2]
        
        # Extract energy per band
        sub_bass = float(np.mean(yf[(freqs >= 20) & (freqs < 60)])) if np.any((freqs >= 20) & (freqs < 60)) else 0
        bass = float(np.mean(yf[(freqs >= 60) & (freqs < 250)])) if np.any((freqs >= 60) & (freqs < 250)) else 0
        mids = float(np.mean(yf[(freqs >= 250) & (freqs < 4000)])) if np.any((freqs >= 250) & (freqs < 4000)) else 0
        highs = float(np.mean(yf[(freqs >= 4000) & (freqs < 16000)])) if np.any((freqs >= 4000) & (freqs < 16000)) else 0
        
        bass_energies.append(sub_bass + bass)
        
        # Beat density: count transient spikes within this 2s window using 50ms sub-chunks
        sub_chunk_size = int(sr * 0.05)  # 50ms
        sub_chunks = len(chunk) // sub_chunk_size
        sub_rms = [float(np.sqrt(np.mean(chunk[j*sub_chunk_size:(j+1)*sub_chunk_size]**2))) for j in range(sub_chunks)]
        if len(sub_rms) > 1:
            avg_sub = np.mean(sub_rms)
            beats_in_seg = sum(1 for s in sub_rms if s > avg_sub * 1.6)
        else:
            beats_in_seg = 0
        
        # Vocal presence: mids-to-bass ratio > 1.5 suggests vocal/melodic content
        vocal_present = mids > (sub_bass + bass) * 1.5 if (sub_bass + bass) > 0 else False
        
        timeline.append({
            "time": seg_time,
            "rms": round(rms, 4),
            "sub_bass": round(sub_bass, 4),
            "bass": round(bass, 4),
            "mids": round(mids, 4),
            "highs": round(highs, 4),
            "beat_density": beats_in_seg,
            "vocal_present": vocal_present
        })
    
    # ================================================================
    # STEP 2: Structural Section Detection
    # Uses energy contour to find distinct sections (Intro, Verse, Chorus, etc.)
    # A section boundary = significant change in average energy over 8s windows
    # ================================================================
    window = 4  # 4 segments = 8 seconds
    sections = []
    current_section_start = 0.0
    prev_energy = np.mean(rms_values[:window]) if len(rms_values) >= window else 0.001
    
    for i in range(window, len(rms_values) - window + 1, window):
        current_energy = float(np.mean(rms_values[i:i + window]))
        energy_change = abs(current_energy - prev_energy) / max(prev_energy, 0.001)
        
        if energy_change > 0.4:  # 40% energy shift = new section
            section_end = round(i * segment_duration, 1)
            
            # Classify the section that just ended
            section_rms = float(np.mean(rms_values[int(current_section_start / segment_duration):i]))
            avg_rms = float(np.mean(rms_values))
            
            if section_rms < avg_rms * 0.4:
                section_type = "Intro/Outro"
            elif section_rms < avg_rms * 0.75:
                section_type = "Verse/Breakdown"
            elif section_rms < avg_rms * 1.2:
                section_type = "Pre-Chorus/Build-up"
            else:
                section_type = "Chorus/Drop"
            
            sections.append({
                "start": current_section_start,
                "end": section_end,
                "type": section_type,
                "avg_energy": round(section_rms, 4)
            })
            current_section_start = section_end
            prev_energy = current_energy
    
    # Add the final section
    final_rms = float(np.mean(rms_values[int(current_section_start / segment_duration):]))
    avg_rms_total = float(np.mean(rms_values))
    if final_rms < avg_rms_total * 0.5:
        final_type = "Outro"
    elif final_rms > avg_rms_total * 1.1:
        final_type = "Chorus/Drop"
    else:
        final_type = "Verse"
    
    sections.append({
        "start": current_section_start,
        "end": round(total_duration, 1),
        "type": final_type,
        "avg_energy": round(final_rms, 4)
    })
    
    # ================================================================
    # STEP 3: Global Song Metrics
    # ================================================================
    # BPM estimation from total beat count
    total_beats = sum(s["beat_density"] for s in timeline)
    approx_bpm = (total_beats / total_duration) * 60.0
    adjusted_bpm = round(max(70.0, min(180.0, approx_bpm)), 1)
    
    # Dynamic range (difference between loudest and quietest sections)
    dynamic_range = round(max(rms_values) / max(min(rms_values), 0.0001), 1)
    
    # Drop timestamps (moments where bass energy spikes 3x above running average)
    drops = []
    if len(bass_energies) > 4:
        running_avg = np.convolve(bass_energies, np.ones(4)/4, mode='valid')
        for i in range(len(running_avg)):
            if i + 4 < len(bass_energies) and bass_energies[i + 4] > running_avg[i] * 3.0:
                drops.append(round((i + 4) * segment_duration, 1))
    
    # Build-up detection (sustained rising energy over 6+ seconds)
    buildups = []
    rising_count = 0
    for i in range(1, len(rms_values)):
        if rms_values[i] > rms_values[i-1] * 1.05:
            rising_count += 1
            if rising_count >= 3:  # 6+ seconds of rising energy
                buildups.append(round((i - rising_count) * segment_duration, 1))
                rising_count = 0
        else:
            rising_count = 0
    
    # Vocal sections (segments where vocals dominate)
    vocal_sections = []
    in_vocal = False
    vocal_start = 0
    for s in timeline:
        if s["vocal_present"] and not in_vocal:
            in_vocal = True
            vocal_start = s["time"]
        elif not s["vocal_present"] and in_vocal:
            in_vocal = False
            if s["time"] - vocal_start >= 4.0:  # Only count 4s+ vocal sections
                vocal_sections.append({"start": vocal_start, "end": s["time"]})
    
    telemetry = {
        "song_metrics": {
            "bpm": adjusted_bpm,
            "total_duration_seconds": round(total_duration, 1),
            "dynamic_range_ratio": dynamic_range,
            "overall_energy": "High" if avg_rms_total > 0.05 else "Medium" if avg_rms_total > 0.02 else "Chill",
            "num_sections_detected": len(sections),
        },
        "structural_sections": sections,
        "spectral_timeline": timeline,
        "events": {
            "drop_timestamps": drops,
            "buildup_timestamps": buildups,
            "vocal_sections": vocal_sections,
        }
    }
    
    elapsed = time.time() - start_time
    print(f"[+] Deep analysis complete! ({elapsed:.2f}s)", flush=True)
    print(f"    Detected {len(sections)} sections, {len(drops)} drops, {len(buildups)} build-ups, {len(vocal_sections)} vocal sections", flush=True)
    
    return telemetry


def save_show(ai_plan, audio_filepath):
    """
    Saves the AI-generated lighting plan and audio file path to current_show.json
    so music_light.py can load it and play the show in perfect sync.
    """
    show_data = {
        "audio_file": os.path.abspath(audio_filepath),
        "lighting_plan": ai_plan
    }
    with open(SHOW_FILE, 'w') as f:
        json.dump(show_data, f, indent=2)
    print(f"\n[+] Show saved to {SHOW_FILE}", flush=True)

    
if __name__ == "__main__":
    total_start = time.time()
    
    import sys
    
    print("=" * 60)
    print("   AI DMX SHOW GENERATOR")
    print("=" * 60)
    
    # Parse CLI arguments
    # Usage: python youtube_analyzer.py <URL> [--skip-download] [--audio <path>] [--start <sec>] [--end <sec>]
    args = sys.argv[1:]
    user_input = None
    skip_download = "--skip-download" in args
    audio_override = None
    trim_start = None
    trim_end = None
    
    for i, arg in enumerate(args):
        if arg == "--audio" and i + 1 < len(args):
            audio_override = args[i + 1]
        elif arg == "--start" and i + 1 < len(args):
            try:
                trim_start = float(args[i + 1])
            except ValueError:
                pass
        elif arg == "--end" and i + 1 < len(args):
            try:
                trim_end = float(args[i + 1])
            except ValueError:
                pass
        elif not arg.startswith("--"):
            user_input = arg.strip()
    
    audio_file = None
    
    if audio_override and os.path.exists(audio_override):
        # Explicit audio file provided — use it directly
        audio_file = audio_override
        print(f"\n[+] Using provided audio: {os.path.basename(audio_file)}", flush=True)
        
    elif user_input and not skip_download:
        # Download from YouTube
        audio_file = download_youtube_audio(user_input)
        if not audio_file:
            print("[-] Download failed. Exiting.")
            exit(1)
            
    elif skip_download:
        # Skip download, use most recent existing WAV
        files = [os.path.join(YOUTUBE_AUDIO_DIR, f) for f in os.listdir(YOUTUBE_AUDIO_DIR) if f.endswith('.wav')]
        if not files:
            print("[-] No audio files found! Cannot skip download without existing files.")
            exit(1)
        audio_file = max(files, key=os.path.getctime)
        print(f"\n[+] Skipping download. Using existing: {os.path.basename(audio_file)}", flush=True)
        
    else:
        # No URL provided, use most recent local WAV
        files = [os.path.join(YOUTUBE_AUDIO_DIR, f) for f in os.listdir(YOUTUBE_AUDIO_DIR) if f.endswith('.wav')]
        if not files:
            print("[-] No audio files found in youtube_audio directory!")
            exit(1)
        audio_file = max(files, key=os.path.getctime)
        print(f"\n[+] Using existing file: {os.path.basename(audio_file)}", flush=True)
    
    # Step 0.5: Trim to user-specified range if requested
    if trim_start is not None or trim_end is not None:
        audio_file = trim_audio_range(audio_file, trim_start, trim_end)
    
    # Step 1: Analyze the audio structure
    telemetry = analyze_audio_structure(audio_file)
    
    print(f"\n[==== AUDIO TELEMETRY (Summary) ====]")
    print(json.dumps({
        "song_metrics": telemetry["song_metrics"],
        "structural_sections": telemetry["structural_sections"],
        "events": telemetry["events"],
        "spectral_timeline_points": len(telemetry.get("spectral_timeline", []))
    }, indent=2))
    
    # Step 2: Send telemetry to Azure OpenAI GPT-5 Nano
    print("\n[+] Sending telemetry to Azure OpenAI 'GPT-5 Nano'...")
    ai_plan = llm_designer.get_gpt_lighting_plan(telemetry)
    
    if ai_plan:
        print("\n[==== AZURE AI DMX LIGHTING SCRIPT ====]")
        print(json.dumps(ai_plan, indent=2))
        
        # Step 3: Save the show to current_show.json
        save_show(ai_plan, audio_file)
        
        print(f"\n[+] Ready! Run 'python music_light.py' to start the synchronized show.")
    else:
        print("\n[-] AI Generation Failed! Check your .env file credentials.")
        
    print(f"\n[+] Total Pipeline Execution Time: {time.time() - total_start:.2f}s", flush=True)


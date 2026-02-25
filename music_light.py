import usb.core
import time
import os
import sys
import json
import numpy as np
import pyaudiowpatch as pyaudio
import logging

# Define logger
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# DMX & LIGHT CONFIGURATION
# ==========================================
# Channel Map: CH1=Master, CH2=Red, CH3=Green, CH4=Blue, CH5=White, CH6=Strobe
dev = usb.core.find(idVendor=0x16C0, idProduct=0x05DC)
if dev is None:
    logger.error("uDMX not found! Please connect the adapter.")
    exit()

logger.info("uDMX found! Setting up audio stream...")

def send_dmx(master, red, green, blue, white=0, strobe=0):
    data = [master, red, green, blue, white, strobe, 0, 0]
    try:
        dev.ctrl_transfer(0x40, 2, 8, 0, data)
    except Exception as e:
        logger.warning(f"Failed to send DMX packet: {e}")

# ==========================================
# AUDIO PROCESSING CONFIG
# ==========================================
SAMPLE_RATE = 44100
BLOCK_SIZE = 1024          # ~23ms per frame at 44.1kHz
MIN_VOLUME_GATE = 0.001    # Silence gate

# Frequency band definitions (Hz) — covers the actual instrument ranges
KICK_LO,  KICK_HI  = 30,  150    # Kick drum body + punch
SNARE_LO, SNARE_HI = 150, 400    # Snare crack + body
MID_LO,   MID_HI   = 400, 2000   # Vocals, guitars, synths
HIHAT_LO, HIHAT_HI = 4000, 10000 # Hi-hats, cymbals, sibilance

# Sensitivity: how much a transient above the moving average triggers intensity
KICK_GAIN  = 6.0   # Kick → primary color flash
SNARE_GAIN = 4.0   # Snare → secondary color flash
MID_GAIN   = 2.0   # Mids → gentle wash
HIHAT_GAIN = 3.0   # Hi-hat → white shimmer

# Beat detection: onset must exceed (moving_avg * threshold) to count
ONSET_THRESHOLD = 1.5    # How far above background a hit must be
ONSET_COOLDOWN  = 0.12   # Minimum seconds between registered beats (prevents doubles)

# Smoothing: attack = how fast lights snap ON, decay = how fast they fade OFF
# 1.0 = instant, 0.0 = frozen.  Drums need instant attack, fast decay.
ATTACK_BEAT  = 0.95  # When a beat is detected: snap ON
DECAY_BEAT   = 0.25  # Between beats: fade out quickly (punchy feel)
ATTACK_WASH  = 0.08  # For mids/vocals: slow, gentle rise
DECAY_WASH   = 0.03  # Slow fade for ambient wash

# AGC (Auto Gain Control) — tracks background level per band
AGC_SPEED = 0.015  # Slow adaptation (don't chase individual beats)

# ==========================================
# CURATED COLOR PALETTES (fallback)
# ==========================================
PALETTES = [
    ((255, 0, 50),   (0, 150, 255)),   # Cyberpunk (Hot Pink & Cyan)
    ((255, 50, 0),   (100, 0, 255)),   # Synthwave (Neon Orange & Deep Purple)
    ((0, 255, 100),  (255, 0, 200)),   # Toxic (Lime Green & Magenta)
]

# ==========================================
# ENGINE STATE
# ==========================================
# Smoothed output channels
out_r, out_g, out_b, out_w, out_master = 0.0, 0.0, 0.0, 0.0, 0.0

# AGC rolling averages (per band)
agc_kick, agc_snare, agc_mid, agc_hihat = 0.0, 0.0, 0.0, 0.0

# Previous frame FFT magnitudes (for spectral flux onset detection)
prev_kick_mag, prev_snare_mag, prev_hihat_mag = 0.0, 0.0, 0.0

# Beat tracking
beat_timestamps = []
total_beat_count = 0
current_palette_idx = 0
last_beat_time = 0.0

# Synced mode: AI cue list
synced_cues = []
synced_start_time = 0.0


def load_ai_show(show_file="current_show.json"):
    """Load AI-generated palettes and cue list from a show JSON file."""
    global PALETTES, current_palette_idx, synced_cues

    if not os.path.exists(show_file):
        logger.warning(f"No {show_file} found! Using default fallback palettes.")
        return None

    try:
        with open(show_file, 'r') as f:
            data = json.load(f)

        plan = data.get("lighting_plan", {})

        # Load color palettes from phrases
        phrases = plan.get("phrases", [])
        if phrases:
            new_palettes = []
            for p in phrases:
                c1 = tuple(p.get("color_1", [255, 255, 255]))
                c2 = tuple(p.get("color_2", [255, 255, 255]))
                new_palettes.append((c1, c2))
            PALETTES = new_palettes
            current_palette_idx = 0
            logger.info(f"Loaded {len(PALETTES)} AI palettes from {show_file}")

        # Load timestamped cues for synced mode
        cues = plan.get("cues", [])
        if cues:
            synced_cues.clear()
            for c in cues:
                synced_cues.append({
                    "start": c.get("start_time", 0),
                    "end": c.get("end_time", 0),
                    "color_1": tuple(c.get("color_1", [255, 255, 255])),
                    "color_2": tuple(c.get("color_2", [255, 255, 255])),
                    "energy": c.get("energy_level", 3),
                    "strobe": c.get("strobe_allowed", False),
                    "behavior": c.get("behavior", "static_wash"),
                    "name": c.get("section_name", ""),
                })
            synced_cues.sort(key=lambda x: x["start"])
            logger.info(f"Loaded {len(synced_cues)} timestamped cues for synced mode")

        return data.get("audio_file")
    except Exception as e:
        logger.error(f"Failed to load AI show: {e}")
        return None


def get_active_cue(elapsed_seconds):
    """Find the AI cue that should be active at the given timestamp."""
    for cue in reversed(synced_cues):
        if elapsed_seconds >= cue["start"]:
            return cue
    return None


def get_band_magnitude(fft_data, fft_freqs, lo_hz, hi_hz):
    """Extract average magnitude for a frequency range from FFT data."""
    idx = np.where((fft_freqs >= lo_hz) & (fft_freqs <= hi_hz))[0]
    if len(idx) == 0:
        return 0.0
    return float(np.mean(fft_data[idx]))


def process_audio(indata, input_format="int16", elapsed_seconds=None):
    """
    Core audio-reactive engine. Processes one audio frame and sends DMX.
    
    Uses SPECTRAL FLUX onset detection for drum hits:
    - Compares current FFT magnitude to previous frame
    - A sudden increase = onset (drum hit, transient)
    - This is far more responsive than simple threshold detection
    
    Separate frequency bands for kick, snare, hi-hat, and mids
    allow each instrument to drive a different light behavior.
    """
    global out_r, out_g, out_b, out_w, out_master
    global agc_kick, agc_snare, agc_mid, agc_hihat
    global prev_kick_mag, prev_snare_mag, prev_hihat_mag
    global beat_timestamps, total_beat_count, current_palette_idx, last_beat_time

    # --- Decode audio bytes to float array ---
    if isinstance(indata, bytes):
        if input_format == "float32":
            audio_data = np.frombuffer(indata, dtype=np.float32)
        else:
            audio_data = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        audio_data = indata

    # Stereo → Mono
    if len(audio_data) > 0 and len(audio_data) % 2 == 0:
        mono = (audio_data[0::2] + audio_data[1::2]) / 2.0
    else:
        mono = audio_data

    # Volume RMS
    volume = float(np.sqrt(np.mean(mono ** 2))) if len(mono) > 0 else 0.0

    if volume < MIN_VOLUME_GATE:
        # Silence — fade to black
        out_r *= 0.9
        out_g *= 0.9
        out_b *= 0.9
        out_w *= 0.9
        out_master *= 0.9
        send_dmx(int(out_master), int(out_r), int(out_g), int(out_b), int(out_w), 0)
        return

    # --- FFT ---
    N = len(mono)
    if N == 0:
        return

    yf = np.fft.rfft(mono)           # Real FFT (more efficient)
    fft_data = np.abs(yf) / N        # Normalize
    fft_freqs = np.fft.rfftfreq(N, 1.0 / SAMPLE_RATE)

    # Extract per-band magnitudes
    kick_mag  = get_band_magnitude(fft_data, fft_freqs, KICK_LO, KICK_HI)
    snare_mag = get_band_magnitude(fft_data, fft_freqs, SNARE_LO, SNARE_HI)
    mid_mag   = get_band_magnitude(fft_data, fft_freqs, MID_LO, MID_HI)
    hihat_mag = get_band_magnitude(fft_data, fft_freqs, HIHAT_LO, HIHAT_HI)

    # --- AGC: track background level per band ---
    agc_kick  += AGC_SPEED * (kick_mag  - agc_kick)
    agc_snare += AGC_SPEED * (snare_mag - agc_snare)
    agc_mid   += AGC_SPEED * (mid_mag   - agc_mid)
    agc_hihat += AGC_SPEED * (hihat_mag - agc_hihat)

    # --- SPECTRAL FLUX ONSET DETECTION ---
    # A "hit" = current magnitude significantly exceeds BOTH the AGC baseline
    # AND the previous frame (spectral flux = frame-to-frame increase)
    kick_flux  = max(0.0, kick_mag  - prev_kick_mag)   # Only positive flux (onsets)
    snare_flux = max(0.0, snare_mag - prev_snare_mag)
    hihat_flux = max(0.0, hihat_mag - prev_hihat_mag)

    # Also check AGC-normalized hit (above background)
    kick_hit  = max(0.0, kick_mag  - agc_kick  * 0.8)
    snare_hit = max(0.0, snare_mag - agc_snare * 0.8)
    mid_hit   = max(0.0, mid_mag   - agc_mid   * 0.5)
    hihat_hit = max(0.0, hihat_mag - agc_hihat * 0.5)

    # Combined onset score = flux + AGC hit (both must contribute)
    kick_onset  = (kick_flux  * 0.6 + kick_hit  * 0.4) * KICK_GAIN
    snare_onset = (snare_flux * 0.6 + snare_hit * 0.4) * SNARE_GAIN
    hihat_onset = (hihat_flux * 0.5 + hihat_hit * 0.5) * HIHAT_GAIN
    mid_level   = mid_hit * MID_GAIN

    # Store for next frame
    prev_kick_mag  = kick_mag
    prev_snare_mag = snare_mag
    prev_hihat_mag = hihat_mag

    # Normalize to 0-1
    kick_i  = min(1.0, kick_onset)
    snare_i = min(1.0, snare_onset)
    hihat_i = min(1.0, hihat_onset)
    mid_i   = min(1.0, mid_level)

    # --- BEAT REGISTRATION ---
    current_time = time.time()
    is_kick_beat = kick_i > 0.3 and (current_time - last_beat_time) > ONSET_COOLDOWN
    is_snare_beat = snare_i > 0.35 and (current_time - last_beat_time) > ONSET_COOLDOWN

    if is_kick_beat or is_snare_beat:
        last_beat_time = current_time
        beat_timestamps.append(current_time)
        total_beat_count += 1

        # Palette rotation every 32 beats (roughly every 8 bars at 120bpm)
        if total_beat_count % 32 == 0:
            current_palette_idx = (current_palette_idx + 1) % len(PALETTES)
            logger.info(f"Palette change → {current_palette_idx}")

    # Clean old timestamps (keep last 3 seconds)
    beat_timestamps = [t for t in beat_timestamps if current_time - t < 3.0]

    # --- DETERMINE COLORS & ENERGY from AI cue ---
    cue_energy = 5  # Default mid-energy
    cue_strobe_ok = False
    if elapsed_seconds is not None and synced_cues:
        cue = get_active_cue(elapsed_seconds)
        if cue:
            kick_color = cue["color_1"]
            accent_color = cue["color_2"]
            cue_energy = cue.get("energy", 5)
            cue_strobe_ok = cue.get("strobe", False)
        else:
            kick_color, accent_color = PALETTES[current_palette_idx]
    else:
        kick_color, accent_color = PALETTES[current_palette_idx]

    # Scale detection sensitivity by AI cue energy (1-10)
    # High-energy sections (chorus/drop) → more reactive to drums
    energy_boost = 0.5 + (cue_energy / 10.0) * 1.0  # Range: 0.6x to 1.5x

    # --- GUARANTEED MINIMUM FLASH ON BEATS ---
    # This is the key fix: even a "soft" kick should produce a visible light pulse.
    # Without this, moderate hits get multiplied by 0.15 intensity = invisible.
    KICK_MIN_FLASH  = 0.6   # Kick beat → at least 60% brightness
    SNARE_MIN_FLASH = 0.5   # Snare beat → at least 50% brightness

    # Boost intensities on registered beats
    if is_kick_beat:
        kick_i = max(kick_i * energy_boost, KICK_MIN_FLASH)
    if is_snare_beat:
        snare_i = max(snare_i * energy_boost, SNARE_MIN_FLASH)

    # --- MIX COLORS ---
    # Kick drives primary color, snare drives accent, mids add gentle wash
    target_r = min(255.0, kick_color[0] * kick_i + accent_color[0] * snare_i + accent_color[0] * mid_i * 0.2)
    target_g = min(255.0, kick_color[1] * kick_i + accent_color[1] * snare_i + accent_color[1] * mid_i * 0.2)
    target_b = min(255.0, kick_color[2] * kick_i + accent_color[2] * snare_i + accent_color[2] * mid_i * 0.2)
    target_w = min(255.0, hihat_i * 150)   # Hi-hats → white shimmer
    target_master = min(255.0, max(40.0, volume * 5000))

    # --- BEAT FLASH: slam master dimmer to full on drum hits ---
    if is_kick_beat or is_snare_beat:
        target_master = 255.0  # Full brightness on every drum hit

    # --- SMOOTHING ---
    is_any_beat = is_kick_beat or is_snare_beat

    # Drums get INSTANT attack (0.95), non-beat frames get gentle wash
    att_rgb = ATTACK_BEAT if is_any_beat else ATTACK_WASH
    # Fast decay between beats for punchy feel; slower in low-energy sections
    dec_rgb = DECAY_BEAT if (kick_i > 0.1 or snare_i > 0.1) else DECAY_WASH

    def ema(current, target, attack, decay):
        speed = attack if target > current else decay
        return current + speed * (target - current)

    out_r = ema(out_r, target_r, att_rgb, dec_rgb)
    out_g = ema(out_g, target_g, att_rgb, dec_rgb)
    out_b = ema(out_b, target_b, att_rgb, dec_rgb)
    out_w = ema(out_w, target_w, 0.7, 0.4)
    out_master = ema(out_master, target_master, 0.8 if is_any_beat else 0.3, 0.15)

    # --- STROBE ---
    strobe = 0
    if cue_strobe_ok or (elapsed_seconds is None):
        # Only strobe when AI cue allows it, or in loopback mode (always allowed)
        if kick_i > 0.7 and snare_i > 0.4:
            strobe = 220
        elif is_kick_beat and kick_i > 0.6:
            strobe = 100  # Light strobe on strong kicks

    send_dmx(int(out_master), int(out_r), int(out_g), int(out_b), int(out_w), strobe)


# ==========================================
# MODE: SYNCED WAV PLAYBACK
# ==========================================
def run_synced_mode(show_file="current_show.json"):
    """Play a saved WAV file with beat-reactive DMX using AI cue timestamps."""
    import wave as wave_mod

    audio_path = load_ai_show(show_file)
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"No valid audio file in {show_file}! Run youtube_analyzer.py first.")
        return

    logger.info(f"[SYNCED] Playing: {os.path.basename(audio_path)}")
    if synced_cues:
        logger.info(f"[SYNCED] Using {len(synced_cues)} AI cues for color changes")

    wf = wave_mod.open(audio_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    logger.info("Starting Audio & DMX Sync... Press Ctrl+C to stop.")
    chunk_size = 1024
    data = wf.readframes(chunk_size)

    frames_played = 0
    sample_rate = wf.getframerate()
    channels = wf.getnchannels()
    last_cue_name = ""

    while data:
        stream.write(data)

        # Calculate exact elapsed time from audio frames (not wall clock — avoids drift)
        elapsed = frames_played / sample_rate
        process_audio(data, input_format="int16", elapsed_seconds=elapsed)

        # Log cue transitions
        if synced_cues:
            cue = get_active_cue(elapsed)
            if cue and cue["name"] != last_cue_name:
                last_cue_name = cue["name"]
                logger.info(f"[CUE @ {elapsed:.1f}s] {cue['name']} → {cue['color_1']}")

        frames_played += chunk_size
        data = wf.readframes(chunk_size)

    logger.info("Song complete!")
    send_dmx(0, 0, 0, 0, 0, 0)
    stream.stop_stream()
    stream.close()
    p.terminate()


# ==========================================
# MODE: LIVE LOOPBACK CAPTURE (WASAPI)
# ==========================================
def run_loopback_mode(show_file=None):
    """Capture system audio via WASAPI loopback and react to it in real-time."""
    if show_file:
        load_ai_show(show_file)

    p = pyaudio.PyAudio()

    try:
        logger.debug("Looking for default WASAPI Loopback device...")
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

        if not default_speakers["isLoopbackDevice"]:
            logger.debug(f"Default speaker: {default_speakers['name']}. Searching loopback pin...")
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    default_speakers = loopback
                    break

        logger.info(f"[LOOPBACK] Captured: {default_speakers['name']}")

    except OSError as e:
        logger.error(f"Couldn't find WASAPI device: {e}")
        return

    logger.info("Capturing system audio... Play some music!")
    logger.info("Press Ctrl+C to stop.\n")

    def py_audio_callback(in_data, frame_count, time_info, status):
        if status:
            logger.warning(f"Stream status: {status}")
        try:
            process_audio(in_data, input_format="float32")
        except Exception as ex:
            logger.error(f"Audio chunk error: {ex}")
        return (in_data, pyaudio.paContinue)

    stream = p.open(format=pyaudio.paFloat32,
                    channels=default_speakers["maxInputChannels"],
                    rate=int(default_speakers["defaultSampleRate"]),
                    frames_per_buffer=BLOCK_SIZE,
                    input=True,
                    input_device_index=default_speakers["index"],
                    stream_callback=py_audio_callback)

    with stream:
        logger.debug("Loopback stream active.")
        while stream.is_active():
            time.sleep(0.1)


# ==========================================
# MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    mode = "synced"
    show_file = "current_show.json"

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1].lower()
            i += 2
        elif args[i] == "--show" and i + 1 < len(args):
            show_file = args[i + 1]
            i += 2
        else:
            i += 1

    try:
        logger.info(f"==== AI Music Light Controller V7 ====")
        logger.info(f"Mode: {mode.upper()} | Show: {show_file}")

        if mode == "loopback":
            run_loopback_mode(show_file)
        else:
            run_synced_mode(show_file)

    except KeyboardInterrupt:
        send_dmx(0, 0, 0, 0, 0, 0)
        logger.info("Lights off. Goodbye!")
    except Exception as e:
        send_dmx(0, 0, 0, 0, 0, 0)
        logger.exception(f"Fatal Error: {e}")

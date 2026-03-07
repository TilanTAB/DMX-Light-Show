"""
DMX Light Show Engine V9 — Class-based architecture (I1 FIX)
All mutable state is encapsulated in DMXEngine.
Hardware init is deferred to run() methods (also fixes C1).
"""
import usb.core
import time
import os
import sys
import json
import math
import numpy as np
import pyaudiowpatch as pyaudio
import logging
from collections import deque
import threading
import queue

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# CONSTANTS (immutable — safe at module level)
# ==========================================
SAMPLE_RATE = 44100
# SYNC FIX: Reduced from 1024→512 to cut inherent audio delay from 21ms→10.7ms.
# At 48kHz WASAPI, 512 samples = 10.7ms per callback.
# Tradeoff: lowest FFT frequency resolution is 48000/512 = 93.75Hz (still covers kick range 30-150Hz).
BLOCK_SIZE = 512
MIN_VOLUME_GATE = 0.001

# Frequency bands
KICK_LO, KICK_HI = 30, 150
SNARE_LO, SNARE_HI = 150, 400
MID_LO, MID_HI = 400, 2000
HIHAT_LO, HIHAT_HI = 4000, 10000

# Gain per band
KICK_GAIN = 6.0
SNARE_GAIN = 4.0
MID_GAIN = 2.0
HIHAT_GAIN = 3.0

# Beat detection
ONSET_COOLDOWN = 0.12
AGC_SPEED = 0.015
LOOPBACK_GAIN_BOOST = 50.0  # WASAPI loopback is extremely quiet (~0.002 vol)
LOOPBACK_VOLUME_GATE = 0.00005  # Near-zero gate — if there's any audio, process it
LOOPBACK_AGC_THRESH = 0.3   # Lower AGC threshold for loopback (vs 0.7 for synced)

# Default palettes: (kick_color, snare_color) — high contrast pairs
DEFAULT_PALETTES = [
    ((255, 0, 50), (0, 150, 255)),     # Red vs Blue
    ((255, 50, 0), (100, 0, 255)),     # Orange vs Purple
    ((0, 255, 100), (255, 0, 200)),    # Green vs Pink
    ((255, 200, 0), (0, 50, 255)),     # Gold vs Deep Blue
    ((0, 255, 255), (255, 0, 80)),     # Cyan vs Red
    ((200, 0, 255), (0, 255, 50)),     # Violet vs Green
    ((255, 100, 0), (0, 200, 255)),    # Amber vs Sky Blue
    ((255, 0, 150), (50, 255, 0)),     # Magenta vs Lime
]

VALID_BEHAVIORS = {
    "blackout_punch", "slow_breathe", "bass_white_blast", "color_chase",
    "buildup_ramp", "static_wash", "strobe_blast", "fast_pulse",
    "beat_reactive", "rainbow_sweep", "instant_flash",
    # Ambient/chill behaviors
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer",
}


# ==========================================
# PURE HELPER FUNCTIONS (no state)
# ==========================================
def get_band_mag(fft_data, fft_freqs, lo, hi):
    idx = np.where((fft_freqs >= lo) & (fft_freqs <= hi))[0]
    return float(np.mean(fft_data[idx])) if len(idx) > 0 else 0.0


def ema(current, target, attack, decay):
    speed = attack if target > current else decay
    return current + speed * (target - current)


def lerp_color(c1, c2, t):
    """Linear interpolate between two RGB tuples. t=0→c1, t=1→c2."""
    t = max(0.0, min(1.0, t))
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
    )


# S2: Gamma correction — LEDs are non-linear. The jump from 0→50 is barely visible,
# while 200→255 is dramatic. A gamma curve (γ≈2.2) makes fades look smooth and
# breathing effects feel natural instead of jerky. Industry standard on all
# professional Martin/Chauvet/ETC fixtures.
GAMMA_LUT = np.array([int(((i / 255.0) ** 2.2) * 255) for i in range(256)], dtype=np.uint8)

def gamma_correct(value):
    """Apply perceptual gamma curve so dimming feels linear to human eyes."""
    return int(GAMMA_LUT[max(0, min(255, int(value)))])


# P0-3: bloom_attack is currently unused after the sync fix removed it from
# beat onset paths. Kept as a utility for future smooth-transition effects
# (e.g., slow color crossfades, ambient glow ramps).
def bloom_attack(current, target, speed=0.85):
    """Fast EMA attack for smooth transitions. NOT used for beat onset (must be instant)."""
    return current + speed * (target - current)


# ============================================================
# DMX ENGINE — All mutable state lives here (I1 FIX)
# ============================================================
class DMXEngine:
    """
    Encapsulates all DMX light engine state. Create one instance,
    call run_synced_mode() or run_loopback_mode().
    Hardware discovery is deferred to first use (fixes C1).
    """

    def __init__(self):
        # DMX device (lazy init)
        self.dev = None
        
        # Async DMX transmission queue (crucial for zero-latency audio)
        # Keeps maxsize=2 so we immediately drop old visual frames if the 
        # physical DMX USB stick gets backlogged, preserving total real-time sync.
        self._dmx_queue = queue.Queue(maxsize=2)
        self._dmx_thread_running = False

        # Output channels
        self.out_r = 0.0
        self.out_g = 0.0
        self.out_b = 0.0
        self.out_w = 0.0
        self.out_master = 0.0
        self.out_strobe = 0.0

        # AGC state
        self.agc_kick = 0.0
        self.agc_snare = 0.0
        self.agc_mid = 0.0
        self.agc_hihat = 0.0

        # Previous magnitudes for spectral flux
        self.prev_kick_mag = 0.0
        self.prev_snare_mag = 0.0
        self.prev_hihat_mag = 0.0

        # Beat tracking
        # M2 FIX: Use deque instead of list to avoid creating a new list every frame
        # maxlen=150 covers ~3 seconds at max 50 beats/sec
        self.beat_timestamps = deque(maxlen=150)
        self.total_beat_count = 0
        self.last_beat_time = 0.0
        self.frame_counter = 0

        # Palette and cue state
        self.palettes = list(DEFAULT_PALETTES)
        self.current_palette_idx = 0
        self.synced_cues = []

        # Auto-behavior detection (non-AI / loopback)
        self.volume_history = deque(maxlen=200)
        self.energy_state = "calm"
        self.energy_state_since = 0.0
        self.drop_cooldown = 0.0

        # Loopback direct-drive state
        self.peak_kick = 0.0
        self.peak_snare = 0.0
        self.peak_mid = 0.0

        # SYNC FIX: Pre-cached Hanning windows keyed by block size.
        self._hanning_cache = {}
        # P1-6+P1-7: Cached FFT frequency bins and band index slices.
        # Avoids recomputing np.fft.rfftfreq + np.where 4x every frame.
        self._fft_freq_cache = {}   # key=(N, sr) → fft_freqs array
        self._band_idx_cache = {}   # key=(N, sr) → dict of band name → index array
        self.peak_hihat = 0.0
        self.last_palette_rotate = 0.0
        self.beat_hold_frames = 0
        self.beat_hold_color = (255, 0, 50)
        self.prev_bps = 0.0
        self.bps_check_time = 0.0
        self.color_phase = 0
        self.last_color_change = 0.0

        # Profile-configurable parameters (defaults = Concert Punchy)
        self.profile_name = "Concert Punchy"
        self.profile_gain_boost = LOOPBACK_GAIN_BOOST
        self.profile_volume_gate = LOOPBACK_VOLUME_GATE
        self.profile_agc_thresh = LOOPBACK_AGC_THRESH
        self.profile_kick_thresh = 0.05
        self.profile_snare_thresh = 0.08
        self.profile_onset_cooldown = ONSET_COOLDOWN
        self.profile_color_cycle_mode = "rhythm"  # rhythm, time, beat
        self.profile_color_cycle_interval = 5.0
        self.profile_rhythm_change_pct = 0.30
        self.profile_deep_bass_enabled = True
        self.profile_deep_bass_thresh = 0.80
        self.profile_decay_speed = 0.75
        self.profile_glow_thresh = 0.55
        self.profile_beat_hold = 4
        self.profile_deep_bass_hold = 5
        # P1-5: Configurable kick-to-mid dominance ratio for vocal suppression.
        # 1.5 = strict (EDM, no vocals). 1.2 = relaxed (pop/rock with vocals).
        self.profile_kick_dominance_ratio = 1.5

        # Playback control state (IPC via files)
        self.playback_position = 0.0
        self.playback_duration = 0.0
        self.playback_state = "stopped"  # stopped, playing, paused
        self.current_cue_name = ""
        self.current_behavior = "beat_reactive"
        self.is_paused = False
        self._state_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playback_state.json")
        self._command_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "playback_command.json")

        # Behavior dispatch table
        self._behavior_map = {
            "blackout_punch": self._render_blackout_punch,
            "slow_breathe": self._render_slow_breathe,
            "bass_white_blast": self._render_bass_white_blast,
            "color_chase": self._render_color_chase,
            "buildup_ramp": self._render_buildup_ramp,
            "static_wash": self._render_static_wash,
            "strobe_blast": self._render_strobe_blast,
            "fast_pulse": self._render_fast_pulse,
            "beat_reactive": self._render_beat_reactive,
            "rainbow_sweep": self._render_rainbow_sweep,
            "instant_flash": self._render_blackout_punch,
            # Ambient/chill behaviors
            "ocean_drift": self._render_ocean_drift,
            "candlelight": self._render_candlelight,
            "sunset_fade": self._render_sunset_fade,
            "aurora_shimmer": self._render_aurora_shimmer,
        }

    # ----------------------------------------------------------
    # PROFILE LOADING
    # ----------------------------------------------------------
    def load_profile(self, profile_path):
        """Load a lighting profile from a JSON file."""
        try:
            with open(profile_path, 'r') as f:
                p = json.load(f)
            self.profile_name = p.get("name", "Unknown")
            self.profile_gain_boost = p.get("gain_boost", self.profile_gain_boost)
            self.profile_volume_gate = p.get("volume_gate", self.profile_volume_gate)
            self.profile_agc_thresh = p.get("agc_thresh", self.profile_agc_thresh)
            self.profile_kick_thresh = p.get("kick_thresh", self.profile_kick_thresh)
            self.profile_snare_thresh = p.get("snare_thresh", self.profile_snare_thresh)
            self.profile_onset_cooldown = p.get("onset_cooldown", self.profile_onset_cooldown)
            self.profile_color_cycle_mode = p.get("color_cycle_mode", self.profile_color_cycle_mode)
            self.profile_color_cycle_interval = p.get("color_cycle_interval", self.profile_color_cycle_interval)
            self.profile_rhythm_change_pct = p.get("rhythm_change_pct", self.profile_rhythm_change_pct)
            self.profile_deep_bass_enabled = p.get("deep_bass_enabled", self.profile_deep_bass_enabled)
            self.profile_deep_bass_thresh = p.get("deep_bass_thresh", self.profile_deep_bass_thresh)
            self.profile_decay_speed = p.get("decay_speed", self.profile_decay_speed)
            self.profile_glow_thresh = p.get("glow_thresh", self.profile_glow_thresh)
            self.profile_beat_hold = p.get("beat_hold_frames", self.profile_beat_hold)
            self.profile_deep_bass_hold = p.get("deep_bass_hold_frames", self.profile_deep_bass_hold)
            self.profile_kick_dominance_ratio = p.get("kick_dominance_ratio", self.profile_kick_dominance_ratio)
            # Load palettes if provided
            if "palettes" in p:
                self.palettes = [(tuple(c1), tuple(c2)) for c1, c2 in p["palettes"]]
            logger.info(f"[PROFILE] Loaded: {self.profile_name}")
        except Exception as e:
            logger.error(f"[PROFILE] Failed to load {profile_path}: {e}")

    # ----------------------------------------------------------
    def _init_hardware(self):
        """Find uDMX adapter. Raises RuntimeError if not found."""
        self.dev = usb.core.find(idVendor=0x16C0, idProduct=0x05DC)
        if self.dev is None:
            raise RuntimeError("uDMX not found! Please connect the adapter.")
        logger.info("uDMX found!")

        # Initialize the asynchronous USB worker driver
        if not self._dmx_thread_running:
            self._dmx_thread_running = True
            t = threading.Thread(target=self._dmx_worker, daemon=True)
            t.start()

    def _dmx_worker(self):
        """
        Background worker that continuously pulls the freshest visual frame 
        from the queue and executes the blocking physical USB transfer.
        """
        while self._dmx_thread_running:
            try:
                # Wait for up to 0.5s for a physical DMX frame update
                data = self._dmx_queue.get(timeout=0.5)
                if self.dev:
                    try:
                        self.dev.ctrl_transfer(0x40, 2, 8, 0, data)
                    except Exception as e:
                        pass # Silently drop the frame, do not stall the thread
            except queue.Empty:
                pass
            except Exception as ex:
                logger.error(f"[DMX WORKER] Error: {ex}")

    def send_dmx(self, master, red, green, blue, white=0, strobe=0):
        # S2: Apply gamma correction to color channels for perceptually linear fading.
        # Master/strobe stay linear (they're intensity controls, not color output).
        data = [
            int(max(0, min(255, master))),     # CH1: Master dimmer (linear)
            gamma_correct(red),                 # CH2: Red (gamma corrected)
            gamma_correct(green),               # CH3: Green (gamma corrected)
            gamma_correct(blue),                # CH4: Blue (gamma corrected)
            gamma_correct(white),               # CH5: White (gamma corrected)
            int(max(0, min(255, strobe))),      # CH6: Strobe (linear)
            0, 0                                # CH7-8: Unused
        ]
        try:
            # Pipelined async send. If the physical USB stick falls >40ms behind, 
            # we forcibly yank the old pending visual frame and insert the fresh one.
            if self._dmx_queue.full():
                try:
                    self._dmx_queue.get_nowait()
                except queue.Empty:
                    pass
            self._dmx_queue.put_nowait(data)
        except Exception:
            pass

    def shutdown(self):
        """C3 FIX: Centralized cleanup — call this on ANY exit path.
        Releases the USB device kernel handle so the next process can find it.
        Without this, libusb holds a stale exclusive claim and usb.core.find()
        returns None on the next run, causing 'uDMX not found'."""
        # 1. Turn off all lights
        if self.dev:
            try:
                self.dev.ctrl_transfer(0x40, 2, 8, 0, [0, 0, 0, 0, 0, 0, 0, 0])
            except Exception:
                pass

        # 2. Stop the background DMX USB worker thread
        self._dmx_thread_running = False

        # 3. Release the USB device handle back to the OS kernel
        if self.dev:
            try:
                usb.util.dispose_resources(self.dev)
                logger.info("[SHUTDOWN] USB device handle released.")
            except Exception as e:
                logger.warning(f"[SHUTDOWN] USB cleanup error: {e}")
            self.dev = None

    # ----------------------------------------------------------
    # IPC: Position reporting + command handling
    # ----------------------------------------------------------
    def _write_playback_state(self):
        """Atomically write current playback state for the API to read."""
        state = {
            "position": round(self.playback_position, 2),
            "duration": round(self.playback_duration, 2),
            "state": "paused" if self.is_paused else self.playback_state,
            "cue_name": self.current_cue_name,
            "behavior": self.current_behavior,
        }
        tmp = self._state_file + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(state, f)
            os.replace(tmp, self._state_file)  # Atomic on Windows
        except Exception:
            pass  # Non-critical — don't crash the audio loop

    def _check_playback_command(self):
        """Check for a command file from the API. Returns command dict or None.
        FIX #2: Atomically rename before reading to prevent lost commands."""
        if not os.path.exists(self._command_file):
            return None
        consumed = self._command_file + ".consumed"
        try:
            os.replace(self._command_file, consumed)  # Atomic move
        except (FileNotFoundError, OSError):
            return None  # File vanished between exists() and replace()
        try:
            with open(consumed, 'r') as f:
                cmd = json.load(f)
            os.remove(consumed)
            logger.info(f"[IPC] Command received: {cmd}")
            return cmd
        except Exception:
            try:
                os.remove(consumed)
            except OSError:
                pass
            return None

    def _cleanup_ipc_files(self):
        """Remove IPC files on shutdown."""
        for path in (self._state_file, self._command_file, self._state_file + ".tmp"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    # ----------------------------------------------------------
    # SHOW LOADING
    # ----------------------------------------------------------
    def load_ai_show(self, show_file="current_show.json"):
        """Load AI-generated palettes and cue list from a show JSON file."""
        if not os.path.exists(show_file):
            logger.warning(f"No {show_file} found! Using default fallback palettes.")
            return None

        try:
            with open(show_file, 'r') as f:
                data = json.load(f)

            plan = data.get("lighting_plan", {})

            phrases = plan.get("phrases", [])
            if phrases:
                new_palettes = []
                for p in phrases:
                    c1 = tuple(p.get("color_1", [255, 255, 255]))
                    c2 = tuple(p.get("color_2", [255, 255, 255]))
                    new_palettes.append((c1, c2))
                self.palettes = new_palettes
                self.current_palette_idx = 0
                logger.info(f"Loaded {len(self.palettes)} AI palettes")

            cues = plan.get("cues", [])
            if cues:
                self.synced_cues.clear()
                for c in cues:
                    self.synced_cues.append({
                        "start": c.get("start_time", 0),
                        "end": c.get("end_time", 0),
                        "color_1": tuple(c.get("color_1", [255, 255, 255])),
                        "color_2": tuple(c.get("color_2", [255, 255, 255])),
                        "energy": c.get("energy_level", 5),
                        "strobe": c.get("strobe_allowed", False),
                        "behavior": c.get("behavior", "beat_reactive"),
                        "dimmer": c.get("master_dimmer_percent", 80),
                        "fade": c.get("fade_speed_seconds", 1.0),
                        "name": c.get("section_name", ""),
                    })
                self.synced_cues.sort(key=lambda x: x["start"])
                logger.info(f"Loaded {len(self.synced_cues)} timestamped cues")

            return data.get("audio_file")
        except Exception as e:
            logger.error(f"Failed to load AI show: {e}")
            return None

    def _get_active_cue(self, elapsed):
        """P1-2 FIX: O(1) bisect lookup instead of O(n) reverse scan.
        Also checks end boundary so we return None when between cues."""
        import bisect
        if not self.synced_cues:
            return None
        # Binary search on start times
        if not hasattr(self, '_cue_starts') or len(self._cue_starts) != len(self.synced_cues):
            self._cue_starts = [c["start"] for c in self.synced_cues]
        idx = bisect.bisect_right(self._cue_starts, elapsed) - 1
        if 0 <= idx < len(self.synced_cues):
            cue = self.synced_cues[idx]
            if elapsed < cue["end"]:
                return cue
        return None

    # ============================================================
    # BEHAVIOR RENDERERS
    # ============================================================

    def _render_blackout_punch(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                               kick_color, accent_color, volume, cue, t):
        """Complete darkness between beats. Flash on hits."""
        # P0-1 FIX: Defensive cue access — cue can be None in edge cases
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0
        strobe_ok = cue.get("strobe", False) if cue else False

        if is_kick:
            self.out_r, self.out_g, self.out_b = kick_color
            self.out_w = 255.0 * dimmer
            self.out_master = 255.0 * dimmer
            self.out_strobe = 200.0 if strobe_ok else 0.0
        elif is_snare:
            self.out_r, self.out_g, self.out_b = accent_color
            self.out_w = 150.0 * dimmer
            self.out_master = 255.0 * dimmer
            self.out_strobe = 200.0 if strobe_ok else 0.0
        else:
            self.out_r = ema(self.out_r, 0, 0, 0.4)
            self.out_g = ema(self.out_g, 0, 0, 0.4)
            self.out_b = ema(self.out_b, 0, 0, 0.4)
            self.out_w = ema(self.out_w, 0, 0, 0.5)
            self.out_master = ema(self.out_master, 0, 0, 0.3)
            self.out_strobe = 0

    def _render_slow_breathe(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                             kick_color, accent_color, volume, cue, t):
        """Slow sinusoidal breathing between two colors."""
        fade_speed = cue.get("fade", 3.0)
        phase = (math.sin(t * math.pi / fade_speed) + 1.0) / 2.0
        color = lerp_color(kick_color, accent_color, phase)

        dimmer = cue.get("dimmer", 40) / 100.0
        brightness = dimmer * (0.6 + 0.4 * phase)

        self.out_r = ema(self.out_r, color[0] * brightness, 0.05, 0.05)
        self.out_g = ema(self.out_g, color[1] * brightness, 0.05, 0.05)
        self.out_b = ema(self.out_b, color[2] * brightness, 0.05, 0.05)
        self.out_w = ema(self.out_w, 30.0 * brightness, 0.03, 0.03)
        self.out_master = ema(self.out_master, 255.0 * brightness, 0.05, 0.05)
        self.out_strobe = 0

    def _render_bass_white_blast(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                                 kick_color, accent_color, volume, cue, t):
        """WHITE LED blasts on every kick. Colored wash underneath from mids."""
        # C1/C2 FIX: Apply AI-generated energy and dimmer
        energy = cue.get("energy", 7) if cue else 7
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0
        energy_scale = 0.5 + (energy / 10.0)  # 0.6 to 1.5

        if is_kick:
            self.out_w = 255.0 * dimmer
            self.out_master = 255.0 * dimmer
        else:
            self.out_w = ema(self.out_w, 0, 0, 0.35)

        wash_brightness = max(mid_i * 0.4 * energy_scale, 0.1)
        snare_boost = 0.6 * energy_scale if is_snare else 0.0

        self.out_r = ema(self.out_r, kick_color[0] * wash_brightness + accent_color[0] * snare_boost, 0.3, 0.08)
        self.out_g = ema(self.out_g, kick_color[1] * wash_brightness + accent_color[1] * snare_boost, 0.3, 0.08)
        self.out_b = ema(self.out_b, kick_color[2] * wash_brightness + accent_color[2] * snare_boost, 0.3, 0.08)
        self.out_master = ema(self.out_master, max(120.0 * dimmer, volume * 4000), 0.5, 0.15)
        self.out_strobe = 0

    def _render_color_chase(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                            kick_color, accent_color, volume, cue, t):
        """Alternates between kick_color and accent_color on EVERY beat."""
        # C1/C2 FIX: Apply AI-generated dimmer
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0

        if is_kick or is_snare:
            use_primary = (self.total_beat_count % 2 == 0)
            color = kick_color if use_primary else accent_color
            self.out_r, self.out_g, self.out_b = color
            self.out_w = 80.0 * dimmer if is_kick else 0.0
            self.out_master = 255.0 * dimmer
        else:
            self.out_r = ema(self.out_r, 0, 0, 0.3)
            self.out_g = ema(self.out_g, 0, 0, 0.3)
            self.out_b = ema(self.out_b, 0, 0, 0.3)
            self.out_w = ema(self.out_w, 0, 0, 0.4)
            self.out_master = ema(self.out_master, 40 * dimmer, 0, 0.15)
        self.out_strobe = 0

    def _render_buildup_ramp(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                             kick_color, accent_color, volume, cue, t):
        """Progressive intensity ramp for buildup sections."""
        # P2-7 FIX: Apply dimmer and energy (was missed in C1/C2 pass)
        dimmer = (cue.get("dimmer", 70) if cue else 70) / 100.0
        energy = cue.get("energy", 5) if cue else 5

        section_start = cue.get("start", 0) if cue else 0
        section_end = cue.get("end", section_start + 8) if cue else 8
        duration = max(1.0, section_end - section_start)
        progress = min(1.0, max(0.0, (t - section_start) / duration))

        brightness = (0.2 + progress * 0.8) * dimmer
        # Higher energy = more aggressive strobe during ramp
        strobe_threshold = 0.7 - (energy / 20.0)  # energy 10 → strobe at 0.2 progress
        strobe_val = progress * 200 if progress > strobe_threshold else 0

        color = lerp_color(kick_color, accent_color, progress)

        self.out_r = ema(self.out_r, color[0] * brightness, 0.15, 0.1)
        self.out_g = ema(self.out_g, color[1] * brightness, 0.15, 0.1)
        self.out_b = ema(self.out_b, color[2] * brightness, 0.15, 0.1)
        self.out_w = ema(self.out_w, 255.0 * progress * kick_i * dimmer if is_kick else self.out_w * 0.9, 0.8, 0.3)
        self.out_master = ema(self.out_master, 255.0 * brightness, 0.1, 0.05)
        self.out_strobe = strobe_val

    def _render_static_wash(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                            kick_color, accent_color, volume, cue, t):
        """Hold a single color wash with subtle volume-linked breathing."""
        dimmer = cue.get("dimmer", 50) / 100.0
        breath = 0.7 + 0.3 * min(1.0, volume * 2000)

        self.out_r = ema(self.out_r, kick_color[0] * dimmer * breath, 0.03, 0.03)
        self.out_g = ema(self.out_g, kick_color[1] * dimmer * breath, 0.03, 0.03)
        self.out_b = ema(self.out_b, kick_color[2] * dimmer * breath, 0.03, 0.03)
        if is_snare:
            self.out_w = 120.0
        else:
            self.out_w = ema(self.out_w, 0, 0, 0.15)
        self.out_master = ema(self.out_master, 255.0 * dimmer * breath, 0.05, 0.05)
        self.out_strobe = 0

    def _render_strobe_blast(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                             kick_color, accent_color, volume, cue, t):
        """Rapid full-white strobe. Maximum sensory impact."""
        # C2 FIX: Apply dimmer (strobe always max energy by design)
        dimmer = (cue.get("dimmer", 100) if cue else 100) / 100.0
        self.out_r, self.out_g, self.out_b = accent_color
        self.out_w = 255.0 * dimmer
        self.out_master = 255.0 * dimmer
        self.out_strobe = 240.0

    def _render_fast_pulse(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                           kick_color, accent_color, volume, cue, t):
        """Rapid beat-synced pulses with hi-hat shimmer."""
        # C1/C2 FIX: Apply AI-generated energy and dimmer
        energy = cue.get("energy", 7) if cue else 7
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0
        decay_speed = 0.3 + (energy / 20.0)  # Higher energy = faster decay = sharper pulses

        if is_kick:
            self.out_r, self.out_g, self.out_b = kick_color
            self.out_w = 200.0 * dimmer
            self.out_master = 255.0 * dimmer
        elif is_snare:
            self.out_r, self.out_g, self.out_b = accent_color
            self.out_w = 100.0 * dimmer
            self.out_master = 255.0 * dimmer
        else:
            self.out_r = ema(self.out_r, 0, 0, decay_speed)
            self.out_g = ema(self.out_g, 0, 0, decay_speed)
            self.out_b = ema(self.out_b, 0, 0, decay_speed)
            self.out_w = ema(self.out_w, hihat_i * 100 * dimmer, 0.6, 0.4)
            self.out_master = ema(self.out_master, 30 * dimmer, 0, 0.25)
        self.out_strobe = 0

    def _render_beat_reactive(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Default beat-reactive mode: kick→color_1, snare→color_2, bass→white.
        In loopback mode, maintains an ambient floor so lights are always alive."""
        energy = cue.get("energy", 5) if cue else 5
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0  # C2 FIX
        energy_boost = 0.5 + (energy / 10.0)
        is_loopback = (self.playback_state == "stopped")  # Loopback never sets playback_state

        # Ambient floor: lights should ALWAYS be somewhat on when music is playing
        if is_loopback:
            ambient = min(1.0, volume * 8000)  # Volume drives base brightness
            ambient_floor = max(0.15, ambient * 0.4)  # Min 15% brightness
        else:
            ambient_floor = 0.0

        k = max(kick_i * energy_boost, 0.6) if is_kick else kick_i
        s = max(snare_i * energy_boost, 0.5) if is_snare else snare_i

        tr = min(255.0, kick_color[0] * k + accent_color[0] * s + accent_color[0] * mid_i * 0.15)
        tg = min(255.0, kick_color[1] * k + accent_color[1] * s + accent_color[1] * mid_i * 0.15)
        tb = min(255.0, kick_color[2] * k + accent_color[2] * s + accent_color[2] * mid_i * 0.15)

        # S6: Color temperature shift based on energy state.
        # Calm sections → warm shift (amber/pink ~2700K feel)
        # High energy → cool shift (blue/violet ~6500K feel)
        # This is how professional LDs create emotional narrative through color alone.
        if self.energy_state == "calm":
            tr = min(255.0, tr * 1.1)   # Boost red warmth
            tg *= 0.85                   # Reduce green
            tb *= 0.6                    # Strongly reduce blue → warm amber
        elif self.energy_state == "high":
            tr *= 0.7                    # Reduce red warmth
            tg *= 0.9                    # Slightly reduce green
            tb = min(255.0, tb * 1.15)  # Boost blue → cool epic feel

        # Apply ambient floor — lights always glow when music plays
        if ambient_floor > 0:
            tr = max(tr, kick_color[0] * ambient_floor)
            tg = max(tg, kick_color[1] * ambient_floor)
            tb = max(tb, kick_color[2] * ambient_floor)

        tw = 255.0 if is_kick else (120.0 if is_snare else max(hihat_i * 100, ambient_floor * 50))
        tm = (255.0 if (is_kick or is_snare) else max(80.0, volume * 8000, ambient_floor * 255)) * dimmer  # C2 FIX

        is_beat = is_kick or is_snare
        att = 0.95 if is_beat else 0.15
        dec = 0.25 if (kick_i > 0.1 or snare_i > 0.1) else 0.06

        self.out_r = ema(self.out_r, tr, att, dec)
        self.out_g = ema(self.out_g, tg, att, dec)
        self.out_b = ema(self.out_b, tb, att, dec)
        self.out_w = ema(self.out_w, tw, 0.9 if is_kick else 0.3, 0.25)
        self.out_master = ema(self.out_master, tm, 0.8 if is_beat else 0.4, 0.12)

        strobe = 0
        if cue and cue.get("strobe", False):
            if kick_i > 0.7 and snare_i > 0.4:
                strobe = 220
        self.out_strobe = strobe

    def _render_rainbow_sweep(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Slowly cycle through hue spectrum. Beats cause brightness pulse."""
        # C1/C2 FIX: Apply AI-generated dimmer
        dimmer = (cue.get("dimmer", 70) if cue else 70) / 100.0

        hue = (t * 0.125) % 1.0
        h_i = int(hue * 6)
        f = hue * 6 - h_i
        q = 1.0 - f
        colors = [
            (255, int(f * 255), 0), (int(q * 255), 255, 0), (0, 255, int(f * 255)),
            (0, int(q * 255), 255), (int(f * 255), 0, 255), (255, 0, int(q * 255)),
        ]
        rgb = colors[h_i % 6]

        brightness = (0.5 + 0.5 * min(1.0, volume * 3000)) * dimmer
        if is_kick:
            brightness = 1.0 * dimmer
            self.out_w = 150.0 * dimmer
        else:
            self.out_w = ema(self.out_w, 0, 0, 0.3)

        self.out_r = ema(self.out_r, rgb[0] * brightness, 0.2, 0.08)
        self.out_g = ema(self.out_g, rgb[1] * brightness, 0.2, 0.08)
        self.out_b = ema(self.out_b, rgb[2] * brightness, 0.2, 0.08)
        self.out_master = ema(self.out_master, 255 * brightness, 0.3, 0.1)
        self.out_strobe = 0

    # ============================================================
    # AMBIENT / CHILL RENDERERS
    # Gentle, non-punchy behaviors for quiet sections, lo-fi,
    # ambient, acoustic, and chill electronic music.
    # ============================================================

    def _render_ocean_drift(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                            kick_color, accent_color, volume, cue, t):
        """Slow wave undulations — like light refracting through water.
        Two overlapping sine waves at different speeds create gentle swelling.
        Bass gives a subtle warm pulse but never flashes."""
        dimmer = (cue.get("dimmer", 50) if cue else 50) / 100.0

        # Two sine waves at non-harmonic ratios → never-repeating pattern
        wave1 = math.sin(t * 0.3) * 0.5 + 0.5       # ~3.3s period
        wave2 = math.sin(t * 0.17 + 1.2) * 0.5 + 0.5  # ~5.9s period
        blend = (wave1 * 0.6 + wave2 * 0.4)  # Combined 0.0-1.0

        color = lerp_color(kick_color, accent_color, blend)
        brightness = (0.3 + 0.3 * blend) * dimmer

        # Bass adds a gentle warmth pulse (never a flash)
        bass_warmth = min(0.15, kick_i * 0.3)
        brightness += bass_warmth

        self.out_r = ema(self.out_r, color[0] * brightness, 0.03, 0.03)
        self.out_g = ema(self.out_g, color[1] * brightness, 0.03, 0.03)
        self.out_b = ema(self.out_b, color[2] * brightness, 0.03, 0.03)
        self.out_w = ema(self.out_w, 20.0 * brightness, 0.02, 0.02)
        self.out_master = ema(self.out_master, 200.0 * brightness, 0.04, 0.03)
        self.out_strobe = 0

    def _render_candlelight(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                            kick_color, accent_color, volume, cue, t):
        """Warm organic flicker — like a room lit by candles.
        Uses pseudo-random noise for natural flicker instead of periodic sine.
        Volume modulates flicker intensity: quiet=steady glow, loud=more movement."""
        dimmer = (cue.get("dimmer", 45) if cue else 45) / 100.0

        # Pseudo-random flicker using multiple incommensurate sine waves
        # (cheaper than actual Perlin noise, visually indistinguishable on LEDs)
        flicker = (
            math.sin(t * 7.3) * 0.15 +
            math.sin(t * 13.1 + 2.0) * 0.10 +
            math.sin(t * 23.7 + 4.5) * 0.05
        )  # Range: roughly -0.3 to +0.3

        # Volume scales flicker range — quiet music = steady, loud = flickery
        vol_scale = min(1.0, volume * 2000)
        flicker_amount = 0.1 + 0.2 * vol_scale  # 0.1 (quiet) to 0.3 (loud)
        brightness = (0.5 + flicker * flicker_amount) * dimmer
        brightness = max(0.2 * dimmer, brightness)  # Never goes dark

        # Warm amber base: (255, 160, 40) blended with kick_color
        warm = lerp_color((255, 160, 40), kick_color, 0.3)

        self.out_r = ema(self.out_r, warm[0] * brightness, 0.08, 0.06)
        self.out_g = ema(self.out_g, warm[1] * brightness, 0.06, 0.04)
        self.out_b = ema(self.out_b, warm[2] * brightness, 0.04, 0.03)
        self.out_w = ema(self.out_w, 40.0 * brightness, 0.05, 0.04)
        self.out_master = ema(self.out_master, 180.0 * brightness, 0.06, 0.04)
        self.out_strobe = 0

    def _render_sunset_fade(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                            kick_color, accent_color, volume, cue, t):
        """Slow cinematic crossfade from color_1 to color_2 over section duration.
        No beat reaction — purely time-based. Mids add subtle white warmth."""
        dimmer = (cue.get("dimmer", 55) if cue else 55) / 100.0

        section_start = cue.get("start", 0) if cue else 0
        section_end = cue.get("end", section_start + 20) if cue else 20
        duration = max(1.0, section_end - section_start)
        progress = min(1.0, max(0.0, (t - section_start) / duration))

        # Smooth S-curve (smoothstep) instead of linear for cinematic feel
        smooth = progress * progress * (3.0 - 2.0 * progress)
        color = lerp_color(kick_color, accent_color, smooth)
        brightness = (0.4 + 0.2 * smooth) * dimmer

        # Mids add subtle white warmth
        mid_warmth = min(0.1, mid_i * 0.2)

        self.out_r = ema(self.out_r, color[0] * brightness, 0.02, 0.02)
        self.out_g = ema(self.out_g, color[1] * brightness, 0.02, 0.02)
        self.out_b = ema(self.out_b, color[2] * brightness, 0.02, 0.02)
        self.out_w = ema(self.out_w, 50.0 * (brightness + mid_warmth), 0.03, 0.02)
        self.out_master = ema(self.out_master, 200.0 * brightness, 0.03, 0.02)
        self.out_strobe = 0

    def _render_aurora_shimmer(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                               kick_color, accent_color, volume, cue, t):
        """Multi-frequency color shimmer — like northern lights.
        Three sine waves at prime ratios modulate R, G, B independently,
        creating a slowly evolving color field that never repeats."""
        dimmer = (cue.get("dimmer", 50) if cue else 50) / 100.0

        # Three independent oscillators at incommensurate frequencies
        r_wave = math.sin(t * 0.23) * 0.5 + 0.5            # ~27s period
        g_wave = math.sin(t * 0.23 * 1.3 + 2.1) * 0.5 + 0.5  # ~21s period
        b_wave = math.sin(t * 0.23 * 1.7 + 4.3) * 0.5 + 0.5  # ~16s period

        # Blend oscillator outputs with the AI-chosen colors
        r = kick_color[0] * r_wave + accent_color[0] * (1 - r_wave)
        g = kick_color[1] * g_wave + accent_color[1] * (1 - g_wave)
        b = kick_color[2] * b_wave + accent_color[2] * (1 - b_wave)

        # Volume gently scales brightness (40-70% range, never harsh)
        vol_brightness = 0.4 + 0.3 * min(1.0, volume * 2000)
        brightness = vol_brightness * dimmer

        self.out_r = ema(self.out_r, r * brightness, 0.04, 0.03)
        self.out_g = ema(self.out_g, g * brightness, 0.04, 0.03)
        self.out_b = ema(self.out_b, b * brightness, 0.04, 0.03)
        self.out_w = ema(self.out_w, 15.0 * brightness, 0.02, 0.02)
        self.out_master = ema(self.out_master, 200.0 * brightness, 0.04, 0.03)
        self.out_strobe = 0

    # ============================================================
    # LOOPBACK DIRECT-DRIVE RENDERER
    # Maps frequency energy directly to light output — no beat
    # onset detection needed. Every frame produces visible light.
    # ============================================================
    def _render_loopback_direct(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                                kick_i, snare_i, hihat_i, mid_i,
                                is_kick, is_snare,
                                color_1, color_2, volume, t, color_idx=0):
        """
        Loopback renderer with rhythm-aware color cycling and deep bass combos.
        - Colors cycle on rhythm changes (tempo shifts, breaks, fills)
        - Deep bass hits trigger dual-color combos (purple, cyan, yellow)
        """
        self.peak_kick = max(kick_mag, self.peak_kick * 0.97)
        self.peak_mid = max(mid_mag, self.peak_mid * 0.97)

        def norm(val, peak):
            if peak < 0.00001:
                return 0.0
            return min(1.0, val / (peak * 0.5))

        bass = norm(kick_mag, self.peak_kick)
        mids = norm(mid_mag, self.peak_mid)

        is_deep_bass = self.profile_deep_bass_enabled and self.peak_kick > 0.00001 and (kick_mag / self.peak_kick) > self.profile_deep_bass_thresh

        # S4: Velocity-sensitive brightness — soft beats get dim, hard beats get blast.
        kick_velocity = min(1.0, kick_i / max(self.profile_kick_thresh, 0.01))
        snare_velocity = min(1.0, snare_i / max(self.profile_snare_thresh, 0.01))
        beat_velocity = max(kick_velocity, snare_velocity)
        velocity_brightness = 120.0 + (135.0 * beat_velocity)

        # ── DEEP BASS COMBO: Dual-color blast ──
        # SYNC FIX: Instant snap (not bloom) on beat onset. Bloom was adding 20-40ms
        # visual delay that made lights feel "late." Beat onset MUST be instant to
        # synchronize with the audio transient the ear just heard.
        if (is_kick or is_snare) and is_deep_bass:
            combo = color_idx % 4
            if combo == 0:
                self.out_r, self.out_g, self.out_b = 255.0, 0, 200.0
            elif combo == 1:
                self.out_r, self.out_g, self.out_b = 255.0, 200.0, 0
            elif combo == 2:
                self.out_r, self.out_g, self.out_b = 0, 200.0, 255.0
            else:
                self.out_r, self.out_g, self.out_b = 200.0, 200.0, 200.0
            self.out_w = 255.0
            self.out_master = velocity_brightness
            self.out_strobe = 0
            self.beat_hold_frames = self.profile_deep_bass_hold
            return

        # ── NORMAL BEAT: Instant color snap ──
        if is_kick or is_snare:
            self.out_r = 255.0 if color_idx == 0 else 0
            self.out_g = 255.0 if color_idx == 2 else 0
            self.out_b = 255.0 if color_idx == 1 else 0
            self.out_w = 255.0 if color_idx == 3 else 0
            self.out_master = velocity_brightness
            self.out_strobe = 0
            self.beat_hold_frames = self.profile_beat_hold
            return

        # ── HOLD after beat ──
        # SYNC FIX: Master stays at full during hold so the flash feels crisp.
        # Previously master was decaying to 60 during hold, making flashes feel mushy.
        # S3 afterglow applies only to color channels, not master brightness.
        if self.beat_hold_frames > 0:
            self.beat_hold_frames -= 1
            # S3: Warm afterglow — colors shift warm as they decay
            self.out_w *= 0.80
            self.out_r *= 0.95
            self.out_g *= 0.88
            self.out_b *= 0.82
            self.out_master = max(self.out_master, 200.0)  # Stay bright during hold
            self.out_strobe = 0
            return

        # ── BETWEEN BEATS: Subtle glow + breathing white ──
        bass_active = bass > self.profile_glow_thresh

        if bass_active:
            glow = (bass - self.profile_glow_thresh) * 0.4
            self.out_r = 255.0 * glow if color_idx == 0 else 0
            self.out_g = 255.0 * glow if color_idx == 2 else 0
            self.out_b = 255.0 * glow if color_idx == 1 else 0
            self.out_w = 255.0 * glow if color_idx == 3 else 0
            self.out_master = max(15, glow * 180)
        else:
            # S3: Warm afterglow tail
            decay = self.profile_decay_speed
            self.out_r *= decay * 1.05
            self.out_g *= decay * 0.95
            self.out_b *= decay * 0.85
            self.out_w *= decay
            self.out_master *= decay

        # S5: Breathing white floor — subtle sine-wave prevents dead room
        breath = math.sin(t * math.pi * 2.0) * 0.5 + 0.5
        white_floor = volume * 800 * breath * 0.12
        self.out_w = max(self.out_w, min(30.0, white_floor))

        self.out_strobe = 0

    # ============================================================
    # AUTO-BEHAVIOR DETECTION (Non-AI / Loopback Mode)
    # ============================================================
    def _detect_auto_behavior(self, volume, kick_i, snare_i, current_time, beats_per_sec):
        """Analyze real-time energy to auto-pick behavior for loopback mode."""
        self.volume_history.append(volume)

        if len(self.volume_history) < 30:
            return "beat_reactive"

        history = list(self.volume_history)
        overall_avg = float(np.mean(history))
        # P0-2 FIX: When <60 samples, past_energy defaults to overall_avg
        # so the past-vs-recent comparison doesn't collapse to always-equal.
        past_energy = float(np.mean(history[:60])) if len(history) >= 60 else overall_avg
        recent_energy = float(np.mean(history[-30:]))

        time_in_state = current_time - self.energy_state_since
        self.drop_cooldown = max(0, self.drop_cooldown - 0.023)

        if self.energy_state == "calm":
            if recent_energy > past_energy * 1.8 and recent_energy > MIN_VOLUME_GATE * 3:
                self.energy_state = "building"
                self.energy_state_since = current_time
            elif beats_per_sec > 2.5 and recent_energy > overall_avg * 1.3:
                self.energy_state = "high"
                self.energy_state_since = current_time

        elif self.energy_state == "building":
            if recent_energy > past_energy * 2.5 and self.drop_cooldown <= 0:
                self.energy_state = "high"
                self.energy_state_since = current_time
                self.drop_cooldown = 5.0
            elif recent_energy < past_energy * 0.6:
                self.energy_state = "calm"
                self.energy_state_since = current_time
            elif time_in_state > 12.0:
                self.energy_state = "high" if beats_per_sec > 2.0 else "calm"
                self.energy_state_since = current_time

        elif self.energy_state == "high":
            if recent_energy < overall_avg * 0.5 and time_in_state > 3.0:
                self.energy_state = "dropping"
                self.energy_state_since = current_time
            elif time_in_state > 20.0:
                self.energy_state = "calm"
                self.energy_state_since = current_time

        elif self.energy_state == "dropping":
            if time_in_state > 4.0:
                self.energy_state = "calm"
                self.energy_state_since = current_time
            elif recent_energy > overall_avg * 1.5:
                self.energy_state = "high"
                self.energy_state_since = current_time

        # AUTO-BEHAVIOR DISPATCH: Picks punchy or chill based on energy + BPS
        if self.energy_state == "calm":
            if beats_per_sec < 0.5 and recent_energy < overall_avg * 0.5:
                # Very quiet — rotate through ambient behaviors for variety
                ambient_pool = ["ocean_drift", "candlelight", "aurora_shimmer", "sunset_fade"]
                ambient_idx = int(current_time / 15.0) % len(ambient_pool)  # Switch every 15s
                return ambient_pool[ambient_idx]
            elif beats_per_sec < 1.0 and recent_energy < overall_avg * 0.7:
                return "slow_breathe"
            elif beats_per_sec < 1.5:
                return "beat_reactive"  # Light beats → standard
            else:
                return "beat_reactive"
        elif self.energy_state == "building":
            return "buildup_ramp"
        elif self.energy_state == "high":
            # High energy → concert punchy modes based on BPS intensity
            if beats_per_sec > 3.0:
                return "blackout_punch"
            elif beats_per_sec > 2.0:
                return "bass_white_blast"
            return "fast_pulse"
        elif self.energy_state == "dropping":
            # Dropping energy → transition to chill
            if time_in_state < 2.0:
                return "sunset_fade"  # Cinematic transition out
            return "slow_breathe"
        return "beat_reactive"

    # ============================================================
    # CORE AUDIO PROCESSING
    # ============================================================
    def process_audio(self, indata, input_format="int16", elapsed_seconds=None, actual_sample_rate=None):
        """Core engine: FFT → onset detection → behavior dispatch → DMX output."""
        self.frame_counter += 1
        is_loopback = (elapsed_seconds is None)
        sr = actual_sample_rate or SAMPLE_RATE

        # --- Decode ---
        if isinstance(indata, bytes):
            if input_format == "float32":
                audio_data = np.frombuffer(indata, dtype=np.float32)
            else:
                audio_data = np.frombuffer(indata, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_data = indata

        if len(audio_data) > 0 and len(audio_data) % 2 == 0:
            mono = (audio_data[0::2] + audio_data[1::2]) / 2.0
        else:
            mono = audio_data

        volume = float(np.sqrt(np.mean(mono ** 2))) if len(mono) > 0 else 0.0

        # Debug logging for first 50 frames in loopback mode
        if is_loopback and self.frame_counter <= 50 and self.frame_counter % 10 == 0:
            logger.debug(f"[LOOPBACK frame {self.frame_counter}] vol={volume:.6f} samples={len(mono)} sr={sr}")


        vol_gate = self.profile_volume_gate if is_loopback else MIN_VOLUME_GATE
        if volume < vol_gate:
            self.out_r *= 0.92; self.out_g *= 0.92; self.out_b *= 0.92
            self.out_w *= 0.92; self.out_master *= 0.92
            self.send_dmx(self.out_master, self.out_r, self.out_g, self.out_b, self.out_w, 0)
            return

        # --- FFT & Windowing ---
        N = len(mono)
        if N == 0:
            return

        # L1 FIX: Apply a Hanning window before FFT to prevent spectral leakage.
        # SYNC FIX: Cache the window array to avoid creating a new one every frame.
        if N not in self._hanning_cache:
            self._hanning_cache[N] = np.hanning(N)
        windowed = mono * self._hanning_cache[N]

        yf = np.fft.rfft(windowed)
        # Multiply by 2.0 to compensate for the Hanning window's 50% amplitude
        # reduction. Without this, real kick drums would be too quiet to break
        # the existing 0.05-0.65 profile thresholds.
        fft_data = (np.abs(yf) / N) * 2.0

        # P1-6+P1-7 FIX: Cache FFT frequency bins AND band index arrays.
        # Saves 5 numpy array allocations per frame (fft_freqs + 4 np.where calls).
        cache_key = (N, sr)
        if cache_key not in self._fft_freq_cache:
            fft_freqs = np.fft.rfftfreq(N, 1.0 / sr)
            self._fft_freq_cache[cache_key] = fft_freqs
            # Precompute band index arrays — these never change for same N+sr
            self._band_idx_cache[cache_key] = {
                'kick': np.where((fft_freqs >= KICK_LO) & (fft_freqs <= KICK_HI))[0],
                'snare': np.where((fft_freqs >= SNARE_LO) & (fft_freqs <= SNARE_HI))[0],
                'mid': np.where((fft_freqs >= MID_LO) & (fft_freqs <= MID_HI))[0],
                'hihat': np.where((fft_freqs >= HIHAT_LO) & (fft_freqs <= HIHAT_HI))[0],
            }
        bands = self._band_idx_cache[cache_key]

        kick_mag = float(np.mean(fft_data[bands['kick']])) if len(bands['kick']) > 0 else 0.0
        snare_mag = float(np.mean(fft_data[bands['snare']])) if len(bands['snare']) > 0 else 0.0
        mid_mag = float(np.mean(fft_data[bands['mid']])) if len(bands['mid']) > 0 else 0.0
        hihat_mag = float(np.mean(fft_data[bands['hihat']])) if len(bands['hihat']) > 0 else 0.0

        self.agc_kick += AGC_SPEED * (kick_mag - self.agc_kick)
        self.agc_snare += AGC_SPEED * (snare_mag - self.agc_snare)
        self.agc_mid += AGC_SPEED * (mid_mag - self.agc_mid)
        self.agc_hihat += AGC_SPEED * (hihat_mag - self.agc_hihat)

        # Spectral flux
        kick_flux = max(0.0, kick_mag - self.prev_kick_mag)
        snare_flux = max(0.0, snare_mag - self.prev_snare_mag)
        hihat_flux = max(0.0, hihat_mag - self.prev_hihat_mag)

        agc_thresh = self.profile_agc_thresh if is_loopback else 0.7
        kick_hit = max(0.0, kick_mag - self.agc_kick * agc_thresh)
        snare_hit = max(0.0, snare_mag - self.agc_snare * agc_thresh)
        mid_hit = max(0.0, mid_mag - self.agc_mid * 0.4)
        hihat_hit = max(0.0, hihat_mag - self.agc_hihat * 0.4)

        gain_mult = self.profile_gain_boost if is_loopback else 1.0
        kick_i = min(1.0, (kick_flux * 0.6 + kick_hit * 0.4) * KICK_GAIN * gain_mult)
        snare_i = min(1.0, (snare_flux * 0.6 + snare_hit * 0.4) * SNARE_GAIN * gain_mult)
        hihat_i = min(1.0, (hihat_flux * 0.5 + hihat_hit * 0.5) * HIHAT_GAIN * gain_mult)
        mid_i = min(1.0, mid_hit * MID_GAIN * gain_mult)

        self.prev_kick_mag = kick_mag
        self.prev_snare_mag = snare_mag
        self.prev_hihat_mag = hihat_mag

        # --- Beat registration ---
        current_time = time.time()
        kick_thresh = self.profile_kick_thresh if is_loopback else 0.3
        snare_thresh = self.profile_snare_thresh if is_loopback else 0.35
        cooldown_ok = (current_time - self.last_beat_time) > self.profile_onset_cooldown

        # VOCAL SUPPRESSION: Two gates that prevent vocals from triggering beats.
        #
        # Gate 1 — Kick-to-mid ratio: Real kick drums have overwhelming energy
        # below 150Hz relative to 400-2000Hz mids. Vocals are the opposite —
        # their fundamental (150-400Hz) and harmonics dominate the mid range.
        # If mid_mag >= kick_mag, the energy is vocal, not percussive.
        # P1-5 FIX: Use profile-configurable ratio (default 1.5 for EDM, lower for pop/rock)
        kick_dominates = kick_mag > (mid_mag * self.profile_kick_dominance_ratio)
        #
        # Gate 2 — Snare flux sharpness: Real snare hits create an explosive
        # transient (huge spectral flux). Vocal consonants create gradual energy
        # changes. We require the flux component to dominate the steady-state
        # magnitude — if most of the energy is steady-state, it's sustained audio
        # (vocals/instruments), not a percussive hit.
        # P1-4 FIX: Add minimum flux threshold to prevent noise triggering
        snare_is_transient = snare_flux > (snare_hit * 0.5) if snare_hit > 0 else snare_flux > 0.001

        is_kick = kick_i > kick_thresh and cooldown_ok and kick_dominates
        is_snare = snare_i > snare_thresh and cooldown_ok and snare_is_transient

        # Periodic diagnostic logging every 100 frames
        if is_loopback and self.frame_counter % 100 == 0:
            logger.info(f"[DIAG] vol={volume:.4f} kick_i={kick_i:.4f} mid_mag={mid_mag:.4f} "
                        f"kick_dom={kick_dominates} snare_trans={snare_is_transient} "
                        f"is_kick={is_kick} beats={self.total_beat_count}")

        if is_kick or is_snare:
            self.last_beat_time = current_time
            self.beat_timestamps.append(current_time)
            self.total_beat_count += 1

            # Only rotate palette in synced mode — loopback has its own R→B→G→W rotation
            if not is_loopback:
                rotation_interval = 16 if self.energy_state == "high" else 32
                if self.total_beat_count % rotation_interval == 0:
                    self.current_palette_idx = (self.current_palette_idx + 1) % len(self.palettes)

        # M2 FIX: Evict old timestamps in-place with O(1) popleft instead of
        # creating a new list every frame (was generating ~47 throwaway lists/sec)
        while self.beat_timestamps and (current_time - self.beat_timestamps[0]) > 3.0:
            self.beat_timestamps.popleft()
        beats_per_sec = len(self.beat_timestamps) / 3.0

        # --- Dispatch ---
        t_start_render = time.perf_counter()
        
        if is_loopback:
            # LOOPBACK: Direct energy-to-light with profile-aware color cycling.
            t = self.frame_counter * BLOCK_SIZE / sr

            # ── Color cycling based on profile mode ──
            cycle_mode = self.profile_color_cycle_mode

            if cycle_mode == "rhythm":
                # Rhythm-change detection: advance color on BPS shifts
                if current_time - self.bps_check_time > 2.0:
                    bps_change = abs(beats_per_sec - self.prev_bps)
                    bps_threshold = max(0.5, self.prev_bps * self.profile_rhythm_change_pct)

                    if bps_change > bps_threshold and self.prev_bps > 0:
                        self.color_phase = (self.color_phase + 1) % 4
                        self.last_color_change = current_time
                        logger.info(f"[RHYTHM] BPS {self.prev_bps:.1f} -> {beats_per_sec:.1f} | Color -> {['R','B','G','W'][self.color_phase]}")

                    self.prev_bps = beats_per_sec
                    self.bps_check_time = current_time

                # Fallback timer
                if current_time - self.last_color_change > self.profile_color_cycle_interval:
                    self.color_phase = (self.color_phase + 1) % 4
                    self.last_color_change = current_time

            elif cycle_mode == "time":
                # Fixed time-based rotation
                self.color_phase = int(t / self.profile_color_cycle_interval) % 4

            elif cycle_mode == "beat":
                # Advance on every 4th beat
                self.color_phase = (self.total_beat_count // 4) % 4

            color_phase = self.color_phase

            kick_color, accent_color = self.palettes[self.current_palette_idx % len(self.palettes)]
            self._render_loopback_direct(
                kick_mag, snare_mag, mid_mag, hihat_mag,
                kick_i, snare_i, hihat_i, mid_i,
                is_kick, is_snare,
                kick_color, accent_color, volume, t, color_phase
            )
        else:
            if self.synced_cues:
                cue = self._get_active_cue(elapsed_seconds)
                if cue:
                    kick_color = cue["color_1"]
                    accent_color = cue["color_2"]
                    behavior = cue.get("behavior", "beat_reactive")
                else:
                    kick_color, accent_color = self.palettes[self.current_palette_idx]
                    behavior = "beat_reactive"
            else:
                kick_color, accent_color = self.palettes[self.current_palette_idx]
                behavior = "beat_reactive"

            # --- Dispatch ---
            t = elapsed_seconds if elapsed_seconds is not None else self.frame_counter * BLOCK_SIZE / SAMPLE_RATE
            renderer = self._behavior_map.get(behavior, self._render_beat_reactive)
            renderer(kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                     kick_color, accent_color, volume, cue or {"energy": 7, "dimmer": 80}, t)

        t_start_usb = time.perf_counter()
        self.send_dmx(self.out_master, self.out_r, self.out_g, self.out_b, self.out_w, self.out_strobe)
        t_end_usb = time.perf_counter()
        
        proc_time = (t_start_usb - t_start_render) * 1000.0
        usb_time = (t_end_usb - t_start_usb) * 1000.0
        total_cb_time = proc_time + usb_time
        
        if is_loopback and total_cb_time > 10.0:
            logger.warning(f"[LATENCY] PyAudio loop took {total_cb_time:.1f}ms (Proc: {proc_time:.1f}ms, USB: {usb_time:.1f}ms)")

    # ==========================================
    # MODE: SYNCED WAV PLAYBACK (I4 FIX: try/finally)
    # ==========================================
    def run_synced_mode(self, show_file="current_show.json"):
        import wave as wave_mod

        self._init_hardware()
        audio_path = self.load_ai_show(show_file)
        if not audio_path or not os.path.exists(audio_path):
            logger.error(f"No valid audio file in {show_file}!")
            return

        logger.info(f"[SYNCED] Playing: {os.path.basename(audio_path)}")
        if self.synced_cues:
            behaviors = set(c["behavior"] for c in self.synced_cues)
            logger.info(f"[SYNCED] {len(self.synced_cues)} cues, behaviors: {behaviors}")

        wf = wave_mod.open(audio_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        sample_rate = wf.getframerate()
        total_frames = wf.getnframes()
        self.playback_duration = total_frames / sample_rate
        self.playback_state = "playing"
        state_write_counter = 0

        try:
            # P1-1 FIX: Use BLOCK_SIZE for consistency with loopback mode
            chunk_size = BLOCK_SIZE
            data = wf.readframes(chunk_size)
            frames_played = 0
            last_cue_name = ""

            while data:
                # --- Check for commands (seek, pause, resume) ---
                cmd = self._check_playback_command()
                if cmd:
                    action = cmd.get("command")
                    if action == "seek":
                        target = float(cmd.get("position", 0))
                        target_frame = int(target * sample_rate)
                        target_frame = max(0, min(target_frame, total_frames - 1))
                        wf.setpos(target_frame)
                        frames_played = target_frame
                        logger.info(f"[SEEK] Jumped to {target:.1f}s (frame {target_frame})")
                        data = wf.readframes(chunk_size)
                        continue
                    elif action == "pause":
                        self.is_paused = True
                        logger.info("[PAUSE]")
                    elif action == "resume":
                        self.is_paused = False
                        logger.info("[RESUME]")
                    elif action == "stop":
                        logger.info("[STOP] Received stop command")
                        break

                # --- Pause loop ---
                if self.is_paused:
                    state_write_counter += 1
                    if state_write_counter % 5 == 0:
                        self.playback_position = frames_played / sample_rate
                        self._write_playback_state()
                    time.sleep(0.05)
                    continue

                # --- Normal playback ---
                stream.write(data)
                elapsed = frames_played / sample_rate
                self.playback_position = elapsed
                self.process_audio(data, input_format="int16", elapsed_seconds=elapsed)

                if self.synced_cues:
                    cue = self._get_active_cue(elapsed)
                    if cue:
                        if cue["name"] != last_cue_name:
                            last_cue_name = cue["name"]
                            logger.info(f"[CUE {elapsed:.1f}s] {cue['name']} → {cue['behavior']}")
                        self.current_cue_name = cue.get("name", "")
                        self.current_behavior = cue.get("behavior", "beat_reactive")

                # Write state every ~10 frames (~230ms at 44.1kHz/1024)
                state_write_counter += 1
                if state_write_counter % 10 == 0:
                    self._write_playback_state()

                frames_played += chunk_size
                data = wf.readframes(chunk_size)

            # Playback finished naturally
            self.playback_state = "stopped"
            self.playback_position = self.playback_duration
            self._write_playback_state()
        finally:
            self.send_dmx(0, 0, 0, 0, 0, 0)
            self.playback_state = "stopped"
            self._write_playback_state()
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            self._cleanup_ipc_files()

    # ==========================================
    # MODE: LIVE LOOPBACK
    # ==========================================
    def run_loopback_mode(self, show_file=None):
        self._init_hardware()
        if show_file:
            self.load_ai_show(show_file)

        p = pyaudio.PyAudio()
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            if not default_speakers["isLoopbackDevice"]:
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
            logger.info(f"[LOOPBACK] {default_speakers['name']}")
        except OSError as e:
            logger.error(f"WASAPI error: {e}")
            return

        # Store actual sample rate for FFT calculations
        device_sr = int(default_speakers["defaultSampleRate"])
        logger.info(f"[LOOPBACK] Device sample rate: {device_sr}Hz")

        def callback(in_data, frame_count, time_info, status):
            try:
                self.process_audio(in_data, input_format="float32", actual_sample_rate=device_sr)
            except Exception as ex:
                logger.error(f"Audio error: {ex}")
            return (in_data, pyaudio.paContinue)

        stream = p.open(format=pyaudio.paFloat32,
                        channels=default_speakers["maxInputChannels"],
                        rate=device_sr,
                        frames_per_buffer=BLOCK_SIZE,
                        input=True,
                        input_device_index=default_speakers["index"],
                        stream_callback=callback)
        try:
            with stream:
                while stream.is_active():
                    time.sleep(0.1)
        finally:
            # C1 FIX: Full resource cleanup to prevent WASAPI port exhaustion.
            # Without this, every loopback restart leaks a COM audio endpoint.
            # After ~15 leaks, Windows refuses new WASAPI connections entirely.
            self.send_dmx(0, 0, 0, 0, 0, 0)
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass  # Stream may already be dead
            p.terminate()  # Release the WASAPI COM port binding
            self._dmx_thread_running = False  # Stop the background USB worker thread
            logger.info("[LOOPBACK] Cleaned up: stream closed, PyAudio terminated, DMX worker stopped.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    mode = "synced"
    show_file = "current_show.json"
    profile_file = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1].lower()
            i += 2
        elif args[i] == "--show" and i + 1 < len(args):
            show_file = args[i + 1]
            i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile_file = args[i + 1]
            i += 2
        else:
            i += 1

    engine = DMXEngine()

    if profile_file:
        engine.load_profile(profile_file)

    try:
        logger.info(f"==== AI Music Light Controller V9 ====")
        logger.info(f"Mode: {mode.upper()} | Show: {show_file} | Profile: {engine.profile_name}")
        if mode == "loopback":
            engine.run_loopback_mode(show_file)
        else:
            engine.run_synced_mode(show_file)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
    finally:
        # C3 FIX: ALWAYS release USB + stop DMX thread, no matter how we exit.
        # This prevents the "uDMX not found" ghost handle bug on next startup.
        engine.shutdown()
        logger.info("Shutdown complete.")

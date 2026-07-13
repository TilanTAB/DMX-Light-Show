"""
DMX Light Show Engine V9 — Class-based architecture (I1 FIX)
All mutable state is encapsulated in DMXEngine.
Hardware init is deferred to run() methods (also fixes C1).
"""
import usb.core
import time
import os
import json
import math
import numpy as np
import logging
from collections import deque
import threading
import queue

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Offline verification: DMX_DRY_RUN=1 skips USB init and records frames on
# self.last_frame instead of transferring. Used by the dry-run playback branch.
DRY_RUN = os.getenv("DMX_DRY_RUN") == "1"

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

# abyssal_bloom renderer tuning (module constants, not profile_* — avoids
# adding new load_profile keys for a single renderer's parameters).
ABYSSAL_FLOOR_MIN = 0.03
ABYSSAL_FLOOR_MAX = 0.10
ABYSSAL_BLOOM_BASE = 0.45
ABYSSAL_BLOOM_MAX = 0.70
ABYSSAL_BLOOM_BASS_GAIN = 0.25
ABYSSAL_BLOOM_RISE = 1.5
ABYSSAL_BLOOM_HOLD = 0.4
ABYSSAL_BLOOM_FALL = 3.5
ABYSSAL_BLOOM_GAP_MIN = 3.0
ABYSSAL_BLOOM_INTERVAL = 11.0
ABYSSAL_BLOOM_NUDGE_THRESH = 0.25
ABYSSAL_GLINT_RISE = 0.2
ABYSSAL_GLINT_FALL = 0.6
ABYSSAL_GLINT_INTERVAL = 18.0
ABYSSAL_GLINT_THRESH = 0.5
ABYSSAL_DISCONTINUITY_THRESHOLD = 1.0  # seconds; a real audio-frame-to-frame
# gap while this behavior stays selected is ~0.01s. Anything bigger means
# this renderer was skipped (ambient_pool rotated away and back) or a seek
# happened -- either way, treat it as "just arrived" so blooms/glints reset
# to rare instead of firing instantly.
# This safety depends on `t` being a sample-derived virtual clock (frame_counter
# or frames_played / sample_rate), NOT time.time() -- a stalled audio thread
# doesn't advance `t` at all, so no wall-clock delay can misfire this. If `t`
# is ever changed to wall-clock, this threshold must be revisited.

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

# cinematic_swell renderer tuning. Film-score hits: a calm drifting floor,
# and a slow eased swell (rise + fall, ~2s total) fired only by STRONG kicks.
# Accumulated-dt like golden_anthem: swell/drift progress advances by clamped
# per-frame dt, so seeks and ambient-rotation re-entry cannot corrupt it.
# Gate on the RAW kick ratio, not _beat_velocity: they do different jobs.
# _beat_velocity GRADES every onset from ratio 1.0 up (beat_velocity_from_ratio,
# dmx_punch.py); this renderer GATES at CINE_TRIGGER_RATIO so weak onsets are
# ignored entirely and the floor stays calm. CINE_FULL_RATIO matches
# PUNCH_FULL_RATIO (3.0) deliberately so "full blast" means the same hit.
CINE_TRIGGER_RATIO = 1.6      # kick_i must exceed 1.6x onset threshold to swell
CINE_FULL_RATIO = 3.0         # ratio at which the swell peaks at max
CINE_RISE_S = 0.5             # eased rise duration (seconds)
CINE_FALL_S = 1.4             # eased fall duration (seconds)
CINE_FLOOR_MIN = 0.08         # never-dark floor (not dimmer-scaled; abyssal PWM lesson)
CINE_PEAK_MAX = 0.90          # hard cap on swell peak brightness fraction
CINE_DRIFT_PERIOD_S = 20.0    # idle floor drifts color_1 <-> color_2 this slowly
CINE_WHITE_PEAK = 100.0       # max white-channel lift at swell peak
CINE_WHITE_KNEE = 0.7         # white ramps in above this swell level (up to PEAK_MAX)
CINE_DT_CLAMP = 0.1           # max seconds of progress per frame
CINE_DISCONTINUITY_THRESHOLD = 1.0  # bigger call-gap => one nominal frame

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
    "ocean_drift", "candlelight", "sunset_fade", "aurora_shimmer", "abyssal_bloom",
    "golden_anthem", "cinematic_swell",
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


def _cine_envelope(age):
    """Eased swell envelope: smoothstep up over CINE_RISE_S, smoothstep down
    over CINE_FALL_S. `age` is seconds since trigger; returns 0..1.
    Returns 0.0 once the swell is finished (age >= rise+fall)."""
    if age < 0.0:
        return 0.0
    if age < CINE_RISE_S:
        x = age / CINE_RISE_S
        return x * x * (3.0 - 2.0 * x)
    fall_age = age - CINE_RISE_S
    if fall_age >= CINE_FALL_S:
        return 0.0
    x = 1.0 - fall_age / CINE_FALL_S
    return x * x * (3.0 - 2.0 * x)


def _cine_rise_age_for(env_frac):
    """Inverse of the rise segment of _cine_envelope: the age in
    [0, CINE_RISE_S] whose envelope equals env_frac. Used by sag-free
    retrigger: resume the rise at the height already on the lights.
    Smoothstep has no closed-form inverse; bisection (monotone on the
    rise) converges to ~5e-7 s in 20 iterations."""
    env_frac = max(0.0, min(1.0, env_frac))
    lo, hi = 0.0, CINE_RISE_S
    for _ in range(20):
        mid = (lo + hi) / 2.0
        x = mid / CINE_RISE_S
        if x * x * (3.0 - 2.0 * x) < env_frac:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# P0-3: bloom_attack is currently unused after the sync fix removed it from
# beat onset paths. Kept as a utility for future smooth-transition effects
# (e.g., slow color crossfades, ambient glow ramps).
def bloom_attack(current, target, speed=0.85):
    """Fast EMA attack for smooth transitions. NOT used for beat onset (must be instant)."""
    return current + speed * (target - current)


# ============================================================
# DMX ENGINE — All mutable state lives here (I1 FIX)
# ============================================================

import bisect
import zlib
from dmx_variety import VarietyEngine
from dmx_punch import velocity_brightness, afterglow, beat_velocity_from_ratio


class DmxEngineBase:
    """Shared engine: hardware, audio/beat pipeline, IPC, renderers, and the
    abstract _dispatch seam. Subclasses (loopback / synced) implement _dispatch
    and their own run loop. See dmx_variety.py for the variety layer."""

    def __init__(self):
        # DMX device (lazy init)
        self.dev = None
        # Async DMX transmission queue (drop old frames to preserve real-time sync)
        self._dmx_queue = queue.Queue(maxsize=2)
        self._dmx_thread_running = False
        # Output channels
        self.out_r = 0.0; self.out_g = 0.0; self.out_b = 0.0
        self.out_w = 0.0; self.out_master = 0.0; self.out_strobe = 0.0
        # AGC state
        self.agc_kick = 0.0; self.agc_snare = 0.0; self.agc_mid = 0.0; self.agc_hihat = 0.0
        # Spectral-flux previous magnitudes
        self.prev_kick_mag = 0.0; self.prev_snare_mag = 0.0; self.prev_hihat_mag = 0.0
        # Beat tracking
        self.beat_timestamps = deque(maxlen=150)
        self.total_beat_count = 0
        self.last_beat_time = 0.0
        self.frame_counter = 0
        self.beats_per_sec = 0.0
        self._beat_velocity = 0.0
        # Palette + cue state
        self.palettes = list(DEFAULT_PALETTES)
        self.current_palette_idx = 0
        self.synced_cues = []
        self._cue_starts = []
        self.show_bpm = 0.0
        self.audio_file = None
        self.last_frame = None             # populated only in DRY_RUN mode
        # FFT caches
        self._hanning_cache = {}
        self._fft_freq_cache = {}
        self._band_idx_cache = {}
        self.peak_hihat = 0.0
        self.last_palette_rotate = 0.0
        # Beat-detection params (base = synced/neutral defaults; loopback overrides)
        self.profile_name = "AI Show"
        self.profile_gain_boost = 1.0
        self.profile_volume_gate = MIN_VOLUME_GATE
        self.profile_agc_thresh = 0.7
        self.profile_kick_thresh = 0.3
        self.profile_snare_thresh = 0.35
        self.profile_onset_cooldown = ONSET_COOLDOWN
        self.profile_kick_dominance_ratio = 1.5
        # Beat-hold shared by the punchy renderers (loopback overrides via profile)
        self.profile_beat_hold = 4
        self.beat_hold_frames = 0
        # Master brightness of the hit that armed the current hold. The hold
        # floor is capped at this so a soft (graded-velocity) hit decays from
        # its own level instead of stepping UP to 200 a frame after the beat.
        self._hold_master = 0.0
        # Variety engine (anti-monotony policy; shared by both modes)
        self.variety = VarietyEngine()
        self._last_section_id = None
        self._evolution_secs = 16.0
        self.loopback_ambient = False
        # energy_state sentinel for IPC (loopback overrides with a real machine)
        self.energy_state = "playing"
        # abyssal_bloom renderer state
        self._ab_bass = 0.0                # smoothed bass activity (EMA of kick_i)
        self._ab_bloom_active = False
        self._ab_bloom_t0 = 0.0            # t when the current/last bloom started
        self._ab_last_bloom_t = 0.0        # placeholder -- always overwritten by the
                                            # discontinuity-reset below before first use
        self._ab_bloom_color = (0, 210, 210)
        self._ab_glint_active = False
        self._ab_glint_t0 = 0.0
        self._ab_last_glint_t = 0.0        # same as _ab_last_bloom_t -- placeholder only
        self._ab_last_render_t = None      # last t this renderer was actually called with (None = never)

        # golden_anthem renderer state
        self._ga_phase = 0.0               # 0..1 position in the swell cycle
        self._ga_energy = 0.0              # smoothed music energy (volume+mids EMA)
        self._ga_last_render_t = None      # last t this renderer was called with
        self._ga_loud_ref = 1e-6           # running loudness peak (self-normalizing, never zero)

        # cinematic_swell renderer state
        self._cs_swell_age = None          # None = idle; else seconds since trigger
        self._cs_swell_peak = 0.0          # velocity-scaled peak fraction for the active swell
        self._cs_drift_phase = 0.0         # 0..1 idle floor drift position
        self._cs_last_render_t = None      # last t this renderer was called with

        # Playback / IPC state
        self.playback_position = 0.0
        self.playback_duration = 0.0
        self.playback_state = "stopped"
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
            "ocean_drift": self._render_ocean_drift,
            "candlelight": self._render_candlelight,
            "sunset_fade": self._render_sunset_fade,
            "aurora_shimmer": self._render_aurora_shimmer,
            "abyssal_bloom": self._render_abyssal_bloom,
            "golden_anthem": self._render_golden_anthem,
            "cinematic_swell": self._render_cinematic_swell,
        }

    def _init_hardware(self):
        """Find uDMX adapter. Raises RuntimeError if not found."""
        if DRY_RUN:
            logger.info("[DRY-RUN] Skipping uDMX init; frames recorded, not sent.")
            self.dev = None
            return
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
        if DRY_RUN:
            self.last_frame = (int(master), int(red), int(green), int(blue), int(white), int(strobe))
            return
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

    def _write_playback_state(self):
        """Atomically write current playback state for the API to read."""
        state = {
            "position": round(self.playback_position, 2),
            "duration": round(self.playback_duration, 2),
            "state": "paused" if self.is_paused else self.playback_state,
            "cue_name": self.current_cue_name,
            "behavior": self.current_behavior,
            "energy_state": self.energy_state,
        }
        tmp = self._state_file + ".tmp"
        try:
            with open(tmp, 'w') as f:
                json.dump(state, f)
            os.replace(tmp, self._state_file)  # Atomic on Windows
        except Exception:
            pass

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

    def _render_blackout_punch(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                               kick_color, accent_color, volume, cue, t):
        """Complete darkness between beats. Flash on hits."""
        # P0-1 FIX: Defensive cue access — cue can be None in edge cases
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0
        strobe_ok = cue.get("strobe", False) if cue else False
        velocity_master = (120.0 + 135.0 * self._beat_velocity) * dimmer

        if is_kick:
            self.out_r, self.out_g, self.out_b = kick_color
            self.out_w = 255.0 * dimmer
            self.out_master = velocity_master
            self.out_strobe = 200.0 if strobe_ok else 0.0
        elif is_snare:
            self.out_r, self.out_g, self.out_b = accent_color
            self.out_w = 150.0 * dimmer
            self.out_master = velocity_master
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
            self._hold_master = self.out_master
        elif self.beat_hold_frames > 0:
            self.beat_hold_frames -= 1
            self.out_w *= 0.80
            # Hold floor never exceeds the arming hit's own brightness.
            self.out_master = max(self.out_master, min(200.0 * dimmer, self._hold_master))
        else:
            self.out_w = ema(self.out_w, 0, 0, 0.35)
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
        velocity_master = (120.0 + 135.0 * self._beat_velocity) * dimmer

        if is_kick:
            self.out_r, self.out_g, self.out_b = kick_color
            self.out_w = 200.0 * dimmer
            self.out_master = velocity_master
        elif is_snare:
            self.out_r, self.out_g, self.out_b = accent_color
            self.out_w = 100.0 * dimmer
            self.out_master = velocity_master
        else:
            self.out_r = ema(self.out_r, 0, 0, decay_speed)
            self.out_g = ema(self.out_g, 0, 0, decay_speed)
            self.out_b = ema(self.out_b, 0, 0, decay_speed)
            self.out_w = ema(self.out_w, hihat_i * 100 * dimmer, 0.6, 0.4)
            self.out_master = ema(self.out_master, 30 * dimmer, 0, 0.25)
        self.out_strobe = 0

    def _render_beat_reactive(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Default beat-reactive mode: kick→color_1, snare→color_2, bass→white."""
        energy = cue.get("energy", 5) if cue else 5
        dimmer = (cue.get("dimmer", 80) if cue else 80) / 100.0  # C2 FIX
        energy_boost = 0.5 + (energy / 10.0)

        k = max(kick_i * energy_boost, 0.6) if is_kick else kick_i
        s = max(snare_i * energy_boost, 0.5) if is_snare else snare_i

        tr = min(255.0, kick_color[0] * k + accent_color[0] * s + accent_color[0] * mid_i * 0.15)
        tg = min(255.0, kick_color[1] * k + accent_color[1] * s + accent_color[1] * mid_i * 0.15)
        tb = min(255.0, kick_color[2] * k + accent_color[2] * s + accent_color[2] * mid_i * 0.15)

        tw = 255.0 if is_kick else (120.0 if is_snare else hihat_i * 100)
        tm = (255.0 if (is_kick or is_snare) else max(80.0, volume * 8000)) * dimmer  # C2 FIX

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
            self._hold_master = self.out_master
        elif self.beat_hold_frames > 0:
            self.beat_hold_frames -= 1
            self.out_r, self.out_g, self.out_b, self.out_w = afterglow(
                self.out_r, self.out_g, self.out_b, self.out_w)
            # Hold floor never exceeds the arming hit's own brightness.
            self.out_master = max(self.out_master, min(200.0 * dimmer, self._hold_master))
        else:
            self.out_master = ema(self.out_master, tm, 0.4, 0.12)

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

    def _render_abyssal_bloom(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                              kick_color, accent_color, volume, cue, t):
        """Deep & sparse ambient: near-black drifting floor, rare bass-reactive
        teal blooms, rarer white glints. Restraint is the effect — only one
        bloom or glint active at a time, with a hard minimum gap between blooms
        regardless of bass energy."""
        # Self-healing reset: if this renderer wasn't called recently (skipped
        # while another ambient behavior was selected) or `t` jumped (a synced
        # seek in either direction), treat it as a fresh arrival rather than
        # letting stale timers fire an instant bloom/glint or desync for tens
        # of seconds. Covers both the loopback re-entry case and the seek case
        # with one mechanism -- no dispatch-layer changes needed.
        if (self._ab_last_render_t is None or
                abs(t - self._ab_last_render_t) > ABYSSAL_DISCONTINUITY_THRESHOLD):
            self._ab_bloom_active = False
            self._ab_glint_active = False
            self._ab_last_bloom_t = t
            self._ab_last_glint_t = t
        self._ab_last_render_t = t

        dimmer = (cue.get("dimmer", 50) if cue else 50) / 100.0

        # --- Smoothed bass activity (not the raw per-frame kick_i) ---
        self._ab_bass = ema(self._ab_bass, min(1.0, kick_i), 0.2, 0.05)

        # --- Layer 1: floor (two slow non-harmonic sines, deep blue <-> violet) ---
        # Deliberately NOT scaled by `dimmer` here: FLOOR_MIN/MAX already ARE the
        # intended final 3-10% floor (the spec's "never fully black" guarantee).
        # Scaling it again by the cue's dimmer (~0.5 typical) would push it below
        # the visible floor of every sibling ambient renderer -- likely reading
        # as fully off on real LED hardware (PWM dead-zone).
        wave1 = math.sin(t * 0.05) * 0.5 + 0.5
        wave2 = math.sin(t * 0.033 + 1.1) * 0.5 + 0.5
        floor_blend = wave1 * 0.6 + wave2 * 0.4
        floor_brightness = (ABYSSAL_FLOOR_MIN +
                            (ABYSSAL_FLOOR_MAX - ABYSSAL_FLOOR_MIN) * floor_blend)
        deep_blue = (10, 20, 120)
        deep_violet = (60, 10, 130)
        floor_color = lerp_color(deep_blue, deep_violet, floor_blend)

        # --- Layer 2: bloom (rare teal swell, bass-reactive but bounded) ---
        bloom_gap = max(ABYSSAL_BLOOM_GAP_MIN,
                        ABYSSAL_BLOOM_INTERVAL * (1.0 - self._ab_bass * 0.6))
        if not self._ab_bloom_active and not self._ab_glint_active:
            time_since_last = t - self._ab_last_bloom_t
            timer_ready = time_since_last >= ABYSSAL_BLOOM_INTERVAL
            nudge_ready = (time_since_last >= bloom_gap and
                          kick_i > ABYSSAL_BLOOM_NUDGE_THRESH)
            if timer_ready or nudge_ready:
                self._ab_bloom_active = True
                self._ab_bloom_t0 = t
                self._ab_bloom_color = lerp_color(accent_color, (0, 210, 210), 0.6)

        bloom_brightness = 0.0
        if self._ab_bloom_active:
            # Clamp against backward time jumps (e.g. a synced-mode seek while
            # a bloom is mid-swell) -- without this, a negative age feeds an
            # unclamped smoothstep and can spike brightness far outside [0,1].
            age = max(0.0, t - self._ab_bloom_t0)
            total = ABYSSAL_BLOOM_RISE + ABYSSAL_BLOOM_HOLD + ABYSSAL_BLOOM_FALL
            if age >= total:
                self._ab_bloom_active = False
                self._ab_last_bloom_t = t
            else:
                if age < ABYSSAL_BLOOM_RISE:
                    p = age / ABYSSAL_BLOOM_RISE
                    env = p * p * (3.0 - 2.0 * p)
                elif age < ABYSSAL_BLOOM_RISE + ABYSSAL_BLOOM_HOLD:
                    env = 1.0
                else:
                    fall_age = age - ABYSSAL_BLOOM_RISE - ABYSSAL_BLOOM_HOLD
                    p = 1.0 - (fall_age / ABYSSAL_BLOOM_FALL)
                    env = p * p * (3.0 - 2.0 * p)
                peak = min(ABYSSAL_BLOOM_MAX,
                          ABYSSAL_BLOOM_BASE + self._ab_bass * ABYSSAL_BLOOM_BASS_GAIN)
                bloom_brightness = env * peak * dimmer

        # --- Layer 3: glint (rarer white flare) ---
        # Mutually exclusive with bloom (spec: "only one bloom OR glint active
        # at a time") -- a glint can't arm while a bloom is mid-swell, and the
        # bloom-arming check above already excludes an active glint too.
        if not self._ab_glint_active and not self._ab_bloom_active:
            time_since_glint = t - self._ab_last_glint_t
            timer_ready = time_since_glint >= ABYSSAL_GLINT_INTERVAL
            hihat_ready = (time_since_glint >= ABYSSAL_BLOOM_GAP_MIN and
                          hihat_i > ABYSSAL_GLINT_THRESH)
            if timer_ready or hihat_ready:
                self._ab_glint_active = True
                self._ab_glint_t0 = t

        glint_brightness = 0.0
        if self._ab_glint_active:
            # Same backward-seek guard as the bloom block above.
            age = max(0.0, t - self._ab_glint_t0)
            total = ABYSSAL_GLINT_RISE + ABYSSAL_GLINT_FALL
            if age >= total:
                self._ab_glint_active = False
                self._ab_last_glint_t = t
            else:
                if age < ABYSSAL_GLINT_RISE:
                    glint_brightness = (age / ABYSSAL_GLINT_RISE) * dimmer
                else:
                    fall_age = age - ABYSSAL_GLINT_RISE
                    glint_brightness = (1.0 - fall_age / ABYSSAL_GLINT_FALL) * dimmer

        # --- Compose: bloom/glint lift above the floor, never darken it ---
        r = max(floor_color[0] * floor_brightness, self._ab_bloom_color[0] * bloom_brightness)
        g = max(floor_color[1] * floor_brightness, self._ab_bloom_color[1] * bloom_brightness)
        b = max(floor_color[2] * floor_brightness, self._ab_bloom_color[2] * bloom_brightness)
        w = 200.0 * glint_brightness
        # 255.0 (not an arbitrary scalar) so floor_brightness's 0.03-0.10 maps
        # directly to "3-10% of full brightness" as the spec literally states.
        master = max(255.0 * floor_brightness, 200.0 * bloom_brightness, 220.0 * glint_brightness)

        self.out_r = ema(self.out_r, r, 0.05, 0.03)
        self.out_g = ema(self.out_g, g, 0.05, 0.03)
        self.out_b = ema(self.out_b, b, 0.05, 0.03)
        self.out_w = ema(self.out_w, w, 0.3, 0.15)
        self.out_master = ema(self.out_master, master, 0.05, 0.03)
        self.out_strobe = 0

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
        # Self-normalizing: `volume` scales differ wildly between modes
        # (loopback RMS ~0.002 vs synced normalized-WAV RMS ~0.05-0.3), so no
        # absolute scale factor can work in both -- it saturates to a binary
        # loud/silent switch. Instead track the song's own running loudness
        # peak (fast attack, ~4%/s decay at ~86 fps) and measure energy
        # relative to it: quiet verse < 1.0, drop/chorus ~= 1.0 in either mode.
        raw = volume + mid_i * 0.02
        self._ga_loud_ref = max(raw, self._ga_loud_ref * 0.9995)
        self._ga_energy = ema(self._ga_energy,
                              min(1.0, raw / self._ga_loud_ref), 0.1, 0.03)

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

    def _render_cinematic_swell(self, kick_i, snare_i, hihat_i, mid_i, is_kick, is_snare,
                                kick_color, accent_color, volume, cue, t):
        """Film-score mode: a dim floor that drifts slowly between the palette
        colors, plus a wide eased swell toward the accent color fired only by
        STRONG kicks (velocity-gated). Rise ~0.5s / fall ~1.4s -- a slow-motion
        impact, never a flash. Accumulated dt (golden_anthem pattern): seeks
        and rotation re-entry advance progress by at most one nominal frame."""
        dimmer = (cue.get("dimmer", 60) if cue else 60) / 100.0

        if (self._cs_last_render_t is None or
                abs(t - self._cs_last_render_t) > CINE_DISCONTINUITY_THRESHOLD):
            dt = 0.012
        else:
            dt = min(max(t - self._cs_last_render_t, 0.0), CINE_DT_CLAMP)
        self._cs_last_render_t = t

        # --- Trigger: strong kicks only, gated on the RAW kick ratio.
        # Deliberately independent of _beat_velocity: velocity GRADES every
        # onset from ratio 1.0; this gate IGNORES onsets below
        # CINE_TRIGGER_RATIO (1.6x) so the calm floor survives weak hits.
        ratio = kick_i / max(self.profile_kick_thresh, 0.01)
        if is_kick and ratio >= CINE_TRIGGER_RATIO:
            strength = min(1.0, (ratio - CINE_TRIGGER_RATIO) /
                           (CINE_FULL_RATIO - CINE_TRIGGER_RATIO))
            new_peak = min(CINE_PEAK_MAX, velocity_brightness(strength) / 255.0)
            if self._cs_swell_age is None:
                self._cs_swell_age = 0.0
                self._cs_swell_peak = new_peak
            else:
                # Retrigger only if the new hit out-peaks what's left of the
                # current swell -- and resume the rise at the age whose
                # envelope matches the height already on the lights, so
                # output never sags backward mid-rise.
                env_now_abs = self._cs_swell_peak * _cine_envelope(self._cs_swell_age)
                if new_peak > env_now_abs:
                    self._cs_swell_age = _cine_rise_age_for(
                        min(1.0, env_now_abs / new_peak))
                    self._cs_swell_peak = new_peak

        # --- Advance swell + idle drift by clamped dt.
        swell = 0.0
        if self._cs_swell_age is not None:
            self._cs_swell_age += dt
            env = _cine_envelope(self._cs_swell_age)
            if self._cs_swell_age >= CINE_RISE_S + CINE_FALL_S:
                self._cs_swell_age = None
            swell = env * self._cs_swell_peak
        self._cs_drift_phase = (self._cs_drift_phase + dt / CINE_DRIFT_PERIOD_S) % 1.0

        # --- Floor: slow triangle-wave drift color_1 <-> color_2.
        tri = 1.0 - abs(2.0 * self._cs_drift_phase - 1.0)
        floor_color = lerp_color(kick_color, accent_color, tri * 0.6)
        # Floor is NOT dimmer-scaled (PWM-visibility lesson from abyssal_bloom).
        floor_b = CINE_FLOOR_MIN

        # --- Swell lifts brightness toward the accent color; never darkens floor.
        swell_color = lerp_color(floor_color, accent_color, min(1.0, swell * 1.5))
        brightness = max(floor_b, swell * dimmer)
        col = swell_color if swell > 0.0 else floor_color

        # EMA lag stretches the effective rise to ~0.7s, reaching ~93% of the
        # envelope target -- intentional aesthetic slack, not a bug.
        self.out_r = ema(self.out_r, col[0] * brightness, 0.08, 0.05)
        self.out_g = ema(self.out_g, col[1] * brightness, 0.08, 0.05)
        self.out_b = ema(self.out_b, col[2] * brightness, 0.08, 0.05)
        # White only near the swell peak (shimmer precedent from golden_anthem).
        # Normalized to PEAK_MAX so full-strength swells reach CINE_WHITE_PEAK.
        white = (CINE_WHITE_PEAK * max(0.0, swell - CINE_WHITE_KNEE) /
                 (CINE_PEAK_MAX - CINE_WHITE_KNEE) * dimmer)
        self.out_w = ema(self.out_w, white, 0.08, 0.06)
        self.out_master = ema(self.out_master, 255.0 * brightness, 0.08, 0.05)
        self.out_strobe = 0

    def load_ai_show(self, show_file="current_show.json"):
        """Load AI-generated palettes and cue list from a show JSON file."""
        if not os.path.exists(show_file):
            logger.warning(f"No {show_file} found! Using default fallback palettes.")
            return None

        try:
            with open(show_file, 'r') as f:
                data = json.load(f)

            plan = data.get("lighting_plan", {})

            # Per-song identity. NOT builtin hash(): Python salts string hashes
            # per process, and every playback is a fresh worker process, so
            # hash() would silently break same-song-same-look replay
            # determinism. zlib.crc32 is stdlib and process-stable.
            # Guarded independently: malformed metadata must degrade (bpm 0.0 =
            # time-based phrasing), never abort loading the cues/palettes below.
            metrics = data.get("song_metrics", {})
            if not isinstance(metrics, dict):
                logger.warning(f"Malformed song_metrics ({type(metrics).__name__}); ignoring")
                metrics = {}
            try:
                self.show_bpm = float(metrics.get("bpm", 0.0) or 0.0)
            except (TypeError, ValueError):
                logger.warning(f"Malformed song_metrics.bpm ({metrics.get('bpm')!r}); "
                               "falling back to 0.0 (time-based phrasing)")
                self.show_bpm = 0.0
            name = plan.get("show_name")
            audio = data.get("audio_file")
            # basename, not abspath: seed must not change when the install
            # moves drives/folders (app.py's slug convention does the same).
            seed_basis = (name if isinstance(name, str) and name
                          else os.path.basename(audio) if isinstance(audio, str) and audio
                          else "")
            self.variety.set_song_seed(zlib.crc32(seed_basis.encode("utf-8")))

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
                        "mood": c.get("mood"),  # carried for the VarietyEngine palette seed (Task 10)
                    })
                self.synced_cues.sort(key=lambda x: x["start"])
                self._cue_starts = [c["start"] for c in self.synced_cues]
                logger.info(f"Loaded {len(self.synced_cues)} timestamped cues")

            return data.get("audio_file")
        except Exception as e:
            logger.error(f"Failed to load AI show: {e}")
            return None

    def _get_active_cue(self, elapsed):
        """P1-2 FIX: O(1) bisect lookup instead of O(n) reverse scan.
        Also checks end boundary so we return None when between cues."""
        if not self.synced_cues:
            return None
        # Binary search on start times (rebuilt lazily if cues changed)
        if len(self._cue_starts) != len(self.synced_cues):
            self._cue_starts = [c["start"] for c in self.synced_cues]
        idx = bisect.bisect_right(self._cue_starts, elapsed) - 1
        if 0 <= idx < len(self.synced_cues):
            cue = self.synced_cues[idx]
            if elapsed < cue["end"]:
                return cue
        return None

    def process_audio(self, indata, elapsed_seconds=None, input_format="int16", actual_sample_rate=None):
        """Core engine: FFT → onset detection → mode dispatch → DMX output.
        Shared by both modes; the per-mode tail lives in the subclass _dispatch()."""
        self.frame_counter += 1
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

        # Debug logging for first 50 frames
        if self.frame_counter <= 50 and self.frame_counter % 10 == 0:
            logger.debug(f"[ENGINE frame {self.frame_counter}] vol={volume:.6f} samples={len(mono)} sr={sr}")

        if volume < self.profile_volume_gate:
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

        kick_hit = max(0.0, kick_mag - self.agc_kick * self.profile_agc_thresh)
        snare_hit = max(0.0, snare_mag - self.agc_snare * self.profile_agc_thresh)
        mid_hit = max(0.0, mid_mag - self.agc_mid * 0.4)
        hihat_hit = max(0.0, hihat_mag - self.agc_hihat * 0.4)

        kick_i = min(1.0, (kick_flux * 0.6 + kick_hit * 0.4) * KICK_GAIN * self.profile_gain_boost)
        snare_i = min(1.0, (snare_flux * 0.6 + snare_hit * 0.4) * SNARE_GAIN * self.profile_gain_boost)
        hihat_i = min(1.0, (hihat_flux * 0.5 + hihat_hit * 0.5) * HIHAT_GAIN * self.profile_gain_boost)
        mid_i = min(1.0, mid_hit * MID_GAIN * self.profile_gain_boost)

        self.prev_kick_mag = kick_mag
        self.prev_snare_mag = snare_mag
        self.prev_hihat_mag = hihat_mag

        # --- Beat registration ---
        current_time = time.time()
        cooldown_ok = (current_time - self.last_beat_time) > self.profile_onset_cooldown

        # VOCAL SUPPRESSION: Two gates that prevent vocals from triggering beats.
        #
        # Gate 1 — Kick-to-mid ratio: Real kick drums have overwhelming energy
        # below 150Hz relative to 400-2000Hz mids. Vocals are the opposite —
        # their fundamental (150-400Hz) and harmonics dominate the mid range.
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

        is_kick = kick_i > self.profile_kick_thresh and cooldown_ok and kick_dominates
        is_snare = snare_i > self.profile_snare_thresh and cooldown_ok and snare_is_transient

        # Periodic diagnostic logging every 100 frames
        if self.frame_counter % 100 == 0:
            logger.info(f"[DIAG] vol={volume:.4f} kick_i={kick_i:.4f} mid_mag={mid_mag:.4f} "
                        f"kick_dom={kick_dominates} snare_trans={snare_is_transient} "
                        f"is_kick={is_kick} beats={self.total_beat_count}")

        if is_kick or is_snare:
            self.last_beat_time = current_time
            self.beat_timestamps.append(current_time)
            self.total_beat_count += 1

        # M2 FIX: Evict old timestamps in-place with O(1) popleft instead of
        # creating a new list every frame (was generating ~47 throwaway lists/sec)
        while self.beat_timestamps and (current_time - self.beat_timestamps[0]) > 3.0:
            self.beat_timestamps.popleft()
        beats_per_sec = len(self.beat_timestamps) / 3.0

        # --- Velocity + BPS for the variety/punch layer, then dispatch ---
        # Velocity measures how far the onset EXCEEDS its detection threshold
        # (see beat_velocity_from_ratio in dmx_punch.py). The old
        # min(1, intensity/thresh) form pinned velocity at 1.0 on every onset
        # frame because is_kick/is_snare already require intensity > thresh.
        kick_ratio = kick_i / max(self.profile_kick_thresh, 0.01)
        snare_ratio = snare_i / max(self.profile_snare_thresh, 0.01)
        kick_velocity = beat_velocity_from_ratio(kick_ratio)
        snare_velocity = beat_velocity_from_ratio(snare_ratio)
        self._beat_velocity = max(kick_velocity, snare_velocity)
        self.beats_per_sec = beats_per_sec

        self._dispatch(kick_mag, snare_mag, mid_mag, hihat_mag,
                       kick_i, snare_i, hihat_i, mid_i,
                       is_kick, is_snare, volume, sr, elapsed_seconds=elapsed_seconds)
        self.send_dmx(self.out_master, self.out_r, self.out_g,
                      self.out_b, self.out_w, self.out_strobe)

    def _dispatch(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                  kick_i, snare_i, hihat_i, mid_i,
                  is_kick, is_snare, volume, sr, elapsed_seconds=None):
        raise NotImplementedError("Subclasses implement _dispatch()")

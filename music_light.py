"""DMX loopback engine — live WASAPI capture. Subclass of DmxEngineBase."""
import sys
import json
import math
import time
import logging
from collections import deque
import numpy as np
import pyaudiowpatch as pyaudio
from dmx_engine import (DmxEngineBase, BLOCK_SIZE, MIN_VOLUME_GATE,
                        LOOPBACK_GAIN_BOOST, LOOPBACK_VOLUME_GATE, LOOPBACK_AGC_THRESH,
                        DRY_RUN)
from dmx_variety import Intent
from dmx_punch import beat_velocity_from_ratio

logger = logging.getLogger(__name__)

# Behaviors the loopback _dispatch can deliver BY NAME (routed through
# _behavior_map). Everything else (the punchy set) collapses into
# _render_loopback_direct, which ignores the behavior name -- so pinning a
# punchy name would report a renderer that isn't actually running. Used by
# both _dispatch routing and load_profile force_behavior validation.
AMBIENT_DISPATCH_BEHAVIORS = {"ocean_drift", "candlelight", "sunset_fade",
                              "aurora_shimmer", "abyssal_bloom", "golden_anthem",
                              "cinematic_swell", "ambient_pulse",
                              "slow_breathe",
                              "static_wash", "buildup_ramp", "rainbow_sweep"}


class DMXEngine(DmxEngineBase):
    """Live loopback engine: energy-state machine + R->B->G->W direct renderer."""

    def __init__(self):
        super().__init__()
        # Loopback beat-detection tuning (WASAPI loopback is extremely quiet)
        self.profile_name = "Concert Punchy"
        self.profile_gain_boost = LOOPBACK_GAIN_BOOST
        self.profile_volume_gate = LOOPBACK_VOLUME_GATE
        self.profile_agc_thresh = LOOPBACK_AGC_THRESH
        self.profile_kick_thresh = 0.05
        self.profile_snare_thresh = 0.08
        # Loopback-only profile params
        self.profile_color_cycle_mode = "rhythm"
        self.profile_color_cycle_interval = 5.0
        self.profile_rhythm_change_pct = 0.30
        self.profile_deep_bass_enabled = True
        self.profile_deep_bass_thresh = 0.80
        self.profile_decay_speed = 0.75
        self.profile_glow_thresh = 0.55
        self.profile_beat_hold = 4
        self.profile_deep_bass_hold = 5
        # Optional: pin the dispatch to one renderer (profile "force_behavior"
        # key). None = auto-behavior detection as always. Added after Cinematic
        # hardware feedback: tuning-only profiles cannot guarantee a mode feel.
        self.profile_force_behavior = None
        # Loopback runtime state
        self.loopback_ambient = True
        self.volume_history = deque(maxlen=200)
        self.energy_state = "calm"
        self.energy_state_since = 0.0
        self.drop_cooldown = 0.0
        self.peak_kick = 0.0
        self.peak_snare = 0.0
        self.peak_mid = 0.0
        self.beat_hold_color = (255, 0, 50)

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
            forced = p.get("force_behavior", None)
            if forced is not None and forced not in AMBIENT_DISPATCH_BEHAVIORS:
                logger.warning(f"[PROFILE] force_behavior '{forced}' is not "
                               "supported for pinning (only per-name ambient "
                               "renderers are) -- falling back to auto-behavior "
                               "detection")
            # Unconditional: a reload without the key (or with a bad value)
            # must clear any stale pin from a previously loaded profile.
            self.profile_force_behavior = forced if forced in AMBIENT_DISPATCH_BEHAVIORS else None
            # Load palettes if provided
            if "palettes" in p:
                self.palettes = [(tuple(c1), tuple(c2)) for c1, c2 in p["palettes"]]
            logger.info(f"[PROFILE] Loaded: {self.profile_name}")
        except Exception as e:
            logger.error(f"[PROFILE] Failed to load {profile_path}: {e}")

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

        # FAST TRANSITION: BPS is the primary signal. If beats are fast,
        # skip the slow volume-ratio state machine and jump directly.
        if self.energy_state == "calm":
            if beats_per_sec > 2.0:
                # Fast beats = immediate escalation, no "building" phase needed
                self.energy_state = "high"
                self.energy_state_since = current_time
            elif beats_per_sec > 1.2 and recent_energy > overall_avg * 0.8:
                self.energy_state = "building"
                self.energy_state_since = current_time
            elif recent_energy > past_energy * 1.5 and recent_energy > MIN_VOLUME_GATE * 3:
                self.energy_state = "building"
                self.energy_state_since = current_time

        elif self.energy_state == "building":
            if beats_per_sec > 2.5 or (recent_energy > past_energy * 2.0 and self.drop_cooldown <= 0):
                self.energy_state = "high"
                self.energy_state_since = current_time
                self.drop_cooldown = 5.0
            elif recent_energy < past_energy * 0.6 and beats_per_sec < 0.8:
                self.energy_state = "calm"
                self.energy_state_since = current_time
            elif time_in_state > 8.0:
                self.energy_state = "high" if beats_per_sec > 1.5 else "calm"
                self.energy_state_since = current_time

        elif self.energy_state == "high":
            if beats_per_sec < 0.5 and recent_energy < overall_avg * 0.5 and time_in_state > 2.0:
                self.energy_state = "dropping"
                self.energy_state_since = current_time
            elif beats_per_sec < 1.0 and time_in_state > 8.0:
                self.energy_state = "dropping"
                self.energy_state_since = current_time
            elif time_in_state > 20.0:
                self.energy_state = "calm"
                self.energy_state_since = current_time

        elif self.energy_state == "dropping":
            if beats_per_sec > 1.5 or recent_energy > overall_avg * 1.5:
                self.energy_state = "high"
                self.energy_state_since = current_time
            elif time_in_state > 3.0:
                self.energy_state = "calm"
                self.energy_state_since = current_time

        # AUTO-BEHAVIOR DISPATCH: Picks punchy or chill based on energy + BPS
        if self.energy_state == "calm":
            if beats_per_sec < 0.5 and recent_energy < overall_avg * 0.5:
                # Very quiet — rotate through ambient behaviors for variety
                ambient_pool = ["ocean_drift", "candlelight", "aurora_shimmer",
                               "sunset_fade", "abyssal_bloom", "golden_anthem",
                               "cinematic_swell", "ambient_pulse"]
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

    def _render_loopback_direct(self, kick_mag, snare_mag, mid_mag, hihat_mag,
                                kick_i, snare_i, hihat_i, mid_i,
                                is_kick, is_snare,
                                color_1, color_2, accent, volume, t):
        """
        Loopback renderer with palette-driven colors and deep bass combos.
        - Colors come from the VarietyEngine's current palette (kick/accent/combo)
        - Deep bass hits trigger a dual-color combo blast (accent + white)
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
        # Velocity measures threshold EXCESS (beat_velocity_from_ratio); the old
        # min(1, intensity/thresh) form pinned it at 1.0 on every onset frame.
        kick_velocity = beat_velocity_from_ratio(kick_i / max(self.profile_kick_thresh, 0.01))
        snare_velocity = beat_velocity_from_ratio(snare_i / max(self.profile_snare_thresh, 0.01))
        beat_velocity = max(kick_velocity, snare_velocity)
        velocity_brightness = 120.0 + (135.0 * beat_velocity)

        # ── DEEP BASS COMBO: Dual-color blast ──
        # SYNC FIX: Instant snap (not bloom) on beat onset. Bloom was adding 20-40ms
        # visual delay that made lights feel "late." Beat onset MUST be instant to
        # synchronize with the audio transient the ear just heard.
        if (is_kick or is_snare) and is_deep_bass:
            # Near-white accents (7/16 palettes, skewing high-energy) would
            # collapse the "dual-color" blast into plain white on top of the
            # white channel -- fall back to color_2 so combos stay visually
            # distinct from normal kick beats.
            combo = accent if sum(accent) < 700 else color_2
            self.out_r, self.out_g, self.out_b = combo
            self.out_w = 255.0
            # Deep-bass combos are the dramatic "special blast" -- keep a 200
            # master floor so graded velocity can't dim them, grade above it.
            self.out_master = max(200.0, velocity_brightness)
            self.out_strobe = 0
            self.beat_hold_frames = self.profile_deep_bass_hold
            self._hold_master = self.out_master
            return

        # ── NORMAL BEAT: Instant color snap ──
        if is_kick or is_snare:
            col = color_1 if is_kick else color_2
            self.out_r, self.out_g, self.out_b = col
            self.out_w = 255.0 if is_kick else 0.0
            self.out_master = velocity_brightness
            self.out_strobe = 0
            self.beat_hold_frames = self.profile_beat_hold
            self._hold_master = self.out_master
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
            # Stay bright during hold -- but never brighter than the hit that
            # armed it (soft graded-velocity hits must not step UP to 200).
            self.out_master = max(self.out_master, min(200.0, self._hold_master))
            self.out_strobe = 0
            return

        # ── BETWEEN BEATS: Subtle glow + breathing white ──
        bass_active = bass > self.profile_glow_thresh

        if bass_active:
            glow = (bass - self.profile_glow_thresh) * 0.4
            self.out_r = color_1[0] * glow
            self.out_g = color_1[1] * glow
            self.out_b = color_1[2] * glow
            self.out_w = 0.0
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
        # Always runs (keeps the energy state machine + Intent mood fresh);
        # a profile force_behavior overrides only the CHOICE, never the machine.
        auto_behavior = self._detect_auto_behavior(volume, kick_i, snare_i, current_time, beats_per_sec)
        if self.profile_force_behavior:
            auto_behavior = self.profile_force_behavior
        self.current_behavior = auto_behavior

        # Write IPC state every ~50 frames (~1s) so the UI shows the sub-mode
        if self.frame_counter % 50 == 0:
            self.playback_state = "loopback"
            self.current_cue_name = f"{self.energy_state} | {auto_behavior}"
            self._write_playback_state()

        # Ambient/chill behaviors → use the standard renderer dispatch
        if auto_behavior in AMBIENT_DISPATCH_BEHAVIORS:
            # ambient_pulse is the beat-locked mode -- it needs pulse headroom,
            # not the dim ambient default.
            cue_dimmer = 80 if auto_behavior == "ambient_pulse" else 50
            cue = {"dimmer": cue_dimmer, "energy": 3, "start": 0, "end": 60,
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

    def run_loopback_mode(self, show_file=None):
        if DRY_RUN:
            # The DRY_RUN offline harness is synced-only. Loopback under this
            # flag would still open a live WASAPI stream but silently discard
            # every DMX frame with no diagnostic output -- warn loudly so a
            # leftover env var (inherited through app.py's subprocess spawn)
            # can't masquerade as "lights mysteriously dead".
            logger.warning("[DRY-RUN] DMX_DRY_RUN=1 is set: loopback will run "
                           "WITHOUT hardware output (frames discarded). This "
                           "harness is intended for synced-mode verification only.")
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


if __name__ == "__main__":
    show_file = None
    profile_file = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--show" and i + 1 < len(args):
            show_file = args[i + 1]; i += 2
        elif args[i] == "--profile" and i + 1 < len(args):
            profile_file = args[i + 1]; i += 2
        else:
            i += 1
    engine = DMXEngine()
    if profile_file:
        engine.load_profile(profile_file)
    try:
        logger.info("==== DMX Loopback Engine ====")
        logger.info(f"Profile: {engine.profile_name} | Show: {show_file}")
        engine.run_loopback_mode(show_file)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except RuntimeError as e:
        logger.error(str(e)); sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal: {e}")
    finally:
        engine.shutdown()
        logger.info("Shutdown complete.")

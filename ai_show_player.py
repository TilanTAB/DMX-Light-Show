"""DMX AI Show Player — synced WAV playback. Subclass of DmxEngineBase."""
import os
import sys
import time
import logging
import pyaudiowpatch as pyaudio
from dmx_engine import DmxEngineBase, BLOCK_SIZE, DRY_RUN
from dmx_variety import Intent

logger = logging.getLogger(__name__)


class DMXEngine(DmxEngineBase):
    """Synced playback: AI cue timeline drives colors/behaviors; inherits the
    shared pipeline + renderers from DmxEngineBase."""

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

        # NOTE: the no-sleep loop below compresses hours of audio-time into
        # seconds of wall-clock. Beat-onset cooldown is wall-clock based
        # (time.time() in process_audio), so beat cadence in DRY_RUN logs is
        # NOT representative of real playback -- use this harness to verify
        # cue/palette/frame wiring, never beat timing or beat-driven velocity.
        if DRY_RUN:
            wf = wave_mod.open(audio_path, 'rb')
            try:
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
            finally:
                wf.close()
            logger.info("[DRY-RUN] Completed synced pass.")
            return

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
                self.process_audio(data, elapsed_seconds=elapsed, input_format="int16", actual_sample_rate=sample_rate)

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

if __name__ == "__main__":
    show_file = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--show" and i + 1 < len(args):
            show_file = args[i + 1]
            i += 2
        else:
            i += 1

    if not show_file:
        print("Usage: ai_show_player.py --show <path/to/show.json>", file=sys.stderr)
        sys.exit(1)

    engine = DMXEngine()
    try:
        logger.info("==== AI Show Player ====")
        logger.info(f"Show: {show_file}")
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
        engine.shutdown()
        logger.info("Shutdown complete.")

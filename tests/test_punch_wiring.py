from dmx_engine import DmxEngineBase
from dmx_punch import beat_velocity_from_ratio


def _engine():
    e = DmxEngineBase()
    e._beat_velocity = 1.0
    return e


def test_bass_white_blast_kick_master_owned_by_velocity():
    # The dilution bug: a trailing unconditional EMA used to overwrite the
    # velocity master on the SAME frame. On a kick with velocity 1.0 and
    # dimmer 0.8, master must be exactly 255 * 0.8.
    e = _engine()
    cue = {"energy": 7, "dimmer": 80}
    e._render_bass_white_blast(1.0, 0, 0, 0, True, False,
                               (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    assert e.out_master == 255.0 * 0.8

def test_bass_white_blast_soft_kick_dimmer_floor():
    e = _engine()
    e._beat_velocity = 0.0
    cue = {"energy": 7, "dimmer": 80}
    e._render_bass_white_blast(0.2, 0, 0, 0, True, False,
                               (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    assert e.out_master == 120.0 * 0.8

def test_beat_reactive_beat_master_owned_by_velocity():
    e = _engine()
    cue = {"energy": 7, "dimmer": 80}
    e._render_beat_reactive(1.0, 0, 0, 0, True, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    assert e.out_master == 255.0 * 0.8

def test_beat_hold_keeps_master_bright():
    e = _engine()
    cue = {"energy": 7, "dimmer": 80}
    e._render_beat_reactive(1.0, 0, 0, 0, True, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    e._render_beat_reactive(0.0, 0, 0, 0, False, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.02)
    assert e.out_master >= 200.0 * 0.8   # hold frame stays bright
    assert e.beat_hold_frames == e.profile_beat_hold - 1


# --- Velocity COMPUTATION (the production side of _beat_velocity) ---
# Old bug: process_audio used min(1, kick_i/thresh), but is_kick already
# requires kick_i > thresh, so velocity was pinned at exactly 1.0 on every
# onset frame. The mapping must measure threshold EXCESS instead.

def test_velocity_barely_over_threshold_is_soft():
    # kick_i = thresh * 1.05 -> ratio 1.05 -> near-zero velocity
    assert beat_velocity_from_ratio(1.05) < 0.1

def test_velocity_at_full_ratio_is_max():
    assert abs(beat_velocity_from_ratio(3.0) - 1.0) < 1e-9
    assert beat_velocity_from_ratio(5.0) == 1.0   # clamps above full

def test_velocity_mid_ratio_is_half():
    assert abs(beat_velocity_from_ratio(2.0) - 0.5) < 1e-9

def test_velocity_below_threshold_clamps_to_zero():
    assert beat_velocity_from_ratio(0.5) == 0.0

def test_velocity_monotone_in_ratio():
    ratios = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0]
    vels = [beat_velocity_from_ratio(r) for r in ratios]
    assert all(b >= a for a, b in zip(vels, vels[1:]))

def test_process_audio_velocity_lines_use_helper():
    # Pin the wiring: process_audio must derive _beat_velocity from the
    # shared helper, not the old pinned-at-1.0 min() formula.
    import inspect, dmx_engine
    src = inspect.getsource(dmx_engine.DmxEngineBase.process_audio)
    assert "beat_velocity_from_ratio" in src
    assert "min(1.0, kick_i / max(self.profile_kick_thresh" not in src

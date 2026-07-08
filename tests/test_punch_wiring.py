from dmx_engine import DmxEngineBase


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

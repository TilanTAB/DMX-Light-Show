from dmx_engine import (DmxEngineBase, _anthem_envelope,
                        ANTHEM_CREST_BASE, ANTHEM_CREST_GAIN, ANTHEM_CREST_MAX)


def test_envelope_bounded_and_shaped():
    vals = [_anthem_envelope(i / 200.0) for i in range(200)]
    assert all(0.0 <= v <= 1.0 for v in vals)
    assert abs(_anthem_envelope(0.0)) < 1e-9          # starts dark
    assert _anthem_envelope(0.45) == 1.0              # crest hold (0.4..0.5)
    assert _anthem_envelope(0.999) < 0.01             # returns to ~0


def test_crest_lift_is_hard_capped():
    assert min(ANTHEM_CREST_MAX, ANTHEM_CREST_BASE + 1.0 * ANTHEM_CREST_GAIN) == ANTHEM_CREST_MAX
    assert ANTHEM_CREST_BASE + 0.0 * ANTHEM_CREST_GAIN == ANTHEM_CREST_BASE


def test_discontinuity_advances_one_nominal_frame():
    # Accumulated-phase design: a 60s time jump (seek / deselected-and-back)
    # must advance the swell by ~one frame, never fast-forward whole cycles.
    e = DmxEngineBase()
    cue = {"dimmer": 60}
    e._render_golden_anthem(0, 0, 0, 0, False, False,
                            (255, 0, 120), (0, 220, 255), 0.0, cue, 10.0)
    before = e._ga_phase
    e._render_golden_anthem(0, 0, 0, 0, False, False,
                            (255, 0, 120), (0, 220, 255), 0.0, cue, 70.0)
    assert e._ga_phase - before < 0.01


def test_floor_keeps_light_alive_at_zero_dimmer():
    e = DmxEngineBase()
    cue = {"dimmer": 0}
    for i in range(60):
        e._render_golden_anthem(0, 0, 0, 0, False, False,
                                (255, 0, 120), (0, 220, 255), 0.0, cue, i * 0.012)
    assert e.out_master > 0.0                          # never fully dark
    assert e.out_strobe == 0


def test_energy_tracks_song_dynamics_not_binary():
    # Regression: energy must be RELATIVE to the song's own loudness, not an
    # absolute scale that saturates to 1.0 the moment any audio plays.
    e = DmxEngineBase()
    cue = {"dimmer": 60}
    t = 0.0
    for _ in range(300):                               # loud passage (chorus)
        e._render_golden_anthem(0, 0, 0, 0.0, False, False,
                                (255, 0, 120), (0, 220, 255), 0.1, cue, t)
        t += 0.012
    loud_energy = e._ga_energy
    for _ in range(600):                               # sustained quiet verse
        e._render_golden_anthem(0, 0, 0, 0.0, False, False,
                                (255, 0, 120), (0, 220, 255), 0.02, cue, t)
        t += 0.012
    quiet_energy = e._ga_energy
    assert loud_energy > 0.9
    assert quiet_energy < 0.6                          # not pinned at max


def test_registered_in_engine():
    e = DmxEngineBase()
    import dmx_engine
    assert "golden_anthem" in dmx_engine.VALID_BEHAVIORS
    assert e._behavior_map["golden_anthem"] == e._render_golden_anthem

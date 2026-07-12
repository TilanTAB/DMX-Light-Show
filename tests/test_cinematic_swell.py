"""cinematic_swell: calm drifting floor + eased swells on strong kicks only.
Pins: trigger threshold, peak cap, smooth rise (no instant flash), decay back
to floor, velocity-scaled peaks, discontinuity guard, registration."""
import os
os.environ["DMX_DRY_RUN"] = "1"

import dmx_engine
from dmx_engine import (
    DmxEngineBase, _cine_envelope, CINE_TRIGGER_VELOCITY, CINE_RISE_S,
    CINE_FALL_S, CINE_FLOOR_MIN, CINE_PEAK_MAX, VALID_BEHAVIORS,
)


def make_engine():
    # test_golden_anthem.py constructs DmxEngineBase() directly -- same here.
    return DmxEngineBase()


def run_frames(eng, n, t0=0.0, velocity=0.0, kick_first=False, dt=0.012):
    """Drive the renderer directly for n frames; returns final t."""
    t = t0
    for i in range(n):
        eng._beat_velocity = velocity if (kick_first and i == 0) else 0.0
        eng._render_cinematic_swell(
            0.0, 0.0, 0.0, 0.1,                  # kick_i, snare_i, hihat_i, mid_i
            kick_first and i == 0, False,         # is_kick, is_snare
            (255, 140, 20), (0, 120, 140),        # kick_color, accent_color
            0.05, {"dimmer": 80}, t)              # volume, cue, t
        t += dt
    return t


def test_weak_hit_does_not_start_swell():
    eng = make_engine()
    run_frames(eng, 50)                                   # settle at floor
    floor_master = eng.out_master
    run_frames(eng, 50, t0=0.6, velocity=CINE_TRIGGER_VELOCITY - 0.1,
               kick_first=True)
    # A sub-threshold kick must not lift master meaningfully above the floor.
    assert eng.out_master < floor_master + 20.0


def test_strong_hit_swells_smoothly_and_caps():
    eng = make_engine()
    run_frames(eng, 50)
    prev = eng.out_master
    max_jump = 0.0
    peak = 0.0
    t = 0.6
    for i in range(200):                                  # ~2.4s of frames
        eng._beat_velocity = 0.9 if i == 0 else 0.0
        eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, i == 0, False,
                                    (255, 140, 20), (0, 120, 140),
                                    0.05, {"dimmer": 80}, t)
        max_jump = max(max_jump, abs(eng.out_master - prev))
        peak = max(peak, eng.out_master)
        prev = eng.out_master
        t += 0.012
    assert peak > 150.0                                   # swell clearly fired
    assert peak <= 255.0 * CINE_PEAK_MAX + 1.0            # hard cap holds
    assert max_jump < 30.0                                # eased, not a flash


def test_swell_decays_back_to_floor():
    eng = make_engine()
    run_frames(eng, 50)
    floor_master = eng.out_master
    # Trigger, then run well past rise+fall (+EMA slack).
    total_frames = int((CINE_RISE_S + CINE_FALL_S) / 0.012) + 300
    run_frames(eng, total_frames, t0=0.6, velocity=0.9, kick_first=True)
    assert abs(eng.out_master - floor_master) < 15.0


def test_stronger_hit_peaks_higher():
    def peak_for(v):
        eng = make_engine()
        run_frames(eng, 50)
        peak, t = 0.0, 0.6
        for i in range(200):
            eng._beat_velocity = v if i == 0 else 0.0
            eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, i == 0, False,
                                        (255, 140, 20), (0, 120, 140),
                                        0.05, {"dimmer": 80}, t)
            peak = max(peak, eng.out_master)
            t += 0.012
        return peak
    assert peak_for(1.0) > peak_for(CINE_TRIGGER_VELOCITY + 0.05) + 10.0


def test_discontinuity_guard_on_seek():
    eng = make_engine()
    run_frames(eng, 50)
    drift_before = eng._cs_drift_phase
    # One frame with t jumped 60s ahead: drift must advance <= one nominal frame.
    eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, False, False,
                                (255, 140, 20), (0, 120, 140),
                                0.05, {"dimmer": 80}, 60.6)
    assert abs(eng._cs_drift_phase - drift_before) < 0.01


def test_floor_never_dark_at_dimmer_zero():
    eng = make_engine()
    t = 0.0
    for _ in range(100):
        eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, False, False,
                                    (255, 140, 20), (0, 120, 140),
                                    0.05, {"dimmer": 0}, t)
        t += 0.012
    assert eng.out_master >= 255.0 * CINE_FLOOR_MIN - 5.0


def test_registered_in_engine():
    assert "cinematic_swell" in VALID_BEHAVIORS
    eng = make_engine()
    assert eng._behavior_map["cinematic_swell"] == eng._render_cinematic_swell

"""cinematic_swell: calm drifting floor + eased swells on strong kicks only.
Pins: ratio-based trigger gate (kick_i vs onset threshold -- deliberately
independent of _beat_velocity, which GRADES every onset from ratio 1.0 via
beat_velocity_from_ratio while this gate IGNORES hits below 1.6x), peak cap,
smooth rise (no instant flash), decay back to floor, ratio-scaled peaks,
sag-free mid-rise retrigger, discontinuity guard, registration."""
import os
os.environ["DMX_DRY_RUN"] = "1"

import dmx_engine
from dmx_engine import (
    DmxEngineBase, _cine_envelope, _cine_rise_age_for,
    CINE_TRIGGER_RATIO, CINE_FULL_RATIO, CINE_RISE_S,
    CINE_FALL_S, CINE_FLOOR_MIN, CINE_PEAK_MAX, VALID_BEHAVIORS,
)


def make_engine():
    # test_golden_anthem.py constructs DmxEngineBase() directly -- same here.
    eng = DmxEngineBase()
    eng.profile_kick_thresh = 0.10  # explicit: ratio math depends on it
    return eng


def run_frames(eng, n, t0=0.0, kick_i=0.0, kick_first=False, dt=0.012):
    """Drive the renderer directly for n frames; returns final t.
    _beat_velocity is set but irrelevant here: the renderer gates on the raw
    kick_i/thresh ratio (CINE_TRIGGER_RATIO), not on graded velocity."""
    t = t0
    for i in range(n):
        hit = kick_first and i == 0
        eng._beat_velocity = 1.0 if hit else 0.0
        eng._render_cinematic_swell(
            kick_i if hit else 0.0, 0.0, 0.0, 0.1,  # kick_i, snare_i, hihat_i, mid_i
            hit, False,                              # is_kick, is_snare
            (255, 140, 20), (0, 120, 140),           # kick_color, accent_color
            0.05, {"dimmer": 80}, t)                 # volume, cue, t
        t += dt
    return t


def test_weak_hit_does_not_start_swell():
    eng = make_engine()
    run_frames(eng, 50)                                   # settle at floor
    floor_master = eng.out_master
    # kick_i=0.12 -> ratio 1.2 < CINE_TRIGGER_RATIO (1.6): a real but weak
    # onset (is_kick True, velocity clamped to 1.0) must not start a swell.
    run_frames(eng, 50, t0=0.6, kick_i=0.12, kick_first=True)
    assert eng.out_master < floor_master + 20.0


def test_strong_hit_swells_smoothly_and_caps():
    eng = make_engine()
    run_frames(eng, 50)
    prev = eng.out_master
    max_jump = 0.0
    peak = 0.0
    t = 0.6
    for i in range(200):                                  # ~2.4s of frames
        hit = i == 0
        eng._beat_velocity = 1.0 if hit else 0.0
        # kick_i=0.32 -> ratio 3.2 >= CINE_FULL_RATIO: full-strength swell.
        eng._render_cinematic_swell(0.32 if hit else 0.0, 0.0, 0.0, 0.1,
                                    hit, False,
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
    run_frames(eng, total_frames, t0=0.6, kick_i=0.32, kick_first=True)
    assert abs(eng.out_master - floor_master) < 15.0


def test_stronger_hit_peaks_higher():
    def peak_for(kick_i):
        eng = make_engine()
        run_frames(eng, 50)
        peak, t = 0.0, 0.6
        for i in range(200):
            hit = i == 0
            eng._beat_velocity = 1.0 if hit else 0.0
            eng._render_cinematic_swell(kick_i if hit else 0.0, 0.0, 0.0, 0.1,
                                        hit, False,
                                        (255, 140, 20), (0, 120, 140),
                                        0.05, {"dimmer": 80}, t)
            peak = max(peak, eng.out_master)
            t += 0.012
        return peak
    # ratio 3.2 (full) must clearly out-peak ratio 2.0 (mid-strength).
    assert peak_for(0.32) > peak_for(0.20) + 10.0


def test_retrigger_mid_rise_never_sags():
    # A second, stronger hit mid-rise must not pull output backward: the
    # rise resumes from the age matching the current height (I2 fix).
    eng = make_engine()
    run_frames(eng, 50)
    t = 0.6
    prev = eng.out_master
    sagged = False
    rise_frames = int(CINE_RISE_S / 0.012)
    for i in range(200):
        hit = i in (0, 20)                    # second hit ~0.24s in, mid-rise
        eng._beat_velocity = 1.0 if hit else 0.0
        kick_i = (0.20 if i == 0 else 0.40) if hit else 0.0
        eng._render_cinematic_swell(kick_i, 0.0, 0.0, 0.1, hit, False,
                                    (255, 140, 20), (0, 120, 140),
                                    0.05, {"dimmer": 80}, t)
        # Monotone from first hit through the end of the retriggered rise;
        # the natural fall afterwards is allowed to decrease, of course.
        if i <= 20 + rise_frames and eng.out_master < prev - 1.0:
            sagged = True
        prev = eng.out_master
        t += 0.012
    assert not sagged


def test_discontinuity_guard_on_seek():
    eng = make_engine()
    # Start a swell so we can also pin the swell-age advance across the jump.
    run_frames(eng, 10, kick_i=0.32, kick_first=True)
    drift_before = eng._cs_drift_phase
    age_before = eng._cs_swell_age
    assert age_before is not None
    # One frame with t jumped 60s ahead: everything advances one nominal frame.
    eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, False, False,
                                (255, 140, 20), (0, 120, 140),
                                0.05, {"dimmer": 80}, 60.6)
    # One nominal frame of drift is 0.012/20 = 0.0006 -- the DT clamp alone
    # (0.1/20 = 0.005) would fail this bound; only the guard passes it.
    assert abs(eng._cs_drift_phase - drift_before) < 0.002
    # Post-review fix: the guard now CLEARS an armed swell on discontinuity
    # (ambient_pulse precedent) -- stronger than the old "advances <= one
    # nominal frame" bound, which let a stale swell resume mid-envelope.
    assert eng._cs_swell_age is None


def test_floor_never_dark_at_dimmer_zero():
    eng = make_engine()
    t = 0.0
    for _ in range(100):
        eng._render_cinematic_swell(0.0, 0.0, 0.0, 0.1, False, False,
                                    (255, 140, 20), (0, 120, 140),
                                    0.05, {"dimmer": 0}, t)
        t += 0.012
    assert eng.out_master >= 255.0 * CINE_FLOOR_MIN - 5.0


def test_rise_age_inversion_matches_envelope():
    # _cine_rise_age_for is the smoothstep inverse on the rise segment.
    for frac in (0.0, 0.1, 0.35, 0.5, 0.72, 0.9, 1.0):
        age = _cine_rise_age_for(frac)
        assert 0.0 <= age <= CINE_RISE_S
        assert abs(_cine_envelope(age) - frac) < 1e-4


def test_registered_in_engine():
    assert "cinematic_swell" in VALID_BEHAVIORS
    eng = make_engine()
    assert eng._behavior_map["cinematic_swell"] == eng._render_cinematic_swell


def test_registered_in_llm_designer_repair_gate():
    # Spec test item 5: the LLM repair pass must not downgrade
    # cinematic_swell to beat_reactive (set-membership silently drops
    # on merges -- this pins llm_designer's own VALID_BEHAVIORS).
    import llm_designer
    assert "cinematic_swell" in llm_designer.VALID_BEHAVIORS
    plan = {"show_name": "t", "cues": [{
        "start_time": 0, "end_time": 10,
        "color_1": [255, 140, 20], "color_2": [0, 120, 140],
        "energy": 5, "strobe": False, "behavior": "cinematic_swell",
        "dimmer": 80, "fade_in": 1, "fade_out": 1,
        "section_name": "intro", "mood": "dark"}]}
    out = llm_designer._validate_and_repair_plan(plan)
    assert out["cues"][0]["behavior"] == "cinematic_swell"


def test_no_phantom_swell_on_reentry():
    # A swell armed just before rotation/seek deselection must NOT resume
    # mid-envelope on re-entry (the ambient_pulse "stale hit" precedent).
    # Pre-fix: the guard reset dt only, and a re-entry 60s later replayed
    # the armed swell at full height (+107 master, simulated).
    eng = make_engine()
    run_frames(eng, 50)                                    # settle at floor
    floor_master = eng.out_master
    # Arm a swell with a strong kick (ratio 3.5 at thresh 0.10), two frames.
    t = run_frames(eng, 2, t0=0.6, kick_i=0.35, kick_first=True)
    assert eng._cs_swell_age is not None                   # armed
    # Rotate away 60s; other renderers drove the lights meanwhile.
    eng.out_master = floor_master
    for i in range(200):                                   # 2.4s of re-entry
        run_frames(eng, 1, t0=t + 60.0 + i * 0.012)
        assert eng.out_master < floor_master + 20.0        # no phantom swell

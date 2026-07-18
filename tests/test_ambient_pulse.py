"""ambient_pulse: layered multi-band beat-locked ambient. Pins: graded kick
pulse with no post-hit step-up, snare color flip without master spike,
gated hi-hat shimmer, mid-driven breathing floor, discontinuity guard,
registration. Built on the threshold-excess _beat_velocity (soft ~0.1,
hard 1.0) merged in 61cb7a3."""
import os
os.environ["DMX_DRY_RUN"] = "1"

from dmx_engine import (
    DmxEngineBase, VALID_BEHAVIORS,
    PULSE_FLOOR_MIN, PULSE_FLOOR_MAX, PULSE_SNARE_FLIP_S, PULSE_HIHAT_THRESH,
)


def make_engine():
    eng = DmxEngineBase()
    return eng


def frame(eng, t, kick=False, snare=False, velocity=0.0,
          mid_i=0.2, hihat_i=0.0, dimmer=80):
    eng._beat_velocity = velocity
    eng._render_ambient_pulse(0.0, 0.0, hihat_i, mid_i, kick, snare,
                              (20, 60, 255), (255, 0, 180),
                              0.05, {"dimmer": dimmer}, t)


def settle(eng, n=100, t0=0.0):
    t = t0
    for _ in range(n):
        frame(eng, t)
        t += 0.012
    return t


def test_kick_pulse_rises_instantly_then_never_steps_up():
    eng = make_engine()
    t = settle(eng)
    floor_master = eng.out_master
    frame(eng, t, kick=True, velocity=1.0)
    hit_master = eng.out_master
    assert hit_master > floor_master + 50.0      # instant, visible rise
    prev = hit_master
    for i in range(120):                          # ~1.4s decay window
        t += 0.012
        frame(eng, t)
        assert eng.out_master <= prev + 1e-9      # monotone: no step-up
        prev = eng.out_master
    assert abs(eng.out_master - floor_master) < 20.0   # back near floor


def test_kick_pulse_graded_by_velocity():
    def peak_for(v):
        eng = make_engine()
        t = settle(eng)
        frame(eng, t, kick=True, velocity=v)
        return eng.out_master
    assert peak_for(1.0) > peak_for(0.2) + 30.0


def test_snare_flips_color_without_master_spike():
    eng = make_engine()
    t = settle(eng)
    master_before = eng.out_master
    r_before = eng.out_r
    frame(eng, t, snare=True, velocity=0.5)
    # Master must not pulse on a snare...
    assert eng.out_master < master_before + 15.0
    # ...but color moves toward the accent (255, 0, 180): red rises.
    for _ in range(10):
        t += 0.012
        frame(eng, t, snare=False)
    assert eng.out_r > r_before + 5.0
    # And it eases back after the flip window.
    for _ in range(int(PULSE_SNARE_FLIP_S / 0.012) + 80):
        t += 0.012
        frame(eng, t)
    assert abs(eng.out_r - r_before) < 15.0


def test_hihat_shimmer_gated_and_fast():
    eng = make_engine()
    t = settle(eng)
    w_quiet = eng.out_w
    frame(eng, t, hihat_i=PULSE_HIHAT_THRESH - 0.1)
    assert eng.out_w <= w_quiet + 1.0             # below gate: nothing
    frame(eng, t + 0.012, hihat_i=PULSE_HIHAT_THRESH + 0.2)
    w_spike = eng.out_w
    assert w_spike > 30.0                          # spike fired
    frame(eng, t + 0.024, hihat_i=0.0)
    assert eng.out_w < w_spike * 0.7               # fast decay


def test_floor_breathes_with_mids_and_never_dark():
    def settled_master(mid, dimmer):
        eng = make_engine()
        t = 0.0
        for _ in range(300):
            frame(eng, t, mid_i=mid, dimmer=dimmer)
            t += 0.012
        return eng.out_master
    assert settled_master(0.9, 80) > settled_master(0.05, 80) + 20.0
    # Floor not dimmer-scaled: still lit at dimmer 0.
    assert settled_master(0.05, 0) >= 255.0 * PULSE_FLOOR_MIN - 5.0
    # Capped: loud mids never exceed the floor max band by much.
    assert settled_master(1.0, 80) <= 255.0 * PULSE_FLOOR_MAX + 10.0


def test_discontinuity_guard_on_seek():
    eng = make_engine()
    settle(eng)
    drift_before = eng._ap_drift_phase
    frame(eng, 60.6)                               # 60s jump, one frame
    # Guard: advance <= one nominal frame (0.012/16 = 0.00075), NOT the
    # dt-clamp alone (0.1/16 = 0.00625) -- bound must discriminate.
    assert abs(eng._ap_drift_phase - drift_before) < 0.002


def test_kick_visible_over_loud_floor():
    # F1 pin: at dimmer 50 with loud mids (floor at PULSE_FLOOR_MAX), max()
    # compositing swallowed sub-floor pulses (soft kick delta 0.0, mid 4.5).
    # Additive compositing must keep every kick visible over any floor.
    def hit_delta(v):
        eng = make_engine()
        t = 0.0
        for _ in range(300):
            frame(eng, t, mid_i=1.0, dimmer=50)
            t += 0.012
        settled = eng.out_master
        frame(eng, t, kick=True, velocity=v, mid_i=1.0, dimmer=50)
        return eng.out_master - settled
    assert hit_delta(0.2) > 15.0                   # soft kick still visible
    assert hit_delta(0.5) > 30.0                   # mid kick clearly visible


def test_no_phantom_pulse_on_reentry():
    eng = make_engine()
    t = settle(eng)
    floor_master = eng.out_master
    frame(eng, t, kick=True, velocity=1.0)
    t += 0.012
    frame(eng, t)
    t += 0.012
    frame(eng, t)
    # Simulate 60s rotated away: another renderer drove the lights back down
    # to ambient levels; only this renderer's private state is stale.
    eng.out_master = floor_master
    frame(eng, t + 60.0)                           # discontinuity, no kick
    assert abs(eng.out_master - floor_master) < 20.0   # no stale-hit replay


def test_registered_in_engine():
    assert "ambient_pulse" in VALID_BEHAVIORS
    eng = make_engine()
    assert eng._behavior_map["ambient_pulse"] == eng._render_ambient_pulse

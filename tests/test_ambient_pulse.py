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


def _loopback_engine():
    # Imports pyaudiowpatch (Windows-only) -- fine on this project's machine.
    import tempfile
    from music_light import DMXEngine
    eng = DMXEngine()
    # Redirect IPC files to temp: driving _dispatch past frame_counter 50
    # triggers _write_playback_state, which must not clobber the repo's
    # tracked playback_state.json.
    eng._state_file = os.path.join(tempfile.gettempdir(), "test_playback_state.json")
    eng._command_file = os.path.join(tempfile.gettempdir(), "test_playback_command.json")
    return eng


def _drive_dispatch(eng, n=40, loud=True):
    """Push frames through _dispatch with loud punchy-looking input."""
    vol = 0.01 if loud else 0.00006
    for i in range(n):
        eng.frame_counter += 1
        eng._dispatch(0.5, 0.2, 0.3, 0.1,
                      0.5 if loud else 0.0, 0.1, 0.2, 0.3,
                      loud and (i % 10 == 0), False,
                      vol, 48000)


def test_force_behavior_pins_dispatch_at_any_energy():
    eng = _loopback_engine()
    eng.profile_force_behavior = "ambient_pulse"
    eng.energy_state = "high"                      # would normally go punchy
    _drive_dispatch(eng, loud=True)
    assert eng.current_behavior == "ambient_pulse"
    # Not just the label: the pinned renderer actually ran (its private
    # accumulated-dt state only moves inside _render_ambient_pulse).
    assert eng._ap_last_render_t is not None
    eng.energy_state = "calm"                      # would normally rotate ambient
    _drive_dispatch(eng, loud=False)
    assert eng.current_behavior == "ambient_pulse"


def test_no_force_behavior_keeps_auto_detection():
    import time as _time
    eng = _loopback_engine()
    assert eng.profile_force_behavior is None      # default: auto
    eng.energy_state = "high"
    # Fresh state timestamp: keeps the state machine from instantly demoting
    # "high" on the huge time_in_state a zero epoch would imply.
    eng.energy_state_since = _time.time()
    _drive_dispatch(eng, loud=True)
    assert eng.current_behavior != "ambient_pulse" # auto picked something else


def _load_profile_dict(eng, prof):
    import json, tempfile, os as _os
    fd, path = tempfile.mkstemp(suffix=".json")
    with _os.fdopen(fd, "w") as f:
        json.dump(prof, f)
    try:
        eng.load_profile(path)
    finally:
        _os.remove(path)


def test_force_behavior_unknown_value_falls_back():
    # Punchy names ARE in VALID_BEHAVIORS but loopback dispatch collapses
    # them all into _render_loopback_direct (the name is ignored) -- pinning
    # must reject them exactly like unknown names, or the IPC label lies.
    for bad in ("no_such_renderer", "strobe_blast"):
        eng = _loopback_engine()
        _load_profile_dict(eng, {"name": "Bad", "force_behavior": bad})
        assert eng.profile_force_behavior is None  # warned + fell back


def test_force_behavior_unhashable_value_does_not_abort_load():
    # A non-string JSON value (natural typo: a list) is unhashable; raw set
    # membership raised TypeError into load_profile's broad except, leaving
    # the profile HALF-applied (palettes skipped) and a stale pin surviving.
    eng = _loopback_engine()
    eng.profile_force_behavior = "ambient_pulse"   # stale pin from "before"
    _load_profile_dict(eng, {"name": "Typo",
                             "force_behavior": ["ambient_pulse"],
                             "palettes": [[[1, 2, 3], [4, 5, 6]]]})
    assert eng.profile_force_behavior is None      # stale pin cleared
    assert eng.palettes == [((1, 2, 3), (4, 5, 6))]  # rest of profile applied


def test_force_behavior_valid_value_loads_from_json():
    # End-to-end through the profile JSON: pins the "force_behavior" key name.
    eng = _loopback_engine()
    _load_profile_dict(eng, {"name": "Good", "force_behavior": "ambient_pulse"})
    assert eng.profile_force_behavior == "ambient_pulse"


def test_variety_still_evolves_when_pinned():
    eng = _loopback_engine()
    eng.profile_force_behavior = "ambient_pulse"
    _drive_dispatch(eng, n=5, loud=True)
    # Force the evolution timer past the threshold and dispatch again.
    eng.variety._section_start_t = -999.0
    _drive_dispatch(eng, n=5, loud=True)
    # begin_section resets _section_start_t to the variety clock (>= 0):
    # proves the force_evolve path still executes under pinning. Would fail
    # if pinning short-circuited the variety tick/begin_section path.
    assert eng.variety._section_start_t > -900.0


def test_registered_in_llm_designer_repair_gate():
    import llm_designer
    assert "ambient_pulse" in llm_designer.VALID_BEHAVIORS
    plan = {"show_name": "t", "cues": [{
        "start_time": 0, "end_time": 10,
        "color_1": [20, 60, 255], "color_2": [255, 0, 180],
        "energy": 5, "strobe": False, "behavior": "ambient_pulse",
        "dimmer": 80, "fade_in": 1, "fade_out": 1,
        "section_name": "groove", "mood": "cool"}]}
    out = llm_designer._validate_and_repair_plan(plan)
    assert out["cues"][0]["behavior"] == "ambient_pulse"


def test_pinnable_set_is_subset_of_valid_behaviors():
    # Pins one direction of the two hand-maintained name lists: every
    # pinnable name must be engine-valid AND actually dispatchable via
    # _behavior_map (a stale entry here would pin to a renderer that
    # cannot run). NOTE: the reverse drift -- a NEW ambient renderer
    # forgotten in AMBIENT_DISPATCH_BEHAVIORS and thus unpinnable -- is
    # NOT catchable by a subset check; guarding it needs an authoritative
    # ambient-renderer list in dmx_engine (deferred).
    import music_light
    from dmx_engine import VALID_BEHAVIORS as ENGINE_VALID
    assert music_light.AMBIENT_DISPATCH_BEHAVIORS <= ENGINE_VALID
    eng = music_light.DMXEngine()
    for name in music_light.AMBIENT_DISPATCH_BEHAVIORS:
        assert name in eng._behavior_map

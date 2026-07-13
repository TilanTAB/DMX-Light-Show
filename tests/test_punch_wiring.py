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

def test_loopback_direct_master_graded_by_kick_strength():
    # music_light's live loopback path had its OWN local copy of the pinned
    # formula: a barely-over-threshold kick and a 3x kick both rendered master
    # 255. The direct renderer must grade brightness by threshold excess.
    from music_light import DMXEngine
    def master_for(kick_i_mult):
        e = DMXEngine()
        e.profile_deep_bass_enabled = False   # take the NORMAL BEAT branch
        kick_i = e.profile_kick_thresh * kick_i_mult
        e._render_loopback_direct(0.5, 0, 0, 0, kick_i, 0.0, 0, 0,
                                  True, False,
                                  (255, 0, 120), (0, 220, 255), (255, 255, 0),
                                  0.05, 1.0)
        return e.out_master
    soft = master_for(1.05)
    hard = master_for(3.0)
    assert soft < 135.0        # barely fired -> near the 120 floor
    assert hard == 255.0       # 3x threshold -> full blast
    assert soft < hard

# --- Beat-hold inversion: the hold floor must never exceed the brightness
# of the hit that armed it. Before this fix a soft onset rendered ~120-199
# on the hit frame and the hold clamp raised it to 200 on the NEXT frame --
# a visible "double pulse" where the light brightens after the beat.

def test_beat_reactive_soft_hit_hold_never_brighter_than_hit():
    e = DmxEngineBase()
    e._beat_velocity = 0.2
    cue = {"energy": 7, "dimmer": 80}
    e._render_beat_reactive(0.3, 0, 0, 0, True, False,
                            (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    hit_master = e.out_master
    for i in range(e.profile_beat_hold):
        e._render_beat_reactive(0.0, 0, 0, 0, False, False,
                                (255, 0, 120), (0, 220, 255), 0.05, cue,
                                1.0 + 0.02 * (i + 1))
        assert e.out_master <= hit_master + 1e-9

def test_bass_white_blast_soft_hit_hold_never_brighter_than_hit():
    e = DmxEngineBase()
    e._beat_velocity = 0.2
    cue = {"energy": 7, "dimmer": 80}
    e._render_bass_white_blast(0.3, 0, 0, 0, True, False,
                               (255, 0, 120), (0, 220, 255), 0.05, cue, 1.0)
    hit_master = e.out_master
    for i in range(e.profile_beat_hold):
        e._render_bass_white_blast(0.0, 0, 0, 0, False, False,
                                   (255, 0, 120), (0, 220, 255), 0.05, cue,
                                   1.0 + 0.02 * (i + 1))
        assert e.out_master <= hit_master + 1e-9

def test_loopback_direct_soft_hit_hold_never_brighter_than_hit():
    from music_light import DMXEngine
    e = DMXEngine()
    e.profile_deep_bass_enabled = False
    kick_i = e.profile_kick_thresh * 1.4   # soft onset: velocity 0.2
    e._render_loopback_direct(0.5, 0, 0, 0, kick_i, 0.0, 0, 0,
                              True, False,
                              (255, 0, 120), (0, 220, 255), (255, 255, 0),
                              0.05, 1.0)
    hit_master = e.out_master
    for i in range(e.profile_beat_hold):
        e._render_loopback_direct(0.1, 0, 0, 0, 0.0, 0.0, 0, 0,
                                  False, False,
                                  (255, 0, 120), (0, 220, 255), (255, 255, 0),
                                  0.05, 1.0 + 0.02 * (i + 1))
        assert e.out_master <= hit_master + 1e-9

def test_loopback_deep_bass_blast_stays_bright_on_soft_kick():
    # The deep-bass combo is a deliberately dramatic special blast (white 255).
    # Graded velocity must not dim it into mush: master keeps a 200 floor,
    # grading only above that.
    from music_light import DMXEngine
    e = DMXEngine()
    e.profile_deep_bass_enabled = True
    # kick_mag == fresh peak_kick -> ratio 1.0 > deep_bass_thresh (0.80)
    kick_i = e.profile_kick_thresh * 1.05   # barely-over-threshold kick
    e._render_loopback_direct(0.5, 0, 0, 0, kick_i, 0.0, 0, 0,
                              True, False,
                              (255, 0, 120), (0, 220, 255), (255, 255, 0),
                              0.05, 1.0)
    assert e.out_w == 255.0            # confirms the deep-bass branch fired
    assert e.out_master >= 200.0

def test_loopback_direct_uses_shared_velocity_helper():
    import inspect, music_light
    src = inspect.getsource(music_light.DMXEngine._render_loopback_direct)
    assert "beat_velocity_from_ratio" in src
    assert "min(1.0, kick_i / max(self.profile_kick_thresh" not in src

def test_process_audio_velocity_lines_use_helper():
    # Pin the wiring: process_audio must derive _beat_velocity from the
    # shared helper, not the old pinned-at-1.0 min() formula.
    import inspect, dmx_engine
    src = inspect.getsource(dmx_engine.DmxEngineBase.process_audio)
    assert "beat_velocity_from_ratio" in src
    assert "min(1.0, kick_i / max(self.profile_kick_thresh" not in src

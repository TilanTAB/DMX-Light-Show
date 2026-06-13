from dmx_variety import VarietyEngine, Intent, PALETTES


def _intent(energy=8, mood=None, sid="s", new=True, bpm=128.0):
    return Intent(energy=energy, mood=mood, section_id=sid,
                  is_new_section=new, bpm=bpm)


def test_begin_section_returns_palette_matching_energy():
    ve = VarietyEngine(seed=1)
    p = ve.begin_section(_intent(energy=2))
    lo, hi = p["energy"]
    assert lo <= 2 <= hi


def test_begin_section_respects_mood():
    ve = VarietyEngine(seed=1)
    p = ve.begin_section(_intent(energy=8, mood="neon"))
    assert p["mood"] == "neon"


def test_anti_repeat_avoids_recent_palettes():
    ve = VarietyEngine(seed=7)
    seen = [ve.begin_section(_intent())["id"] for _ in range(5)]
    # no palette repeats within a window of 4 selections
    for k in range(4, len(seen)):
        assert seen[k] not in seen[k - 4:k]


def test_seeding_is_deterministic():
    a = VarietyEngine(seed=42)
    b = VarietyEngine(seed=42)
    seq_a = [a.begin_section(_intent())["id"] for _ in range(6)]
    seq_b = [b.begin_section(_intent())["id"] for _ in range(6)]
    assert seq_a == seq_b


def test_different_seeds_can_diverge():
    # Deterministic per seed; across a 6-section run two seeds should differ.
    seq_a = VarietyEngine(seed=1)
    seq_b = VarietyEngine(seed=2)
    run_a = [seq_a.begin_section(_intent())["id"] for _ in range(6)]
    run_b = [seq_b.begin_section(_intent())["id"] for _ in range(6)]
    assert run_a != run_b


def test_seed_color_picks_nearest_family():
    # An intent carrying a magenta-ish seed_color should land on a neon/pink
    # family whose primary is close to it, deterministically (no rng).
    ve = VarietyEngine(seed=999)
    ve.begin_section(_intent(energy=8, mood=None))  # warm up recent buffer
    chosen = ve.begin_section(Intent(energy=8, mood=None, section_id="x",
                                     is_new_section=True, bpm=128.0,
                                     seed_color=[255, 0, 120]))
    # nearest primary to [255,0,120] among energy-8 candidates is neon_pink
    assert chosen["id"] == "neon_pink"


def test_relax_when_library_exhausted():
    tiny = PALETTES[:2]
    ve = VarietyEngine(palettes=tiny, seed=1)
    # more selections than the library size must not crash
    ids = [ve.begin_section(_intent(energy=tiny[0]["energy"][0]))["id"] for _ in range(5)]
    assert len(ids) == 5

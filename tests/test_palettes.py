from dmx_variety import PALETTES

def test_palettes_are_well_formed():
    assert len(PALETTES) >= 12
    ids = [p["id"] for p in PALETTES]
    assert len(ids) == len(set(ids)), "palette ids must be unique"
    for p in PALETTES:
        assert set(p) >= {"id", "mood", "energy", "primary", "secondary", "accent"}
        lo, hi = p["energy"]
        assert 1 <= lo <= hi <= 10
        for key in ("primary", "secondary", "accent"):
            r, g, b = p[key]
            assert all(0 <= c <= 255 for c in (r, g, b))

def test_palettes_cover_all_energy_bands():
    # every energy level 1..10 must be servable by at least one palette
    for e in range(1, 11):
        assert any(p["energy"][0] <= e <= p["energy"][1] for p in PALETTES)

def test_palettes_cover_core_moods():
    moods = {p["mood"] for p in PALETTES}
    assert {"warm", "cool", "neon", "euphoric", "dark"} <= moods

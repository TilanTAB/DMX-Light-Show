from dmx_variety import Intent

def test_intent_holds_fields():
    i = Intent(energy=8, mood="neon", section_id="chorus1",
               is_new_section=True, bpm=128.0, strobe_allowed=True)
    assert i.energy == 8
    assert i.mood == "neon"
    assert i.section_id == "chorus1"
    assert i.is_new_section is True
    assert i.bpm == 128.0
    assert i.strobe_allowed is True

def test_intent_defaults():
    i = Intent(energy=3, mood="warm", section_id="intro")
    assert i.is_new_section is False
    assert i.bpm == 0.0
    assert i.strobe_allowed is False
    assert i.seed_color is None

def test_intent_carries_seed_color():
    i = Intent(energy=8, mood="neon", section_id="drop", seed_color=[255, 0, 120])
    assert i.seed_color == [255, 0, 120]

from llm_designer import _default_mood_for_energy


def test_default_mood_for_energy_bands():
    assert _default_mood_for_energy(1) == "warm"
    assert _default_mood_for_energy(4) == "cool"
    assert _default_mood_for_energy(7) == "neon"
    assert _default_mood_for_energy(10) == "euphoric"

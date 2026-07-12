import json
import os
import tempfile

from dmx_engine import DmxEngineBase
from dmx_variety import Intent


def _write_show(dirpath, name="Test Song"):
    path = os.path.join(dirpath, "show.json")
    show = {
        "audio_file": "x.wav",
        "song_metrics": {"bpm": 128.0},
        "lighting_plan": {
            "show_name": name,
            "cues": [{
                "start_time": 0, "end_time": 10,
                "color_1": [255, 0, 120], "color_2": [0, 220, 255],
                "energy_level": 8, "strobe_allowed": False,
                "behavior": "beat_reactive", "master_dimmer_percent": 80,
                "fade_speed_seconds": 1.0, "section_name": "A", "mood": "neon",
            }],
            "phrases": [],
        },
    }
    with open(path, "w") as f:
        json.dump(show, f)
    return path


def _palette_run(show_path, n=4):
    engine = DmxEngineBase()
    engine.load_ai_show(show_path)
    return [engine.variety.begin_section(
                Intent(energy=8, mood=None, section_id=str(i), is_new_section=True)
            )["id"] for i in range(n)]


def test_same_show_same_palette_sequence_across_processes():
    # zlib.crc32 seeding: two fresh engines (simulating two worker processes)
    # must produce the identical palette sequence for the same show. Python's
    # builtin hash() would fail this across real processes (per-process salt).
    with tempfile.TemporaryDirectory() as d:
        p = _write_show(d)
        assert _palette_run(p) == _palette_run(p)


def test_different_shows_can_diverge():
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        a = _palette_run(_write_show(d1, name="Song A"), n=6)
        b = _palette_run(_write_show(d2, name="Song B"), n=6)
        assert a != b


def test_bpm_is_read_from_show_file():
    with tempfile.TemporaryDirectory() as d:
        engine = DmxEngineBase()
        engine.load_ai_show(_write_show(d))
        assert engine.show_bpm == 128.0


def test_malformed_bpm_degrades_without_killing_the_load():
    # One bad metadata field must NOT abort cue/palette loading (repair-don't-
    # reject convention). Reproduced pre-fix: float("abc") aborted everything.
    with tempfile.TemporaryDirectory() as d:
        path = _write_show(d)
        show = json.load(open(path))
        show["song_metrics"] = {"bpm": "abc"}
        json.dump(show, open(path, "w"))
        engine = DmxEngineBase()
        result = engine.load_ai_show(path)
        assert result == "x.wav"                 # load completed, not aborted
        assert len(engine.synced_cues) == 1      # cues survived
        assert engine.show_bpm == 0.0            # degraded, not crashed


def test_non_dict_song_metrics_ignored():
    with tempfile.TemporaryDirectory() as d:
        path = _write_show(d)
        show = json.load(open(path))
        show["song_metrics"] = "corrupt"
        json.dump(show, open(path, "w"))
        engine = DmxEngineBase()
        engine.load_ai_show(path)
        assert engine.show_bpm == 0.0
        assert len(engine.synced_cues) == 1

"""Variety & evolution layer: Intent contract, palette library, VarietyEngine.

Shared by both engines. The two directors (loopback energy state machine and
synced LLM cue) each build an Intent; the VarietyEngine consumes it identically.
"""
import random
from collections import deque


class Intent:
    """Normalized lighting intent emitted by either director."""
    __slots__ = ("energy", "mood", "section_id", "is_new_section",
                 "bpm", "strobe_allowed", "seed_color")

    def __init__(self, energy, mood, section_id,
                 is_new_section=False, bpm=0.0, strobe_allowed=False,
                 seed_color=None):
        self.energy = energy
        self.mood = mood
        self.section_id = section_id
        self.is_new_section = is_new_section
        self.bpm = bpm
        self.strobe_allowed = strobe_allowed
        self.seed_color = seed_color  # optional [R,G,B] hint (LLM color_1); None in loopback


# Curated palette families. Each: primary (kick color), secondary (snare/accent
# color), accent (combo/stab color). mood + energy range drive selection.
PALETTES = [
    {"id": "volcanic",   "mood": "warm",     "energy": [6, 10], "primary": [255, 60, 10],  "secondary": [255, 150, 0],  "accent": [255, 255, 255]},
    {"id": "ember",      "mood": "warm",     "energy": [2, 6],  "primary": [255, 90, 30],  "secondary": [200, 40, 60],  "accent": [255, 200, 120]},
    {"id": "candle",     "mood": "warm",     "energy": [1, 4],  "primary": [255, 140, 40], "secondary": [180, 70, 20],  "accent": [255, 220, 150]},
    {"id": "arctic",     "mood": "cool",     "energy": [4, 8],  "primary": [0, 180, 255],  "secondary": [10, 30, 180],  "accent": [255, 255, 255]},
    {"id": "deep_ocean", "mood": "cool",     "energy": [1, 5],  "primary": [10, 30, 180],  "secondary": [0, 120, 140],  "accent": [120, 200, 255]},
    {"id": "glacier",    "mood": "cool",     "energy": [3, 7],  "primary": [120, 200, 255],"secondary": [40, 90, 200],  "accent": [255, 255, 255]},
    {"id": "neon_pink",  "mood": "neon",     "energy": [6, 10], "primary": [255, 0, 120],  "secondary": [0, 220, 255],  "accent": [255, 255, 255]},
    {"id": "acid",       "mood": "neon",     "energy": [6, 10], "primary": [180, 255, 0],  "secondary": [255, 0, 200],  "accent": [255, 255, 255]},
    {"id": "violet_haze","mood": "neon",     "energy": [4, 9],  "primary": [130, 0, 255],  "secondary": [0, 220, 255],  "accent": [255, 120, 255]},
    {"id": "sunburst",   "mood": "euphoric", "energy": [5, 10], "primary": [255, 200, 0],  "secondary": [255, 0, 120],  "accent": [255, 255, 255]},
    {"id": "rave",       "mood": "euphoric", "energy": [7, 10], "primary": [0, 255, 100],  "secondary": [255, 0, 200],  "accent": [255, 255, 255]},
    {"id": "prism",      "mood": "euphoric", "energy": [4, 9],  "primary": [255, 0, 0],    "secondary": [0, 100, 255],  "accent": [0, 255, 100]},
    {"id": "midnight",   "mood": "dark",     "energy": [1, 5],  "primary": [40, 0, 80],    "secondary": [0, 40, 90],    "accent": [120, 80, 200]},
    {"id": "blood",      "mood": "dark",     "energy": [5, 10], "primary": [120, 0, 0],    "secondary": [200, 0, 40],   "accent": [255, 60, 60]},
    {"id": "forest",     "mood": "cool",     "energy": [1, 5],  "primary": [0, 120, 60],   "secondary": [40, 90, 30],   "accent": [150, 255, 180]},
    {"id": "aurora",     "mood": "euphoric", "energy": [2, 6],  "primary": [0, 255, 150],  "secondary": [80, 0, 255],   "accent": [0, 220, 255]},
]


def _shift_hue(rgb, degrees):
    """Rotate an RGB color's hue by `degrees`. Cheap, dependency-free."""
    import colorsys
    r, g, b = (c / 255.0 for c in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [int(r * 255), int(g * 255), int(b * 255)]


def _color_distance(a, b):
    """Euclidean distance between two RGB triples."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _nearest_palette(candidates, color):
    """The candidate whose primary is closest to `color`. Deterministic."""
    return min(candidates, key=lambda p: _color_distance(p["primary"], color))


class VarietyEngine:
    """Owns all anti-monotony policy: palette selection (anti-repeat + per-song
    seed + optional LLM color seed) and phrase-grid texture evolution.
    Mode-agnostic — fed via Intent."""

    PHRASE_LEN_BEATS = 8
    TIME_PHRASE_FALLBACK_S = 4.0

    def __init__(self, palettes=PALETTES, seed=None):
        self._palettes = list(palettes)
        self._recent = deque(maxlen=4)          # recent palette ids (anti-repeat)
        self._rng = random.Random(seed)
        self.current_palette = self._palettes[0]
        self.phrase_index = 0
        self._beats_in_section = 0
        self._last_t = 0.0
        self._section_start_t = 0.0
        self._last_phrase_t = 0.0

    def set_song_seed(self, seed):
        self._rng = random.Random(seed)

    def begin_section(self, intent):
        """Pick a fresh palette for a new section and reset phrase state.

        Selection precedence:
          1. energy range + anti-repeat (always),
          2. mood match (only when there is no seed_color — color wins over mood),
          3. if seed_color given -> nearest palette by primary color (deterministic);
             else -> seeded random choice (per-song identity).
        Relaxes filters step-by-step if nothing matches, so it never deadlocks."""
        in_energy = [p for p in self._palettes
                     if p["energy"][0] <= intent.energy <= p["energy"][1]]
        fresh = [p for p in in_energy if p["id"] not in self._recent] or in_energy

        if intent.seed_color is None and intent.mood is not None:
            mood_match = [p for p in fresh if p["mood"] == intent.mood]
            candidates = mood_match or fresh
        else:
            candidates = fresh

        if not candidates:
            candidates = [p for p in self._palettes if p["id"] not in self._recent] \
                or list(self._palettes)

        if intent.seed_color is not None:
            chosen = _nearest_palette(candidates, intent.seed_color)
        else:
            chosen = self._rng.choice(candidates)

        self._recent.append(chosen["id"])
        self.current_palette = chosen
        self.phrase_index = 0
        self._beats_in_section = 0
        self._section_start_t = self._last_t
        self._last_phrase_t = self._last_t
        return chosen

    def on_phrase_boundary(self):
        """Advance one phrase; texture moves are derived from phrase_index so
        the section *develops* deterministically (beat-quantized = intentional)."""
        self.phrase_index += 1

    def current_colors(self):
        """(color_1, color_2, accent) for this frame, modulated by phrase_index.

        Phase pattern (cycles every 4 phrases):
          0: primary / secondary / accent
          1: primary / accent    / secondary   (swap accent in)
          2: secondary / primary / accent       (flip roles)
          3: primary / hue-shifted secondary / accent
        """
        p = self.current_palette
        prim, sec, acc = p["primary"], p["secondary"], p["accent"]
        phase = self.phrase_index % 4
        if phase == 0:
            return list(prim), list(sec), list(acc)
        if phase == 1:
            return list(prim), list(acc), list(sec)
        if phase == 2:
            return list(sec), list(prim), list(acc)
        return list(prim), _shift_hue(sec, 40), list(acc)

    def tick(self, is_beat, bpm, t):
        """Per-frame update. Returns {'phrase_boundary': bool,
        'seconds_in_section': float}. Directors call this every frame and read
        current_colors(); they decide when to call begin_section()."""
        self._last_t = t
        boundary = False

        if bpm and bpm > 0:
            if is_beat:
                self._beats_in_section += 1
                if self._beats_in_section % self.PHRASE_LEN_BEATS == 0:
                    self.on_phrase_boundary()
                    boundary = True
                    self._last_phrase_t = t
        else:
            # No reliable beat grid -> fall back to a wall-clock phrase interval.
            if t - self._last_phrase_t >= self.TIME_PHRASE_FALLBACK_S:
                self.on_phrase_boundary()
                boundary = True
                self._last_phrase_t = t

        return {"phrase_boundary": boundary,
                "seconds_in_section": t - self._section_start_t}

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

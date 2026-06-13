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

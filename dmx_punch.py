"""Shared punch primitives: the velocity/beat-hold/afterglow math that made
loopback feel crisp, lifted out of _render_loopback_direct so synced renderers
can use it too. Pure functions — no engine state, easy to unit-test."""

VELOCITY_FLOOR = 120.0
VELOCITY_FULL = 255.0


def velocity_brightness(beat_velocity):
    """Map a 0..1 beat velocity to master brightness 120..255.
    Soft beats stay dim; hard beats blast. Values >1 clamp to full."""
    v = max(0.0, min(1.0, beat_velocity))
    return VELOCITY_FLOOR + (VELOCITY_FULL - VELOCITY_FLOOR) * v


def afterglow(r, g, b, w):
    """One frame of warm-shifted decay (R slowest, B fastest), matching the
    _render_loopback_direct tail (R 0.95 / G 0.88 / B 0.82 / W 0.80)."""
    return r * 0.95, g * 0.88, b * 0.82, w * 0.80

"""Shared punch primitives: the velocity/beat-hold/afterglow math that made
loopback feel crisp, lifted out of _render_loopback_direct so synced renderers
can use it too. Pure functions — no engine state, easy to unit-test."""

VELOCITY_FLOOR = 120.0
VELOCITY_FULL = 255.0

# _beat_velocity measures onset strength ABOVE the detection threshold.
# Raw min(1, kick_i/thresh) clamps to exactly 1.0 on every onset frame
# (is_kick already requires kick_i > thresh), which made velocity_brightness
# a constant 255 in live playback. Map the threshold-excess ratio instead:
# ratio 1.0 (barely fired) -> velocity 0.0, ratio >= PUNCH_FULL_RATIO -> 1.0.
PUNCH_FULL_RATIO = 3.0


def beat_velocity_from_ratio(ratio):
    """Map an intensity/threshold ratio to a 0..1 beat velocity.
    ratio <= 1.0 (at or below the onset threshold) -> 0.0;
    ratio >= PUNCH_FULL_RATIO -> 1.0; linear in between."""
    return min(1.0, max(0.0, (ratio - 1.0) / (PUNCH_FULL_RATIO - 1.0)))


def velocity_brightness(beat_velocity):
    """Map a 0..1 beat velocity to master brightness 120..255.
    Soft beats stay dim; hard beats blast. Values >1 clamp to full."""
    v = max(0.0, min(1.0, beat_velocity))
    return VELOCITY_FLOOR + (VELOCITY_FULL - VELOCITY_FLOOR) * v


def afterglow(r, g, b, w):
    """One frame of warm-shifted decay (R slowest, B fastest), matching the
    _render_loopback_direct tail (R 0.95 / G 0.88 / B 0.82 / W 0.80)."""
    return r * 0.95, g * 0.88, b * 0.82, w * 0.80

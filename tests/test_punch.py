from dmx_punch import velocity_brightness, afterglow


def test_velocity_brightness_scales_between_floor_and_full():
    assert velocity_brightness(0.0) == 120.0          # soft beat -> dim floor
    assert velocity_brightness(1.0) == 255.0          # hard beat -> full
    mid = velocity_brightness(0.5)
    assert 120.0 < mid < 255.0


def test_velocity_brightness_clamps_above_one():
    assert velocity_brightness(5.0) == 255.0


def test_afterglow_shifts_warm():
    # warm shift: red decays slowest, blue fastest
    r, g, b, w = afterglow(200.0, 200.0, 200.0, 200.0)
    assert r > g > b
    assert w < 200.0

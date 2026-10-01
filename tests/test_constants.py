from hectorconfig_py.constants import load_constants


def test_shared_constants_are_loaded():
    constants = load_constants()
    assert constants.plate_radius == 226.0
    assert constants.fov == 452.0
    assert len(constants.ceg_positions) == 3
    assert constants.dngalprobes == 19
    assert constants.ngprobesmax == 6

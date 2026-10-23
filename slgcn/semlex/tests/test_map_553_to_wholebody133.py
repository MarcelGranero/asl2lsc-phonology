"""Invariants of the 553->133 conversion and its fit to upstream selection."""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01_data_prep"))
from map_553_to_wholebody133 import (BODY_MAP, FULL_MAP, LEFT_HAND_MAP,
                                     RIGHT_HAND_MAP, SCALE, convert)

sys.path.insert(0, str(REPO / "external/SAM-SLR-v2/SL-GCN/data_gen"))
from sign_gendata import selected_joints


def test_mapping_counts():
    assert len(BODY_MAP) == 17
    assert len(LEFT_HAND_MAP) == len(RIGHT_HAND_MAP) == 21
    wb = [w for w, _ in FULL_MAP]
    mp = [m for _, m in FULL_MAP]
    assert len(set(wb)) == len(wb) and len(set(mp)) == len(mp)


def test_selected_27_fully_populated():
    populated = {w for w, _ in FULL_MAP}
    missing = [j for j in selected_joints["27"] if j not in populated]
    assert not missing, f"'27' selects unpopulated wholebody slots: {missing}"


def test_convert_shapes_and_channels():
    rng = np.random.default_rng(0)
    pose = rng.random((5, 553, 3)).astype(np.float16)
    pose[0, 532:553] = np.nan          # right hand missing in frame 0
    out = convert(pose, channels="presence")
    assert out.shape == (5, 133, 3) and out.dtype == np.float32
    # x scaled to 512-space
    assert out[:, 0, 0].max() <= SCALE and out[:, 0, 0].min() >= 0
    # right-hand slots: zero coords + presence 0 in frame 0, presence 1 later
    assert (out[0, 112:133, :] == 0).all()
    assert (out[1, 112:133, 2] == 1).all()
    # face/feet slots untouched
    assert (out[:, 17:91, :] == 0).all()


def test_body_offsets_are_pose_block():
    from map_553_to_wholebody133 import BODY_MAP, POSE_OFFSET
    for _, m in BODY_MAP:
        assert POSE_OFFSET <= m < POSE_OFFSET + 33


def test_convert_xyz_mode():
    pose = np.full((2, 553, 3), 0.5, dtype=np.float16)
    out = convert(pose, channels="xyz")
    assert np.isclose(out[0, 0, 2], 0.5, atol=1e-3)   # z unscaled
    assert np.isclose(out[0, 0, 0], 0.5 * SCALE, atol=0.5)


if __name__ == "__main__":
    for fn in [test_mapping_counts, test_selected_27_fully_populated,
               test_convert_shapes_and_channels, test_convert_xyz_mode,
               test_body_offsets_are_pose_block]:
        fn()
        print(f"{fn.__name__} OK")

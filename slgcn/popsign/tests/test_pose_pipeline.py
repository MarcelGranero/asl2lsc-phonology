"""Unit checks for the parquet -> wholebody133 data boundary.

Run: python -m pytest tests/ -q   (or: python tests/test_pose_pipeline.py)

These cover the parts we wrote; upstream code is exercised by the smoke run.
The synthetic-parquet tests need no cluster data; the tests marked `real`
read one real clip and are skipped when the GTI path is unavailable.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "01_data_prep"))

from data_config import LANDMARKS_DIR  # noqa: E402
from map_543_to_wholebody133 import (BLOCK_OFFSET, LEFT_HAND_OFFSET,  # noqa: E402
                                     N_LANDMARKS, POSE_OFFSET,
                                     RIGHT_HAND_OFFSET, SCALE, convert)
from read_parquet_pose import pose_stats, read_pose  # noqa: E402

BLOCKS = [("face", 468), ("pose", 33), ("left_hand", 21), ("right_hand", 21)]


def make_parquet(path, n_frames=3, nan_left_hand=False, first_frame=0):
    """Synthetic clip: every landmark's x encodes its own dense index."""
    frame, ltype, lidx, xs, ys, zs = [], [], [], [], [], []
    for t in range(n_frames):
        for name, n in BLOCKS:
            for i in range(n):
                dense = BLOCK_OFFSET[name] + i
                frame.append(t + first_frame)
                ltype.append(name)
                lidx.append(i)
                nan = nan_left_hand and name == "left_hand"
                xs.append(np.nan if nan else dense / 1000.0)
                ys.append(np.nan if nan else t / 100.0)
                zs.append(np.nan if nan else 0.5)
    pq.write_table(pa.table({
        "frame": pa.array(frame, pa.int16()),
        "row_id": pa.array([f"{f}-{t}-{i}" for f, t, i
                            in zip(frame, ltype, lidx)]),
        "type": pa.array(ltype),
        "landmark_index": pa.array(lidx, pa.int16()),
        "x": pa.array(xs, pa.float64()),
        "y": pa.array(ys, pa.float64()),
        "z": pa.array(zs, pa.float64()),
    }), path)
    return path


def test_read_pose_shape_and_scatter(tmp_path):
    p = make_parquet(tmp_path / "c.parquet", n_frames=4)
    pose = read_pose(p)
    assert pose.shape == (4, N_LANDMARKS, 3)
    # x encodes the dense index -> every landmark landed in the right slot
    expected = np.arange(N_LANDMARKS) / 1000.0
    np.testing.assert_allclose(pose[0, :, 0], expected, rtol=1e-5)


def test_read_pose_block_offsets(tmp_path):
    """The whole port hinges on these three offsets; pin them explicitly."""
    assert BLOCK_OFFSET == {"face": 0, "pose": 468, "left_hand": 501,
                            "right_hand": 522}
    assert (POSE_OFFSET, LEFT_HAND_OFFSET, RIGHT_HAND_OFFSET) == (468, 501, 522)
    # SEM-LEX's 553-point layout is exactly +10 (the refined-face iris block)
    assert POSE_OFFSET + 10 == 478


def test_read_pose_nonzero_first_frame(tmp_path):
    """A clip whose frames start at 5 must not gain 5 leading NaN frames."""
    p = make_parquet(tmp_path / "c.parquet", n_frames=3, first_frame=5)
    pose = read_pose(p)
    assert pose.shape[0] == 3
    assert np.isfinite(pose[0, POSE_OFFSET, 0])


def test_read_pose_rejects_unknown_type(tmp_path):
    p = tmp_path / "bad.parquet"
    pq.write_table(pa.table({
        "frame": pa.array([0], pa.int16()), "row_id": pa.array(["0-x-0"]),
        "type": pa.array(["elbow"]), "landmark_index": pa.array([0], pa.int16()),
        "x": pa.array([0.0]), "y": pa.array([0.0]), "z": pa.array([0.0]),
    }), p)
    with pytest.raises(ValueError, match="unknown landmark type"):
        read_pose(p)


def test_convert_shape_and_face_slots(tmp_path):
    pose = read_pose(make_parquet(tmp_path / "c.parquet"))
    wb = convert(pose)
    assert wb.shape == (3, 133, 3) and wb.dtype == np.float32
    # feet + face slots are never populated and never read by '27'
    assert np.all(wb[:, 17:91, :] == 0)


def test_convert_scales_by_512(tmp_path):
    pose = read_pose(make_parquet(tmp_path / "c.parquet"))
    wb = convert(pose)
    # wholebody slot 0 (nose) <- dense POSE_OFFSET, whose x is offset/1000
    assert wb[0, 0, 0] == pytest.approx(POSE_OFFSET / 1000.0 * SCALE)
    # wholebody 91 (left wrist) <- dense LEFT_HAND_OFFSET
    assert wb[0, 91, 0] == pytest.approx(LEFT_HAND_OFFSET / 1000.0 * SCALE)
    assert wb[0, 112, 0] == pytest.approx(RIGHT_HAND_OFFSET / 1000.0 * SCALE)


def test_presence_channel_marks_missing_hand(tmp_path):
    pose = read_pose(make_parquet(tmp_path / "c.parquet", nan_left_hand=True))
    wb = convert(pose)
    assert np.all(wb[:, 91:112, 2] == 0.0)   # left hand absent
    assert np.all(wb[:, 112:133, 2] == 1.0)  # right hand present
    assert np.all(wb[:, 91:112, :2] == 0.0)  # NaN -> 0 after scaling


def test_convert_xyz_channel_keeps_z(tmp_path):
    pose = read_pose(make_parquet(tmp_path / "c.parquet"))
    wb = convert(pose, channels="xyz")
    assert wb[0, 0, 2] == pytest.approx(0.5)


def test_convert_rejects_wrong_width():
    with pytest.raises(AssertionError):
        convert(np.zeros((2, 553, 3), dtype=np.float32))  # SEM-LEX layout


@pytest.mark.skipif(not LANDMARKS_DIR.exists(), reason="GTI data unavailable")
def test_real_clip_roundtrip():
    p = next(iter(sorted(LANDMARKS_DIR.glob("*.parquet"))))
    pose = read_pose(p)
    assert pose.shape[1:] == (N_LANDMARKS, 3)
    stats = pose_stats(pose)
    assert stats["n_frames"] == pose.shape[0]
    assert 0.0 <= stats["frac_nan_left_hand"] <= 1.0
    wb = convert(pose)
    assert wb.shape == (pose.shape[0], 133, 3)
    assert np.all(np.isfinite(wb))  # no NaN may reach upstream
    assert np.all(np.isin(wb[:, :, 2], [0.0, 1.0]))




# --- phase-2 LR schedule across a resume (regression) -----------------------
# A resumed job must continue on ABSOLUTE epochs (--start-epoch): otherwise
# warmup re-runs at every resume and step:[150,200] is measured from the
# resume, so the decays never fire.

def _lr_at():
    import sys
    sys.path.insert(0, str(REPO / "04_phase2_multihead"))
    from train_multihead import lr_at
    return lr_at


def test_lr_decays_fire_at_absolute_epochs():
    lr_at = _lr_at()
    base, steps, warm = 0.1, [150, 200], 20
    assert lr_at(149, base, steps, warm) == pytest.approx(0.1)
    assert lr_at(150, base, steps, warm) == pytest.approx(0.01)
    assert lr_at(199, base, steps, warm) == pytest.approx(0.01)
    assert lr_at(200, base, steps, warm) == pytest.approx(0.001)


def test_resume_lr_is_continuous_not_restarted():
    """A job resuming at absolute epoch 72 must keep lr=0.1, not re-warm up."""
    lr_at = _lr_at()
    base, steps, warm = 0.1, [150, 200], 20
    # with --start-epoch the loop passes ABSOLUTE epochs
    assert lr_at(72, base, steps, warm) == pytest.approx(0.1)
    # the old (buggy) behaviour passed the LOCAL index and re-entered warmup
    local_index = 0
    assert lr_at(local_index, base, steps, warm) == pytest.approx(0.005)
    assert lr_at(72, base, steps, warm) != lr_at(local_index, base, steps, warm)


def test_resume_covers_the_remaining_absolute_range():
    """--num-epoch <remaining> + --start-epoch <done> tiles 0..total-1 exactly."""
    total, done = 250, 72
    start, n = done, total - done
    covered = list(range(start, start + n))
    assert covered[0] == 72 and covered[-1] == total - 1
    assert len(covered) == n
    # and the epochs before the resume were covered by the first job
    assert list(range(0, done)) + covered == list(range(total))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))

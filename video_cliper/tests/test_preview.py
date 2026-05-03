from unittest.mock import MagicMock, patch

import numpy as np

import preview


@patch("preview.cv2.VideoCapture")
def test_seek_seconds_sets_pos_msec(mock_cap_cls):
    instance = MagicMock()
    instance.isOpened.return_value = True
    instance.read.side_effect = [
        (True, np.zeros((2, 2, 3), dtype=np.uint8)),
    ]
    mock_cap_cls.return_value = instance

    vp = preview.VideoPreview()
    assert vp.open("C:/fake/video.mp4", duration_sec=10.0, fps=30.0) is True
    vp.seek_seconds(1.25)
    instance.set.assert_called()
    assert any(
        c[0][0] == 0 and abs(float(c[0][1]) - 1250.0) < 1.0 for c in instance.set.call_args_list
    )

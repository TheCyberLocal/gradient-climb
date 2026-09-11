from types import SimpleNamespace

import numpy as np
import pytest

from gradientclimb.capture.screen import WindowCapture
from gradientclimb.capture.windows import ClientRect, WindowUnavailable


class Camera:
    width, height = 100, 80

    def __init__(self):
        self.frame = np.zeros((20, 30, 3), dtype=np.uint8)
        self.calls = []
        self.released = False

    def grab(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame

    def release(self):
        self.released = True


def capture_with(camera):
    guard = SimpleNamespace(target=SimpleNamespace(client_rect=ClientRect(10, 20, 30, 20)))
    capture = WindowCapture(guard, backend="dxcam")
    capture._dxcam = camera
    return capture


def test_dxcam_copy_and_exact_region():
    camera = Camera()
    capture = capture_with(camera)
    result = capture._read(ClientRect(10, 20, 30, 20))
    camera.frame[:] = 255
    assert not result.any()
    assert camera.calls == [{"region": (10, 20, 40, 40), "new_frame_only": False}]
    capture.close()
    capture.close()
    assert camera.released


@pytest.mark.parametrize(
    "rect", [ClientRect(-1, 0, 10, 10), ClientRect(95, 0, 10, 10), ClientRect(0, 75, 10, 10)]
)
def test_dxcam_rejects_cross_output_capture(rect):
    camera = Camera()
    capture = capture_with(camera)
    with pytest.raises(WindowUnavailable, match="outside"):
        capture._read(rect)
    assert not camera.calls


def test_dxcam_no_frame_is_not_an_observation():
    camera = Camera()
    camera.frame = None
    with pytest.raises(WindowUnavailable, match="no available"):
        capture_with(camera)._read(ClientRect(10, 20, 30, 20))

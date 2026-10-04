from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from spektrafilm.utils import raw_file_processor


def _assert_valid_rgb(image: np.ndarray) -> None:
    assert image.dtype == np.float32
    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.size > 0
    assert np.all(np.isfinite(image))


def test_load_and_process_raw_file_smoke_without_external_raw(monkeypatch) -> None:
    raw_image = np.array(
        [
            [[8192, 8192, 8192], [12288, 8192, 6144]],
            [[16384, 12288, 8192], [8192, 12288, 16384]],
        ],
        dtype=np.uint16,
    )

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **_kwargs):
            return raw_image

    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )

    daylight = raw_file_processor.load_and_process_raw_file('synthetic.nef', white_balance='daylight')
    custom_daylight = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='custom',
        temperature=6504.0,
        tint=1.0,
    )
    custom_tungsten = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=1.0,
    )
    custom_tungsten_tinted = raw_file_processor.load_and_process_raw_file(
        'synthetic.nef',
        white_balance='custom',
        temperature=3200.0,
        tint=0.85,
    )

    for image in (daylight, custom_daylight, custom_tungsten, custom_tungsten_tinted):
        _assert_valid_rgb(image)
        assert image.shape == daylight.shape

    # 6504 K should be the daylight identity point for the simplified WB path.
    np.testing.assert_allclose(custom_daylight, daylight, atol=1e-6)

    # A significantly warmer temperature setting should change the result.
    assert not np.allclose(custom_tungsten, daylight, atol=1e-4)

    # Tint is implemented as a direct green scaling, so it must reduce mean green.
    green_mean = float(np.mean(custom_tungsten[..., 1]))
    tinted_green_mean = float(np.mean(custom_tungsten_tinted[..., 1]))
    assert tinted_green_mean < green_mean


def _patch_rawpy(monkeypatch, raw_image: np.ndarray, sizes) -> None:
    """Stub rawpy with a reader returning ``raw_image`` and reporting ``sizes``."""

    class Reader:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def postprocess(self, **_kwargs):
            return raw_image

    Reader.sizes = sizes
    monkeypatch.setattr(
        raw_file_processor,
        'rawpy',
        SimpleNamespace(
            imread=lambda path: Reader(),
            ColorSpace=SimpleNamespace(ACES='ACES'),
        ),
    )


def test_load_and_process_raw_file_applies_the_inset_active_area_crop(monkeypatch) -> None:
    """A sensor whose active area is inset in the raw frame must be cropped.

    The Sony A1 II reports usable pixels inset inside a larger frame that also
    carries a masked optical-black border; rawpy's ``postprocess`` returns the
    whole frame, so the loader has to apply LibRaw's inset crop itself.
    """

    # 5x5 frame; only the 3x3 block at (1, 1) is active, the rest is masked.
    raw_image = np.zeros((5, 5, 3), dtype=np.uint16)
    raw_image[1:4, 1:4] = 12000
    sizes = SimpleNamespace(
        crop_width=3, crop_height=3, crop_top_margin=1, crop_left_margin=1,
        width=5, height=5,
    )
    _patch_rawpy(monkeypatch, raw_image, sizes)

    image = raw_file_processor.load_and_process_raw_file('synthetic.arw')

    assert image.shape == (3, 3, 3)
    _assert_valid_rgb(image)
    np.testing.assert_allclose(image, np.full((3, 3, 3), 12000 / 65535.0, dtype=np.float32))


def test_load_and_process_raw_file_keeps_the_frame_when_no_inset_crop(monkeypatch) -> None:
    """Without an inset crop the loader must leave the frame untouched."""

    raw_image = np.full((4, 4, 3), 12000, dtype=np.uint16)
    sizes = SimpleNamespace(
        crop_width=0, crop_height=0, crop_top_margin=0, crop_left_margin=0,
        width=4, height=4,
    )
    _patch_rawpy(monkeypatch, raw_image, sizes)

    image = raw_file_processor.load_and_process_raw_file('synthetic.nef')

    assert image.shape == (4, 4, 3)
    _assert_valid_rgb(image)


def test_active_area_crop_ignores_a_box_that_does_not_fit(monkeypatch) -> None:
    """A crop box larger than what postprocess returned must be ignored."""

    raw_image = np.full((4, 4, 3), 12000, dtype=np.uint16)
    sizes = SimpleNamespace(
        crop_width=9, crop_height=9, crop_top_margin=0, crop_left_margin=0,
        width=9, height=9,
    )
    _patch_rawpy(monkeypatch, raw_image, sizes)

    image = raw_file_processor.load_and_process_raw_file('synthetic.nef')

    assert image.shape == (4, 4, 3)

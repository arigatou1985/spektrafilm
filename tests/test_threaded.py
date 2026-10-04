"""Tests for the row-chunked thread pool."""

from __future__ import annotations

import numpy as np
import pytest

from spektrafilm.utils import threaded

pytestmark = pytest.mark.unit


def _per_pixel(array: np.ndarray) -> np.ndarray:
    """A non-linear, per-pixel kernel with no image-wide statistics."""

    values = np.asarray(array, dtype=np.float64)
    return values * 2.0 ** np.abs(values) + 1.0


def test_map_rows_matches_a_direct_call() -> None:
    """Chunking a per-pixel kernel must not change a single value."""

    array = np.random.default_rng(0).random((256, 64, 3))
    assert threaded.map_rows(_per_pixel, array).shape == array.shape
    np.testing.assert_array_equal(threaded.map_rows(_per_pixel, array), _per_pixel(array))


def test_map_rows_leaves_flat_and_small_inputs_on_the_direct_path() -> None:
    """A flat ``(3,)`` sample and a tiny image must not be split.

    The first axis is only "rows" for arrays that have a trailing channel axis;
    a dispatcher that slices ``[:1]`` on a flat sample would corrupt it.
    """

    flat = np.array([1.0, -0.1, -0.05])
    np.testing.assert_array_equal(threaded.map_rows(_per_pixel, flat), _per_pixel(flat))

    tiny = np.random.default_rng(1).random((8, 4, 3))
    np.testing.assert_array_equal(threaded.map_rows(_per_pixel, tiny), _per_pixel(tiny))


def test_map_rows_with_one_worker_is_the_direct_path() -> None:
    array = np.random.default_rng(2).random((128, 32, 3))
    np.testing.assert_array_equal(
        threaded.map_rows(_per_pixel, array, workers=1), _per_pixel(array)
    )


def _cctf_encode(array: np.ndarray) -> np.ndarray:
    """A same-space CCTF encode: the call that toggles colour's global scale."""

    colour = pytest.importorskip("colour")
    return np.asarray(
        colour.RGB_to_RGB(
            array, "sRGB", "sRGB",
            apply_cctf_decoding=False, apply_cctf_encoding=True,
        )
    )


def test_map_rows_preserves_colours_domain_range_scale() -> None:
    """Chunking must not leak colour's process-global domain-range scale.

    ``colour`` saves and restores that global without a lock, so concurrent
    transfer functions can interleave the two and leave it flipped. A leaked
    scale then changes the branch taken by every later colour conversion in the
    process, so the pool has to put the caller's value back.
    """

    colour = pytest.importorskip("colour")
    array = np.random.default_rng(3).random((256, 32, 3))
    before = colour.get_domain_range_scale()

    threaded.map_rows(_cctf_encode, array, workers=4)

    assert colour.get_domain_range_scale() == before


def test_map_rows_does_not_poison_later_colour_conversions() -> None:
    """A chunked transfer function must not change later CAM16 results.

    This is the end-to-end shape of the bug the scale guard prevents:
    ``XYZ_to_CAM16UCS`` scales XYZ by 100 under the "reference" scale and not
    under "ignore", so a leaked scale moved every subsequent CAM16 result by
    tens of units.
    """

    colour = pytest.importorskip("colour")
    from spektrafilm.utils import gamut_compression

    xyz = np.random.default_rng(4).random((64, 3))
    xyz_w = gamut_compression._output_cs_whitepoint_xyz("sRGB")
    kwargs = {"XYZ_w": xyz_w, "L_A": gamut_compression._CAM16UCS_L_A,
              "Y_b": gamut_compression._CAM16UCS_Y_B}
    reference = np.asarray(colour.XYZ_to_CAM16UCS(xyz, **kwargs))

    array = np.random.default_rng(5).random((256, 32, 3))
    threaded.map_rows(_cctf_encode, array, workers=4)

    np.testing.assert_array_equal(np.asarray(colour.XYZ_to_CAM16UCS(xyz, **kwargs)), reference)


def test_worker_count_is_capped() -> None:
    """Threads share one memory bus, so the pool is capped rather than unbounded."""

    assert 1 <= threaded.worker_count() <= threaded.MAX_WORKERS

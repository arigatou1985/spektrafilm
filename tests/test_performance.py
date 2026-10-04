"""Performance regression tests.

These assert *ratios*, never absolute times: an absolute millisecond budget
would fail on a slower machine and pass on a broken one. What is pinned here is
the algorithmic shape of the hot paths, which is what actually regressed before.

Run them with::

    pytest tests/test_performance.py            # includes the slow ones
    pytest tests/test_performance.py -m benchmark
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from spektrafilm.model.grain import layer_particle_model

pytestmark = [pytest.mark.slow, pytest.mark.benchmark]

PLANE_SHAPE = (1000, 1000)


def _fastest_ms(call, *, repeats: int = 3) -> float:
    """Best-of-N wall time in milliseconds, after one warm-up call.

    The minimum is used rather than the mean: it is the sample least polluted by
    the scheduler and by other processes on the machine.
    """

    call()
    best = float('inf')
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def _grain_call(density_value: float, n_particles: float):
    plane = np.full(PLANE_SHAPE, density_value, dtype=np.float64)

    def call():
        return layer_particle_model(
            plane,
            density_max=2.2,
            n_particles_per_pixel=n_particles,
            grain_uniformity=0.98,
            seed=0,
            blur_particle=1.0,
            use_fast_stats=True,
        )

    return call


@pytest.mark.parametrize('n_particles', [20.0, 150.0, 600.0])
def test_grain_cost_does_not_blow_up_on_saturated_density(n_particles: float) -> None:
    """A fully exposed pixel must not cost orders of magnitude more than a mid one.

    The old Poisson -> Binomial chain inverted the binomial CDF numerically from
    ``k = 0``. At ``p -> 1`` the first term ``(1 - p) ** n`` underflows to zero,
    so the walk never reached the target and ran the full ``n`` trials for every
    pixel: measured 613 ms per megapixel against 3.7 ms for a mid density. The
    single-Poisson form is O(1) everywhere.
    """

    midway = _fastest_ms(_grain_call(1.0, n_particles))       # p ~ 0.45
    saturated = _fastest_ms(_grain_call(2.2, n_particles))    # p = 1 (clipped)

    assert saturated < midway * 5, (
        f'saturated density costs {saturated:.1f} ms vs {midway:.1f} ms at mid density '
        f'({saturated / midway:.1f}x) with n_particles={n_particles}'
    )


def test_grain_cost_does_not_scale_with_the_particle_count() -> None:
    """Particle count is a precision knob, not a cost knob.

    Cost used to grow with the number of particles because the per-pixel work was
    a trial loop / CDF walk over ``n``. Measured at a saturated density, where
    the old inversion walk was longest.
    """

    few = _fastest_ms(_grain_call(2.2, 20.0))
    many = _fastest_ms(_grain_call(2.2, 600.0))

    assert many < few * 4, (
        f'n=600 costs {many:.1f} ms vs {few:.1f} ms for n=20 ({many / few:.1f}x)'
    )

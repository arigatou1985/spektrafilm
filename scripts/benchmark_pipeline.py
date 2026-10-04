#!/usr/bin/env python3
"""Benchmark the simulation pipeline.

Reports wall time, CPU time (and therefore how many cores were busy), peak
memory and the per-stage breakdown for one or more resolutions. ``--hotspots``
also instruments the two functions that dominate the profile. ``--json``
archives a run so two revisions can be diffed.

Examples
--------
    scripts/benchmark_pipeline.py                       # 2.7 MP, GUI-like settings
    scripts/benchmark_pipeline.py --mp 12.2 --repeat 1  # one full-frame pass
    scripts/benchmark_pipeline.py --mp 2.7 12.2 --json run.json
    scripts/benchmark_pipeline.py --hotspots

Notes
-----
The timings are absolute and therefore machine specific; compare runs made on
the same machine. Settings are the runtime defaults plus the fast-path switches
the GUI forces, so the numbers reflect what a user actually waits for.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / 'img' / 'test' / 'portrait_leaves_32bit_linear_prophoto_rgb.tif'
DEFAULT_MP = 2.7

# The GUI forces these on in params_mapper._apply_settings.
FAST_PATH_SETTINGS = {
    'use_enlarger_lut': True,
    'use_scanner_lut': True,
    'lut_resolution': 17,
    'use_fast_stats': True,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Benchmark the spektrafilm simulation pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mp', type=float, nargs='+', default=[DEFAULT_MP],
                        help='target megapixels (the test image is rescaled to it)')
    parser.add_argument('--repeat', type=int, default=2,
                        help='timed runs per resolution (the last one is reported)')
    parser.add_argument('--image', type=Path, default=DEFAULT_IMAGE,
                        help='source image, rescaled to the requested megapixels')
    parser.add_argument('--hotspots', action='store_true',
                        help='also time the dominant inner functions')
    parser.add_argument('--json', type=Path, default=None,
                        help='write the results to this file as JSON')
    return parser.parse_args(argv)


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def peak_rss_gib() -> float:
    # ru_maxrss is bytes on macOS, kilobytes on Linux.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024 ** 3 if sys.platform == 'darwin' else value / 1024 ** 2


def instrument(module, name: str, records: dict[str, list[float]]) -> None:
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            records.setdefault(name, []).append(time.perf_counter() - start)

    setattr(module, name, wrapper)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from skimage.transform import resize
    from spektrafilm.model import grain as grain_module
    from spektrafilm.runtime.api import Simulator, digest_params, init_params
    from spektrafilm.runtime.stages import scanning as scanning_module
    from spektrafilm.utils.io import load_image_oiio
    from spektrafilm.utils.numba_warmup import warmup

    image = np.asarray(load_image_oiio(str(args.image))[..., :3], dtype=np.float64)
    warmup()

    def build_params():
        params = init_params(print_profile='kodak_portra_endura')
        params.io.input_cctf_decoding = False
        params.camera.auto_exposure = True
        params.enlarger.print_exposure_compensation = True
        params.film_render.grain.active = True
        params.print_render.glare.active = True
        for name, value in FAST_PATH_SETTINGS.items():
            setattr(params.settings, name, value)
        return params

    reports = []
    print(f'image: {args.image.name} {image.shape[:2]}  |  repeat={args.repeat}\n')
    header = (f'{"MP":>7}{"wall s":>9}{"cpu s":>9}{"cores":>8}'
              f'{"s/MP":>8}{"peak RSS GiB":>14}')
    print(header)

    for target_mp in args.mp:
        factor = (target_mp / (image.shape[0] * image.shape[1] / 1e6)) ** 0.5
        frame = image if abs(factor - 1.0) < 1e-9 else resize(
            image, (int(image.shape[0] * factor), int(image.shape[1] * factor)),
            preserve_range=True, anti_aliasing=True).astype(np.float64)
        megapixels = frame.shape[0] * frame.shape[1] / 1e6

        records: dict[str, list[float]] = {}
        if args.hotspots:
            instrument(grain_module, 'layer_particle_model', records)
            instrument(scanning_module, 'compress_rgb', records)

        simulator = Simulator(digest_params(build_params()))
        simulator.process(frame)  # warm caches for this shape

        for _ in range(max(args.repeat, 1)):
            records.clear()
            wall_start, cpu_start = time.perf_counter(), cpu_seconds()
            simulator.process(frame)
            wall = time.perf_counter() - wall_start
            cpu = cpu_seconds() - cpu_start

        print(f'{megapixels:>7.2f}{wall:>9.2f}{cpu:>9.2f}{cpu / wall:>8.2f}'
              f'{wall / megapixels:>8.2f}{peak_rss_gib():>14.2f}')
        print(simulator.format_timings())

        report = {
            'megapixels': megapixels,
            'wall_s': wall,
            'cpu_s': cpu,
            'cores_busy': cpu / wall,
            'peak_rss_gib': peak_rss_gib(),
            'stages': dict(simulator.get_timings()),
        }
        if records:
            print('hot functions:')
            report['hotspots'] = {}
            for name, times in records.items():
                total = sum(times)
                report['hotspots'][name] = {'calls': len(times), 'total_s': total,
                                            'max_s': max(times)}
                print(f'  {name:<24} calls={len(times):<3} total={total * 1000:8.1f} ms'
                      f'  each={[round(t * 1000, 1) for t in times]}')
        reports.append(report)
        print()

    if args.json is not None:
        args.json.write_text(json.dumps(reports, indent=2))
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

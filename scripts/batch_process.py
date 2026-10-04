#!/usr/bin/env python3
"""Batch-process a folder of images through the spektrafilm pipeline.

The script drives the same runtime path as the GUI, so for the same settings a
batch render matches what the GUI shows. Settings come from the first of these
that exists:

* ``--state FILE`` — a GUI project file (``Save current to file`` in the GUI).
  Use this when you have dialled in a look interactively.
* otherwise the GUI's saved default (``Save current as default``), which is
  exactly what the GUI itself starts from.
* otherwise built-in defaults.

``--film`` / ``--print`` override the profile selection on top of whichever of
the above applied. The fast-path switches the GUI forces (enlarger/scanner
LUTs, fast stats) are always enabled.

RAW files go through the GUI's RAW loader (rawpy -> linear ACES -> the state's
input colour space). Rendered files (TIFF/PNG/JPEG/EXR) are read as-is, so the
input colour space and CCTF decoding must describe what is actually in them --
override with ``--input-color-space`` / ``--input-cctf-decoding`` when needed.

Examples
--------
    # reproduce the GUI look exactly, 16-bit TIFF, into ./out
    python scripts/batch_process.py ~/Pictures/trip --state ~/my_look.json --out ./out

    # defaults, half resolution, JPEG
    python scripts/batch_process.py ./raw --out ./out --upscale 0.5 --format jpg

    # see what would happen without rendering anything
    python scripts/batch_process.py ./raw --out ./out --dry-run

Notes
-----
Images are processed one at a time on purpose. A 50 MP render peaks around
40 GiB of RAM, so running several in parallel would exhaust memory long before
it saved wall-clock time. There is no ``--jobs`` flag for that reason.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

RAW_EXTENSIONS = frozenset({
    '.3fr', '.arw', '.cr2', '.cr3', '.crw', '.dng', '.erf', '.fff', '.iiq',
    '.k25', '.kdc', '.mef', '.mos', '.mrw', '.nef', '.nrw', '.orf', '.pef',
    '.raf', '.raw', '.rw2', '.rwl', '.sr2', '.srf', '.srw', '.x3f',
})
IMAGE_EXTENSIONS = frozenset({'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.exr'})
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | IMAGE_EXTENSIONS

# The GUI runs inside napari's QApplication. Qt only appends the
# organisation/application name to the app-config location when a QApplication
# exists, so the saved default state ends up under .../napari/napari/. A plain
# script must reproduce that or it looks one directory up and finds nothing.
NAPARI_ORGANISATION = 'napari'
NAPARI_APPLICATION = 'napari'

# The GUI forces these on in params_mapper._apply_settings; mirror it so a
# batch render without --state costs the same and matches the GUI's output.
FAST_PATH_SETTINGS = {
    'use_enlarger_lut': True,
    'use_scanner_lut': True,
    'lut_resolution': 17,
    'use_fast_stats': True,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Batch-process images with the spektrafilm pipeline.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('inputs', nargs='+', type=Path,
                        help='files and/or directories to process')
    parser.add_argument('--out', type=Path, required=True,
                        help='output directory (created if missing)')
    parser.add_argument('--state', type=Path, default=None,
                        help='GUI project file to take the settings from')
    parser.add_argument('--film', default=None, help='film profile label')
    parser.add_argument('--print', dest='print_profile', default=None,
                        help='print paper profile label')
    parser.add_argument('--format', default='tif',
                        choices=['tif', 'tiff', 'png', 'jpg', 'jpeg', 'exr'],
                        help='output format')
    parser.add_argument('--bit-depth', type=int, default=16, choices=[8, 16, 32],
                        help='bit depth for TIFF/EXR (ignored for PNG/JPEG)')
    parser.add_argument('--upscale', type=float, default=None,
                        help='override io.upscale_factor (e.g. 0.5 for half size)')
    parser.add_argument('--input-color-space', default=None,
                        help='input colour space for rendered (non-RAW) files')
    parser.add_argument('--input-cctf-decoding', dest='input_cctf_decoding',
                        action='store_true', default=None,
                        help='treat rendered input as CCTF-encoded and decode it')
    parser.add_argument('--no-input-cctf-decoding', dest='input_cctf_decoding',
                        action='store_false', default=None,
                        help='treat rendered input as linear')
    parser.add_argument('--recursive', action='store_true',
                        help='recurse into subdirectories')
    parser.add_argument('--suffix', default='',
                        help='inserted before the output extension')
    parser.add_argument('--skip-existing', action='store_true',
                        help='leave outputs that already exist untouched')
    parser.add_argument('--dry-run', action='store_true',
                        help='list the planned work and exit')
    return parser.parse_args(argv)


def collect_inputs(inputs: list[Path], *, recursive: bool) -> list[Path]:
    """Expand directories into the supported files they contain, sorted."""

    found: list[Path] = []
    for entry in inputs:
        if entry.is_dir():
            pattern = '**/*' if recursive else '*'
            found.extend(
                path for path in sorted(entry.glob(pattern))
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        elif entry.is_file():
            found.append(entry)
        else:
            raise SystemExit(f'not found: {entry}')
    return found


def gui_default_state_file(default_gui_state_path) -> Path | None:
    """Path of the GUI's saved default state, or None when there is none.

    Qt only appends the organisation/application name to the app-config
    location while a QApplication exists. The GUI runs inside napari's, so its
    file lives under ``<config>/napari/napari/``; a bare script would compute
    ``<config>/gui_default_state.json`` instead and silently miss it.
    """

    candidate = Path(default_gui_state_path())
    if candidate.is_file():
        return candidate
    try:
        from PySide6.QtCore import QCoreApplication
        app = QCoreApplication.instance() or QCoreApplication([])
        app.setOrganizationName(NAPARI_ORGANISATION)
        app.setApplicationName(NAPARI_APPLICATION)
    except Exception:  # noqa: BLE001 - Qt missing or unusable, fall through
        return None
    candidate = Path(default_gui_state_path())
    return candidate if candidate.is_file() else None


def build_params(args, state, load_profile, build_params_from_state):
    """Build the runtime parameters for the batch."""

    params = build_params_from_state(state)
    if args.film is not None:
        params.film = load_profile(args.film)
    if args.print_profile is not None:
        params.print = load_profile(args.print_profile)
    if args.upscale is not None:
        params.io.upscale_factor = args.upscale
    if args.input_color_space is not None:
        params.io.input_color_space = args.input_color_space
    if args.input_cctf_decoding is not None:
        params.io.input_cctf_decoding = args.input_cctf_decoding

    return params


def load_input(path: Path, *, params, state, load_image_oiio, load_raw):
    """Load one input as float RGB, mirroring the GUI's two load paths."""

    if path.suffix.lower() in RAW_EXTENSIONS:
        raw_state = state.gui_only.load_raw if state is not None else None
        return load_raw(
            str(path),
            output_colorspace=params.io.input_color_space,
            output_cctf_encoding=params.io.input_cctf_decoding,
            white_balance=raw_state.white_balance if raw_state else 'as_shot',
            temperature=raw_state.temperature if raw_state else None,
            tint=raw_state.tint if raw_state else None,
            lens_correction=raw_state.lens_correction if raw_state else False,
        )
    return load_image_oiio(str(path))[..., :3]


def _candidate_interpreters() -> list[Path]:
    """Interpreters that may hold the spektrafilm dependencies, best first."""

    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    override = os.environ.get('SPEKTRAFILM_PYTHON')
    if override:
        candidates.append(Path(override))
    candidates.append(repo_root / '.venv' / 'bin' / 'python')
    candidates.append(Path.home() / '.venv' / 'bin' / 'python')
    return candidates


def ensure_environment() -> None:
    """Re-run under a virtualenv interpreter when this one lacks the deps.

    People reasonably type ``python scripts/batch_process.py`` and land on the
    system interpreter, which does not have spektrafilm installed. Rather than
    failing with ModuleNotFoundError, hand the process to an interpreter that
    does have it.
    """

    if importlib.util.find_spec('colour') is not None:
        return

    script = str(Path(__file__).resolve())
    if os.environ.get('SPEKTRAFILM_BATCH_REEXEC') != '1':
        for candidate in _candidate_interpreters():
            # Compare literal paths: a venv's bin/python is usually a symlink to
            # the base interpreter, so resolving would wrongly treat it as the
            # interpreter we are already running under.
            if not candidate.is_file() or str(candidate) == sys.executable:
                continue
            os.environ['SPEKTRAFILM_BATCH_REEXEC'] = '1'
            os.execv(str(candidate), [str(candidate), script, *sys.argv[1:]])

    raise SystemExit(
        'spektrafilm is not importable from this interpreter.\n'
        'Run the script with the project virtualenv, e.g.:\n'
        f'  ~/.venv/bin/python {Path(__file__).name} ...'
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    sources = collect_inputs(args.inputs, recursive=args.recursive)
    if not sources:
        print('no supported files found', file=sys.stderr)
        return 1

    suffix = f'{args.suffix}.{args.format}'
    outputs = [args.out / f'{path.stem}{suffix}' for path in sources]

    collisions = {path for path in outputs if outputs.count(path) > 1}
    if collisions:
        print('these inputs map to the same output, rename or split them:', file=sys.stderr)
        for path in sorted(collisions):
            print(f'  {path}', file=sys.stderr)
        return 2

    if args.dry_run:
        print(f'would process {len(sources)} file(s) into {args.out}:')
        for source, destination in zip(sources, outputs):
            note = ' (exists, would skip)' if args.skip_existing and destination.exists() else ''
            print(f'  {source} -> {destination}{note}')
        return 0

    # Imported lazily so --help and --dry-run stay instant.
    ensure_environment()
    import colour
    from spektrafilm import load_profile
    from spektrafilm.runtime.api import Simulator, digest_params
    from spektrafilm.utils.io import (
        load_image_oiio, read_image_metadata, save_image_oiio, write_image_metadata,
    )
    from spektrafilm.utils.numba_warmup import warmup
    from spektrafilm.utils.raw_file_processor import load_and_process_raw_file
    from spektrafilm_gui.params_mapper import build_params_from_state
    from spektrafilm_gui.persistence import default_gui_state_path, load_gui_state_from_path

    if args.state is not None:
        state = load_gui_state_from_path(args.state)
        settings_source = f'project file {args.state}'
    else:
        saved = gui_default_state_file(default_gui_state_path)
        if saved is not None:
            state = load_gui_state_from_path(saved)
            settings_source = f'GUI saved default ({saved})'
        else:
            from spektrafilm_gui.persistence import load_default_gui_state
            state = load_default_gui_state()
            settings_source = ('built-in defaults (no saved GUI default found; '
                               'pass --state FILE for a project file)')

    params = build_params(args, state, load_profile, build_params_from_state)

    # The GUI hardcodes "the output layer is CCTF-encoded" and converts to the
    # workflow's saving colour space at save time; mirror that exactly.
    output_color_space = params.io.output_color_space
    saving_color_space = state.simulation.workflow.saving_color_space
    saving_cctf_encoding = state.simulation.workflow.saving_cctf_encoding

    args.out.mkdir(parents=True, exist_ok=True)
    print(f'settings: {settings_source}')
    print(f'profiles: film={params.film.info.stock}  print={params.print.info.stock}')
    print(f'saving  : {saving_color_space}, cctf_encoding={saving_cctf_encoding}, '
          f'{args.format} bit_depth={args.bit_depth}')
    print(f'files   : {len(sources)}')

    warmup_started = time.perf_counter()
    warmup()
    print(f'warmup  : {time.perf_counter() - warmup_started:.1f} s')

    simulator = Simulator(digest_params(params))

    written = skipped = failed = 0
    batch_started = time.perf_counter()
    for index, (source, destination) in enumerate(zip(sources, outputs), start=1):
        label = f'[{index}/{len(sources)}] {source.name}'
        if args.skip_existing and destination.exists():
            print(f'{label} -> skipped (exists)')
            skipped += 1
            continue

        started = time.perf_counter()
        try:
            image = load_input(source, params=params, state=state,
                               load_image_oiio=load_image_oiio,
                               load_raw=load_and_process_raw_file)
            result = simulator.process(image)

            if (output_color_space != saving_color_space
                    or not saving_cctf_encoding):
                result = colour.RGB_to_RGB(
                    result, output_color_space, saving_color_space,
                    apply_cctf_decoding=True,
                    apply_cctf_encoding=saving_cctf_encoding,
                )

            save_image_oiio(str(destination), result, bit_depth=args.bit_depth,
                            color_space=saving_color_space,
                            cctf_encoding=saving_cctf_encoding)
            write_image_metadata(str(destination), read_image_metadata(str(source)),
                                 saving_color_space=saving_color_space,
                                 saving_cctf_encoding=saving_cctf_encoding)
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f'{label} -> FAILED: {type(exc).__name__}: {exc}', file=sys.stderr)
            failed += 1
            continue

        elapsed = time.perf_counter() - started
        size_mb = destination.stat().st_size / 1024 ** 2
        print(f'{label} -> {destination.name}  {elapsed:.1f} s  {size_mb:.1f} MiB')
        written += 1

    total = time.perf_counter() - batch_started
    print(f'\ndone: {written} written, {skipped} skipped, {failed} failed '
          f'in {total:.1f} s ({total / max(written, 1):.1f} s/file)')
    print(f'output: {args.out}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

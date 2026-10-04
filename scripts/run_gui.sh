#!/usr/bin/env bash
# Launch the spektrafilm interactive GUI (entry point: spektrafilm_gui.app:main).
#
# Usage:
#   scripts/run_gui.sh            run in the foreground
#   scripts/run_gui.sh --detach   run in the background, logging to the cache dir
#   scripts/run_gui.sh --check    resolve interpreter/paths and verify imports, then exit
#   scripts/run_gui.sh --help     show this message
#
# Environment overrides:
#   SPEKTRAFILM_PYTHON  Python interpreter to use
#                       (default: ./.venv/bin/python, ~/.venv/bin/python, then python3)
#   SPEKTRAFILM_CACHE   writable dir for matplotlib/numba caches and logs
#                       (default: ${TMPDIR:-/tmp}/spektrafilm-gui)
#
# Note: --detach uses nohup, which is enough in an ordinary terminal. If you
# launch this from a harness that reaps its own process tree, the child is
# killed when that call returns. On macOS you can hand the job to launchd
# instead; launchd may be denied read access to a checkout under ~/Downloads,
# so point it at the installed entry point rather than this script:
#   launchctl submit -l com.spektrafilm.gui -- "$HOME/.venv/bin/spektrafilm"
#   launchctl remove com.spektrafilm.gui   # to stop it

set -euo pipefail

usage() {
  cat <<'EOF'
Launch the spektrafilm interactive GUI (entry point: spektrafilm_gui.app:main).

Usage:
  scripts/run_gui.sh            run in the foreground
  scripts/run_gui.sh --detach   run in the background, logging to the cache dir
  scripts/run_gui.sh --check    resolve interpreter/paths and verify imports, then exit
  scripts/run_gui.sh --help     show this message

Environment overrides:
  SPEKTRAFILM_PYTHON  Python interpreter to use
                      (default: ./.venv/bin/python, ~/.venv/bin/python, then python3)
  SPEKTRAFILM_CACHE   writable dir for matplotlib/numba caches and logs
                      (default: ${TMPDIR:-/tmp}/spektrafilm-gui)
EOF
}

detach=0
check=0
for arg in "$@"; do
  case "$arg" in
    --detach) detach=1 ;;
    --check) check=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "run_gui.sh: unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# --- interpreter ---------------------------------------------------------
python=""
if [[ -n "${SPEKTRAFILM_PYTHON:-}" ]]; then
  python="$SPEKTRAFILM_PYTHON"
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  python="$repo_root/.venv/bin/python"
elif [[ -x "$HOME/.venv/bin/python" ]]; then
  python="$HOME/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  python="$(command -v python3)"
fi

if [[ -z "$python" || ! -x "$python" ]]; then
  echo "run_gui.sh: no usable Python interpreter (set SPEKTRAFILM_PYTHON)" >&2
  exit 1
fi

# Import the checkout from source even when the package is not pip-installed.
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

# --- writable caches -----------------------------------------------------
default_cache_root="${TMPDIR:-/tmp}"
default_cache_root="${default_cache_root%/}"
cache_dir="${SPEKTRAFILM_CACHE:-$default_cache_root/spektrafilm-gui}"
mkdir -p "$cache_dir/matplotlib" "$cache_dir/numba"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$cache_dir/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$cache_dir/numba}"

if (( check )); then
  echo "repo_root:  $repo_root"
  echo "python:     $python"
  echo "PYTHONPATH: $PYTHONPATH"
  echo "cache_dir:  $cache_dir"
  "$python" -c 'import spektrafilm_gui.app as app; print("import OK:", app.main)'
  exit 0
fi

# --- launch --------------------------------------------------------------
if (( detach )); then
  log="$cache_dir/gui.log"
  nohup "$python" -c 'from spektrafilm_gui.app import main; main()' >"$log" 2>&1 &
  echo "spektrafilm GUI started (pid $!); log: $log"
else
  exec "$python" -c 'from spektrafilm_gui.app import main; main()'
fi

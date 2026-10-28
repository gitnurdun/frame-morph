#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$root/lib:${LD_LIBRARY_PATH:-}"
export PATH="$HOME/.local/bin:$PATH"
if [[ ! -x "$root/bin/framemorph" ]]; then echo 'FrameMorph is not built. Run scripts/build-local.sh first.' >&2; exit 1; fi
cd "$root"
exec "$root/bin/framemorph" "$@"

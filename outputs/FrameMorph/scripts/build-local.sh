#!/usr/bin/env bash
# Isolated Ubuntu 24.04 build. No sudo, Conda changes, or Linux GPU driver install.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
tag="$(printf '%s' "$root" | sha256sum | cut -c1-12)"
cache="${FRAMEMORPH_BUILD_ROOT:-/var/tmp/framemorph-$tag}"
mkdir -p "$cache/debs/partial" "$cache/sysroot" "$cache/toolkit" "$cache/downloads"
if [[ ! -f "$cache/.deps-ready" ]]; then
  apt-get -y --download-only --no-install-recommends -o Debug::NoLocking=true -o Dir::Cache::archives="$cache/debs" install libopencv-core-dev libopencv-imgproc-dev libopencv-imgcodecs-dev libfltk1.3-dev nlohmann-json3-dev ninja-build libfreetype-dev
  for f in "$cache"/debs/*.deb; do dpkg-deb -x "$f" "$cache/sysroot"; done
  touch "$cache/.deps-ready"
fi
if [[ ! -f "$cache/.cuda-ready" ]]; then
  python3 - "$cache" <<'PY'
import hashlib,json,pathlib,sys,tarfile,urllib.request
cache=pathlib.Path(sys.argv[1]);base='https://developer.download.nvidia.com/compute/cuda/redist/'
manifest=json.load(urllib.request.urlopen(base+'redistrib_12.8.1.json',timeout=90))
for part in ('cuda_nvcc','cuda_cudart','cuda_cccl'):
    item=manifest[part]['linux-x86_64']; archive=cache/'downloads'/(part+'.tar.xz')
    if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest()!=item['sha256']:
        print('Downloading',part,flush=True);urllib.request.urlretrieve(base+item['relative_path'],archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=item['sha256']:raise RuntimeError('CUDA checksum mismatch')
    # NVIDIA archives contain a single top-level directory; retain relative symlinks.
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            name=pathlib.PurePosixPath(member.name)
            if len(name.parts)<=1:continue
            member.name=str(pathlib.PurePosixPath(*name.parts[1:]))
            if pathlib.PurePosixPath(member.name).is_absolute() or '..' in pathlib.PurePosixPath(member.name).parts:raise RuntimeError('Unsafe archive path')
            tar.extract(member,cache/'toolkit',filter='data')
PY
  ln -sfn lib "$cache/toolkit/lib64"
  touch "$cache/.cuda-ready"
fi
export PATH="$cache/toolkit/bin:$cache/sysroot/usr/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$cache/sysroot/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
cmake -S "$root" -B "$cache/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$cache/toolkit/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="${FRAMEMORPH_CUDA_ARCH:-86}" -DFRAMEMORPH_SYSROOT="$cache/sysroot" -DCMAKE_BUILD_RPATH="$cache/sysroot/usr/lib/x86_64-linux-gnu"
cmake --build "$cache/build" -j "${FRAMEMORPH_JOBS:-4}"
ctest --test-dir "$cache/build" --output-on-failure
python3 "$root/scripts/package-runtime.py" "$cache/build" "$root" "$cache/sysroot"
echo "Ready: bash '$root/run.sh'"

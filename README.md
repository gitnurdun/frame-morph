# FrameMorph

FrameMorph is a C++ and CUDA desktop animation editor for Ubuntu/WSLg. It turns image or authored keyframes into animated videos and provides shape drawing, mathematical expressions, updaters, image editing, LaTeX, SVG and source-code artwork. The desktop interface uses FLTK; OpenCV and FFmpeg support image processing and export.

The application source, full documentation, help, tests and build scripts are under [`outputs/FrameMorph`](outputs/FrameMorph/). That directory's README describes authoring and its optional LaTeX/SVG backends.

## Build

The source targets C++17, CUDA, FLTK, OpenCV, FreeType, nlohmann/json, CMake and Ninja. The isolated Ubuntu build script can fetch its own development dependencies and CUDA toolkit without changing a Conda environment or installing a Linux NVIDIA driver:

```bash
cd outputs/FrameMorph
bash scripts/build-local.sh
bash run.sh
```

See the application README and `scripts/build-local.sh` for details and optional backend requirements. GPU availability, FFmpeg, and WSLg are required for the corresponding rendering, export and desktop workflows.

The current source contains an experimental texture-minification change that has not completed its regression tests. The previously verified v0.8 binary and local projects are not included in this source-only repository. No project license has been specified; third-party license notices are retained in `outputs/FrameMorph/third-party-licenses/`.

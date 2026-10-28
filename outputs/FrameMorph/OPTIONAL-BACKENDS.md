# Optional artwork construction backends

The C++/CUDA timeline renderer does not depend on Python or LaTeX during playback/export of cached projects. Formula, code and SVG construction uses `scripts/media_render.py` to make portable transparent artwork.

- Fast math uses the existing Matplotlib/Mathtext and Pillow installation.
- Full LaTeX uses the existing TinyTeX `pdflatex` with `standalone`, `amsmath`, `amssymb` and `xcolor`, plus the existing Python PyMuPDF installation to rasterize the resulting PDF.
- SVG construction uses the same approved installed PyMuPDF backend.
- Code artwork uses existing Pillow plus Python standard-library tokenization/lexing. Source is never executed.

The user explicitly approved reusing installed TinyTeX and PyMuPDF for this feature. No package was installed; the Manim environment and NVIDIA driver were not changed. The default interpreter is the existing `~/anaconda3/bin/python` when present; override with `FRAMEMORPH_MATH_PYTHON`. The launcher adds the existing `~/.local/bin` tools to PATH; TinyTeX is also detected at `~/.TinyTeX/bin/x86_64-linux/pdflatex`.

PyMuPDF is offered under AGPL or commercial licensing. It is not bundled in this application archive. A redistributed/commercial product must satisfy the licenses of the components it uses; the choice of an installed optional backend is not a claim that distribution licensing has been resolved.

Full LaTeX accepts mathematical fragments, not arbitrary documents or custom preambles. File/shell commands are excluded from the intended input, shell escape is disabled, and a compile has a 20-second timeout. This is a desktop math tool rather than a sandbox for untrusted TeX. Helper jobs have a 35-second outer timeout, and cancelled previews terminate their own helper process group. The isolated temporary directory is removed after completion or error.

SVG source is limited to 2 MB. External resource references, scripts and foreignObject are rejected; supported SVG content is converted and rasterized. This does not promise full browser SVG/CSS/filter compatibility. Images are saved in project assets and source remains in metadata, so existing projects can reopen and export without rebuilding optional content.

The implementation uses the documented [PyMuPDF image-rendering API](https://pymupdf.readthedocs.io/en/latest/recipes-images.html) and [page pixmaps](https://pymupdf.readthedocs.io/en/latest/page.html). LaTeX process options follow [TeX Live guidance](https://www.tug.org/texlive/doc/texlive-en/texlive-en.html).
